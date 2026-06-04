from __future__ import annotations

import argparse
import itertools
import json
import shutil
import warnings
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

from src.features import fix_length, normalize_signal, read_signal_csv

DEFAULT_FREQUENCY = "default"


def load_config(path: str | Path) -> dict:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a YAML mapping: {config_path}")
    return config


def scan_single_source_files(
    single_dir: str | Path,
) -> tuple[list[str], dict[str, dict[str, list[Path]]]]:
    """Scan source classes while preserving their frequency-condition groups.

    CSV files directly under a source directory belong to the synthetic
    ``default`` condition. CSV files one directory deeper use that directory's
    name as their frequency condition. Deeper directories are intentionally not
    scanned because the supported layouts are explicit and predictable.
    """
    root = Path(single_dir)
    if not root.exists():
        raise FileNotFoundError(
            f"Single-source data directory not found: {root}. "
            "Run python -m src.create_demo_single_data or add real CSV files."
        )
    if not root.is_dir():
        raise ValueError(f"single_dir must be a directory: {root}")

    class_dirs = sorted((path for path in root.iterdir() if path.is_dir()), key=lambda path: path.name)
    class_names: list[str] = []
    files_by_class: dict[str, dict[str, list[Path]]] = {}
    for class_dir in class_dirs:
        files_by_frequency: dict[str, list[Path]] = {}

        direct_csv_files = sorted(class_dir.glob("*.csv"), key=lambda path: path.name)
        if direct_csv_files:
            files_by_frequency[DEFAULT_FREQUENCY] = direct_csv_files

        frequency_dirs = sorted(
            (path for path in class_dir.iterdir() if path.is_dir()),
            key=lambda path: path.name,
        )
        for frequency_dir in frequency_dirs:
            csv_files = sorted(frequency_dir.glob("*.csv"), key=lambda path: path.name)
            if csv_files:
                files_by_frequency[frequency_dir.name] = csv_files
            else:
                warnings.warn(f"Skipping empty frequency directory: {frequency_dir}", stacklevel=2)

        if not files_by_frequency:
            warnings.warn(f"Skipping source directory with no CSV files: {class_dir}", stacklevel=2)
            continue
        class_names.append(class_dir.name)
        files_by_class[class_dir.name] = files_by_frequency

    if not class_names:
        raise ValueError(f"No class subdirectories with CSV files found under: {root}")
    return class_names, files_by_class


def split_counts(num_samples: int, train_ratio: float, val_ratio: float) -> dict[str, int]:
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}")
    if train_ratio <= 0 or val_ratio < 0 or train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio must be > 0, val_ratio must be >= 0, and their sum must be < 1")

    train_count = int(num_samples * train_ratio)
    val_count = int(num_samples * val_ratio)
    test_count = num_samples - train_count - val_count
    if min(train_count, val_count, test_count) <= 0:
        raise ValueError(
            "Each split must contain at least one sample. "
            f"Got train={train_count}, val={val_count}, test={test_count}."
        )
    return {"train": train_count, "val": val_count, "test": test_count}


def prepare_output_dirs(mixed_dir: str | Path) -> Path:
    root = Path(mixed_dir)
    if root.exists():
        shutil.rmtree(root)
    for split in ("train", "val", "test"):
        (root / split / "x").mkdir(parents=True, exist_ok=True)
        (root / split / "y").mkdir(parents=True, exist_ok=True)
        (root / split / "metadata").mkdir(parents=True, exist_ok=True)
    return root


def relative_path_string(path: Path) -> str:
    """Return a portable relative path for metadata whenever possible."""
    try:
        relative_path = path.resolve().relative_to(Path.cwd().resolve())
    except ValueError:
        relative_path = (
            Path(*([".."] * len(Path.cwd().resolve().parts)))
            / path.resolve().relative_to(path.resolve().anchor)
        )
    return relative_path.as_posix()


def _float_pair(config: dict, key: str, default: tuple[float, float]) -> tuple[float, float]:
    value = config.get(key, default)
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"augmentation.{key} must contain exactly two numbers")
    low, high = float(value[0]), float(value[1])
    if low > high:
        raise ValueError(f"augmentation.{key} low value must be <= high value")
    return low, high


def scan_background_noise_files(config: dict) -> list[Path]:
    augmentation_config = config.get("augmentation", {})
    background_dir = Path(augmentation_config.get("background_noise_dir", "data/unknown/background_noise"))
    if not bool(augmentation_config.get("background_noise_enabled", False)) or not background_dir.exists():
        return []
    if not background_dir.is_dir():
        raise ValueError(f"background_noise_dir must be a directory: {background_dir}")
    return sorted(path for path in background_dir.rglob("*.csv") if path.is_file())


def apply_mix_normalization(signal: np.ndarray, mode: str) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float32)
    if mode == "none":
        return signal.astype(np.float32, copy=False)
    if mode == "zscore":
        return normalize_signal(signal)
    if mode == "rms":
        rms = float(np.sqrt(np.mean(np.square(signal)))) if signal.size else 0.0
        if rms < 1e-8:
            return signal.astype(np.float32, copy=False)
        return (signal / rms).astype(np.float32, copy=False)
    raise ValueError(f"Unsupported augmentation.mix_normalization: {mode}")


def add_snr_noise(signal: np.ndarray, snr_db_range: tuple[float, float], rng: np.random.Generator) -> np.ndarray:
    signal_power = float(np.mean(np.square(signal)))
    if signal_power < 1e-12:
        return signal
    snr_db = float(rng.uniform(snr_db_range[0], snr_db_range[1]))
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    noise = rng.normal(0.0, np.sqrt(noise_power), size=signal.shape).astype(np.float32)
    return (signal + noise).astype(np.float32, copy=False)


def make_mixed_sample(
    class_names: list[str],
    files_by_class: dict[str, dict[str, list[Path]]],
    signal_length: int,
    max_sources_per_mix: int,
    noise_std: float,
    rng: np.random.Generator,
    selected_indices: np.ndarray | None = None,
    augmentation_config: dict | None = None,
    background_noise_files: list[Path] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict], dict]:
    if max_sources_per_mix <= 0:
        raise ValueError(f"max_sources_per_mix must be positive, got {max_sources_per_mix}")

    augmentation_config = augmentation_config or {}
    augmentation_enabled = bool(augmentation_config.get("enabled", False))
    gain_range = _float_pair(augmentation_config, "gain_range", (0.1, 2.0)) if augmentation_enabled else (0.3, 1.5)
    noise_snr_db_range = _float_pair(augmentation_config, "noise_snr_db_range", (10.0, 40.0))
    dc_offset_range = _float_pair(augmentation_config, "dc_offset_range", (-0.05, 0.05))
    background_scale_range = _float_pair(
        augmentation_config, "background_noise_scale_range", (0.01, 0.2)
    )
    mix_normalization = str(augmentation_config.get("mix_normalization", "zscore" if not augmentation_enabled else "rms"))
    source_dropout_prob = float(augmentation_config.get("source_dropout_prob", 0.0))
    if not 0.0 <= source_dropout_prob < 1.0:
        raise ValueError(f"source_dropout_prob must be in [0, 1), got {source_dropout_prob}")

    num_classes = len(class_names)
    if selected_indices is None:
        source_count = int(rng.integers(1, min(max_sources_per_mix, num_classes) + 1))
        selected_indices = rng.choice(num_classes, size=source_count, replace=False)
    else:
        selected_indices = np.asarray(selected_indices, dtype=int)
        if selected_indices.ndim != 1 or not 1 <= len(selected_indices) <= max_sources_per_mix:
            raise ValueError("selected_indices must contain between 1 and max_sources_per_mix class indices")
        if len(np.unique(selected_indices)) != len(selected_indices):
            raise ValueError("selected_indices must not contain duplicates")
        if np.any(selected_indices < 0) or np.any(selected_indices >= num_classes):
            raise ValueError(f"selected_indices must be between 0 and {num_classes - 1}")

    mixed = np.zeros(signal_length, dtype=np.float32)
    label = np.zeros(num_classes, dtype=np.float32)
    sources: list[dict] = []
    kept_source_indices: list[int] = []

    for class_index in selected_indices:
        if source_dropout_prob > 0 and len(selected_indices) > 1 and rng.random() < source_dropout_prob:
            continue
        kept_source_indices.append(int(class_index))

    if not kept_source_indices:
        kept_source_indices = [int(selected_indices[int(rng.integers(0, len(selected_indices)))])]

    for class_index in kept_source_indices:
        class_name = class_names[int(class_index)]
        files_by_frequency = files_by_class[class_name]
        frequencies = sorted(files_by_frequency)
        frequency = frequencies[int(rng.integers(0, len(frequencies)))]
        frequency_files = files_by_frequency[frequency]
        csv_path = frequency_files[int(rng.integers(0, len(frequency_files)))]
        signal = read_signal_csv(csv_path)
        signal = fix_length(
            signal,
            signal_length,
            random_crop=bool(augmentation_config.get("random_crop", True)),
            rng=rng,
        )
        signal = normalize_signal(signal)
        gain = float(rng.uniform(gain_range[0], gain_range[1]))
        signal = signal * gain
        polarity = -1.0 if bool(augmentation_config.get("random_polarity_flip", False)) and rng.random() < 0.5 else 1.0
        signal = signal * polarity
        shift = 0
        if bool(augmentation_config.get("random_time_shift", True)):
            shift = int(rng.integers(0, signal_length))
            signal = np.roll(signal, shift)
        mixed += signal.astype(np.float32, copy=False)
        label[int(class_index)] = 1.0
        sources.append(
            {
                "source_name": class_name,
                "frequency": frequency,
                "file": relative_path_string(csv_path),
                "gain": gain,
                "polarity": polarity,
                "shift": shift,
            }
        )

    augmentations: dict = {"mix_normalization": mix_normalization}
    if augmentation_enabled and bool(augmentation_config.get("background_noise_enabled", False)) and background_noise_files:
        background_path = background_noise_files[int(rng.integers(0, len(background_noise_files)))]
        background = read_signal_csv(background_path)
        background = fix_length(background, signal_length, random_crop=True, rng=rng)
        background = normalize_signal(background)
        background_scale = float(rng.uniform(background_scale_range[0], background_scale_range[1]))
        mixed += background.astype(np.float32, copy=False) * background_scale
        augmentations["background_noise"] = {
            "file": relative_path_string(background_path),
            "scale": background_scale,
        }

    if augmentation_enabled:
        if bool(augmentation_config.get("random_dc_offset", False)):
            dc_offset = float(rng.uniform(dc_offset_range[0], dc_offset_range[1]))
            mixed += dc_offset
            augmentations["dc_offset"] = dc_offset
        mixed = add_snr_noise(mixed, noise_snr_db_range, rng)
        augmentations["snr_db_range"] = list(noise_snr_db_range)
    elif noise_std > 0:
        mixed += rng.normal(0.0, noise_std, size=signal_length).astype(np.float32)

    return apply_mix_normalization(mixed, mix_normalization), label, sources, augmentations

def make_balanced_multilabel_plan(
    num_samples: int,
    num_classes: int,
    max_sources_per_mix: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Build a shuffled plan balancing every non-empty source combination.

    For three classes and ``max_sources_per_mix == 3`` this yields the seven
    expected combinations: single-source labels, pair labels, and the full
    three-source label. Counts differ by at most one when ``num_samples`` is not
    divisible by the number of combinations.
    """
    if num_classes <= 0:
        raise ValueError(f"num_classes must be positive, got {num_classes}")
    if max_sources_per_mix <= 0:
        raise ValueError(f"max_sources_per_mix must be positive, got {max_sources_per_mix}")

    max_sources = min(max_sources_per_mix, num_classes)
    label_indices = [
        np.asarray(indices, dtype=int)
        for source_count in range(1, max_sources + 1)
        for indices in itertools.combinations(range(num_classes), source_count)
    ]
    counts = np.full(len(label_indices), num_samples // len(label_indices), dtype=int)
    counts[: num_samples % len(label_indices)] += 1
    plan = [indices for indices, count in zip(label_indices, counts) for _ in range(int(count))]
    rng.shuffle(plan)
    return plan


def synthesize_dataset(config: dict) -> None:
    data_config = config.get("data", {})
    single_dir = Path(data_config.get("single_dir", "data/single"))
    mixed_dir = Path(data_config.get("mixed_dir", "data/mixed"))
    signal_length = int(data_config.get("signal_length", 4096))
    num_samples = int(data_config.get("num_samples", 300))
    max_sources_per_mix = int(data_config.get("max_sources_per_mix", 3))
    balanced_generation = bool(data_config.get("balanced_generation", False))
    noise_std = float(data_config.get("noise_std", 0.02))
    seed = int(config.get("seed", 42))
    augmentation_config = config.get("augmentation", {})

    class_names, files_by_class = scan_single_source_files(single_dir)
    counts = split_counts(
        num_samples,
        float(data_config.get("train_ratio", 0.8)),
        float(data_config.get("val_ratio", 0.1)),
    )
    output_root = prepare_output_dirs(mixed_dir)

    rng = np.random.default_rng(seed)
    background_noise_files = scan_background_noise_files(config)
    if bool(augmentation_config.get("background_noise_enabled", False)) and not background_noise_files:
        print("warning: background_noise_enabled is true but no CSV files were found under data/unknown/background_noise")
    if balanced_generation and max_sources_per_mix < 1:
        raise ValueError("balanced_generation requires max_sources_per_mix >= 1")

    class_names_path = output_root / "class_names.json"
    with class_names_path.open("w", encoding="utf-8") as handle:
        json.dump(class_names, handle, ensure_ascii=False, indent=2)

    for split, count in counts.items():
        balanced_plan = (
            make_balanced_multilabel_plan(count, len(class_names), max_sources_per_mix, rng)
            if balanced_generation
            else None
        )
        for index in tqdm(range(count), desc=f"synthesizing {split}"):
            sample_id = f"mixed_{index:06d}"
            x, y, sources, augmentations = make_mixed_sample(
                class_names,
                files_by_class,
                signal_length,
                max_sources_per_mix,
                noise_std,
                rng,
                selected_indices=None if balanced_plan is None else balanced_plan[index],
                augmentation_config=augmentation_config,
                background_noise_files=background_noise_files,
            )
            np.save(output_root / split / "x" / f"{sample_id}.npy", x)
            np.save(output_root / split / "y" / f"{sample_id}.npy", y)
            metadata = {
                "sample_id": sample_id,
                "sources": sources,
                "label": y.astype(int).tolist(),
                "augmentations": augmentations,
            }
            metadata_path = output_root / split / "metadata" / f"{sample_id}.json"
            with metadata_path.open("w", encoding="utf-8") as handle:
                json.dump(metadata, handle, ensure_ascii=False, indent=2)

    print(f"Wrote mixed dataset to {output_root}")
    print(f"Classes: {', '.join(class_names)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthesize multi-label mixed noise samples.")
    parser.add_argument("--config", type=Path, default=Path("configs/train.yaml"), help="Path to YAML config.")
    parser.add_argument(
        "--mode",
        choices=("balanced_multilabel",),
        default="balanced_multilabel",
        help="Compatibility option; balanced multi-label generation is configured in YAML.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    synthesize_dataset(load_config(args.config))


if __name__ == "__main__":
    main()
