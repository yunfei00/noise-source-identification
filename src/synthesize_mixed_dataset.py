from __future__ import annotations

import argparse
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


def make_mixed_sample(
    class_names: list[str],
    files_by_class: dict[str, dict[str, list[Path]]],
    signal_length: int,
    max_sources_per_mix: int,
    noise_std: float,
    rng: np.random.Generator,
    selected_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    if max_sources_per_mix <= 0:
        raise ValueError(f"max_sources_per_mix must be positive, got {max_sources_per_mix}")

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

    for class_index in selected_indices:
        class_name = class_names[int(class_index)]
        files_by_frequency = files_by_class[class_name]
        frequencies = sorted(files_by_frequency)
        frequency = frequencies[int(rng.integers(0, len(frequencies)))]
        frequency_files = files_by_frequency[frequency]
        csv_path = frequency_files[int(rng.integers(0, len(frequency_files)))]
        signal = read_signal_csv(csv_path)
        signal = fix_length(signal, signal_length, random_crop=True, rng=rng)
        signal = normalize_signal(signal)
        gain = float(rng.uniform(0.3, 1.5))
        signal = signal * gain
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
                "shift": shift,
            }
        )

    if noise_std > 0:
        mixed += rng.normal(0.0, noise_std, size=signal_length).astype(np.float32)

    return normalize_signal(mixed), label, sources


def make_balanced_two_class_plan(num_samples: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Build a shuffled plan with approximately one third of each two-class label."""
    label_indices = (np.array([0]), np.array([1]), np.array([0, 1]))
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

    class_names, files_by_class = scan_single_source_files(single_dir)
    counts = split_counts(
        num_samples,
        float(data_config.get("train_ratio", 0.8)),
        float(data_config.get("val_ratio", 0.1)),
    )
    output_root = prepare_output_dirs(mixed_dir)

    rng = np.random.default_rng(seed)
    balanced_plan: list[np.ndarray] | None = None
    if balanced_generation and len(class_names) == 2:
        if max_sources_per_mix < 2:
            raise ValueError("balanced_generation requires max_sources_per_mix >= 2")
        balanced_plan = make_balanced_two_class_plan(num_samples, rng)
    elif balanced_generation:
        warnings.warn(
            "balanced_generation is only supported for exactly two source classes; "
            f"found {len(class_names)} classes, falling back to random generation.",
            stacklevel=2,
        )

    class_names_path = output_root / "class_names.json"
    with class_names_path.open("w", encoding="utf-8") as handle:
        json.dump(class_names, handle, ensure_ascii=False, indent=2)

    plan_index = 0
    for split, count in counts.items():
        for index in tqdm(range(count), desc=f"synthesizing {split}"):
            sample_id = f"mixed_{index:06d}"
            x, y, sources = make_mixed_sample(
                class_names,
                files_by_class,
                signal_length,
                max_sources_per_mix,
                noise_std,
                rng,
                selected_indices=None if balanced_plan is None else balanced_plan[plan_index],
            )
            plan_index += 1
            np.save(output_root / split / "x" / f"{sample_id}.npy", x)
            np.save(output_root / split / "y" / f"{sample_id}.npy", y)
            metadata = {
                "sample_id": sample_id,
                "sources": sources,
                "label": y.astype(int).tolist(),
            }
            metadata_path = output_root / split / "metadata" / f"{sample_id}.json"
            with metadata_path.open("w", encoding="utf-8") as handle:
                json.dump(metadata, handle, ensure_ascii=False, indent=2)

    print(f"Wrote mixed dataset to {output_root}")
    print(f"Classes: {', '.join(class_names)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthesize multi-label mixed noise samples.")
    parser.add_argument("--config", type=Path, default=Path("configs/train.yaml"), help="Path to YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    synthesize_dataset(load_config(args.config))


if __name__ == "__main__":
    main()
