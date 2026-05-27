from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

from src.features import fix_length, normalize_signal, read_signal_csv


def load_config(path: str | Path) -> dict:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a YAML mapping: {config_path}")
    return config


def scan_single_source_files(single_dir: str | Path) -> tuple[list[str], dict[str, list[Path]]]:
    root = Path(single_dir)
    if not root.exists():
        raise FileNotFoundError(
            f"Single-source data directory not found: {root}. "
            "Run python -m src.create_demo_single_data or add real CSV files."
        )
    if not root.is_dir():
        raise ValueError(f"single_dir must be a directory: {root}")

    class_dirs = sorted([path for path in root.iterdir() if path.is_dir()], key=lambda path: path.name)
    class_names: list[str] = []
    files_by_class: dict[str, list[Path]] = {}
    for class_dir in class_dirs:
        csv_files = sorted(class_dir.glob("*.csv"))
        if not csv_files:
            continue
        class_names.append(class_dir.name)
        files_by_class[class_dir.name] = csv_files

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
    return root


def make_mixed_sample(
    class_names: list[str],
    files_by_class: dict[str, list[Path]],
    signal_length: int,
    max_sources_per_mix: int,
    noise_std: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if max_sources_per_mix <= 0:
        raise ValueError(f"max_sources_per_mix must be positive, got {max_sources_per_mix}")

    num_classes = len(class_names)
    source_count = int(rng.integers(1, min(max_sources_per_mix, num_classes) + 1))
    selected_indices = rng.choice(num_classes, size=source_count, replace=False)

    mixed = np.zeros(signal_length, dtype=np.float32)
    label = np.zeros(num_classes, dtype=np.float32)

    for class_index in selected_indices:
        class_name = class_names[int(class_index)]
        class_files = files_by_class[class_name]
        csv_path = class_files[int(rng.integers(0, len(class_files)))]
        signal = read_signal_csv(csv_path)
        signal = fix_length(signal, signal_length, random_crop=True, rng=rng)
        signal = normalize_signal(signal)
        signal = np.roll(signal, int(rng.integers(0, signal_length)))
        signal = signal * float(rng.uniform(0.3, 1.5))
        mixed += signal.astype(np.float32, copy=False)
        label[int(class_index)] = 1.0

    if noise_std > 0:
        mixed += rng.normal(0.0, noise_std, size=signal_length).astype(np.float32)

    return normalize_signal(mixed), label


def synthesize_dataset(config: dict) -> None:
    data_config = config.get("data", {})
    single_dir = Path(data_config.get("single_dir", "data/single"))
    mixed_dir = Path(data_config.get("mixed_dir", "data/mixed"))
    signal_length = int(data_config.get("signal_length", 4096))
    num_samples = int(data_config.get("num_samples", 300))
    max_sources_per_mix = int(data_config.get("max_sources_per_mix", 3))
    noise_std = float(data_config.get("noise_std", 0.02))
    seed = int(config.get("seed", 42))

    class_names, files_by_class = scan_single_source_files(single_dir)
    counts = split_counts(
        num_samples,
        float(data_config.get("train_ratio", 0.8)),
        float(data_config.get("val_ratio", 0.1)),
    )
    output_root = prepare_output_dirs(mixed_dir)

    class_names_path = output_root / "class_names.json"
    with class_names_path.open("w", encoding="utf-8") as handle:
        json.dump(class_names, handle, ensure_ascii=False, indent=2)

    rng = np.random.default_rng(seed)
    for split, count in counts.items():
        for index in tqdm(range(count), desc=f"synthesizing {split}"):
            x, y = make_mixed_sample(
                class_names,
                files_by_class,
                signal_length,
                max_sources_per_mix,
                noise_std,
                rng,
            )
            np.save(output_root / split / "x" / f"{index:06d}.npy", x)
            np.save(output_root / split / "y" / f"{index:06d}.npy", y)

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
