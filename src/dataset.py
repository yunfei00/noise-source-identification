from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.features import compute_stft_feature, fix_length, normalize_signal, read_signal_csv

_SOURCE_NAME_RE = re.compile(r"source_\d+")
_UNKNOWN_GROUP_PREFIXES = ("unknown", "background")


def is_unknown_group(group: str) -> bool:
    return group.startswith(_UNKNOWN_GROUP_PREFIXES)


def parse_group_label(group: str, class_names: list[str]) -> np.ndarray:
    if is_unknown_group(group):
        return np.zeros(len(class_names), dtype=np.float32)
    source_names = set(_SOURCE_NAME_RE.findall(group))
    if not source_names or not (group.endswith("_only") or group.endswith("_mix")):
        raise ValueError(f"Cannot parse label from real-data group: {group}")
    unknown_sources = sorted(source_names.difference(class_names))
    if unknown_sources:
        raise ValueError(
            f"Group '{group}' contains sources not present in class_names: {', '.join(unknown_sources)}"
        )
    return np.asarray([1.0 if name in source_names else 0.0 for name in class_names], dtype=np.float32)


class MixedNoiseDataset(Dataset):
    """Dataset for synthesized mixed noise signals."""

    def __init__(self, mixed_dir: str | Path, split: str, config: dict):
        if split not in {"train", "val", "test"}:
            raise ValueError(f"split must be one of train/val/test, got {split}")

        self.root = Path(mixed_dir) / split
        self.x_dir = self.root / "x"
        self.y_dir = self.root / "y"
        if not self.x_dir.exists() or not self.y_dir.exists():
            raise FileNotFoundError(
                f"Missing split directories under {self.root}. "
                "Run python -m src.synthesize_mixed_dataset first."
            )

        self.x_files = sorted(self.x_dir.glob("*.npy"))
        self.y_files = sorted(self.y_dir.glob("*.npy"))
        if not self.x_files:
            raise ValueError(f"No x .npy files found in {self.x_dir}")
        if len(self.x_files) != len(self.y_files):
            raise ValueError(
                f"x/y file count mismatch for split {split}: "
                f"{len(self.x_files)} x files, {len(self.y_files)} y files"
            )

        self._configure_features(config)

    def _configure_features(self, config: dict) -> None:
        data_config = config.get("data", {})
        stft_config = config.get("stft", {})
        self.sample_rate = int(data_config.get("sample_rate", 1_000_000))
        self.signal_length = int(data_config.get("signal_length", 4096))
        self.nperseg = int(stft_config.get("nperseg", 256))
        self.noverlap = int(stft_config.get("noverlap", 128))
        self.target_freq_bins = int(stft_config.get("target_freq_bins", 128))
        self.target_time_bins = int(stft_config.get("target_time_bins", 64))

    def __len__(self) -> int:
        return len(self.x_files)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        signal = np.load(self.x_files[index]).astype(np.float32, copy=False)
        label = np.load(self.y_files[index]).astype(np.float32, copy=False)
        return self._to_tensors(signal, label)

    def _to_tensors(self, signal: np.ndarray, label: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        feature = compute_stft_feature(
            signal,
            sample_rate=self.sample_rate,
            nperseg=self.nperseg,
            noverlap=self.noverlap,
            target_freq_bins=self.target_freq_bins,
            target_time_bins=self.target_time_bins,
        )
        x = torch.from_numpy(feature).unsqueeze(0).float()
        y = torch.from_numpy(label).float()
        return x, y


class RealNoiseDataset(MixedNoiseDataset):
    """Dataset for recursive real training CSV folders with first-level group labels."""

    def __init__(self, real_dir: str | Path, class_names: list[str], config: dict):
        self.root = Path(real_dir)
        if not self.root.exists():
            raise FileNotFoundError(f"Real training directory not found: {self.root}")
        if not self.root.is_dir():
            raise ValueError(f"real_train dir must be a directory: {self.root}")
        self.class_names = class_names
        self.samples: list[tuple[Path, np.ndarray]] = []
        for group_dir in sorted((path for path in self.root.iterdir() if path.is_dir()), key=lambda p: p.name):
            csv_files = sorted(path for path in group_dir.rglob("*.csv") if path.is_file())
            if not csv_files:
                print(f"warning: no CSV files found recursively under real_train group: {group_dir}")
                continue
            label = parse_group_label(group_dir.name, class_names)
            self.samples.extend((csv_path, label) for csv_path in csv_files)
        if not self.samples:
            raise ValueError(f"No recursive real_train CSV files found under: {self.root}")
        self._configure_features(config)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        csv_path, label = self.samples[index]
        signal = read_signal_csv(csv_path)
        signal = fix_length(signal, self.signal_length)
        signal = normalize_signal(signal)
        return self._to_tensors(signal, label.astype(np.float32, copy=False))
