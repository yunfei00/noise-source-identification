from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.features import compute_stft_feature


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

        data_config = config.get("data", {})
        stft_config = config.get("stft", {})
        self.sample_rate = int(data_config.get("sample_rate", 1_000_000))
        self.nperseg = int(stft_config.get("nperseg", 256))
        self.noverlap = int(stft_config.get("noverlap", 128))
        self.target_freq_bins = int(stft_config.get("target_freq_bins", 128))
        self.target_time_bins = int(stft_config.get("target_time_bins", 64))

    def __len__(self) -> int:
        return len(self.x_files)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        signal = np.load(self.x_files[index]).astype(np.float32, copy=False)
        label = np.load(self.y_files[index]).astype(np.float32, copy=False)

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
