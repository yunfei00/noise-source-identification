from __future__ import annotations

import csv
import hashlib
import json
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
    if not source_names or (group not in class_names and not group.endswith("_mix")):
        raise ValueError(f"Cannot parse label from real-data group: {group}")
    unknown_sources = sorted(source_names.difference(class_names))
    if unknown_sources:
        raise ValueError(
            f"Group '{group}' contains sources not present in class_names: {', '.join(unknown_sources)}"
        )
    return np.asarray([1.0 if name in source_names else 0.0 for name in class_names], dtype=np.float32)


def parse_label_text(label_text: str) -> np.ndarray:
    stripped = label_text.strip()
    if not (stripped.startswith("[") and stripped.endswith("]")):
        raise ValueError(f"Invalid label text: {label_text}")
    body = stripped[1:-1].strip()
    if not body:
        return np.asarray([], dtype=np.float32)
    return np.asarray([float(token.strip()) for token in body.split(",")], dtype=np.float32)


class SyntheticNpyDataset(Dataset):
    """Dataset for synthesized mixed noise signals stored as x/y .npy files."""

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


MixedNoiseDataset = SyntheticNpyDataset


class RealCsvDataset(SyntheticNpyDataset):
    """Lazy real CSV dataset backed by a unified real_dataset_split.csv or generated index."""

    def __init__(
        self,
        real_dir: str | Path,
        class_names: list[str],
        config: dict,
        split: str | None = None,
        index_path: str | Path | None = None,
    ):
        self.root = Path(real_dir)
        if not self.root.exists():
            raise FileNotFoundError(f"Real dataset root not found: {self.root}")
        if not self.root.is_dir():
            raise ValueError(f"real dataset root must be a directory: {self.root}")
        self.class_names = class_names
        self._configure_features(config)
        self.cache_config = config.get("cache", {})
        self.cache_enabled = bool(self.cache_config.get("enabled", False))
        self.cache_dir = Path(self.cache_config.get("dir", "data/cache_stft"))
        self.cache_rebuild = bool(self.cache_config.get("rebuild", False))
        self.feature_config_hash = self._feature_config_hash()

        self.samples = self._load_samples(index_path, split)
        if not self.samples:
            split_text = f" split={split}" if split else ""
            raise ValueError(f"No real CSV samples found under {self.root}{split_text}")

    def _load_samples(self, index_path: str | Path | None, split: str | None) -> list[dict[str, object]]:
        if index_path is not None and Path(index_path).exists():
            return self._load_samples_from_index(Path(index_path), split)
        if index_path is not None and split is not None:
            raise FileNotFoundError(f"Real split/index file not found: {index_path}")
        return self._scan_samples()

    def _load_samples_from_index(self, index_path: Path, split: str | None) -> list[dict[str, object]]:
        samples: list[dict[str, object]] = []
        with index_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if split is not None and row.get("split") != split:
                    continue
                label = parse_label_text(row["label"])
                if label.shape[0] != len(self.class_names):
                    raise ValueError(
                        f"Label length mismatch in {index_path}: {row['label']} vs class_names={self.class_names}"
                    )
                path_value = Path(row["file"])
                sample_path = path_value if path_value.exists() else self.root / path_value
                samples.append(
                    {
                        "path": sample_path,
                        "file": row["file"],
                        "source_root": row.get("source_root", ""),
                        "group": row.get("group", ""),
                        "condition_path": row.get("condition_path", ""),
                        "label": label,
                    }
                )
        return samples

    def _scan_samples(self) -> list[dict[str, object]]:
        samples: list[dict[str, object]] = []
        for group_dir in sorted((path for path in self.root.iterdir() if path.is_dir()), key=lambda p: p.name):
            csv_files = sorted(path for path in group_dir.rglob("*.csv") if path.is_file())
            if not csv_files:
                print(f"warning: no CSV files found recursively under real_dataset group: {group_dir}")
                continue
            try:
                label = parse_group_label(group_dir.name, self.class_names)
            except ValueError as exc:
                print(f"warning: skipping unparseable real_dataset group {group_dir.name}: {exc}")
                continue
            for csv_path in csv_files:
                samples.append(
                    {
                        "path": csv_path,
                        "file": csv_path.relative_to(self.root).as_posix(),
                        "source_root": "real_dataset",
                        "group": group_dir.name,
                        "condition_path": csv_path.relative_to(group_dir).parent.as_posix() if csv_path.relative_to(group_dir).parent.as_posix() != "." else "",
                        "label": label,
                    }
                )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        label = np.asarray(sample["label"], dtype=np.float32)
        feature = self._load_or_compute_feature(Path(sample["path"]), str(sample["file"]))
        x = torch.from_numpy(feature).unsqueeze(0).float()
        y = torch.from_numpy(label).float()
        return x, y

    def _feature_config_hash(self) -> str:
        payload = {
            "sample_rate": self.sample_rate,
            "signal_length": self.signal_length,
            "nperseg": self.nperseg,
            "noverlap": self.noverlap,
            "target_freq_bins": self.target_freq_bins,
            "target_time_bins": self.target_time_bins,
        }
        return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]

    def _cache_path(self, relative_file: str) -> Path:
        key = hashlib.sha1(f"{relative_file}|{self.feature_config_hash}".encode("utf-8")).hexdigest()
        return self.cache_dir / f"{key}.npy"

    def _load_or_compute_feature(self, csv_path: Path, relative_file: str) -> np.ndarray:
        if self.cache_enabled:
            cache_path = self._cache_path(relative_file)
            if cache_path.exists() and not self.cache_rebuild:
                return np.load(cache_path).astype(np.float32, copy=False)

        signal = read_signal_csv(csv_path)
        signal = fix_length(signal, self.signal_length)
        signal = normalize_signal(signal)
        feature = compute_stft_feature(
            signal,
            sample_rate=self.sample_rate,
            nperseg=self.nperseg,
            noverlap=self.noverlap,
            target_freq_bins=self.target_freq_bins,
            target_time_bins=self.target_time_bins,
        )

        if self.cache_enabled:
            cache_path = self._cache_path(relative_file)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, feature.astype(np.float32, copy=False))
        return feature


RealNoiseDataset = RealCsvDataset
