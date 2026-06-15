from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import f1_score
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler
from torch.utils.data import Dataset

from src.build_real_index import build_real_index, discover_class_names
from src.dataset import RealCsvDataset, SyntheticNpyDataset
from src.model_cnn import NoiseCNN
from src.split_real_dataset import split_real_dataset

DEFAULT_CLASS_NAMES = ["source_1", "source_3", "source_5"]


def load_config(path: str | Path) -> dict:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a YAML mapping: {config_path}")
    return config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")
    return device


def _configured_class_names(config: dict | None) -> list[str] | None:
    configured = (config or {}).get("data", {}).get("class_names")
    if configured is None:
        return None
    if not isinstance(configured, list) or not configured or not all(isinstance(name, str) for name in configured):
        raise ValueError("data.class_names must be a non-empty list of strings")
    return configured


def load_class_names(
    mixed_dir: str | Path,
    config: dict | None = None,
    *,
    prefer_config: bool = False,
) -> list[str]:
    configured = _configured_class_names(config)
    single_dir = Path((config or {}).get("data", {}).get("single_dir", "data/single"))
    single_class_names = discover_class_names(single_dir) if single_dir.exists() else []
    if prefer_config:
        if single_class_names:
            return single_class_names
        raise ValueError(f"No class_names found under real single_dir: {single_dir}")

    class_names_path = Path(mixed_dir) / "class_names.json"
    if class_names_path.exists():
        with class_names_path.open("r", encoding="utf-8") as handle:
            class_names = json.load(handle)
        if not isinstance(class_names, list) or not all(isinstance(name, str) for name in class_names):
            raise ValueError(f"class_names.json must be a list of strings: {class_names_path}")
        return class_names

    if configured is not None:
        return configured
    if single_class_names:
        return single_class_names
    return list(DEFAULT_CLASS_NAMES)



def _label_to_combo_key(label: np.ndarray) -> str:
    return "[" + ",".join(str(int(value)) for value in label.astype(int).tolist()) + "]"


def _dataset_labels(dataset: Dataset) -> np.ndarray:
    """Return labels for supported datasets without computing STFT features."""
    if isinstance(dataset, ConcatDataset):
        parts = [_dataset_labels(child) for child in dataset.datasets]
        return np.vstack(parts).astype(np.float32)
    if isinstance(dataset, RealCsvDataset):
        return np.vstack([np.asarray(sample["label"], dtype=np.float32) for sample in dataset.samples])
    if isinstance(dataset, SyntheticNpyDataset):
        return np.vstack([np.load(path).astype(np.float32, copy=False) for path in dataset.y_files])
    labels: list[np.ndarray] = []
    for _, y in dataset:
        labels.append(y.detach().cpu().numpy().astype(np.float32, copy=False))
    if not labels:
        raise ValueError("Cannot build sampler or loss weights from an empty dataset")
    return np.vstack(labels).astype(np.float32)


def build_label_balanced_sampler(dataset: Dataset, strategy: str, seed: int) -> WeightedRandomSampler | None:
    if strategy == "none":
        return None
    labels = _dataset_labels(dataset).astype(np.int32)
    if labels.ndim != 2 or labels.shape[0] == 0:
        raise ValueError("Expected a non-empty 2D label matrix for weighted sampling")
    if strategy == "label_combo":
        keys = [_label_to_combo_key(label) for label in labels]
        counts = Counter(keys)
        weights = [1.0 / counts[key] for key in keys]
    elif strategy == "source_balance":
        pos_counts = labels.sum(axis=0).astype(np.float64)
        neg_counts = labels.shape[0] - pos_counts
        pos_weights = labels.shape[0] / np.maximum(pos_counts, 1.0)
        neg_weights = labels.shape[0] / np.maximum(neg_counts, 1.0)
        sample_weights = labels * pos_weights + (1 - labels) * neg_weights
        weights = sample_weights.mean(axis=1).astype(np.float64).tolist()
    else:
        raise ValueError(f"Unsupported sample_weight_strategy: {strategy}")
    generator = torch.Generator()
    generator.manual_seed(seed)
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True, generator=generator)


def compute_pos_weight(labels: np.ndarray, class_names: list[str], device: torch.device) -> torch.Tensor:
    labels = labels.astype(np.float32)
    pos_counts = labels.sum(axis=0)
    neg_counts = labels.shape[0] - pos_counts
    pos_weight = neg_counts / np.maximum(pos_counts, 1.0)
    for class_name, pos_count, neg_count, weight in zip(class_names, pos_counts, neg_counts, pos_weight):
        print(f"{class_name} pos_count={int(pos_count)} neg_count={int(neg_count)} pos_weight={float(weight):.6g}")
    return torch.as_tensor(pos_weight, dtype=torch.float32, device=device)


class AsymmetricBCEWithLogitsLoss(nn.Module):
    def __init__(
        self,
        pos_weight: torch.Tensor | None = None,
        fp_penalty: float = 1.5,
        fn_penalty: float = 1.0,
    ) -> None:
        super().__init__()
        self.register_buffer("pos_weight", pos_weight if pos_weight is not None else None)
        self.fp_penalty = float(fp_penalty)
        self.fn_penalty = float(fn_penalty)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weight,
            reduction="none",
        )
        weights = torch.where(
            targets > 0.5,
            torch.full_like(targets, self.fn_penalty),
            torch.full_like(targets, self.fp_penalty),
        )
        return (loss * weights).mean()


def build_criterion(config: dict, train_labels: np.ndarray, class_names: list[str], device: torch.device) -> nn.Module:
    loss_config = config.get("loss", {})
    loss_type = str(loss_config.get("type", "bce"))
    pos_weight = None
    if bool(loss_config.get("use_pos_weight", False)):
        pos_weight = compute_pos_weight(train_labels, class_names, device)
    if loss_type == "bce":
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    if loss_type == "asymmetric_bce":
        return AsymmetricBCEWithLogitsLoss(
            pos_weight=pos_weight,
            fp_penalty=float(loss_config.get("fp_penalty", 1.5)),
            fn_penalty=float(loss_config.get("fn_penalty", 1.0)),
        )
    raise ValueError(f"Unsupported loss.type: {loss_type}")

def training_data_config(config: dict) -> tuple[str, float, float]:
    data_mode_config = config.get("training_data", {})
    mode = str(data_mode_config.get("mode", "synthetic_only"))
    if mode not in {"synthetic_only", "hybrid", "real_only"}:
        raise ValueError(f"training_data.mode must be synthetic_only, hybrid, or real_only; got {mode}")
    synthetic_ratio = float(data_mode_config.get("synthetic_ratio", 0.3 if mode == "hybrid" else 1.0))
    real_ratio = float(data_mode_config.get("real_ratio", 0.7 if mode == "hybrid" else 0.0))
    if mode == "synthetic_only":
        synthetic_ratio, real_ratio = 1.0, 0.0
    elif mode == "real_only":
        synthetic_ratio, real_ratio = 0.0, 1.0
    if synthetic_ratio < 0 or real_ratio < 0:
        raise ValueError("training_data synthetic_ratio and real_ratio must be non-negative")
    return mode, synthetic_ratio, real_ratio


def prepare_real_split(config: dict, class_names: list[str]) -> Path | None:
    mode, _, _ = training_data_config(config)
    real_data_config = config.get("real_data", {})
    legacy_split_config = config.get("real_split", {})
    enabled = bool(real_data_config.get("enabled", mode == "real_only" or legacy_split_config.get("enabled", False)))
    if not enabled and mode != "real_only":
        return None

    data_config = config.get("data", {})
    paths_config = config.get("paths", {})
    report_dir = Path(paths_config.get("report_dir", "outputs/reports"))
    single_dir = Path(real_data_config.get("single_dir", data_config.get("single_dir", "data/single")))
    combo_dir = Path(real_data_config.get("combo_dir", "data/real_dataset"))
    index_path = Path(real_data_config.get("index_file", report_dir / "real_dataset_index.csv"))
    split_path = Path(real_data_config.get("split_file", report_dir / "real_dataset_split.csv"))

    rebuild_real_files = mode == "real_only"

    if rebuild_real_files or not index_path.exists():
        if rebuild_real_files:
            print(f"real_only mode scans all real CSV files; rebuilding {index_path}")
        else:
            print(f"real dataset index not found; building {index_path}")
        build_real_index(
            single_dir=single_dir,
            combo_dir=combo_dir,
            output=index_path,
            class_names=class_names,
            include_single=True,
            include_combo=True,
        )

    if rebuild_real_files or not split_path.exists():
        if rebuild_real_files:
            print(f"real_only mode splits all indexed real samples; rebuilding {split_path}")
        else:
            print(f"real dataset split not found; building {split_path}")
        split_real_dataset(
            index=index_path,
            output=split_path,
            train_ratio=float(legacy_split_config.get("train_ratio", 0.7)),
            val_ratio=float(legacy_split_config.get("val_ratio", 0.15)),
            test_ratio=float(legacy_split_config.get("test_ratio", 0.15)),
            seed=int(legacy_split_config.get("seed", config.get("seed", 42))),
        )
    return split_path


def real_split_counts(split_path: str | Path | None) -> dict[str, int]:
    counts = {
        "total_real_samples": 0,
        "train_samples": 0,
        "val_samples": 0,
        "test_samples": 0,
        "single_samples": 0,
        "combo_samples": 0,
    }
    if split_path is None:
        return counts

    path = Path(split_path)
    if not path.exists():
        return counts

    source_root_counts: Counter[str] = Counter()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        split_counts: Counter[str] = Counter()
        for row in reader:
            split_counts[row.get("split", "")] += 1
            source_root_counts[row.get("source_root", "")] += 1

    counts["total_real_samples"] = sum(split_counts.values())
    counts["train_samples"] = int(split_counts.get("train", 0))
    counts["val_samples"] = int(split_counts.get("val", 0))
    counts["test_samples"] = int(split_counts.get("test", 0))
    counts["single_samples"] = int(source_root_counts.get("single", 0))
    counts["combo_samples"] = int(source_root_counts.get("real_dataset", 0))
    return counts


def _try_synthetic_dataset(config: dict, split: str) -> SyntheticNpyDataset | None:
    data_config = config.get("data", {})
    try:
        return SyntheticNpyDataset(data_config.get("mixed_dir", "data/mixed"), split, config)
    except (FileNotFoundError, ValueError) as exc:
        print(f"warning: synthetic {split} dataset unavailable: {exc}")
        return None


def _try_real_dataset(
    config: dict,
    split: str,
    class_names: list[str],
    split_path: Path | None = None,
) -> RealCsvDataset | None:
    real_data_config = config.get("real_data", {})
    data_config = config.get("data", {})
    real_dir = real_data_config.get("dataset_root", ".")
    try:
        return RealCsvDataset(real_dir, class_names, config, split=split if split_path else None, index_path=split_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"warning: real {split} dataset unavailable: {exc}")
        return None


def _weighted_sampler(
    synthetic_len: int,
    real_len: int,
    synthetic_ratio: float,
    real_ratio: float,
    seed: int,
) -> WeightedRandomSampler:
    total = synthetic_len + real_len
    ratio_sum = synthetic_ratio + real_ratio
    synthetic_prob = synthetic_ratio / ratio_sum
    real_prob = real_ratio / ratio_sum
    weights = [synthetic_prob / synthetic_len] * synthetic_len + [real_prob / real_len] * real_len
    generator = torch.Generator()
    generator.manual_seed(seed)
    return WeightedRandomSampler(weights, num_samples=total, replacement=True, generator=generator)


def build_dataset_and_sampler(
    config: dict,
    split: str,
    class_names: list[str],
    split_path: Path | None = None,
) -> tuple[torch.utils.data.Dataset, WeightedRandomSampler | None, dict[str, int]]:
    mode, synthetic_ratio, real_ratio = training_data_config(config)
    stats = {"synthetic_samples": 0, "real_samples": 0}

    if mode == "synthetic_only":
        synthetic = SyntheticNpyDataset(config.get("data", {}).get("mixed_dir", "data/mixed"), split, config)
        stats["synthetic_samples"] = len(synthetic)
        return synthetic, None, stats

    if mode == "real_only":
        real = _try_real_dataset(config, split, class_names, split_path)
        if real is None:
            raise ValueError(f"real_only mode requires non-empty split={split} samples in real_data.split_file")
        stats["real_samples"] = len(real)
        sampler = None
        train_config = config.get("train", {})
        if split == "train" and bool(train_config.get("use_weighted_sampler", False)):
            sampler = build_label_balanced_sampler(
                real,
                str(train_config.get("sample_weight_strategy", "label_combo")),
                int(config.get("seed", 42)),
            )
        return real, sampler, stats

    if split not in {"train", "val"}:
        synthetic = SyntheticNpyDataset(config.get("data", {}).get("mixed_dir", "data/mixed"), split, config)
        stats["synthetic_samples"] = len(synthetic)
        return synthetic, None, stats

    synthetic = _try_synthetic_dataset(config, split)
    real = _try_real_dataset(config, split, class_names, split_path)
    if real is None:
        if synthetic is None:
            raise ValueError("hybrid mode requires at least one non-empty synthetic or real dataset")
        print("warning: hybrid mode found no real samples; falling back to synthetic_only")
        stats["synthetic_samples"] = len(synthetic)
        return synthetic, None, stats
    if synthetic is None:
        print("warning: hybrid mode found no synthetic samples; falling back to real_only")
        stats["real_samples"] = len(real)
        return real, None, stats

    stats["synthetic_samples"] = len(synthetic)
    stats["real_samples"] = len(real)
    dataset = ConcatDataset([synthetic, real])
    sampler = None
    if split == "train":
        train_config = config.get("train", {})
        if bool(train_config.get("use_weighted_sampler", False)):
            sampler = build_label_balanced_sampler(
                dataset,
                str(train_config.get("sample_weight_strategy", "label_combo")),
                int(config.get("seed", 42)),
            )
        else:
            sampler = _weighted_sampler(len(synthetic), len(real), synthetic_ratio, real_ratio, int(config.get("seed", 42)))
    return dataset, sampler, stats


def make_loader(
    config: dict,
    split: str,
    shuffle: bool,
    class_names: list[str] | None = None,
    real_split_path: str | Path | None = None,
) -> DataLoader:
    data_config = config.get("data", {})
    train_config = config.get("train", {})
    if class_names is None:
        class_names = load_class_names(data_config.get("mixed_dir", "data/mixed"), config)
    dataset, sampler, _ = build_dataset_and_sampler(
        config,
        split,
        class_names,
        Path(real_split_path) if real_split_path is not None else None,
    )

    return DataLoader(
        dataset,
        batch_size=int(train_config.get("batch_size", 32)),
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=int(train_config.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
    )

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_items = 0
    all_probs: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    with torch.set_grad_enabled(is_train):
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            batch_size = x.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_items += batch_size
            all_probs.append(torch.sigmoid(logits).detach().cpu().numpy())
            all_targets.append(y.detach().cpu().numpy())

    if total_items == 0:
        raise ValueError("DataLoader produced no batches")

    return total_loss / total_items, np.vstack(all_probs), np.vstack(all_targets)


def compute_validation_metrics(probs: np.ndarray, targets: np.ndarray, threshold: float) -> dict[str, float]:
    preds = (probs >= threshold).astype(np.int32)
    targets = targets.astype(np.int32)
    exact_matches = np.all(preds == targets, axis=1)
    over_predictions = np.any(preds > targets, axis=1)
    under_predictions = np.any(preds < targets, axis=1)
    return {
        "micro_f1": float(f1_score(targets, preds, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(targets, preds, average="macro", zero_division=0)),
        "exact_match": float(exact_matches.mean()) if len(exact_matches) else 0.0,
        "over_prediction_rate": float(over_predictions.mean()) if len(over_predictions) else 0.0,
        "under_prediction_rate": float(under_predictions.mean()) if len(under_predictions) else 0.0,
    }


class EarlyStopping:
    def __init__(self, mode: str = "max", patience: int = 15, min_delta: float = 0.001) -> None:
        if mode not in {"min", "max"}:
            raise ValueError(f"Early stopping mode must be 'min' or 'max', got {mode}")
        if patience < 1:
            raise ValueError(f"Early stopping patience must be >= 1, got {patience}")
        if min_delta < 0:
            raise ValueError(f"Early stopping min_delta must be >= 0, got {min_delta}")
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.best_metric: float | None = None
        self.bad_epochs = 0

    def step(self, metric: float) -> bool:
        if self.best_metric is None or self._is_improvement(metric):
            self.best_metric = metric
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return self.bad_epochs >= self.patience

    def _is_improvement(self, metric: float) -> bool:
        if self.best_metric is None:
            return True
        if self.mode == "max":
            return metric > self.best_metric + self.min_delta
        return metric < self.best_metric - self.min_delta


def save_training_history(path: str | Path, history: list[dict[str, float | int]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["epoch", "train_loss", "val_loss", "micro_f1", "macro_f1", "exact_match", "over_prediction_rate", "under_prediction_rate", "lr"]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    class_names: list[str],
    config: dict,
    epoch: int,
    best_metric: float,
) -> None:
    checkpoint = {
        "model_state": model.state_dict(),
        "class_names": class_names,
        "config": config,
        "epoch": epoch,
        "best_metric": best_metric,
    }
    torch.save(checkpoint, Path(path))


def train(config: dict) -> None:
    seed = int(config.get("seed", 42))
    set_seed(seed)

    data_config = config.get("data", {})
    train_config = config.get("train", {})
    early_stopping_config = config.get("early_stopping", {})
    scheduler_config = config.get("scheduler", {})
    paths_config = config.get("paths", {})

    mode, synthetic_ratio, real_ratio = training_data_config(config)
    class_names = load_class_names(
        data_config.get("mixed_dir", "data/mixed"),
        config,
        prefer_config=mode == "real_only",
    )
    device = resolve_device(str(train_config.get("device", "auto")))
    checkpoint_dir = Path(paths_config.get("checkpoint_dir", "outputs/checkpoints"))
    report_dir = Path(paths_config.get("report_dir", "outputs/reports"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    real_split_path = prepare_real_split(config, class_names)
    real_counts = real_split_counts(real_split_path)
    train_dataset, train_sampler, train_stats = build_dataset_and_sampler(config, "train", class_names, real_split_path)
    val_dataset, val_sampler, val_stats = build_dataset_and_sampler(config, "val", class_names, real_split_path)
    if val_sampler is not None:
        val_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(train_config.get("batch_size", 32)),
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=int(train_config.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(train_config.get("batch_size", 32)),
        shuffle=False,
        num_workers=int(train_config.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
    )

    model = NoiseCNN(num_classes=len(class_names)).to(device)
    train_labels = _dataset_labels(train_dataset)
    criterion = build_criterion(config, train_labels, class_names, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(train_config.get("lr", 1e-3)))
    threshold = float(train_config.get("threshold", 0.5))
    epochs = int(train_config.get("epochs", 200))

    scheduler = None
    if bool(scheduler_config.get("enabled", False)):
        scheduler_type = str(scheduler_config.get("type", "ReduceLROnPlateau"))
        if scheduler_type != "ReduceLROnPlateau":
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode=str(early_stopping_config.get("mode", "max")),
            factor=float(scheduler_config.get("factor", 0.5)),
            patience=int(scheduler_config.get("patience", 5)),
        )

    early_stopping = None
    if bool(early_stopping_config.get("enabled", False)):
        monitor = str(early_stopping_config.get("monitor", "micro_f1"))
        supported_monitors = {"val_loss", "micro_f1", "macro_f1", "exact_match", "over_prediction_rate", "under_prediction_rate"}
        if monitor not in supported_monitors:
            raise ValueError(f"Unsupported early stopping monitor: {monitor}")
        early_stopping = EarlyStopping(
            mode=str(early_stopping_config.get("mode", "max")),
            patience=int(early_stopping_config.get("patience", 15)),
            min_delta=float(early_stopping_config.get("min_delta", 0.001)),
        )

    synthetic_samples = train_stats["synthetic_samples"] + val_stats["synthetic_samples"]
    total_real_samples = real_counts["total_real_samples"] or train_stats["real_samples"] + val_stats["real_samples"]
    train_samples = real_counts["train_samples"] or train_stats["real_samples"]
    val_samples = real_counts["val_samples"] or val_stats["real_samples"]
    test_samples = real_counts["test_samples"]
    single_samples = real_counts["single_samples"]
    combo_samples = real_counts["combo_samples"]

    print(f"device={device}")
    print(f"training_data.mode={mode}")
    print(f"class_names={class_names}")
    print(f"total_samples={total_real_samples}")
    print(f"train_samples={train_samples}")
    print(f"val_samples={val_samples}")
    print(f"test_samples={test_samples}")
    print(f"single_samples={single_samples}")
    print(f"combo_samples={combo_samples}")
    print(f"batch_size={int(train_config.get('batch_size', 32))}")
    print(f"epochs={epochs}")
    print(f"synthetic_samples={synthetic_samples}")
    print(f"synthetic_ratio={synthetic_ratio}")
    print(f"real_ratio={real_ratio}")
    print(f"num_classes={len(class_names)}")

    best_metric = -1.0
    best_epoch = 0
    history: list[dict[str, float | int]] = []
    history_path = report_dir / "training_history.csv"
    for epoch in range(1, epochs + 1):
        train_loss, _, _ = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_probs, val_targets = run_epoch(model, val_loader, criterion, device)
        val_metrics = compute_validation_metrics(val_probs, val_targets, threshold)
        micro_f1 = val_metrics["micro_f1"]
        macro_f1 = val_metrics["macro_f1"]
        exact_match = val_metrics["exact_match"]
        over_prediction_rate = val_metrics["over_prediction_rate"]
        under_prediction_rate = val_metrics["under_prediction_rate"]
        monitor = str(early_stopping_config.get("monitor", "micro_f1"))
        current_metric = val_loss if monitor == "val_loss" else val_metrics[monitor]

        if (early_stopping_config.get("mode", "max") == "min" and (best_epoch == 0 or current_metric < best_metric)) or (early_stopping_config.get("mode", "max") != "min" and current_metric > best_metric):
            best_metric = current_metric
            best_epoch = epoch
            save_checkpoint(
                checkpoint_dir / "best.pt",
                model,
                class_names,
                config,
                epoch,
                best_metric,
            )

        if scheduler is not None:
            scheduler.step(current_metric)
        learning_rate = float(optimizer.param_groups[0]["lr"])

        save_checkpoint(
            checkpoint_dir / "last.pt",
            model,
            class_names,
            config,
            epoch,
            best_metric,
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "micro_f1": micro_f1,
                "macro_f1": macro_f1,
                "exact_match": exact_match,
                "over_prediction_rate": over_prediction_rate,
                "under_prediction_rate": under_prediction_rate,
                "lr": learning_rate,
            }
        )
        save_training_history(history_path, history)

        print(
            f"Epoch={epoch:03d} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"micro_f1={micro_f1:.4f} "
            f"macro_f1={macro_f1:.4f} "
            f"exact_match={exact_match:.4f} "
            f"over_prediction_rate={over_prediction_rate:.4f} "
            f"under_prediction_rate={under_prediction_rate:.4f} "
            f"learning_rate={learning_rate:.6g}"
        )

        if early_stopping is not None and early_stopping.step(current_metric):
            print("Early stopping triggered.")
            print(f"Best {monitor}={best_metric:.4f}")
            print(f"Best epoch={best_epoch}")
            break


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a multi-label noise source classifier.")
    parser.add_argument("--config", type=Path, default=Path("configs/train.yaml"), help="Path to YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train(load_config(args.config))


if __name__ == "__main__":
    main()
