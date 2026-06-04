from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import f1_score
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import ConcatDataset, DataLoader, Subset

from src.dataset import MixedNoiseDataset, RealNoiseDataset
from src.model_cnn import NoiseCNN


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


def load_class_names(mixed_dir: str | Path) -> list[str]:
    class_names_path = Path(mixed_dir) / "class_names.json"
    if not class_names_path.exists():
        raise FileNotFoundError(
            f"Missing {class_names_path}. Run python -m src.synthesize_mixed_dataset first."
        )
    with class_names_path.open("r", encoding="utf-8") as handle:
        class_names = json.load(handle)
    if not isinstance(class_names, list) or not all(isinstance(name, str) for name in class_names):
        raise ValueError(f"class_names.json must be a list of strings: {class_names_path}")
    return class_names


def make_loader(
    config: dict,
    split: str,
    shuffle: bool,
    class_names: list[str] | None = None,
) -> DataLoader:
    data_config = config.get("data", {})
    train_config = config.get("train", {})
    synthetic_dataset = MixedNoiseDataset(data_config.get("mixed_dir", "data/mixed"), split, config)
    dataset = synthetic_dataset

    real_train_config = config.get("real_train", {})
    if split == "train" and bool(real_train_config.get("enabled", False)):
        if class_names is None:
            class_names = load_class_names(data_config.get("mixed_dir", "data/mixed"))
        ratio = float(real_train_config.get("ratio", 0.2))
        if not 0.0 < ratio < 1.0:
            raise ValueError(f"real_train.ratio must be between 0 and 1, got {ratio}")
        real_dataset = RealNoiseDataset(real_train_config.get("dir", "data/real_train"), class_names, config)
        target_real_count = max(1, int(round(len(synthetic_dataset) * ratio / (1.0 - ratio))))
        rng = np.random.default_rng(int(config.get("seed", 42)))
        if target_real_count <= len(real_dataset):
            real_indices = rng.choice(len(real_dataset), size=target_real_count, replace=False).tolist()
        else:
            real_indices = rng.choice(len(real_dataset), size=target_real_count, replace=True).tolist()
        dataset = ConcatDataset([synthetic_dataset, Subset(real_dataset, real_indices)])
        print(
            f"real_train enabled: synthetic={len(synthetic_dataset)} "
            f"real_selected={target_real_count} ratio≈{target_real_count / len(dataset):.3f}"
        )

    return DataLoader(
        dataset,
        batch_size=int(train_config.get("batch_size", 32)),
        shuffle=shuffle,
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


def compute_f1(probs: np.ndarray, targets: np.ndarray, threshold: float) -> tuple[float, float]:
    preds = (probs >= threshold).astype(np.int32)
    targets = targets.astype(np.int32)
    micro = f1_score(targets, preds, average="micro", zero_division=0)
    macro = f1_score(targets, preds, average="macro", zero_division=0)
    return float(micro), float(macro)


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
    fieldnames = ["epoch", "train_loss", "val_loss", "micro_f1", "macro_f1", "lr"]
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

    class_names = load_class_names(data_config.get("mixed_dir", "data/mixed"))
    device = resolve_device(str(train_config.get("device", "auto")))
    checkpoint_dir = Path(paths_config.get("checkpoint_dir", "outputs/checkpoints"))
    report_dir = Path(paths_config.get("report_dir", "outputs/reports"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_loader = make_loader(config, "train", shuffle=True, class_names=class_names)
    val_loader = make_loader(config, "val", shuffle=False, class_names=class_names)

    model = NoiseCNN(num_classes=len(class_names)).to(device)
    criterion = nn.BCEWithLogitsLoss()
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
            mode="max",
            factor=float(scheduler_config.get("factor", 0.5)),
            patience=int(scheduler_config.get("patience", 5)),
        )

    early_stopping = None
    if bool(early_stopping_config.get("enabled", False)):
        monitor = str(early_stopping_config.get("monitor", "micro_f1"))
        if monitor != "micro_f1":
            raise ValueError(f"Unsupported early stopping monitor: {monitor}")
        early_stopping = EarlyStopping(
            mode=str(early_stopping_config.get("mode", "max")),
            patience=int(early_stopping_config.get("patience", 15)),
            min_delta=float(early_stopping_config.get("min_delta", 0.001)),
        )

    print(f"device={device}")
    print(f"classes={class_names}")

    best_metric = -1.0
    best_epoch = 0
    history: list[dict[str, float | int]] = []
    history_path = report_dir / "training_history.csv"
    for epoch in range(1, epochs + 1):
        train_loss, _, _ = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_probs, val_targets = run_epoch(model, val_loader, criterion, device)
        micro_f1, macro_f1 = compute_f1(val_probs, val_targets, threshold)

        if micro_f1 > best_metric:
            best_metric = micro_f1
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
            scheduler.step(micro_f1)
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
            f"learning_rate={learning_rate:.6g}"
        )

        if early_stopping is not None and early_stopping.step(micro_f1):
            print("Early stopping triggered.")
            print(f"Best micro_f1={best_metric:.4f}")
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
