from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader

from src.dataset import MixedNoiseDataset
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


def make_loader(config: dict, split: str, shuffle: bool) -> DataLoader:
    data_config = config.get("data", {})
    train_config = config.get("train", {})
    dataset = MixedNoiseDataset(data_config.get("mixed_dir", "data/mixed"), split, config)
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
    paths_config = config.get("paths", {})

    class_names = load_class_names(data_config.get("mixed_dir", "data/mixed"))
    device = resolve_device(str(train_config.get("device", "auto")))
    checkpoint_dir = Path(paths_config.get("checkpoint_dir", "outputs/checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_loader = make_loader(config, "train", shuffle=True)
    val_loader = make_loader(config, "val", shuffle=False)

    model = NoiseCNN(num_classes=len(class_names)).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(train_config.get("lr", 1e-3)))
    threshold = float(train_config.get("threshold", 0.5))
    epochs = int(train_config.get("epochs", 3))

    print(f"device={device}")
    print(f"classes={class_names}")

    best_metric = -1.0
    for epoch in range(1, epochs + 1):
        train_loss, _, _ = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_probs, val_targets = run_epoch(model, val_loader, criterion, device)
        micro_f1, macro_f1 = compute_f1(val_probs, val_targets, threshold)

        if micro_f1 > best_metric:
            best_metric = micro_f1
            save_checkpoint(
                checkpoint_dir / "best.pt",
                model,
                class_names,
                config,
                epoch,
                best_metric,
            )

        save_checkpoint(
            checkpoint_dir / "last.pt",
            model,
            class_names,
            config,
            epoch,
            best_metric,
        )

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"micro_f1={micro_f1:.4f} "
            f"macro_f1={macro_f1:.4f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a multi-label noise source classifier.")
    parser.add_argument("--config", type=Path, default=Path("configs/train.yaml"), help="Path to YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train(load_config(args.config))


if __name__ == "__main__":
    main()
