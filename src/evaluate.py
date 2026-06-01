from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_recall_fscore_support

from src.infer import load_checkpoint
from src.model_cnn import NoiseCNN
from src.train import make_loader, resolve_device

DEFAULT_THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7)


def collect_probabilities(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the model over a dataset split and return probabilities and labels."""
    model.eval()
    all_probs: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device))
            all_probs.append(torch.sigmoid(logits).cpu().numpy())
            all_targets.append(y.cpu().numpy())

    if not all_probs:
        raise ValueError("DataLoader produced no batches")
    return np.vstack(all_probs), np.vstack(all_targets).astype(np.int32)


def compute_metrics(
    probs: np.ndarray,
    targets: np.ndarray,
    class_names: list[str],
    threshold: float,
) -> dict:
    """Compute per-class and aggregate multi-label metrics at one threshold."""
    if probs.shape != targets.shape:
        raise ValueError(f"Probability/target shape mismatch: {probs.shape} vs {targets.shape}")
    if probs.ndim != 2 or probs.shape[1] != len(class_names):
        raise ValueError(
            f"Expected predictions with {len(class_names)} classes, got shape {probs.shape}"
        )

    preds = (probs >= threshold).astype(np.int32)
    precision, recall, f1, support = precision_recall_fscore_support(
        targets,
        preds,
        average=None,
        zero_division=0,
    )
    per_class = [
        {
            "class_name": class_name,
            "precision": float(class_precision),
            "recall": float(class_recall),
            "f1": float(class_f1),
            "support": int(class_support),
        }
        for class_name, class_precision, class_recall, class_f1, class_support in zip(
            class_names, precision, recall, f1, support
        )
    ]
    overall = {
        "micro_f1": float(f1_score(targets, preds, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(targets, preds, average="macro", zero_division=0)),
        "sample_f1": float(f1_score(targets, preds, average="samples", zero_division=0)),
    }
    return {"threshold": threshold, "per_class": per_class, "overall": overall}


def print_metrics(metrics_by_threshold: list[dict]) -> None:
    """Print readable per-class and aggregate metric tables."""
    for metrics in metrics_by_threshold:
        print(f"\nthreshold={metrics['threshold']:.1f}")
        print(f"{'class':<24} {'precision':>10} {'recall':>10} {'f1':>10} {'support':>10}")
        for class_metrics in metrics["per_class"]:
            print(
                f"{class_metrics['class_name']:<24} "
                f"{class_metrics['precision']:>10.4f} "
                f"{class_metrics['recall']:>10.4f} "
                f"{class_metrics['f1']:>10.4f} "
                f"{class_metrics['support']:>10d}"
            )
        overall = metrics["overall"]
        print(
            "overall: "
            f"micro_f1={overall['micro_f1']:.4f} "
            f"macro_f1={overall['macro_f1']:.4f} "
            f"sample_f1={overall['sample_f1']:.4f}"
        )


def evaluate(
    model_path: str | Path,
    split: str = "test",
    device_name: str = "auto",
    report_path: str | Path | None = None,
) -> dict:
    """Evaluate a checkpoint on a synthesized multi-label dataset split."""
    device = resolve_device(device_name)
    checkpoint = load_checkpoint(model_path, map_location=device)
    class_names = checkpoint.get("class_names")
    config = checkpoint.get("config")
    if not isinstance(class_names, list) or not class_names or not all(
        isinstance(name, str) for name in class_names
    ):
        raise ValueError("Checkpoint is missing valid class_names")
    if not isinstance(config, dict):
        raise ValueError("Checkpoint is missing config")

    loader = make_loader(config, split, shuffle=False)
    model = NoiseCNN(num_classes=len(class_names)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    probs, targets = collect_probabilities(model, loader, device)
    metrics_by_threshold = [
        compute_metrics(probs, targets, class_names, threshold) for threshold in DEFAULT_THRESHOLDS
    ]

    output_path = Path(report_path or "outputs/reports/eval_report.json")
    report = {
        "model": str(model_path),
        "split": split,
        "num_samples": int(targets.shape[0]),
        "class_names": class_names,
        "thresholds": metrics_by_threshold,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(f"device={device}")
    print(f"model={model_path}")
    print(f"split={split} samples={targets.shape[0]}")
    print_metrics(metrics_by_threshold)
    print(f"\nreport={output_path}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a multi-label noise source classifier.")
    parser.add_argument("--model", type=Path, required=True, help="Path to model checkpoint.")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test", help="Dataset split.")
    parser.add_argument("--device", default="auto", help="Device: auto, cpu, or cuda.")
    parser.add_argument("--report", type=Path, help="Report path (default: outputs/reports/eval_report.json).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate(args.model, args.split, args.device, args.report)


if __name__ == "__main__":
    main()
