from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from src.evaluate import collect_probabilities, compute_metrics, _real_loader_and_groups
from src.infer import load_checkpoint
from src.model_cnn import NoiseCNN
from src.train import make_loader, resolve_device

METRIC_KEYS = ("exact_match", "micro_f1", "macro_f1", "sample_f1")


def threshold_values(start: float, end: float, step: float) -> list[float]:
    if step <= 0:
        raise ValueError("step must be positive")
    if end < start:
        raise ValueError("end must be greater than or equal to start")
    values: list[float] = []
    current = start
    epsilon = step / 1_000_000
    while current <= end + epsilon:
        values.append(round(current, 10))
        current += step
    return values


def _load_probs_and_targets(
    model_path: str | Path,
    split: str,
    real_split: str | None,
    device_name: str,
) -> tuple[np.ndarray, np.ndarray, list[str], str]:
    device = resolve_device(device_name)
    checkpoint = load_checkpoint(model_path, map_location=device)
    class_names = checkpoint.get("class_names")
    config = checkpoint.get("config")
    if not isinstance(class_names, list) or not class_names or not all(isinstance(name, str) for name in class_names):
        raise ValueError("Checkpoint is missing valid class_names")
    if not isinstance(config, dict):
        raise ValueError("Checkpoint is missing config")

    if real_split is not None:
        loader, _, _ = _real_loader_and_groups(config, class_names, real_split)
        eval_split = f"real_{real_split}"
    else:
        loader = make_loader(config, split, shuffle=False, class_names=class_names)
        eval_split = split

    model = NoiseCNN(num_classes=len(class_names)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    probs, targets = collect_probabilities(model, loader, device)
    return probs, targets, class_names, eval_split


def search_thresholds(
    model_path: str | Path,
    split: str = "test",
    real_split: str | None = None,
    metric: str = "exact_match",
    start: float = 0.3,
    end: float = 0.8,
    step: float = 0.05,
    output: str | Path = "outputs/reports/threshold_search.csv",
    device_name: str = "auto",
) -> list[dict[str, str]]:
    if metric not in METRIC_KEYS:
        raise ValueError(f"metric must be one of {', '.join(METRIC_KEYS)}")

    probs, targets, class_names, eval_split = _load_probs_and_targets(model_path, split, real_split, device_name)
    rows: list[dict[str, str]] = []
    for threshold in threshold_values(start, end, step):
        metrics = compute_metrics(probs, targets, class_names, threshold)
        overall = metrics["overall"]
        row = {
            "threshold": f"{threshold:.10g}",
            "metric": metric,
            "metric_value": f"{overall[metric]:.10g}",
            "micro_f1": f"{overall['micro_f1']:.10g}",
            "macro_f1": f"{overall['macro_f1']:.10g}",
            "sample_f1": f"{overall['sample_f1']:.10g}",
            "exact_match": f"{overall['exact_match']:.10g}",
            "over_prediction_rate": f"{overall['over_prediction_rate']:.10g}",
            "under_prediction_rate": f"{overall['under_prediction_rate']:.10g}",
        }
        rows.append(row)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "threshold",
        "metric",
        "metric_value",
        "micro_f1",
        "macro_f1",
        "sample_f1",
        "exact_match",
        "over_prediction_rate",
        "under_prediction_rate",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    best = max(rows, key=lambda row: float(row["metric_value"])) if rows else None
    print(f"model={model_path}")
    print(f"split={eval_split} samples={len(targets)}")
    print(f"metric={metric}")
    if best is not None:
        print(f"best_threshold={best['threshold']} best_{metric}={best['metric_value']}")
    print(f"output={output_path}")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search multi-label decision thresholds for a trained checkpoint.")
    parser.add_argument("--model", type=Path, required=True, help="Path to model checkpoint.")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test", help="Synthetic dataset split.")
    parser.add_argument("--real-split", choices=("train", "val", "test"), help="Evaluate a real_dataset_split.csv split.")
    parser.add_argument("--metric", choices=METRIC_KEYS, default="exact_match", help="Metric to maximize.")
    parser.add_argument("--start", type=float, default=0.3, help="Inclusive start threshold.")
    parser.add_argument("--end", type=float, default=0.8, help="Inclusive end threshold.")
    parser.add_argument("--step", type=float, default=0.05, help="Threshold step size.")
    parser.add_argument("--output", type=Path, default=Path("outputs/reports/threshold_search.csv"), help="Output CSV path.")
    parser.add_argument("--device", default="auto", help="Device: auto, cpu, or cuda.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    search_thresholds(
        model_path=args.model,
        split=args.split,
        real_split=args.real_split,
        metric=args.metric,
        start=args.start,
        end=args.end,
        step=args.step,
        output=args.output,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
