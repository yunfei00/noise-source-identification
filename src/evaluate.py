from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader

from src.dataset import RealCsvDataset
from src.infer import load_checkpoint
from src.model_cnn import NoiseCNN
from src.train import make_loader, prepare_real_split, resolve_device

DEFAULT_THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7)



def normalize_thresholds(threshold: float | list[float] | np.ndarray, num_classes: int) -> np.ndarray:
    values = np.asarray(threshold, dtype=np.float32)
    if values.ndim == 0:
        return np.full(num_classes, float(values), dtype=np.float32)
    if values.shape != (num_classes,):
        raise ValueError(f"Expected {num_classes} threshold(s), got shape {values.shape}")
    return values


def thresholds_for_report(threshold: float | list[float] | np.ndarray, class_names: list[str]) -> float | dict[str, float]:
    values = normalize_thresholds(threshold, len(class_names))
    if np.allclose(values, values[0]):
        return float(values[0])
    return {class_name: float(value) for class_name, value in zip(class_names, values)}


def load_thresholds_json(path: str | Path, class_names: list[str]) -> np.ndarray:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if "thresholds" in payload and isinstance(payload["thresholds"], dict):
        payload = payload["thresholds"]
    return np.asarray([float(payload[class_name]) for class_name in class_names], dtype=np.float32)

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
    threshold: float | list[float] | np.ndarray,
) -> dict:
    """Compute per-class and aggregate multi-label metrics at one threshold."""
    if probs.shape != targets.shape:
        raise ValueError(f"Probability/target shape mismatch: {probs.shape} vs {targets.shape}")
    if probs.ndim != 2 or probs.shape[1] != len(class_names):
        raise ValueError(
            f"Expected predictions with {len(class_names)} classes, got shape {probs.shape}"
        )

    threshold_values = normalize_thresholds(threshold, len(class_names))
    preds = (probs >= threshold_values.reshape(1, -1)).astype(np.int32)
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
    exact_matches = np.all(preds == targets, axis=1)
    over_predictions = np.any(preds > targets, axis=1)
    under_predictions = np.any(preds < targets, axis=1)
    overall = {
        "micro_f1": float(f1_score(targets, preds, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(targets, preds, average="macro", zero_division=0)),
        "sample_f1": float(f1_score(targets, preds, average="samples", zero_division=0)),
        "exact_match": float(exact_matches.mean()) if len(exact_matches) else 0.0,
        "over_prediction_rate": float(over_predictions.mean()) if len(over_predictions) else 0.0,
        "under_prediction_rate": float(under_predictions.mean()) if len(under_predictions) else 0.0,
    }
    return {"threshold": thresholds_for_report(threshold_values, class_names), "per_class": per_class, "overall": overall}


def label_to_text(label: np.ndarray) -> str:
    return "[" + ",".join(str(int(value)) for value in label.tolist()) + "]"


def _ratio_from_condition(condition_path: str) -> str | None:
    for part in Path(condition_path).parts:
        if part.startswith("ratio_"):
            return part
    return None


def compute_real_breakdowns(
    probs: np.ndarray,
    targets: np.ndarray,
    groups: list[str],
    class_names: list[str],
    threshold: float | list[float] | np.ndarray,
    condition_paths: list[str] | None = None,
) -> dict:
    threshold_values = normalize_thresholds(threshold, len(class_names))
    preds = (probs >= threshold_values.reshape(1, -1)).astype(np.int32)
    exact = np.all(preds == targets, axis=1)
    group_accuracy: dict[str, dict[str, float | int]] = {}
    for group in sorted(set(groups)):
        indices = [index for index, value in enumerate(groups) if value == group]
        group_accuracy[group] = {
            "num_samples": len(indices),
            "accuracy": float(exact[indices].mean()) if indices else 0.0,
        }

    label_indices: dict[str, list[int]] = defaultdict(list)
    for index, target in enumerate(targets):
        label_indices[label_to_text(target)].append(index)
    label_accuracy = {
        label: {
            "num_samples": len(indices),
            "accuracy": float(exact[indices].mean()) if indices else 0.0,
        }
        for label, indices in sorted(label_indices.items())
    }

    precision, recall, f1, support = precision_recall_fscore_support(
        targets,
        preds,
        average=None,
        zero_division=0,
    )
    per_source = {
        class_name: {
            "precision": float(class_precision),
            "recall": float(class_recall),
            "f1": float(class_f1),
            "support": int(class_support),
        }
        for class_name, class_precision, class_recall, class_f1, class_support in zip(
            class_names, precision, recall, f1, support
        )
    }
    ratio_accuracy: dict[str, dict[str, float | int]] = {}
    if condition_paths is not None:
        ratio_indices: dict[str, list[int]] = defaultdict(list)
        for index, condition_path in enumerate(condition_paths):
            ratio = _ratio_from_condition(condition_path)
            if ratio is not None:
                ratio_indices[ratio].append(index)
        ratio_accuracy = {
            ratio: {
                "num_samples": len(indices),
                "accuracy": float(exact[indices].mean()) if indices else 0.0,
            }
            for ratio, indices in sorted(ratio_indices.items())
        }
    combo_confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    requested_true_labels = {"[0,1,0]", "[0,1,1]", "[1,1,0]", "[1,0,1]"}
    for target, pred in zip(targets, preds):
        true_label = label_to_text(target)
        if true_label in requested_true_labels:
            combo_confusion[true_label][label_to_text(pred)] += 1

    source_1_index = class_names.index("source_1") if "source_1" in class_names else 0
    no_source_1 = targets[:, source_1_index] == 0
    source_1_false_positive_rate = (
        float((preds[no_source_1, source_1_index] == 1).mean()) if np.any(no_source_1) else 0.0
    )
    true_source_counts = targets.sum(axis=1)
    true_double = true_source_counts == 2
    predicted_triple = preds.sum(axis=1) == len(class_names)
    double_to_triple_rate = float(predicted_triple[true_double].mean()) if np.any(true_double) else 0.0

    return {
        "threshold": thresholds_for_report(threshold_values, class_names),
        "group_accuracy": group_accuracy,
        "label_accuracy": label_accuracy,
        "per_source": per_source,
        "ratio_accuracy": ratio_accuracy,
        "combo_confusion": {true: dict(pred_counts) for true, pred_counts in sorted(combo_confusion.items())},
        "source_1_false_positive_rate": source_1_false_positive_rate,
        "double_source_predicted_as_triple_rate": double_to_triple_rate,
    }


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
            f"sample_f1={overall['sample_f1']:.4f} "
            f"exact_match={overall['exact_match']:.4f} "
            f"over_prediction_rate={overall['over_prediction_rate']:.4f} "
            f"under_prediction_rate={overall['under_prediction_rate']:.4f}"
        )


def print_real_breakdowns(breakdowns: dict) -> None:
    print("\nreal group accuracy:")
    for group, metrics in breakdowns["group_accuracy"].items():
        print(f"  {group}: accuracy={metrics['accuracy']:.4f} samples={metrics['num_samples']}")
    print("\nreal label accuracy:")
    for label, metrics in breakdowns["label_accuracy"].items():
        print(f"  {label}: accuracy={metrics['accuracy']:.4f} samples={metrics['num_samples']}")
    print("\nreal per-source metrics:")
    print(f"{'source':<16} {'precision':>10} {'recall':>10} {'f1':>10} {'support':>10}")
    for source_name, metrics in breakdowns["per_source"].items():
        print(
            f"{source_name:<16} "
            f"{metrics['precision']:>10.4f} "
            f"{metrics['recall']:>10.4f} "
            f"{metrics['f1']:>10.4f} "
            f"{metrics['support']:>10d}"
        )
    if breakdowns.get("ratio_accuracy"):
        print("\nreal ratio accuracy:")
        for ratio, metrics in breakdowns["ratio_accuracy"].items():
            print(f"  {ratio}: accuracy={metrics['accuracy']:.4f} samples={metrics['num_samples']}")
    print(f"\nsource_1_false_positive_rate={breakdowns.get('source_1_false_positive_rate', 0.0):.4f}")
    print(f"double_source_predicted_as_triple_rate={breakdowns.get('double_source_predicted_as_triple_rate', 0.0):.4f}")
    if breakdowns.get("combo_confusion"):
        print("\ncombo confusion:")
        for true_label, pred_counts in breakdowns["combo_confusion"].items():
            print(f"  true {true_label}: {pred_counts}")


def _real_loader_and_groups(config: dict, class_names: list[str], real_split: str) -> tuple[DataLoader, list[str], list[str]]:
    split_path = prepare_real_split(config, class_names)
    if split_path is None:
        split_path = Path(config.get("real_data", {}).get("split_file", Path(config.get("paths", {}).get("report_dir", "outputs/reports")) / "real_dataset_split.csv"))
    dataset = RealCsvDataset(
        config.get("real_data", {}).get("dataset_root", "."),
        class_names,
        config,
        split=real_split,
        index_path=split_path,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config.get("train", {}).get("batch_size", 32)),
        shuffle=False,
        num_workers=int(config.get("train", {}).get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
    )
    groups = [str(sample["group"]) for sample in dataset.samples]
    condition_paths = [str(sample.get("condition_path", "")) for sample in dataset.samples]
    return loader, groups, condition_paths



def write_error_analysis(
    path: str | Path,
    probs: np.ndarray,
    targets: np.ndarray,
    groups: list[str],
    condition_paths: list[str],
    class_names: list[str],
    threshold: float | list[float] | np.ndarray,
) -> None:
    threshold_values = normalize_thresholds(threshold, len(class_names))
    preds = (probs >= threshold_values.reshape(1, -1)).astype(np.int32)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "index",
        "group",
        "condition_path",
        "true_label",
        "pred_label",
        "exact_match",
        "over_prediction",
        "under_prediction",
    ]
    fieldnames.extend(f"prob_{name}" for name in class_names)
    fieldnames.extend(f"true_{name}" for name in class_names)
    fieldnames.extend(f"pred_{name}" for name in class_names)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, (prob, target, pred) in enumerate(zip(probs, targets, preds)):
            row = {
                "index": index,
                "group": groups[index] if index < len(groups) else "",
                "condition_path": condition_paths[index] if index < len(condition_paths) else "",
                "true_label": label_to_text(target),
                "pred_label": label_to_text(pred),
                "exact_match": int(np.all(pred == target)),
                "over_prediction": int(np.any(pred > target)),
                "under_prediction": int(np.any(pred < target)),
            }
            row.update({f"prob_{name}": f"{float(value):.10g}" for name, value in zip(class_names, prob)})
            row.update({f"true_{name}": int(value) for name, value in zip(class_names, target)})
            row.update({f"pred_{name}": int(value) for name, value in zip(class_names, pred)})
            writer.writerow(row)

def evaluate(
    model_path: str | Path,
    split: str = "test",
    device_name: str = "auto",
    report_path: str | Path | None = None,
    real_split: str | None = None,
    threshold: float | None = None,
    thresholds_json: str | Path | None = None,
) -> dict:
    """Evaluate a checkpoint on a synthesized split or a real dataset split."""
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

    requested_real_split = real_split
    eval_split = split
    groups: list[str] | None = None
    condition_paths: list[str] | None = None
    if requested_real_split is not None:
        loader, groups, condition_paths = _real_loader_and_groups(config, class_names, requested_real_split)
        eval_split = f"real_{requested_real_split}"
    else:
        loader = make_loader(config, split, shuffle=False, class_names=class_names)

    model = NoiseCNN(num_classes=len(class_names)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    probs, targets = collect_probabilities(model, loader, device)
    metrics_by_threshold = [
        compute_metrics(probs, targets, class_names, threshold) for threshold in DEFAULT_THRESHOLDS
    ]

    if thresholds_json is not None:
        selected_threshold = load_thresholds_json(thresholds_json, class_names)
    else:
        selected_threshold = float(config.get("train", {}).get("threshold", 0.5) if threshold is None else threshold)
    real_breakdowns = None
    if groups is not None:
        real_breakdowns = compute_real_breakdowns(probs, targets, groups, class_names, selected_threshold, condition_paths)

    output_path = Path(report_path or "outputs/reports/eval_report.json")
    report = {
        "model": str(model_path),
        "split": eval_split,
        "num_samples": int(targets.shape[0]),
        "class_names": class_names,
        "selected_threshold": thresholds_for_report(selected_threshold, class_names),
        "thresholds": metrics_by_threshold,
    }
    if real_breakdowns is not None:
        report["real_breakdowns"] = real_breakdowns
        write_error_analysis(
            Path(output_path).with_name("error_analysis.csv"),
            probs,
            targets,
            groups or [],
            condition_paths or [],
            class_names,
            selected_threshold,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(f"device={device}")
    print(f"model={model_path}")
    print(f"split={eval_split} samples={targets.shape[0]}")
    print_metrics(metrics_by_threshold)
    if real_breakdowns is not None:
        print_real_breakdowns(real_breakdowns)
    print(f"\nreport={output_path}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a multi-label noise source classifier.")
    parser.add_argument("--model", type=Path, required=True, help="Path to model checkpoint.")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test", help="Dataset split.")
    parser.add_argument("--real-split", choices=("train", "val", "test"), help="Evaluate a split from real_dataset_split.csv.")
    parser.add_argument("--device", default="auto", help="Device: auto, cpu, or cuda.")
    parser.add_argument("--report", type=Path, help="Report path (default: outputs/reports/eval_report.json).")
    parser.add_argument("--threshold", type=float, help="Threshold for real-data breakdowns (default: config train.threshold).")
    parser.add_argument("--thresholds-json", type=Path, help="Per-class thresholds JSON from search_thresholds.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate(args.model, args.split, args.device, args.report, args.real_split, args.threshold, args.thresholds_json)


if __name__ == "__main__":
    main()
