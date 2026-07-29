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
from src.model_cnn import build_model
from src.noise_source_runtime.device import resolve_device
from src.train import make_loader, prepare_real_split

DEFAULT_THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7)
EXPECTED_LABELS = (
    "[1,0,0]",
    "[0,1,0]",
    "[0,0,1]",
    "[1,1,0]",
    "[1,0,1]",
    "[0,1,1]",
    "[1,1,1]",
)
EXPECTED_GROUPS = (
    "source_1",
    "source_3",
    "source_5",
    "source_1_source_3_mix",
    "source_1_source_5_mix",
    "source_3_source_5_mix",
    "source_1_source_3_source_5_mix",
)


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


def _format_threshold(threshold: float | dict[str, float]) -> str:
    if isinstance(threshold, dict):
        return ",".join(f"{name}={value:.3g}" for name, value in threshold.items())
    return f"{threshold:.3g}"


def parse_thresholds_arg(text: str, class_names: list[str]) -> np.ndarray:
    values: dict[str, float] = {}
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid threshold item '{item}', expected name=value")
        name, value = item.split("=", 1)
        name = name.strip()
        if name not in class_names:
            raise ValueError(f"Unknown class in --thresholds: {name}")
        values[name] = float(value.strip())
    missing = [class_name for class_name in class_names if class_name not in values]
    if missing:
        raise ValueError(f"--thresholds is missing class(es): {', '.join(missing)}")
    return np.asarray([values[class_name] for class_name in class_names], dtype=np.float32)


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
            x = x.to(device)
            if bool(getattr(model, "auxiliary_heads", False)) and getattr(
                model, "prediction_mode", "multilabel"
            ) == "structured":
                outputs = model.forward_with_auxiliary(x)
                # Evaluation metrics for structured checkpoints use the final
                # legal-combination decode, preserving historical behavior.
                probabilities = model.decoded_labels_from_outputs(outputs)
            else:
                probabilities = torch.sigmoid(model(x))
            all_probs.append(probabilities.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    if not all_probs:
        raise ValueError("DataLoader produced no batches")
    return np.vstack(all_probs), np.vstack(all_targets).astype(np.int32)


def predictions_at_threshold(
    probs: np.ndarray,
    class_names: list[str],
    threshold: float | list[float] | np.ndarray,
) -> np.ndarray:
    threshold_values = normalize_thresholds(threshold, len(class_names))
    return (probs >= threshold_values.reshape(1, -1)).astype(np.int32)


def compute_metrics(
    probs: np.ndarray,
    targets: np.ndarray,
    class_names: list[str],
    threshold: float | list[float] | np.ndarray,
) -> dict:
    """Compute per-source and aggregate multi-label metrics at one threshold setting."""
    if probs.shape != targets.shape:
        raise ValueError(f"Probability/target shape mismatch: {probs.shape} vs {targets.shape}")
    if probs.ndim != 2 or probs.shape[1] != len(class_names):
        raise ValueError(
            f"Expected predictions with {len(class_names)} classes, got shape {probs.shape}"
        )

    threshold_values = normalize_thresholds(threshold, len(class_names))
    preds = predictions_at_threshold(probs, class_names, threshold_values)
    targets = targets.astype(np.int32)
    precision, recall, f1, support = precision_recall_fscore_support(
        targets,
        preds,
        average=None,
        zero_division=0,
    )

    per_class: list[dict[str, float | int | str]] = []
    per_source: dict[str, dict[str, float | int]] = {}
    for index, (class_name, class_precision, class_recall, class_f1, class_support) in enumerate(
        zip(class_names, precision, recall, f1, support)
    ):
        target_col = targets[:, index]
        pred_col = preds[:, index]
        true_positive = int(((target_col == 1) & (pred_col == 1)).sum())
        false_positive = int(((target_col == 0) & (pred_col == 1)).sum())
        true_negative = int(((target_col == 0) & (pred_col == 0)).sum())
        false_negative = int(((target_col == 1) & (pred_col == 0)).sum())
        negative_count = false_positive + true_negative
        positive_count = true_positive + false_negative
        false_positive_rate = float(false_positive / negative_count) if negative_count else 0.0
        false_negative_rate = float(false_negative / positive_count) if positive_count else 0.0
        source_metrics = {
            "threshold": float(threshold_values[index]),
            "precision": float(class_precision),
            "recall": float(class_recall),
            "f1": float(class_f1),
            "support": int(class_support),
            "positive_count": int(positive_count),
            "negative_count": int(negative_count),
            "true_positive": true_positive,
            "false_positive": false_positive,
            "true_negative": true_negative,
            "false_negative": false_negative,
            "false_positive_rate": false_positive_rate,
            "false_negative_rate": false_negative_rate,
        }
        per_source[class_name] = source_metrics
        per_class.append({"class_name": class_name, **source_metrics})

    exact_matches = np.all(preds == targets, axis=1)
    over_predictions = np.any(preds > targets, axis=1)
    under_predictions = np.any(preds < targets, axis=1)
    true_double = targets.sum(axis=1) == 2
    predicted_triple = preds.sum(axis=1) == 3
    double_to_triple_rate = float(predicted_triple[true_double].mean()) if np.any(true_double) else 0.0

    source5_over_prediction_rate = 0.0
    if "source_5" in class_names:
        source5_index = class_names.index("source_5")
        no_source5 = targets[:, source5_index] == 0
        source5_over_prediction_rate = (
            float((preds[no_source5, source5_index] == 1).mean()) if np.any(no_source5) else 0.0
        )

    overall = {
        "micro_f1": float(f1_score(targets, preds, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(targets, preds, average="macro", zero_division=0)),
        "sample_f1": float(f1_score(targets, preds, average="samples", zero_division=0)),
        "exact_match": float(exact_matches.mean()) if len(exact_matches) else 0.0,
        "over_prediction_rate": float(over_predictions.mean()) if len(over_predictions) else 0.0,
        "under_prediction_rate": float(under_predictions.mean()) if len(under_predictions) else 0.0,
        "double_source_predict_as_triple_rate": double_to_triple_rate,
        "double_source_predicted_as_triple_rate": double_to_triple_rate,
        "source5_over_prediction_rate": source5_over_prediction_rate,
        "source5_false_positive_rate": source5_over_prediction_rate,
    }
    return {
        "threshold": thresholds_for_report(threshold_values, class_names),
        "per_class": per_class,
        "per_source": per_source,
        "overall": overall,
    }


def label_to_text(label: np.ndarray | list[int] | list[float]) -> str:
    return "[" + ",".join(str(int(value)) for value in np.asarray(label).astype(int).tolist()) + "]"


def _ordered_keys(expected: tuple[str, ...], observed: set[str]) -> list[str]:
    keys = [key for key in expected if key in observed or key in expected]
    keys.extend(sorted(observed.difference(expected)))
    return keys


def compute_group_accuracy(preds: np.ndarray, targets: np.ndarray, groups: list[str]) -> dict[str, dict[str, float | int]]:
    exact = np.all(preds == targets, axis=1)
    observed = set(groups)
    group_accuracy: dict[str, dict[str, float | int]] = {}
    for group in _ordered_keys(EXPECTED_GROUPS, observed):
        indices = [index for index, value in enumerate(groups) if value == group]
        group_accuracy[group] = {
            "num_samples": len(indices),
            "exact_match_accuracy": float(exact[indices].mean()) if indices else 0.0,
        }
    return group_accuracy


def compute_label_accuracy(preds: np.ndarray, targets: np.ndarray) -> dict[str, dict[str, float | int]]:
    exact = np.all(preds == targets, axis=1)
    label_indices: dict[str, list[int]] = defaultdict(list)
    for index, target in enumerate(targets):
        label_indices[label_to_text(target)].append(index)
    labels = _ordered_keys(EXPECTED_LABELS, set(label_indices))
    return {
        label: {
            "num_samples": len(label_indices.get(label, [])),
            "exact_match_accuracy": float(exact[label_indices[label]].mean()) if label_indices.get(label) else 0.0,
        }
        for label in labels
    }


def _ratio_from_condition(condition_path: str) -> str | None:
    for part in Path(condition_path).parts:
        if part.startswith("ratio_"):
            return part
        if part.startswith("radio_"):
            return "ratio_" + part[len("radio_") :]
    return None


def compute_ratio_accuracy(
    preds: np.ndarray,
    targets: np.ndarray,
    condition_paths: list[str] | None,
) -> dict[str, dict[str, float | int]]:
    if condition_paths is None:
        return {}
    exact = np.all(preds == targets, axis=1)
    ratio_indices: dict[str, list[int]] = defaultdict(list)
    for index, condition_path in enumerate(condition_paths):
        ratio = _ratio_from_condition(condition_path)
        if ratio is not None:
            ratio_indices[ratio].append(index)
    return {
        ratio: {
            "num_samples": len(indices),
            "exact_match_accuracy": float(exact[indices].mean()) if indices else 0.0,
        }
        for ratio, indices in sorted(ratio_indices.items())
    }


def compute_combo_confusion(targets: np.ndarray, preds: np.ndarray) -> dict[str, dict[str, int]]:
    true_labels = [label_to_text(target) for target in targets]
    pred_labels = [label_to_text(pred) for pred in preds]
    true_order = _ordered_keys(EXPECTED_LABELS, set(true_labels))
    pred_order = _ordered_keys(EXPECTED_LABELS, set(pred_labels))
    matrix: dict[str, dict[str, int]] = {
        true_label: {pred_label: 0 for pred_label in pred_order}
        for true_label in true_order
    }
    for true_label, pred_label in zip(true_labels, pred_labels):
        matrix.setdefault(true_label, {label: 0 for label in pred_order})
        if pred_label not in matrix[true_label]:
            matrix[true_label][pred_label] = 0
        matrix[true_label][pred_label] += 1
    return matrix


def compute_source_1_source_3_errors(targets: np.ndarray, preds: np.ndarray) -> dict[str, float | int | dict[str, float]]:
    true_labels = [label_to_text(target) for target in targets]
    pred_labels = [label_to_text(pred) for pred in preds]
    indices = [index for index, label in enumerate(true_labels) if label == "[1,1,0]"]
    total = len(indices)
    requested_predictions = ("[1,1,1]", "[0,1,1]", "[1,0,1]", "[0,1,0]", "[1,0,0]")
    prediction_counts = {
        pred_label: sum(1 for index in indices if pred_labels[index] == pred_label)
        for pred_label in ("[1,1,0]", *requested_predictions)
    }
    prediction_rates = {
        pred_label: float(count / total) if total else 0.0
        for pred_label, count in prediction_counts.items()
    }
    return {
        "true_label": "[1,1,0]",
        "num_samples": total,
        "exact_match_accuracy": prediction_rates["[1,1,0]"],
        "predicted_as_[1,1,1]_rate": prediction_rates["[1,1,1]"],
        "predicted_as_[0,1,1]_rate": prediction_rates["[0,1,1]"],
        "predicted_as_[1,0,1]_rate": prediction_rates["[1,0,1]"],
        "predicted_as_[0,1,0]_rate": prediction_rates["[0,1,0]"],
        "predicted_as_[1,0,0]_rate": prediction_rates["[1,0,0]"],
        "prediction_counts": prediction_counts,
        "prediction_rates": prediction_rates,
    }


def compute_real_breakdowns(
    probs: np.ndarray,
    targets: np.ndarray,
    groups: list[str],
    class_names: list[str],
    threshold: float | list[float] | np.ndarray,
    condition_paths: list[str] | None = None,
) -> dict:
    threshold_values = normalize_thresholds(threshold, len(class_names))
    preds = predictions_at_threshold(probs, class_names, threshold_values)
    selected_metrics = compute_metrics(probs, targets, class_names, threshold_values)
    return {
        "threshold": thresholds_for_report(threshold_values, class_names),
        "group_accuracy": compute_group_accuracy(preds, targets, groups),
        "label_accuracy": compute_label_accuracy(preds, targets),
        "per_source": selected_metrics["per_source"],
        "ratio_accuracy": compute_ratio_accuracy(preds, targets, condition_paths),
        "combo_confusion": compute_combo_confusion(targets, preds),
        "double_source_predict_as_triple_rate": selected_metrics["overall"]["double_source_predict_as_triple_rate"],
        "source5_over_prediction_rate": selected_metrics["overall"]["source5_over_prediction_rate"],
        "source_1_source_3_mix_errors": compute_source_1_source_3_errors(targets, preds),
    }


def print_metrics(metrics_by_threshold: list[dict]) -> None:
    """Print readable per-class and aggregate metric tables."""
    for metrics in metrics_by_threshold:
        print(f"\nthreshold={_format_threshold(metrics['threshold'])}")
        print(
            f"{'class':<16} {'precision':>10} {'recall':>10} {'f1':>10} "
            f"{'fpr':>10} {'fnr':>10} {'support':>10}"
        )
        for class_metrics in metrics["per_class"]:
            print(
                f"{class_metrics['class_name']:<16} "
                f"{class_metrics['precision']:>10.4f} "
                f"{class_metrics['recall']:>10.4f} "
                f"{class_metrics['f1']:>10.4f} "
                f"{class_metrics['false_positive_rate']:>10.4f} "
                f"{class_metrics['false_negative_rate']:>10.4f} "
                f"{class_metrics['support']:>10d}"
            )
        overall = metrics["overall"]
        print(
            "overall: "
            f"micro_f1={overall['micro_f1']:.4f} "
            f"macro_f1={overall['macro_f1']:.4f} "
            f"sample_f1={overall['sample_f1']:.4f} "
            f"exact_match={overall['exact_match']:.4f} "
            f"source5_over_prediction_rate={overall['source5_over_prediction_rate']:.4f} "
            f"double_source_predict_as_triple_rate={overall['double_source_predict_as_triple_rate']:.4f}"
        )


def print_real_breakdowns(breakdowns: dict) -> None:
    print("\nreal group exact match accuracy:")
    for group, metrics in breakdowns["group_accuracy"].items():
        print(f"  {group}: accuracy={metrics['exact_match_accuracy']:.4f} samples={metrics['num_samples']}")
    print("\nreal label exact match accuracy:")
    for label, metrics in breakdowns["label_accuracy"].items():
        print(f"  {label}: accuracy={metrics['exact_match_accuracy']:.4f} samples={metrics['num_samples']}")
    print("\nreal per-source metrics:")
    print(f"{'source':<16} {'precision':>10} {'recall':>10} {'f1':>10} {'fpr':>10} {'fnr':>10}")
    for source_name, metrics in breakdowns["per_source"].items():
        print(
            f"{source_name:<16} "
            f"{metrics['precision']:>10.4f} "
            f"{metrics['recall']:>10.4f} "
            f"{metrics['f1']:>10.4f} "
            f"{metrics['false_positive_rate']:>10.4f} "
            f"{metrics['false_negative_rate']:>10.4f}"
        )
    source5_metrics = breakdowns["per_source"].get("source_5")
    if source5_metrics:
        print(
            "\nsource_5 focus: "
            f"false_positive_rate={source5_metrics['false_positive_rate']:.4f} "
            f"recall={source5_metrics['recall']:.4f} "
            f"precision={source5_metrics['precision']:.4f}"
        )
    if breakdowns.get("ratio_accuracy"):
        print("\nreal ratio exact match accuracy:")
        for ratio, metrics in breakdowns["ratio_accuracy"].items():
            print(f"  {ratio}: accuracy={metrics['exact_match_accuracy']:.4f} samples={metrics['num_samples']}")
    print(f"\nsource5_over_prediction_rate={breakdowns.get('source5_over_prediction_rate', 0.0):.4f}")
    print(f"double_source_predict_as_triple_rate={breakdowns.get('double_source_predict_as_triple_rate', 0.0):.4f}")
    print("\nsource_1_source_3_mix critical errors:")
    for key, value in breakdowns["source_1_source_3_mix_errors"].items():
        if isinstance(value, float):
            print(f"  {key}={value:.4f}")
        elif not isinstance(value, dict):
            print(f"  {key}={value}")
    if breakdowns.get("combo_confusion"):
        print("\ncombo confusion:")
        for true_label, pred_counts in breakdowns["combo_confusion"].items():
            print(f"  true {true_label}: {pred_counts}")


def _real_loader_and_groups(config: dict, class_names: list[str], real_split: str) -> tuple[DataLoader, list[str], list[str], list[str]]:
    split_path = prepare_real_split(config, class_names)
    if split_path is None:
        split_path = Path(
            config.get("real_data", {}).get(
                "split_file",
                Path(config.get("paths", {}).get("report_dir", "outputs/reports")) / "real_dataset_split.csv",
            )
        )
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
    files = [str(sample.get("file", "")) for sample in dataset.samples]
    groups = [str(sample["group"]) for sample in dataset.samples]
    condition_paths = [
        str(sample.get("condition_path", "")).strip() or file_path
        for sample, file_path in zip(dataset.samples, files)
    ]
    return loader, groups, condition_paths, files


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
    preds = predictions_at_threshold(probs, class_names, threshold_values)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "index",
        "group",
        "condition_path",
        "true_label",
        "pred_label",
        "true_source_count",
        "pred_source_count",
        "exact_match",
        "over_prediction",
        "under_prediction",
    ]
    fieldnames.extend(f"prob_{name}" for name in class_names)
    fieldnames.extend(f"threshold_{name}" for name in class_names)
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
                "true_source_count": int(target.sum()),
                "pred_source_count": int(pred.sum()),
                "exact_match": int(np.all(pred == target)),
                "over_prediction": int(np.any(pred > target)),
                "under_prediction": int(np.any(pred < target)),
            }
            row.update({f"prob_{name}": f"{float(value):.10g}" for name, value in zip(class_names, prob)})
            row.update({f"threshold_{name}": f"{float(value):.10g}" for name, value in zip(class_names, threshold_values)})
            row.update({f"true_{name}": int(value) for name, value in zip(class_names, target)})
            row.update({f"pred_{name}": int(value) for name, value in zip(class_names, pred)})
            writer.writerow(row)


def write_combo_confusion_csv(path: str | Path, combo_confusion: dict[str, dict[str, int]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["true_label", "pred_label", "count", "true_total", "rate"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for true_label, pred_counts in combo_confusion.items():
            true_total = sum(pred_counts.values())
            for pred_label, count in pred_counts.items():
                writer.writerow(
                    {
                        "true_label": true_label,
                        "pred_label": pred_label,
                        "count": count,
                        "true_total": true_total,
                        "rate": f"{(count / true_total if true_total else 0.0):.10g}",
                    }
                )


def write_110_to_111_analysis(
    path: str | Path,
    probs: np.ndarray,
    targets: np.ndarray,
    groups: list[str],
    condition_paths: list[str],
    class_names: list[str],
    threshold: float | list[float] | np.ndarray,
    files: list[str] | None = None,
) -> dict[str, float | int]:
    threshold_values = normalize_thresholds(threshold, len(class_names))
    preds = predictions_at_threshold(probs, class_names, threshold_values)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "file",
        "group",
        "condition_path",
        "true_label",
        "pred_label",
        "source_1_prob",
        "source_3_prob",
        "source_5_prob",
        "source_1_threshold",
        "source_3_threshold",
        "source_5_threshold",
        "source_5_margin",
    ]
    class_index = {name: index for index, name in enumerate(class_names)}
    source5_values: list[float] = []
    source5_margins: list[float] = []
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, (prob, target, pred) in enumerate(zip(probs, targets, preds)):
            if label_to_text(target) != "[1,1,0]" or label_to_text(pred) != "[1,1,1]":
                continue
            source5_prob = float(prob[class_index["source_5"]])
            source5_margin = source5_prob - float(threshold_values[class_index["source_5"]])
            source5_values.append(source5_prob)
            source5_margins.append(source5_margin)
            writer.writerow(
                {
                    "file": files[index] if files is not None and index < len(files) else "",
                    "group": groups[index] if index < len(groups) else "",
                    "condition_path": condition_paths[index] if index < len(condition_paths) else "",
                    "true_label": label_to_text(target),
                    "pred_label": label_to_text(pred),
                    "source_1_prob": f"{float(prob[class_index['source_1']]):.10g}",
                    "source_3_prob": f"{float(prob[class_index['source_3']]):.10g}",
                    "source_5_prob": f"{source5_prob:.10g}",
                    "source_1_threshold": f"{float(threshold_values[class_index['source_1']]):.10g}",
                    "source_3_threshold": f"{float(threshold_values[class_index['source_3']]):.10g}",
                    "source_5_threshold": f"{float(threshold_values[class_index['source_5']]):.10g}",
                    "source_5_margin": f"{source5_margin:.10g}",
                }
            )
    values = np.asarray(source5_values, dtype=np.float32)
    margins = np.asarray(source5_margins, dtype=np.float32)
    return {
        "count": int(values.size),
        "source5_prob_mean": float(values.mean()) if values.size else 0.0,
        "source5_prob_median": float(np.median(values)) if values.size else 0.0,
        "source5_prob_p90": float(np.percentile(values, 90)) if values.size else 0.0,
        "source5_prob_min": float(values.min()) if values.size else 0.0,
        "source5_prob_max": float(values.max()) if values.size else 0.0,
        "source5_margin_mean": float(margins.mean()) if margins.size else 0.0,
    }


def evaluate(
    model_path: str | Path,
    split: str = "test",
    device_name: str = "auto",
    report_path: str | Path | None = None,
    real_split: str | None = None,
    threshold: float | None = None,
    thresholds_json: str | Path | None = None,
    thresholds: str | None = None,
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
    files: list[str] | None = None
    if requested_real_split is not None:
        loader, groups, condition_paths, files = _real_loader_and_groups(config, class_names, requested_real_split)
        eval_split = f"real_{requested_real_split}"
    else:
        loader = make_loader(config, split, shuffle=False, class_names=class_names)

    model = build_model(num_classes=len(class_names), config=config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    probs, targets = collect_probabilities(model, loader, device)
    metrics_by_threshold = [
        compute_metrics(probs, targets, class_names, default_threshold)
        for default_threshold in DEFAULT_THRESHOLDS
    ]

    if thresholds is not None:
        selected_threshold = parse_thresholds_arg(thresholds, class_names)
        threshold_source = "cli_thresholds"
    elif thresholds_json is not None:
        selected_threshold = load_thresholds_json(thresholds_json, class_names)
        threshold_source = str(thresholds_json)
    else:
        selected_threshold = float(config.get("train", {}).get("threshold", 0.5) if threshold is None else threshold)
        threshold_source = "config_train_threshold" if threshold is None else "cli_threshold"

    threshold_values = normalize_thresholds(selected_threshold, len(class_names))
    preds = predictions_at_threshold(probs, class_names, threshold_values)
    selected_metrics = compute_metrics(probs, targets, class_names, threshold_values)
    combo_confusion = compute_combo_confusion(targets, preds)
    source_1_source_3_mix_errors = compute_source_1_source_3_errors(targets, preds)

    output_path = Path(report_path or "outputs/reports/eval_report.json")
    error_analysis_path = output_path.with_name("error_analysis.csv")
    combo_confusion_path = output_path.with_name("combo_confusion.csv")
    analysis_110_to_111_path = output_path.with_name("analysis_110_to_111.csv")
    write_error_analysis(
        error_analysis_path,
        probs,
        targets,
        groups or [],
        condition_paths or [],
        class_names,
        threshold_values,
    )
    write_combo_confusion_csv(combo_confusion_path, combo_confusion)
    analysis_110_to_111 = write_110_to_111_analysis(
        analysis_110_to_111_path,
        probs,
        targets,
        groups or [],
        condition_paths or [],
        class_names,
        threshold_values,
        files,
    )

    real_breakdowns = None
    if groups is not None:
        real_breakdowns = compute_real_breakdowns(
            probs,
            targets,
            groups,
            class_names,
            threshold_values,
            condition_paths,
        )

    report = {
        "model": str(model_path),
        "split": eval_split,
        "num_samples": int(targets.shape[0]),
        "class_names": class_names,
        "threshold_config": {
            "source": threshold_source,
            "values": thresholds_for_report(threshold_values, class_names),
        },
        "selected_threshold": thresholds_for_report(threshold_values, class_names),
        "selected_metrics": selected_metrics,
        "per_source": selected_metrics["per_source"],
        "overall": selected_metrics["overall"],
        "combo_confusion": combo_confusion,
        "source_1_source_3_mix_errors": source_1_source_3_mix_errors,
        "analysis_110_to_111": analysis_110_to_111,
        "thresholds": metrics_by_threshold,
        "artifacts": {
            "eval_report_json": str(output_path),
            "error_analysis_csv": str(error_analysis_path),
            "combo_confusion_csv": str(combo_confusion_path),
            "analysis_110_to_111_csv": str(analysis_110_to_111_path),
        },
    }
    if real_breakdowns is not None:
        report["real_breakdowns"] = real_breakdowns
        report["group_accuracy"] = real_breakdowns["group_accuracy"]
        report["label_accuracy"] = real_breakdowns["label_accuracy"]
        report["ratio_accuracy"] = real_breakdowns["ratio_accuracy"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(f"device={device}")
    print(f"model={model_path}")
    print(f"split={eval_split} samples={targets.shape[0]}")
    print_metrics(metrics_by_threshold)
    print("\nselected threshold summary:")
    print_metrics([selected_metrics])
    if real_breakdowns is not None:
        print_real_breakdowns(real_breakdowns)
    print(f"\nreport={output_path}")
    print(f"error_analysis={error_analysis_path}")
    print(f"combo_confusion={combo_confusion_path}")
    print(f"analysis_110_to_111={analysis_110_to_111_path}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a multi-label noise source classifier.")
    parser.add_argument("--model", type=Path, required=True, help="Path to model checkpoint.")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test", help="Dataset split.")
    parser.add_argument("--real-split", choices=("train", "val", "test"), help="Evaluate a split from real_dataset_split.csv.")
    parser.add_argument("--device", default="auto", help="Device: auto, cpu, or cuda.")
    parser.add_argument("--report", type=Path, help="Report path (default: outputs/reports/eval_report.json).")
    parser.add_argument("--threshold", type=float, help="Uniform threshold (default: config train.threshold).")
    parser.add_argument("--thresholds", help="Per-class thresholds, e.g. source_1=0.55,source_3=0.45,source_5=0.75.")
    parser.add_argument("--thresholds-json", type=Path, help="Per-class thresholds JSON from search_thresholds.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate(
        model_path=args.model,
        split=args.split,
        device_name=args.device,
        report_path=args.report,
        real_split=args.real_split,
        threshold=args.threshold,
        thresholds_json=args.thresholds_json,
        thresholds=args.thresholds,
    )


if __name__ == "__main__":
    main()
