from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support

from src.features import compute_stft_feature, fix_length, normalize_signal, read_signal_csv
from src.infer import load_checkpoint
from src.model_cnn import NoiseCNN
from src.train import resolve_device

_SOURCE_NAME_RE = re.compile(r"source_\d+")
_UNKNOWN_GROUP_PREFIXES = ("unknown", "background")


def _validate_class_names(class_names: Any) -> list[str]:
    if not isinstance(class_names, list) or not class_names or not all(
        isinstance(name, str) for name in class_names
    ):
        raise ValueError("Checkpoint is missing valid class_names")
    return class_names


def label_to_text(label: np.ndarray) -> str:
    return "[" + ",".join(str(int(value)) for value in label.tolist()) + "]"


def is_unknown_group(group: str) -> bool:
    """Return whether a real-data group should be treated as an unknown source."""
    return group.startswith(_UNKNOWN_GROUP_PREFIXES)


def parse_true_label(group: str, class_names: list[str]) -> np.ndarray:
    """Parse a first-level real-data group name into a multi-label target vector."""
    if is_unknown_group(group):
        return np.zeros(len(class_names), dtype=np.int32)

    source_names = set(_SOURCE_NAME_RE.findall(group))
    if not source_names:
        raise ValueError(
            f"Cannot parse true label from group '{group}'. Expected names like "
            "source_1, source_3, source_1_source_5_mix, or "
            "source_1_source_3_source_5_mix."
        )
    if group not in class_names and not group.endswith("_mix"):
        raise ValueError(
            f"Cannot parse true label from group '{group}'. Group must be a single source directory like source_1 or a combo group ending with _mix."
        )

    unknown_sources = sorted(source_names.difference(class_names))
    if unknown_sources:
        raise ValueError(
            f"Group '{group}' contains sources not present in checkpoint class_names: "
            f"{', '.join(unknown_sources)}"
        )

    return np.asarray(
        [1 if class_name in source_names else 0 for class_name in class_names],
        dtype=np.int32,
    )


def iter_group_csv_files(input_dir: str | Path) -> list[tuple[str, Path]]:
    """Return (first-level group, csv path) pairs using recursive group scanning."""
    root = Path(input_dir)
    if not root.exists():
        raise FileNotFoundError(f"Input directory not found: {root}")
    if not root.is_dir():
        raise ValueError(f"Expected input directory, got: {root}")

    grouped_files: list[tuple[str, Path]] = []
    group_dirs = sorted((path for path in root.iterdir() if path.is_dir()), key=lambda path: path.name)
    for group_dir in group_dirs:
        csv_files = sorted(path for path in group_dir.rglob("*.csv") if path.is_file())
        if not csv_files:
            print(f"warning: no CSV files found recursively under group directory: {group_dir}")
            continue
        grouped_files.extend((group_dir.name, csv_path) for csv_path in csv_files)
    return grouped_files


def find_csv_files(input_dir: str | Path) -> list[Path]:
    """Backward-compatible recursive CSV finder."""
    return [csv_path for _, csv_path in iter_group_csv_files(input_dir)]


def _feature_from_csv(csv_path: Path, data_config: dict, stft_config: dict) -> np.ndarray:
    signal = read_signal_csv(csv_path)
    signal = fix_length(signal, int(data_config.get("signal_length", 4096)))
    signal = normalize_signal(signal)
    return compute_stft_feature(
        signal,
        sample_rate=int(data_config.get("sample_rate", 1_000_000)),
        nperseg=int(stft_config.get("nperseg", 256)),
        noverlap=int(stft_config.get("noverlap", 128)),
        target_freq_bins=int(stft_config.get("target_freq_bins", 128)),
        target_time_bins=int(stft_config.get("target_time_bins", 64)),
    )


def load_model_for_inference(
    model_path: str | Path,
    device: torch.device,
) -> tuple[NoiseCNN, list[str], dict]:
    checkpoint = load_checkpoint(model_path, map_location=device)
    class_names = _validate_class_names(checkpoint.get("class_names"))
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise ValueError("Checkpoint is missing config")

    model = NoiseCNN(num_classes=len(class_names)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, class_names, config


def infer_csv_probabilities(
    model: NoiseCNN,
    csv_path: Path,
    config: dict,
    device: torch.device,
) -> np.ndarray:
    data_config = config.get("data", {})
    stft_config = config.get("stft", {})
    feature = _feature_from_csv(csv_path, data_config, stft_config)
    x = torch.from_numpy(feature).unsqueeze(0).unsqueeze(0).float().to(device)
    with torch.no_grad():
        logits = model(x)
        probabilities = torch.sigmoid(logits).squeeze(0).cpu().numpy()
    return probabilities.astype(np.float32, copy=False)


def classify_prediction(probabilities: np.ndarray, threshold: float, unknown_threshold: float) -> tuple[np.ndarray, str]:
    max_prob = float(np.max(probabilities)) if probabilities.size else 0.0
    if max_prob < unknown_threshold:
        return np.zeros_like(probabilities, dtype=np.int32), "unknown"
    pred_label = (probabilities >= threshold).astype(np.int32)
    if not np.any(pred_label):
        return pred_label, "uncertain"
    return pred_label, "known"


def build_unknown_summary(
    rows: list[dict[str, Any]],
    preds: np.ndarray,
    probabilities: np.ndarray,
    class_names: list[str],
) -> dict[str, Any]:
    unknown_indices = [index for index, row in enumerate(rows) if is_unknown_group(str(row["group"]))]
    num_unknown = len(unknown_indices)
    if not num_unknown:
        return {
            "num_samples": 0,
            "full_rejection_accuracy": 0.0,
            "misclassified_as_source_counts": {class_name: 0 for class_name in class_names},
            "max_prob_mean": 0.0,
            "max_prob_max": 0.0,
        }

    unknown_preds = preds[unknown_indices]
    unknown_probabilities = probabilities[unknown_indices]
    rejected = np.all(unknown_preds == 0, axis=1)
    max_probabilities = np.max(unknown_probabilities, axis=1)

    return {
        "num_samples": num_unknown,
        "full_rejection_accuracy": float(rejected.mean()),
        "misclassified_as_source_counts": {
            class_name: int(unknown_preds[:, class_index].sum())
            for class_index, class_name in enumerate(class_names)
        },
        "max_prob_mean": float(max_probabilities.mean()),
        "max_prob_max": float(max_probabilities.max()),
    }


def _group_bucket(group: str, class_names: list[str]) -> str:
    if group in class_names:
        return group
    if group.endswith("_mix"):
        return "mix"
    return group


def build_summary(
    rows: list[dict[str, Any]],
    targets: np.ndarray,
    preds: np.ndarray,
    probabilities: np.ndarray,
    class_names: list[str],
    threshold: float,
    unknown_threshold: float,
) -> dict[str, Any]:
    total_samples = len(rows)
    exact_matches = np.all(preds == targets, axis=1) if total_samples else np.asarray([], dtype=bool)
    precision, recall, f1, support = precision_recall_fscore_support(
        targets,
        preds,
        average=None,
        zero_division=0,
    )

    groups = sorted({str(row["group"]) for row in rows})
    group_accuracy: dict[str, dict[str, float | int]] = {}
    group_probability_means: dict[str, dict[str, float]] = {}
    for group in groups:
        indices = [index for index, row in enumerate(rows) if row["group"] == group]
        group_matches = exact_matches[indices] if len(indices) else np.asarray([], dtype=bool)
        group_accuracy[group] = {
            "num_samples": len(indices),
            "accuracy": float(group_matches.mean()) if len(indices) else 0.0,
        }
        group_probability_means[group] = {
            f"{class_name}_prob_mean": float(probabilities[indices, class_index].mean()) if len(indices) else 0.0
            for class_index, class_name in enumerate(class_names)
        }

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

    misclassification_counters: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        bucket = _group_bucket(str(row["group"]), class_names)
        if bucket in {*class_names, "mix"}:
            misclassification_counters[bucket][str(row["pred_label"])] += 1
    misclassification_stats = {
        bucket: dict(counter) for bucket, counter in sorted(misclassification_counters.items())
    }

    exact_accuracy = float(exact_matches.mean()) if total_samples else 0.0
    return {
        "threshold": float(threshold),
        "unknown_threshold": float(unknown_threshold),
        "num_samples": total_samples,
        "exact_match_accuracy": exact_accuracy,
        "overall_exact_match_accuracy": exact_accuracy,
        "group_accuracy": group_accuracy,
        "per_source": per_source,
        "group_probability_means": group_probability_means,
        "misclassification_stats": misclassification_stats,
        "unknown": build_unknown_summary(rows, preds, probabilities, class_names),
    }


def write_report_csv(rows: list[dict[str, Any]], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    probability_fields = [
        fieldname for fieldname in rows[0] if fieldname.endswith("_prob") and fieldname != "max_prob"
    ] if rows else []
    fieldnames = [
        "file",
        "group",
        "true_label",
        "pred_label",
        *sorted(set(probability_fields)),
        "max_prob",
        "result_type",
        "correct",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            output_row = dict(row)
            output_row["correct"] = str(bool(output_row["correct"])).lower()
            writer.writerow(output_row)


def write_summary_json(summary: dict[str, Any], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def print_summary(summary: dict[str, Any]) -> None:
    print(f"total_samples={summary['num_samples']}")
    print(f"overall_exact_match_accuracy={summary['overall_exact_match_accuracy']:.4f}")
    print("\ngroup accuracy:")
    for group, metrics in summary["group_accuracy"].items():
        print(f"  {group}: accuracy={metrics['accuracy']:.4f} samples={metrics['num_samples']}")
    print("\ngroup probability means:")
    for group, metrics in summary["group_probability_means"].items():
        joined = " ".join(f"{key}={value:.4f}" for key, value in metrics.items())
        print(f"  {group}: {joined}")
    print("\nmisclassification stats:")
    for group, counts in summary["misclassification_stats"].items():
        print(f"  {group}: {counts}")
    unknown = summary.get("unknown", {})
    print("\nunknown rejection:")
    print(f"  samples={unknown.get('num_samples', 0)}")
    print(f"  full_rejection_accuracy={unknown.get('full_rejection_accuracy', 0.0):.4f}")
    print(f"  max_prob_mean={unknown.get('max_prob_mean', 0.0):.4f}")
    print(f"  max_prob_max={unknown.get('max_prob_max', 0.0):.4f}")
    print("  misclassified_as_source_counts:")
    for source_name, count in unknown.get("misclassified_as_source_counts", {}).items():
        print(f"    {source_name}: {count}")

    print("\nper-source metrics:")
    print(f"{'source':<16} {'precision':>10} {'recall':>10} {'f1':>10} {'support':>10}")
    for source_name, metrics in summary["per_source"].items():
        print(
            f"{source_name:<16} "
            f"{metrics['precision']:>10.4f} "
            f"{metrics['recall']:>10.4f} "
            f"{metrics['f1']:>10.4f} "
            f"{metrics['support']:>10d}"
        )


def infer_folder(
    model_path: str | Path,
    input_dir: str | Path,
    output_path: str | Path,
    threshold: float = 0.5,
    device_name: str = "auto",
    unknown_threshold: float = 0.35,
) -> dict[str, Any]:
    device = resolve_device(device_name)
    model, class_names, config = load_model_for_inference(model_path, device)

    grouped_csv_files = iter_group_csv_files(input_dir)
    if not grouped_csv_files:
        raise ValueError(f"No CSV files found under first-level group directories: {input_dir}")

    root = Path(input_dir)
    rows: list[dict[str, Any]] = []
    targets: list[np.ndarray] = []
    preds: list[np.ndarray] = []
    probabilities_by_file: list[np.ndarray] = []

    for group, csv_path in grouped_csv_files:
        true_label = parse_true_label(group, class_names)
        probabilities = infer_csv_probabilities(model, csv_path, config, device)
        pred_label, result_type = classify_prediction(probabilities, threshold, unknown_threshold)
        correct = bool(np.array_equal(pred_label, true_label))

        row = {
            "file": csv_path.relative_to(root).as_posix(),
            "group": group,
            "true_label": label_to_text(true_label),
            "pred_label": label_to_text(pred_label),
            "max_prob": f"{float(np.max(probabilities)):.6f}",
            "result_type": result_type,
            "correct": correct,
        }
        row.update(
            {
                f"{class_name}_prob": f"{float(probabilities[class_index]):.6f}"
                for class_index, class_name in enumerate(class_names)
            }
        )
        rows.append(row)
        targets.append(true_label)
        preds.append(pred_label)
        probabilities_by_file.append(probabilities)

    target_array = np.vstack(targets).astype(np.int32)
    pred_array = np.vstack(preds).astype(np.int32)
    probability_array = np.vstack(probabilities_by_file).astype(np.float32)
    summary = build_summary(
        rows, target_array, pred_array, probability_array, class_names, threshold, unknown_threshold
    )
    summary_path = Path(output_path).with_suffix(".summary.json")
    summary.update(
        {
            "model": str(model_path),
            "input_dir": str(input_dir),
            "output": str(output_path),
            "summary_output": str(summary_path),
            "class_names": class_names,
        }
    )

    write_report_csv(rows, output_path)
    write_summary_json(summary, summary_path)

    print(f"device={device}")
    print(f"model={model_path}")
    print(f"input_dir={input_dir}")
    print(f"report={output_path}")
    print(f"summary={summary_path}\n")
    print_summary(summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch infer real dataset CSV folders.")
    parser.add_argument("--model", type=Path, required=True, help="Path to model checkpoint.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Root directory containing real dataset group folders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/reports/infer_folder_report.csv"),
        help="Path to output CSV report.",
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="Multi-label decision threshold.")
    parser.add_argument(
        "--unknown-threshold",
        type=float,
        default=0.35,
        help="Classify samples below this max probability as unknown.",
    )
    parser.add_argument("--device", default="auto", help="Device: auto, cpu, or cuda.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    infer_folder(
        args.model,
        args.input_dir,
        args.output,
        args.threshold,
        args.device,
        args.unknown_threshold,
    )


if __name__ == "__main__":
    main()
