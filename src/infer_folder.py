from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support

from src.features import compute_stft_feature, fix_length, normalize_signal, read_signal_csv
from src.infer import load_checkpoint
from src.model_cnn import NoiseCNN
from src.train import resolve_device

_REQUIRED_REPORT_CLASSES = ("source_1", "source_3")
_SOURCE_NAME_RE = re.compile(r"source_\d+")
_UNKNOWN_GROUP_PREFIX = "unknown"


def _validate_class_names(class_names: Any) -> list[str]:
    if not isinstance(class_names, list) or not class_names or not all(
        isinstance(name, str) for name in class_names
    ):
        raise ValueError("Checkpoint is missing valid class_names")
    missing = [name for name in _REQUIRED_REPORT_CLASSES if name not in class_names]
    if missing:
        raise ValueError(
            "Checkpoint class_names must include "
            f"{', '.join(_REQUIRED_REPORT_CLASSES)}; missing: {', '.join(missing)}"
        )
    return class_names


def _label_to_text(label: np.ndarray) -> str:
    return "[" + ",".join(str(int(value)) for value in label.tolist()) + "]"


def is_unknown_group(group: str) -> bool:
    """Return whether a real-test group should be treated as an unknown source."""
    return group.startswith(_UNKNOWN_GROUP_PREFIX)


def parse_true_label(group: str, class_names: list[str]) -> np.ndarray:
    """Parse a real-test group directory name into a multi-label target vector."""
    if is_unknown_group(group):
        return np.zeros(len(class_names), dtype=np.int32)

    source_names = set(_SOURCE_NAME_RE.findall(group))
    if not source_names:
        raise ValueError(
            f"Cannot parse true label from group '{group}'. Expected names like "
            "source_1_only, source_3_only, or source_1_source_3_mix."
        )
    if not (group.endswith("_only") or group.endswith("_mix")):
        raise ValueError(
            f"Cannot parse true label from group '{group}'. Group must end with _only or _mix."
        )
    return np.asarray(
        [1 if class_name in source_names else 0 for class_name in class_names],
        dtype=np.int32,
    )


def find_csv_files(input_dir: str | Path) -> list[Path]:
    root = Path(input_dir)
    if not root.exists():
        raise FileNotFoundError(f"Input directory not found: {root}")
    if not root.is_dir():
        raise ValueError(f"Expected input directory, got: {root}")
    return sorted(path for path in root.rglob("*.csv") if path.is_file())


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


def build_unknown_summary(
    rows: list[dict[str, Any]],
    preds: np.ndarray,
    probabilities: np.ndarray,
    class_names: list[str],
) -> dict[str, Any]:
    """Build rejection metrics for real-test groups whose names start with unknown."""
    unknown_indices = [
        index for index, row in enumerate(rows) if is_unknown_group(str(row["group"]))
    ]
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


def build_summary(
    rows: list[dict[str, Any]],
    targets: np.ndarray,
    preds: np.ndarray,
    probabilities: np.ndarray,
    class_names: list[str],
    threshold: float,
) -> dict[str, Any]:
    total_samples = len(rows)
    exact_matches = np.all(preds == targets, axis=1) if total_samples else np.asarray([], dtype=bool)
    precision, recall, f1, support = precision_recall_fscore_support(
        targets,
        preds,
        average=None,
        zero_division=0,
    )

    groups = sorted({row["group"] for row in rows})
    group_accuracy = {}
    for group in groups:
        group_matches = [bool(row["correct"]) for row in rows if row["group"] == group]
        group_accuracy[group] = {
            "num_samples": len(group_matches),
            "accuracy": float(sum(group_matches) / len(group_matches)) if group_matches else 0.0,
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

    return {
        "threshold": float(threshold),
        "num_samples": total_samples,
        "exact_match_accuracy": float(exact_matches.mean()) if total_samples else 0.0,
        "group_accuracy": group_accuracy,
        "per_source": per_source,
        "unknown": build_unknown_summary(rows, preds, probabilities, class_names),
    }


def write_report_csv(rows: list[dict[str, Any]], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "file",
        "group",
        "true_label",
        "pred_label",
        "source_1_prob",
        "source_3_prob",
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
    print(f"exact_match_accuracy={summary['exact_match_accuracy']:.4f}")
    print("\ngroup accuracy:")
    for group, metrics in summary["group_accuracy"].items():
        print(f"  {group}: accuracy={metrics['accuracy']:.4f} samples={metrics['num_samples']}")
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
) -> dict[str, Any]:
    device = resolve_device(device_name)
    model, class_names, config = load_model_for_inference(model_path, device)
    source_1_index = class_names.index("source_1")
    source_3_index = class_names.index("source_3")

    csv_files = find_csv_files(input_dir)
    if not csv_files:
        raise ValueError(f"No CSV files found under input directory: {input_dir}")

    root = Path(input_dir)
    rows: list[dict[str, Any]] = []
    targets: list[np.ndarray] = []
    preds: list[np.ndarray] = []
    probabilities_by_file: list[np.ndarray] = []

    for csv_path in csv_files:
        group = csv_path.parent.name
        true_label = parse_true_label(group, class_names)
        probabilities = infer_csv_probabilities(model, csv_path, config, device)
        pred_label = (probabilities >= threshold).astype(np.int32)
        correct = bool(np.array_equal(pred_label, true_label))

        rows.append(
            {
                "file": str(csv_path.relative_to(root)),
                "group": group,
                "true_label": _label_to_text(true_label),
                "pred_label": _label_to_text(pred_label),
                "source_1_prob": f"{float(probabilities[source_1_index]):.6f}",
                "source_3_prob": f"{float(probabilities[source_3_index]):.6f}",
                "correct": correct,
            }
        )
        targets.append(true_label)
        preds.append(pred_label)
        probabilities_by_file.append(probabilities)

    target_array = np.vstack(targets).astype(np.int32)
    pred_array = np.vstack(preds).astype(np.int32)
    probability_array = np.vstack(probabilities_by_file).astype(np.float32)
    summary = build_summary(
        rows, target_array, pred_array, probability_array, class_names, threshold
    )
    summary.update(
        {
            "model": str(model_path),
            "input_dir": str(input_dir),
            "output": str(output_path),
            "summary_output": str(Path(output_path).with_name("real_test_summary.json")),
            "class_names": class_names,
        }
    )

    write_report_csv(rows, output_path)
    summary_path = Path(output_path).with_name("real_test_summary.json")
    write_summary_json(summary, summary_path)

    print(f"device={device}")
    print(f"model={model_path}")
    print(f"input_dir={input_dir}")
    print(f"report={output_path}")
    print(f"summary={summary_path}\n")
    print_summary(summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch infer real validation CSV folders.")
    parser.add_argument("--model", type=Path, required=True, help="Path to model checkpoint.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Root directory containing real-test group folders.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Path to output CSV report.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Multi-label decision threshold.")
    parser.add_argument("--device", default="auto", help="Device: auto, cpu, or cuda.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    infer_folder(args.model, args.input_dir, args.output, args.threshold, args.device)


if __name__ == "__main__":
    main()
