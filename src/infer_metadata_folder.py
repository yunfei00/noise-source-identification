from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.infer_folder import feature_from_csv, load_model_for_inference
from src.model_cnn import NoiseCNN
from src.train import resolve_device


DEFAULT_OUTPUT = Path("outputs/reports/metadata_inference.csv")


def iter_recursive_csv_files(input_dir: str | Path) -> list[Path]:
    root = Path(input_dir)
    if not root.exists():
        raise FileNotFoundError(f"Input directory not found: {root}")
    if not root.is_dir():
        raise ValueError(f"Expected an input directory, got: {root}")
    return sorted(path for path in root.rglob("*.csv") if path.is_file())


def label_to_text(label: np.ndarray) -> str:
    return "[" + ",".join(str(int(value)) for value in label.tolist()) + "]"


def label_to_sources(label: np.ndarray, class_names: list[str]) -> str:
    present = [name for name, value in zip(class_names, label) if int(value)]
    return "+".join(present) if present else "none"


def _all_label_matrix(num_classes: int, include_empty: bool) -> np.ndarray:
    start = 0 if include_empty else 1
    return np.asarray(
        [
            [(value >> (num_classes - 1 - index)) & 1 for index in range(num_classes)]
            for value in range(start, 2**num_classes)
        ],
        dtype=np.int32,
    )


def _normalized_entropy(probabilities: torch.Tensor) -> torch.Tensor:
    if probabilities.shape[1] <= 1:
        return torch.zeros(probabilities.shape[0], device=probabilities.device)
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=1)
    return entropy / float(np.log(probabilities.shape[1]))


def predict_feature_batch(
    model: NoiseCNN,
    features: np.ndarray,
    threshold: float,
) -> dict[str, np.ndarray]:
    """Predict labels plus calibrated distribution diagnostics for one feature batch."""
    if features.ndim != 4:
        raise ValueError(f"Expected [batch,channels,freq,time] features, got {features.shape}")
    device = next(model.parameters()).device
    inputs = torch.from_numpy(features).float().to(device)

    with torch.no_grad():
        if model.auxiliary_heads and model.prediction_mode == "structured":
            outputs = model.forward_with_auxiliary(inputs)
            combo_probabilities = model.structured_combo_probabilities(outputs)
            combo_labels = model.combo_labels.to(device=device, dtype=torch.float32)
            source_probabilities = combo_probabilities @ combo_labels
            predicted_indices = combo_probabilities.argmax(dim=1)
            predicted_labels = combo_labels[predicted_indices].to(dtype=torch.int32)
        else:
            source_probabilities = torch.sigmoid(model(inputs))
            combo_labels = torch.from_numpy(
                _all_label_matrix(model.num_classes, include_empty=True)
            ).to(device=device, dtype=torch.float32)
            positive = source_probabilities.clamp(1e-7, 1.0 - 1e-7).log().unsqueeze(1)
            negative = (1.0 - source_probabilities).clamp(1e-7, 1.0).log().unsqueeze(1)
            label_matrix = combo_labels.unsqueeze(0)
            combo_log_probabilities = (
                label_matrix * positive + (1.0 - label_matrix) * negative
            ).sum(dim=2)
            combo_probabilities = torch.softmax(combo_log_probabilities, dim=1)
            predicted_labels = (source_probabilities >= threshold).to(dtype=torch.int32)
            powers = 2 ** torch.arange(
                model.num_classes - 1,
                -1,
                -1,
                device=device,
                dtype=torch.int64,
            )
            predicted_indices = (predicted_labels.to(torch.int64) * powers).sum(dim=1)

        top_count = min(2, combo_probabilities.shape[1])
        top_probabilities, top_indices = torch.topk(combo_probabilities, k=top_count, dim=1)
        if top_count == 1:
            second_probabilities = torch.zeros_like(top_probabilities[:, 0])
            second_indices = top_indices[:, 0]
        else:
            second_probabilities = top_probabilities[:, 1]
            second_indices = top_indices[:, 1]

        row_indices = torch.arange(features.shape[0], device=device)
        predicted_confidences = combo_probabilities[row_indices, predicted_indices]
        confidence_margins = predicted_confidences - second_probabilities
        entropies = _normalized_entropy(combo_probabilities)

    return {
        "source_probabilities": source_probabilities.cpu().numpy().astype(np.float32),
        "predicted_labels": predicted_labels.cpu().numpy().astype(np.int32),
        "predicted_confidences": predicted_confidences.cpu().numpy().astype(np.float32),
        "second_labels": combo_labels[second_indices].cpu().numpy().astype(np.int32),
        "second_confidences": second_probabilities.cpu().numpy().astype(np.float32),
        "confidence_margins": confidence_margins.cpu().numpy().astype(np.float32),
        "normalized_entropies": entropies.cpu().numpy().astype(np.float32),
    }


def _distribution(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if not array.size:
        return {}
    return {
        "min": float(np.min(array)),
        "p05": float(np.percentile(array, 5)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def _counts_and_rates(counter: Counter[str], total: int) -> dict[str, dict[str, float | int]]:
    return {
        name: {
            "count": int(count),
            "rate": float(count / total) if total else 0.0,
        }
        for name, count in sorted(counter.items())
    }


def build_distribution_summary(
    rows: list[dict[str, Any]],
    failures: list[dict[str, str]],
    class_names: list[str],
    confidence_threshold: float,
) -> dict[str, Any]:
    total = len(rows)
    combination_counts = Counter(str(row["predicted_sources"]) for row in rows)
    expected_combination_names = [
        label_to_sources(label, class_names)
        for label in _all_label_matrix(len(class_names), include_empty=False)
    ]
    for name in expected_combination_names:
        combination_counts.setdefault(name, 0)
    source_counts = {
        class_name: sum(int(row[f"{class_name}_present"]) for row in rows)
        for class_name in class_names
    }
    source_probability_means = {
        class_name: (
            float(np.mean([float(row[f"{class_name}_probability"]) for row in rows]))
            if rows
            else 0.0
        )
        for class_name in class_names
    }

    by_top_level_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_top_level_rows[str(row["top_level_folder"])].append(row)
    by_top_level = {}
    for folder, folder_rows in sorted(by_top_level_rows.items()):
        folder_counts = Counter(str(row["predicted_sources"]) for row in folder_rows)
        for name in expected_combination_names:
            folder_counts.setdefault(name, 0)
        by_top_level[folder] = {
            "file_count": len(folder_rows),
            "combination_distribution": _counts_and_rates(folder_counts, len(folder_rows)),
        }

    low_confidence_rows = [
        {
            "file": str(row["file"]),
            "predicted_sources": str(row["predicted_sources"]),
            "confidence": float(row["confidence"]),
            "second_prediction": str(row["second_prediction"]),
            "confidence_margin": float(row["confidence_margin"]),
            "normalized_entropy": float(row["normalized_entropy"]),
        }
        for row in rows
        if float(row["confidence"]) < confidence_threshold
    ]
    source_distribution = {
        class_name: {
            "present_count": int(source_counts[class_name]),
            "present_rate": float(source_counts[class_name] / total) if total else 0.0,
            "mean_probability": source_probability_means[class_name],
        }
        for class_name in class_names
    }

    recommendations: list[str] = []
    if failures:
        recommendations.append("Inspect inference_failures first; these CSV files were not included in the distribution.")
    if total and len(low_confidence_rows) / total >= 0.10:
        recommendations.append(
            "At least 10% of files are low-confidence; manually review them and compare their dB ranges with the training audit."
        )
    if total and combination_counts:
        dominant_name, dominant_count = combination_counts.most_common(1)[0]
        if dominant_count / total >= 0.80:
            recommendations.append(
                f"Prediction is dominated by {dominant_name} ({dominant_count / total:.1%}); check input distribution shift or class imbalance."
            )
    if not recommendations:
        recommendations.append(
            "No automatic distribution warning was triggered; still inspect low-confidence files before using predictions operationally."
        )

    return {
        "processed_files": total,
        "failed_files": len(failures),
        "class_names": class_names,
        "combination_distribution": _counts_and_rates(combination_counts, total),
        "source_distribution": source_distribution,
        "confidence_distribution": _distribution([float(row["confidence"]) for row in rows]),
        "confidence_margin_distribution": _distribution(
            [float(row["confidence_margin"]) for row in rows]
        ),
        "normalized_entropy_distribution": _distribution(
            [float(row["normalized_entropy"]) for row in rows]
        ),
        "low_confidence_threshold": confidence_threshold,
        "low_confidence_count": len(low_confidence_rows),
        "low_confidence_rate": float(len(low_confidence_rows) / total) if total else 0.0,
        "low_confidence_files": low_confidence_rows,
        "by_top_level_folder": by_top_level,
        "inference_failures": failures,
        "recommendations": recommendations,
    }


def write_prediction_csv(
    rows: list[dict[str, Any]],
    class_names: list[str],
    output: str | Path,
) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "file",
        "top_level_folder",
        "predicted_label",
        "predicted_sources",
        "confidence",
        "second_prediction",
        "second_confidence",
        "confidence_margin",
        "normalized_entropy",
        "status",
    ]
    for class_name in class_names:
        fieldnames.extend((f"{class_name}_present", f"{class_name}_probability"))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(summary: dict[str, Any], output: str | Path) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def print_summary(summary: dict[str, Any]) -> None:
    print(f"processed_files={summary['processed_files']}")
    print(f"failed_files={summary['failed_files']}")
    print(f"class_names={summary['class_names']}")
    print("combination_distribution:")
    for name, metrics in summary["combination_distribution"].items():
        print(f"  {name}: count={metrics['count']} rate={metrics['rate']:.4f}")
    print("source_distribution:")
    for name, metrics in summary["source_distribution"].items():
        print(
            f"  {name}: present_count={metrics['present_count']} "
            f"present_rate={metrics['present_rate']:.4f} "
            f"mean_probability={metrics['mean_probability']:.4f}"
        )
    confidence = summary["confidence_distribution"]
    if confidence:
        print(
            "confidence: "
            f"mean={confidence['mean']:.4f} median={confidence['median']:.4f} "
            f"p05={confidence['p05']:.4f}"
        )
    print(
        f"low_confidence={summary['low_confidence_count']} "
        f"rate={summary['low_confidence_rate']:.4f}"
    )
    print("recommendations:")
    for recommendation in summary["recommendations"]:
        print(f"  - {recommendation}")


def infer_metadata_folder(
    model_path: str | Path,
    input_dir: str | Path,
    output: str | Path = DEFAULT_OUTPUT,
    summary_output: str | Path | None = None,
    *,
    device_name: str = "auto",
    batch_size: int = 32,
    threshold: float = 0.5,
    confidence_threshold: float = 0.6,
    progress_every: int = 500,
) -> dict[str, Any]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be in [0, 1]")

    root = Path(input_dir)
    csv_files = iter_recursive_csv_files(root)
    if not csv_files:
        raise ValueError(f"No CSV files found recursively under: {root}")

    device = resolve_device(device_name)
    model, class_names, config = load_model_for_inference(model_path, device)
    data_config = config.get("data", {})
    stft_config = config.get("stft", {})
    preprocessing_config = config.get("preprocessing", {})

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    feature_buffer: list[np.ndarray] = []
    path_buffer: list[Path] = []

    def flush_batch() -> None:
        if not feature_buffer:
            return
        features = np.stack(feature_buffer).astype(np.float32, copy=False)
        predictions = predict_feature_batch(model, features, threshold)
        for index, csv_path in enumerate(path_buffer):
            relative = csv_path.relative_to(root)
            predicted_label = predictions["predicted_labels"][index]
            second_label = predictions["second_labels"][index]
            confidence = float(predictions["predicted_confidences"][index])
            row: dict[str, Any] = {
                "file": relative.as_posix(),
                "top_level_folder": relative.parts[0] if len(relative.parts) > 1 else "(root)",
                "predicted_label": label_to_text(predicted_label),
                "predicted_sources": label_to_sources(predicted_label, class_names),
                "confidence": confidence,
                "second_prediction": label_to_sources(second_label, class_names),
                "second_confidence": float(predictions["second_confidences"][index]),
                "confidence_margin": float(predictions["confidence_margins"][index]),
                "normalized_entropy": float(predictions["normalized_entropies"][index]),
                "status": "low_confidence" if confidence < confidence_threshold else "accepted",
            }
            for class_index, class_name in enumerate(class_names):
                row[f"{class_name}_present"] = int(predicted_label[class_index])
                row[f"{class_name}_probability"] = float(
                    predictions["source_probabilities"][index, class_index]
                )
            rows.append(row)
        feature_buffer.clear()
        path_buffer.clear()

    for file_index, csv_path in enumerate(csv_files, start=1):
        try:
            feature = feature_from_csv(
                csv_path,
                data_config,
                stft_config,
                preprocessing_config,
            )
        except Exception as exc:
            failures.append(
                {
                    "file": csv_path.relative_to(root).as_posix(),
                    "error": str(exc),
                }
            )
            continue
        feature_buffer.append(feature)
        path_buffer.append(csv_path)
        if len(feature_buffer) >= batch_size:
            flush_batch()
        if progress_every > 0 and file_index % progress_every == 0:
            print(
                f"progress={file_index}/{len(csv_files)} "
                f"predicted={len(rows)} failed={len(failures)}"
            )
    flush_batch()

    rows.sort(key=lambda row: str(row["file"]))
    summary = build_distribution_summary(
        rows,
        failures,
        class_names,
        confidence_threshold,
    )
    summary.update(
        {
            "model": str(Path(model_path).resolve()),
            "input_dir": str(root.resolve()),
            "discovered_csv_files": len(csv_files),
            "prediction_output": str(Path(output).resolve()),
            "prediction_mode": model.prediction_mode,
            "batch_size": batch_size,
            "multilabel_threshold": threshold,
        }
    )
    if summary_output is None:
        summary_output = Path(output).with_suffix(".summary.json")
    summary["summary_output"] = str(Path(summary_output).resolve())

    write_prediction_csv(rows, class_names, output)
    write_summary(summary, summary_output)
    print_summary(summary)
    print(f"prediction_output={Path(output).resolve()}")
    print(f"summary_output={Path(summary_output).resolve()}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recursively infer unlabeled CSV metadata and report prediction distributions."
    )
    parser.add_argument("--model", type=Path, required=True, help="Trained checkpoint path.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Folder recursively containing CSV files.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Per-file prediction CSV.")
    parser.add_argument("--summary-output", type=Path, default=None, help="Distribution summary JSON.")
    parser.add_argument("--device", default="auto", help="Device: auto, cpu, or cuda.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Only used by legacy independent multilabel checkpoints.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.6,
        help="Flag predictions below this combination confidence for manual review.",
    )
    parser.add_argument("--progress-every", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    infer_metadata_folder(
        model_path=args.model,
        input_dir=args.input_dir,
        output=args.output,
        summary_output=args.summary_output,
        device_name=args.device,
        batch_size=args.batch_size,
        threshold=args.threshold,
        confidence_threshold=args.confidence_threshold,
        progress_every=args.progress_every,
    )


if __name__ == "__main__":
    main()
