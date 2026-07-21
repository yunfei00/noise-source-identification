from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from src.dataset import parse_label_text

DEFAULT_CLASS_NAMES = ("source_1", "source_3", "source_5")
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
EXPECTED_RATIOS = (
    "ratio_1_1",
    "ratio_1_2",
    "ratio_1_4",
    "ratio_2_1",
    "ratio_3_1",
    "ratio_4_1",
    "ratio_1_1_1",
    "ratio_1_2_1",
    "ratio_1_2_4",
    "ratio_4_2_1",
)
SPLITS = ("train", "val", "test")


def label_to_text(label: np.ndarray | list[int] | list[float]) -> str:
    return "[" + ",".join(str(int(value)) for value in np.asarray(label).astype(int).tolist()) + "]"


def _ratio_from_condition(condition_path: str) -> str | None:
    for part in Path(condition_path).parts:
        if part.startswith("ratio_"):
            return part
        if part.startswith("radio_"):
            return "ratio_" + part[len("radio_") :]
    return None


def _ordered_counts(counter: Counter[str], expected: tuple[str, ...]) -> dict[str, int]:
    ordered = {key: int(counter.get(key, 0)) for key in expected}
    for key in sorted(set(counter).difference(expected)):
        ordered[key] = int(counter[key])
    return ordered


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return float(numerator / denominator)


def _infer_class_names(fieldnames: list[str] | None) -> list[str]:
    if not fieldnames:
        return list(DEFAULT_CLASS_NAMES)
    found = [name for name in DEFAULT_CLASS_NAMES if name in fieldnames]
    return found if found else list(DEFAULT_CLASS_NAMES)


def read_split_rows(split_path: str | Path) -> tuple[list[dict[str, Any]], list[str]]:
    path = Path(split_path)
    if not path.exists():
        raise FileNotFoundError(f"split file not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        class_names = _infer_class_names(reader.fieldnames)
        for row in reader:
            if "label" not in row or not row["label"]:
                raise ValueError(f"Missing label column in {path}")
            label = parse_label_text(row["label"]).astype(np.int32)
            if label.shape[0] != len(class_names):
                raise ValueError(
                    f"Label length mismatch: {row['label']} vs class_names={class_names}"
                )
            rows.append(
                {
                    **row,
                    "_label": label,
                    "_label_text": label_to_text(label),
                    "_split": row.get("split", ""),
                    "_group": row.get("group", ""),
                    "_ratio": _ratio_from_condition(row.get("condition_path", "")),
                }
            )
    return rows, class_names


def summarize_rows(rows: list[dict[str, Any]], class_names: list[str]) -> dict[str, Any]:
    labels = [np.asarray(row["_label"], dtype=np.int32) for row in rows]
    label_counts = _ordered_counts(Counter(row["_label_text"] for row in rows), EXPECTED_LABELS)
    group_counts = _ordered_counts(Counter(row["_group"] for row in rows if row["_group"]), EXPECTED_GROUPS)
    ratio_counts = _ordered_counts(Counter(row["_ratio"] for row in rows if row["_ratio"]), EXPECTED_RATIOS)
    total = len(rows)

    if labels:
        label_matrix = np.vstack(labels).astype(np.int32)
    else:
        label_matrix = np.zeros((0, len(class_names)), dtype=np.int32)

    source_stats: dict[str, dict[str, float | int]] = {}
    positive_ratios: list[float] = []
    for index, class_name in enumerate(class_names):
        positive_count = int(label_matrix[:, index].sum()) if total else 0
        negative_count = int(total - positive_count)
        positive_ratio = float(positive_count / total) if total else 0.0
        positive_ratios.append(positive_ratio)
        source_stats[class_name] = {
            "positive_count": positive_count,
            "negative_count": negative_count,
            "positive_ratio": positive_ratio,
        }

    expected_label_values = [label_counts[label] for label in EXPECTED_LABELS]
    expected_group_values = [group_counts[group] for group in EXPECTED_GROUPS]
    max_label_count = max(expected_label_values) if expected_label_values else 0
    min_label_count = min(expected_label_values) if expected_label_values else 0
    max_group_count = max(expected_group_values) if expected_group_values else 0
    min_group_count = min(expected_group_values) if expected_group_values else 0

    return {
        "num_samples": total,
        "label_counts": label_counts,
        "source_stats": source_stats,
        "group_counts": group_counts,
        "ratio_counts": ratio_counts,
        "avg_true_sources_per_sample": float(label_matrix.sum(axis=1).mean()) if total else 0.0,
        "max_group_count": int(max_group_count),
        "min_group_count": int(min_group_count),
        "max_group_count_over_min_group_count": _safe_ratio(max_group_count, min_group_count),
        "label_imbalance_ratio": _safe_ratio(max_label_count, min_label_count),
        "source_positive_ratio_gap": float(max(positive_ratios) - min(positive_ratios)) if positive_ratios else 0.0,
    }


def _warning_lines(summary: dict[str, Any], scope: str) -> list[str]:
    warnings: list[str] = []
    total = int(summary["num_samples"])
    if total == 0:
        warnings.append(f"[{scope}] split has no samples")
        return warnings

    source_stats: dict[str, dict[str, float | int]] = summary["source_stats"]
    for source_name, metrics in source_stats.items():
        positive_ratio = float(metrics["positive_ratio"])
        if positive_ratio >= 0.75:
            warnings.append(f"[{scope}] {source_name} positive_ratio is high: {positive_ratio:.4f}")

    label_counts: dict[str, int] = summary["label_counts"]
    nonzero_counts = [count for count in label_counts.values() if count > 0]
    median_count = float(np.median(nonzero_counts)) if nonzero_counts else 0.0
    for label in EXPECTED_LABELS:
        count = int(label_counts.get(label, 0))
        if count == 0:
            warnings.append(f"[{scope}] label combo {label} has zero samples")
        elif median_count and count < median_count * 0.25:
            warnings.append(
                f"[{scope}] label combo {label} is much lower than median: count={count} median={median_count:.1f}"
            )

    triple_count = int(label_counts.get("[1,1,1]", 0))
    if triple_count / total >= 0.25:
        warnings.append(f"[{scope}] [1,1,1] count is high: count={triple_count} ratio={triple_count / total:.4f}")

    source_1_3_count = int(label_counts.get("[1,1,0]", 0))
    other_double_counts = [
        int(label_counts.get("[1,0,1]", 0)),
        int(label_counts.get("[0,1,1]", 0)),
    ]
    positive_other_double_counts = [count for count in other_double_counts if count > 0]
    if positive_other_double_counts and source_1_3_count < min(positive_other_double_counts) * 0.5:
        warnings.append(
            f"[{scope}] [1,1,0] is much lower than other double-source combos: "
            f"count={source_1_3_count} others={other_double_counts}"
        )
    return warnings


def analyze_label_distribution(split_path: str | Path, output: str | Path) -> dict[str, Any]:
    rows, class_names = read_split_rows(split_path)
    rows_by_split = {
        split: [row for row in rows if row["_split"] == split]
        for split in SPLITS
    }
    split_summaries = {
        split: summarize_rows(split_rows, class_names)
        for split, split_rows in rows_by_split.items()
    }
    overall_summary = summarize_rows(rows, class_names)

    warnings: list[str] = []
    warnings.extend(_warning_lines(overall_summary, "overall"))
    for split, summary in split_summaries.items():
        warnings.extend(_warning_lines(summary, split))

    report = {
        "input": str(split_path),
        "class_names": class_names,
        "splits": split_summaries,
        "overall": overall_summary,
        "warnings": warnings,
    }

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    csv_path = output_path.with_suffix(".csv")
    write_distribution_csv(csv_path, report)

    print(f"input={split_path}")
    print(f"json={output_path}")
    print(f"csv={csv_path}")
    print(f"total_samples={overall_summary['num_samples']}")
    print("warnings:")
    if warnings:
        for warning in warnings:
            print(f"  warning: {warning}")
    else:
        print("  (none)")
    return report


def write_distribution_csv(path: str | Path, report: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "split",
        "section",
        "key",
        "count",
        "value",
        "positive_count",
        "negative_count",
        "positive_ratio",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for split, summary in {**report["splits"], "overall": report["overall"]}.items():
            for label, count in summary["label_counts"].items():
                writer.writerow({"split": split, "section": "label_counts", "key": label, "count": count})
            for source_name, metrics in summary["source_stats"].items():
                writer.writerow(
                    {
                        "split": split,
                        "section": "source_stats",
                        "key": source_name,
                        "positive_count": metrics["positive_count"],
                        "negative_count": metrics["negative_count"],
                        "positive_ratio": f"{float(metrics['positive_ratio']):.10g}",
                    }
                )
            for group, count in summary["group_counts"].items():
                writer.writerow({"split": split, "section": "group_counts", "key": group, "count": count})
            for ratio, count in summary["ratio_counts"].items():
                writer.writerow({"split": split, "section": "ratio_counts", "key": ratio, "count": count})
            for key in (
                "avg_true_sources_per_sample",
                "max_group_count",
                "min_group_count",
                "max_group_count_over_min_group_count",
                "label_imbalance_ratio",
                "source_positive_ratio_gap",
            ):
                value = summary[key]
                writer.writerow(
                    {
                        "split": split,
                        "section": "summary",
                        "key": key,
                        "value": "" if value is None else f"{float(value):.10g}",
                    }
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze real dataset label, source, group, and ratio distribution.")
    parser.add_argument("--split", type=Path, required=True, help="Path to real_dataset_split.csv.")
    parser.add_argument("--output", type=Path, default=Path("outputs/reports/label_distribution.json"), help="Output JSON path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analyze_label_distribution(args.split, args.output)


if __name__ == "__main__":
    main()
