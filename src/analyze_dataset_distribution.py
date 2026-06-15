from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from src.build_real_index import parse_label_text

SPLITS = ("train", "val", "test")
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
EXPECTED_LABELS = (
    "[1,0,0]",
    "[0,1,0]",
    "[0,0,1]",
    "[1,1,0]",
    "[1,0,1]",
    "[0,1,1]",
    "[1,1,1]",
)
DEFAULT_CLASS_NAMES = ("source_1", "source_3", "source_5")


def label_to_text(label: list[int]) -> str:
    return "[" + ",".join(str(int(value)) for value in label) + "]"


def ratio_from_condition(condition_path: str) -> str | None:
    for part in Path(condition_path).parts:
        if part.startswith("ratio_"):
            return part
    return None


def zero_counter(keys: tuple[str, ...]) -> dict[str, int]:
    return {key: 0 for key in keys}


def analyze_distribution(split: str | Path) -> dict:
    split_path = Path(split)
    if not split_path.exists():
        raise FileNotFoundError(f"Split CSV not found: {split_path}")

    with split_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    source_columns = [name for name in DEFAULT_CLASS_NAMES if name in fieldnames]
    if not source_columns:
        source_columns = list(DEFAULT_CLASS_NAMES)

    split_counts: Counter[str] = Counter()
    source_counts: dict[str, Counter[str]] = {split_name: Counter() for split_name in SPLITS}
    label_combo_counts: dict[str, Counter[str]] = {split_name: Counter() for split_name in SPLITS}
    group_counts: dict[str, Counter[str]] = {split_name: Counter() for split_name in SPLITS}
    ratio_counts: dict[str, Counter[str]] = {split_name: Counter() for split_name in SPLITS}

    overall_source_counts: Counter[str] = Counter()
    overall_label_counts: Counter[str] = Counter()
    overall_group_counts: Counter[str] = Counter()
    overall_ratio_counts: Counter[str] = Counter()

    for row in rows:
        split_name = row.get("split", "")
        split_counts[split_name] += 1
        if "label" in row and row["label"]:
            label = parse_label_text(row["label"])
        else:
            label = [int(float(row.get(source, 0) or 0)) for source in source_columns]
        label_text = label_to_text(label)
        group = row.get("group", "")
        ratio = ratio_from_condition(row.get("condition_path", ""))

        for index, source in enumerate(source_columns):
            if index < len(label) and int(label[index]) == 1:
                source_counts.setdefault(split_name, Counter())[source] += 1
                overall_source_counts[source] += 1
        label_combo_counts.setdefault(split_name, Counter())[label_text] += 1
        overall_label_counts[label_text] += 1
        group_counts.setdefault(split_name, Counter())[group] += 1
        overall_group_counts[group] += 1
        if ratio is not None:
            ratio_counts.setdefault(split_name, Counter())[ratio] += 1
            overall_ratio_counts[ratio] += 1

    by_split = {}
    for split_name in SPLITS:
        by_split[split_name] = {
            "num_samples": int(split_counts.get(split_name, 0)),
            "source_positive_counts": {source: int(source_counts[split_name].get(source, 0)) for source in source_columns},
            "label_combo_counts": {label: int(label_combo_counts[split_name].get(label, 0)) for label in EXPECTED_LABELS},
            "group_counts": {group: int(group_counts[split_name].get(group, 0)) for group in EXPECTED_GROUPS},
            "ratio_counts": {ratio: int(ratio_counts[split_name].get(ratio, 0)) for ratio in EXPECTED_RATIOS},
        }

    total = len(rows)
    source_1_count = int(overall_source_counts.get("source_1", 0))
    triple_count = int(overall_label_counts.get("[1,1,1]", 0))
    diagnostics = {
        "source_1_positive_fraction": float(source_1_count / total) if total else 0.0,
        "triple_combo_fraction": float(triple_count / total) if total else 0.0,
        "source_3_only_count": int(overall_group_counts.get("source_3", 0)),
        "source_3_source_5_mix_count": int(overall_group_counts.get("source_3_source_5_mix", 0)),
        "split_distribution": {split_name: float(split_counts.get(split_name, 0) / total) if total else 0.0 for split_name in SPLITS},
    }

    return {
        "split_file": split_path.as_posix(),
        "num_samples": total,
        "class_names": source_columns,
        "split_counts": {split_name: int(split_counts.get(split_name, 0)) for split_name in SPLITS},
        "by_split": by_split,
        "overall": {
            "source_positive_counts": {source: int(overall_source_counts.get(source, 0)) for source in source_columns},
            "label_combo_counts": {label: int(overall_label_counts.get(label, 0)) for label in EXPECTED_LABELS},
            "group_counts": {group: int(overall_group_counts.get(group, 0)) for group in EXPECTED_GROUPS},
            "ratio_counts": {ratio: int(overall_ratio_counts.get(ratio, 0)) for ratio in EXPECTED_RATIOS},
        },
        "diagnostics": diagnostics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze real dataset split distributions.")
    parser.add_argument("--split", type=Path, required=True, help="Path to real_dataset_split.csv.")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = analyze_distribution(args.split)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"split={args.split}")
    print(f"output={args.output}")
    print(json.dumps(report["diagnostics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
