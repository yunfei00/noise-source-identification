from __future__ import annotations

import argparse
import csv
import json
import random
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

DEFAULT_INPUT = Path("outputs/reports/real_dataset_split.csv")
DEFAULT_OUTPUT = Path("outputs/reports/real_dataset_split_balanced.csv")
DEFAULT_SUMMARY_OUTPUT = Path("outputs/reports/balanced_train_summary.json")


def parse_quota(text: str) -> dict[str, int]:
    quota: dict[str, int] = {}
    if not text.strip():
        return quota
    for item in text.split(","):
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"Invalid quota item '{item}', expected COMBO=N")
        combo = key.strip()
        if not combo or any(ch not in "01" for ch in combo):
            raise ValueError(f"Invalid combo key in quota: {combo!r}")
        count = int(value.strip())
        if count < 0:
            raise ValueError(f"Quota must be non-negative for combo {combo}")
        quota[combo] = count
    return quota


def label_to_combo(label_text: str) -> str:
    stripped = label_text.strip()
    if not (stripped.startswith("[") and stripped.endswith("]")):
        raise ValueError(f"Invalid label text: {label_text}")
    return "".join(str(int(float(token.strip()))) for token in stripped[1:-1].split(",") if token.strip())


def source_positive_ratio(rows: list[dict[str, str]]) -> dict[str, float]:
    if not rows:
        return {}
    source_fields = [field for field in rows[0] if field.startswith("source_") and field != "source_root"]
    if not source_fields:
        n = len(label_to_combo(rows[0]["label"]))
        source_fields = [f"source_{i + 1}" for i in range(n)]
    ratios: dict[str, float] = {}
    for idx, field in enumerate(source_fields):
        positives = 0
        for row in rows:
            if field in row and row[field] != "":
                positives += int(float(row[field]))
            else:
                positives += int(label_to_combo(row["label"])[idx])
        ratios[field] = positives / len(rows)
    return ratios


def create_balanced_split(input_path: str | Path, output_path: str | Path, quota: dict[str, int], seed: int = 42) -> dict:
    input_path = Path(input_path)
    output_path = Path(output_path)
    with input_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Input split has no header: {input_path}")
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    rng = random.Random(seed)
    train_by_combo: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("split") == "train":
            train_by_combo[label_to_combo(row["label"])].append(row)

    selected_ids: set[int] = set()
    insufficient: list[str] = []
    for combo, target in quota.items():
        bucket = list(train_by_combo.get(combo, []))
        rng.shuffle(bucket)
        if len(bucket) < target:
            message = f"combo {combo} has {len(bucket)} train sample(s), below quota {target}; using all available"
            warnings.warn(message, stacklevel=2)
            insufficient.append(message)
            selected = bucket
        else:
            selected = bucket[:target]
        selected_ids.update(id(row) for row in selected)

    output_rows: list[dict[str, str]] = []
    for row in rows:
        out = dict(row)
        out["selected_for_train"] = "true" if row.get("split") == "train" and id(row) in selected_ids else "false"
        output_rows.append(out)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_fields = [field for field in fieldnames if field != "selected_for_train"] + ["selected_for_train"]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=out_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(output_rows)

    train_before = [row for row in rows if row.get("split") == "train"]
    train_after = [row for row in output_rows if row.get("split") == "train" and row.get("selected_for_train") == "true"]
    before_ratios = source_positive_ratio(train_before)
    after_ratios = source_positive_ratio(train_after)
    summary = {
        "before_train_combo_counts": dict(sorted(Counter(label_to_combo(row["label"]) for row in train_before).items())),
        "after_train_combo_counts": dict(sorted(Counter(label_to_combo(row["label"]) for row in train_after).items())),
        "before_source_positive_ratio": before_ratios,
        "after_source_positive_ratio": after_ratios,
        "source5_positive_ratio_before": before_ratios.get("source_5"),
        "source5_positive_ratio_after": after_ratios.get("source_5"),
        "total_train_before": len(train_before),
        "total_train_after": len(train_after),
        "quota_config": quota,
        "insufficient_combo_warnings": insufficient,
    }
    summary_path = output_path.parent / DEFAULT_SUMMARY_OUTPUT.name
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select a quota-balanced real train split while preserving val/test rows.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--quota", required=True, type=parse_quota)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    create_balanced_split(args.input, args.output, args.quota, args.seed)


if __name__ == "__main__":
    main()
