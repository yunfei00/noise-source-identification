from __future__ import annotations

import argparse
import csv
import random
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

DEFAULT_INDEX = Path("outputs/reports/real_dataset_index.csv")
DEFAULT_OUTPUT = Path("outputs/reports/real_dataset_split.csv")
SPLITS = ("train", "val", "test")


def read_index(index: str | Path) -> tuple[list[dict[str, str]], list[str]]:
    index_path = Path(index)
    if not index_path.exists():
        raise FileNotFoundError(f"Real dataset index not found: {index_path}")
    with index_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Index has no header: {index_path}")
        rows = list(reader)
        fieldnames = list(reader.fieldnames)
    if not rows:
        warnings.warn(f"Index contains no rows: {index_path}", stacklevel=2)
    required = {"file", "source_root", "group", "condition_path", "label"}
    missing = required.difference(fieldnames)
    if missing:
        raise ValueError(f"Index missing required field(s): {', '.join(sorted(missing))}")
    return rows, fieldnames


def _split_counts(n: int, train_ratio: float, val_ratio: float, test_ratio: float) -> tuple[int, int, int]:
    if n <= 0:
        return 0, 0, 0
    ratios = np.asarray([train_ratio, val_ratio, test_ratio], dtype=np.float64)
    if np.any(ratios < 0) or float(ratios.sum()) <= 0:
        raise ValueError("split ratios must be non-negative and sum to a positive value")
    ratios = ratios / ratios.sum()
    raw = ratios * n
    counts = np.floor(raw).astype(int)
    remainder = n - int(counts.sum())
    if remainder:
        order = np.argsort(-(raw - counts))
        for index in order[:remainder]:
            counts[index] += 1
    if n >= 3:
        for index in range(3):
            if ratios[index] > 0 and counts[index] == 0:
                donor = int(np.argmax(counts))
                if counts[donor] > 1:
                    counts[donor] -= 1
                    counts[index] += 1
    return int(counts[0]), int(counts[1]), int(counts[2])


def ratio_key(condition_path: str) -> str:
    for part in Path(condition_path).parts:
        if part.startswith("ratio_"):
            return part
    return "__no_ratio__"


def _assign_bucket(
    rows: list[dict[str, str]],
    rng: random.Random,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> list[dict[str, str]]:
    shuffled = list(rows)
    rng.shuffle(shuffled)
    train_count, val_count, test_count = _split_counts(len(shuffled), train_ratio, val_ratio, test_ratio)
    assignments = ["train"] * train_count + ["val"] * val_count + ["test"] * test_count
    output: list[dict[str, str]] = []
    for row, split in zip(shuffled, assignments, strict=True):
        split_row = dict(row)
        split_row["split"] = split
        output.append(split_row)
    return output


def split_rows(
    rows: list[dict[str, str]],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> list[dict[str, str]]:
    """Split rows by group, and by ratio condition within each group when present."""
    rng = random.Random(seed)
    buckets: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        buckets[(row["group"], ratio_key(row.get("condition_path", "")))].append(row)

    split_output: list[dict[str, str]] = []
    for (group, ratio), bucket_rows in sorted(buckets.items()):
        if len(bucket_rows) < 3:
            warnings.warn(
                f"bucket group={group} ratio={ratio} has only {len(bucket_rows)} sample(s); "
                "not every split can receive this bucket.",
                stacklevel=2,
            )
        split_output.extend(_assign_bucket(bucket_rows, rng, train_ratio, val_ratio, test_ratio))
    return sorted(split_output, key=lambda row: (row["split"], row["source_root"], row["group"], row["condition_path"], row["file"]))


def write_split(rows: list[dict[str, str]], output: str | Path, input_fieldnames: list[str]) -> None:
    fieldnames = [field for field in input_fieldnames if field != "split"] + ["split"]
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def print_split_summary(rows: list[dict[str, str]]) -> None:
    print(f"total_real_samples={len(rows)}")
    print(f"split samples={len(rows)}")
    print("split counts:")
    split_counts = Counter(row["split"] for row in rows)
    for split in SPLITS:
        print(f"  {split}: {split_counts.get(split, 0)}")

    print("group split counts:")
    groups = sorted({row["group"] for row in rows})
    for group in groups:
        counts = Counter(row["split"] for row in rows if row["group"] == group)
        print(f"  {group}: train={counts.get('train', 0)} val={counts.get('val', 0)} test={counts.get('test', 0)}")

    ratio_rows = [row for row in rows if ratio_key(row.get("condition_path", "")) != "__no_ratio__"]
    if ratio_rows:
        print("ratio split counts:")
        for ratio in sorted({ratio_key(row.get("condition_path", "")) for row in ratio_rows}):
            counts = Counter(row["split"] for row in ratio_rows if ratio_key(row.get("condition_path", "")) == ratio)
            print(f"  {ratio}: train={counts.get('train', 0)} val={counts.get('val', 0)} test={counts.get('test', 0)}")


def split_real_dataset(
    index: str | Path = DEFAULT_INDEX,
    output: str | Path = DEFAULT_OUTPUT,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> list[dict[str, str]]:
    rows, fieldnames = read_index(index)
    split_output = split_rows(rows, train_ratio, val_ratio, test_ratio, seed)
    write_split(split_output, output, fieldnames)
    print(f"index={index}")
    print(f"split={output}")
    print_split_summary(split_output)
    return split_output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create train/val/test splits for a unified real dataset index.")
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX, help="Input real dataset index CSV.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output split CSV.")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_real_dataset(args.index, args.output, args.train_ratio, args.val_ratio, args.test_ratio, args.seed)


if __name__ == "__main__":
    main()
