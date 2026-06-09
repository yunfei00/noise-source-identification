from __future__ import annotations

import argparse
import csv
import random
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from src.dataset import parse_group_label

DEFAULT_CLASS_NAMES = ["source_1", "source_3", "source_5"]
INDEX_FIELDNAMES = ["file", "group", "condition_path", "label", *DEFAULT_CLASS_NAMES]
SPLIT_FIELDNAMES = ["file", "group", "label", "split"]


def label_to_text(label: np.ndarray | list[float] | list[int]) -> str:
    values = [int(value) for value in np.asarray(label).astype(int).tolist()]
    return "[" + ",".join(str(value) for value in values) + "]"


def parse_label_text(label_text: str) -> list[int]:
    stripped = label_text.strip()
    if not (stripped.startswith("[") and stripped.endswith("]")):
        raise ValueError(f"Invalid label text: {label_text}")
    body = stripped[1:-1].strip()
    if not body:
        return []
    return [int(float(token.strip())) for token in body.split(",")]


def scan_real_train(
    real_dir: str | Path,
    class_names: list[str] | None = None,
) -> list[dict[str, str]]:
    """Recursively scan real_train CSV files and parse labels from first-level groups."""
    root = Path(real_dir)
    class_names = class_names or DEFAULT_CLASS_NAMES
    rows: list[dict[str, str]] = []

    if not root.exists():
        warnings.warn(f"real_train directory does not exist: {root}", stacklevel=2)
        return rows
    if not root.is_dir():
        raise ValueError(f"real_train path must be a directory: {root}")

    group_dirs = sorted((path for path in root.iterdir() if path.is_dir()), key=lambda path: path.name)
    if not group_dirs:
        warnings.warn(f"real_train directory is empty: {root}", stacklevel=2)
        return rows

    for group_dir in group_dirs:
        group = group_dir.name
        try:
            label = parse_group_label(group, class_names)
        except ValueError as exc:
            warnings.warn(f"Skipping unparseable real_train group '{group}': {exc}", stacklevel=2)
            continue

        csv_files = sorted(path for path in group_dir.rglob("*.csv") if path.is_file())
        if not csv_files:
            warnings.warn(f"No CSV files found recursively under real_train group: {group_dir}", stacklevel=2)
            continue

        label_text = label_to_text(label)
        source_values = {class_name: str(int(label[index])) for index, class_name in enumerate(class_names)}
        for csv_path in csv_files:
            relative_file = csv_path.relative_to(root).as_posix()
            condition_parts = Path(relative_file).parts[1:-1]
            row = {
                "file": relative_file,
                "group": group,
                "condition_path": Path(*condition_parts).as_posix() if condition_parts else "",
                "label": label_text,
            }
            for class_name in DEFAULT_CLASS_NAMES:
                row[class_name] = source_values.get(class_name, "0")
            rows.append(row)

    if not rows:
        warnings.warn(f"No parseable real_train CSV files found under: {root}", stacklevel=2)
    return rows


def write_index(rows: list[dict[str, str]], output: str | Path) -> None:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=INDEX_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def print_counts(rows: list[dict[str, str]]) -> None:
    group_counts = Counter(row["group"] for row in rows)
    label_counts = Counter(row["label"] for row in rows)

    print("group counts:")
    if not group_counts:
        print("  (none)")
    for group, count in sorted(group_counts.items()):
        print(f"  {group}: {count}")

    print("label counts:")
    if not label_counts:
        print("  (none)")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count}")


def _split_counts(n: int, train_ratio: float, val_ratio: float, test_ratio: float) -> tuple[int, int, int]:
    if n <= 0:
        return 0, 0, 0
    ratios = np.asarray([train_ratio, val_ratio, test_ratio], dtype=np.float64)
    if np.any(ratios < 0) or float(ratios.sum()) <= 0:
        raise ValueError("real_split ratios must be non-negative and sum to a positive value")
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


def split_real_rows(
    rows: list[dict[str, str]],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    split_by_group: bool = True,
) -> list[dict[str, str]]:
    rng = random.Random(seed)
    split_rows: list[dict[str, str]] = []
    buckets: dict[str, list[dict[str, str]]] = defaultdict(list)
    if split_by_group:
        for row in rows:
            buckets[row["group"]].append(row)
    else:
        buckets["__all__"] = list(rows)

    for bucket_name, bucket_rows in sorted(buckets.items()):
        shuffled = list(bucket_rows)
        rng.shuffle(shuffled)
        n = len(shuffled)
        if n < 3:
            warnings.warn(
                f"real_split bucket '{bucket_name}' has only {n} sample(s); "
                "not every split can receive this group.",
                stacklevel=2,
            )
        train_count, val_count, test_count = _split_counts(n, train_ratio, val_ratio, test_ratio)
        assignments = (
            ["train"] * train_count
            + ["val"] * val_count
            + ["test"] * test_count
        )
        for row, split in zip(shuffled, assignments, strict=True):
            split_rows.append(
                {
                    "file": row["file"],
                    "group": row["group"],
                    "label": row["label"],
                    "split": split,
                }
            )

    return sorted(split_rows, key=lambda row: (row["split"], row["group"], row["file"]))


def write_split(rows: list[dict[str, str]], output: str | Path) -> None:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SPLIT_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def build_real_index(
    real_dir: str | Path,
    output: str | Path,
    class_names: list[str] | None = None,
    split_output: str | Path | None = "outputs/reports/real_train_split.csv",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    split_by_group: bool = True,
) -> list[dict[str, str]]:
    rows = scan_real_train(real_dir, class_names)
    write_index(rows, output)
    print(f"index={output} samples={len(rows)}")
    print_counts(rows)
    if split_output is not None:
        split_rows = split_real_rows(rows, train_ratio, val_ratio, test_ratio, seed, split_by_group)
        write_split(split_rows, split_output)
        print(f"split={split_output} samples={len(split_rows)}")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build recursive real_train index and split CSV files.")
    parser.add_argument("--dir", type=Path, default=Path("data/real_train"), help="real_train root directory.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/reports/real_train_index.csv"),
        help="Output real_train index CSV path.",
    )
    parser.add_argument(
        "--split-output",
        type=Path,
        default=Path("outputs/reports/real_train_split.csv"),
        help="Output real_train split CSV path.",
    )
    parser.add_argument("--no-split", action="store_true", help="Only write the index CSV.")
    parser.add_argument("--class-names", nargs="+", default=DEFAULT_CLASS_NAMES, help="Ordered source class names.")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-split-by-group", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_real_index(
        args.dir,
        args.output,
        class_names=args.class_names,
        split_output=None if args.no_split else args.split_output,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        split_by_group=not args.no_split_by_group,
    )


if __name__ == "__main__":
    main()
