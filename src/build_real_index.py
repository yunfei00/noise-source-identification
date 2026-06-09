from __future__ import annotations

import argparse
import csv
import json
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from src.dataset import parse_group_label

DEFAULT_SINGLE_DIR = Path("data/single")
DEFAULT_REAL_TRAIN_DIR = Path("data/real_train")
DEFAULT_OUTPUT = Path("outputs/reports/real_dataset_index.csv")
DEFAULT_SUMMARY_OUTPUT = Path("outputs/reports/real_dataset_summary.json")


def discover_class_names(single_dir: str | Path) -> list[str]:
    """Discover ordered source classes from first-level directories in data/single."""
    root = Path(single_dir)
    if not root.exists():
        warnings.warn(f"single directory does not exist: {root}", stacklevel=2)
        return []
    if not root.is_dir():
        raise ValueError(f"single path must be a directory: {root}")
    return sorted(path.name for path in root.iterdir() if path.is_dir())


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


def _relative_to_cwd(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _condition_path(csv_path: Path, group_dir: Path) -> str:
    try:
        parts = csv_path.relative_to(group_dir).parts[:-1]
    except ValueError:
        parts = ()
    return Path(*parts).as_posix() if parts else ""


def _source_columns(label: np.ndarray, class_names: list[str]) -> dict[str, str]:
    return {class_name: str(int(label[index])) for index, class_name in enumerate(class_names)}


def _warn(message: str, warnings_list: list[str]) -> None:
    warnings_list.append(message)
    warnings.warn(message, stacklevel=3)


def scan_single(
    single_dir: str | Path,
    class_names: list[str],
    warnings_list: list[str] | None = None,
) -> list[dict[str, str]]:
    """Recursively scan data/single and create one-hot rows from source directories."""
    warnings_list = warnings_list if warnings_list is not None else []
    root = Path(single_dir)
    rows: list[dict[str, str]] = []
    if not root.exists():
        _warn(f"single directory does not exist: {root}", warnings_list)
        return rows
    if not root.is_dir():
        raise ValueError(f"single path must be a directory: {root}")

    group_dirs = sorted((path for path in root.iterdir() if path.is_dir()), key=lambda path: path.name)
    if not group_dirs:
        _warn(f"single directory is empty: {root}", warnings_list)
        return rows

    class_index = {class_name: index for index, class_name in enumerate(class_names)}
    for group_dir in group_dirs:
        group = group_dir.name
        if group not in class_index:
            _warn(f"Skipping unparseable single group '{group}': not present in class_names", warnings_list)
            continue
        csv_files = sorted(path for path in group_dir.rglob("*.csv") if path.is_file())
        if not csv_files:
            _warn(f"No CSV files found recursively under single group: {group_dir}", warnings_list)
            continue

        label = np.zeros(len(class_names), dtype=np.float32)
        label[class_index[group]] = 1.0
        label_text = label_to_text(label)
        source_values = _source_columns(label, class_names)
        for csv_path in csv_files:
            row = {
                "file": _relative_to_cwd(csv_path),
                "source_root": "single",
                "group": group,
                "condition_path": _condition_path(csv_path, group_dir),
                "label": label_text,
            }
            row.update(source_values)
            rows.append(row)
    return rows


def scan_real_train(
    real_dir: str | Path,
    class_names: list[str],
    warnings_list: list[str] | None = None,
) -> list[dict[str, str]]:
    """Recursively scan data/real_train and parse labels from first-level mix groups."""
    warnings_list = warnings_list if warnings_list is not None else []
    root = Path(real_dir)
    rows: list[dict[str, str]] = []

    if not root.exists():
        _warn(f"real_train directory does not exist: {root}", warnings_list)
        return rows
    if not root.is_dir():
        raise ValueError(f"real_train path must be a directory: {root}")

    group_dirs = sorted((path for path in root.iterdir() if path.is_dir()), key=lambda path: path.name)
    if not group_dirs:
        _warn(f"real_train directory is empty: {root}", warnings_list)
        return rows

    for group_dir in group_dirs:
        group = group_dir.name
        try:
            label = parse_group_label(group, class_names)
        except ValueError as exc:
            _warn(f"Skipping unparseable real_train group '{group}': {exc}", warnings_list)
            continue

        csv_files = sorted(path for path in group_dir.rglob("*.csv") if path.is_file())
        if not csv_files:
            _warn(f"No CSV files found recursively under real_train group: {group_dir}", warnings_list)
            continue

        label_text = label_to_text(label)
        source_values = _source_columns(label, class_names)
        for csv_path in csv_files:
            row = {
                "file": _relative_to_cwd(csv_path),
                "source_root": "real_train",
                "group": group,
                "condition_path": _condition_path(csv_path, group_dir),
                "label": label_text,
            }
            row.update(source_values)
            rows.append(row)
    return rows


def index_fieldnames(class_names: list[str]) -> list[str]:
    return ["file", "source_root", "group", "condition_path", "label", *class_names]


def write_index(rows: list[dict[str, str]], output: str | Path, class_names: list[str]) -> None:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=index_fieldnames(class_names), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _ratio_from_condition(condition_path: str) -> str | None:
    for part in Path(condition_path).parts:
        if part.startswith("ratio_"):
            return part
    return None


def summarize_rows(rows: list[dict[str, str]], class_names: list[str], warnings_list: list[str]) -> dict[str, Any]:
    group_counts = Counter(row["group"] for row in rows)
    label_counts = Counter(row["label"] for row in rows)
    source_root_counts = Counter(row["source_root"] for row in rows)
    ratio_counts = Counter(
        ratio for row in rows if (ratio := _ratio_from_condition(row.get("condition_path", ""))) is not None
    )
    total_real_samples = len(rows)
    return {
        "class_names": class_names,
        "total_samples": total_real_samples,
        "total_real_samples": total_real_samples,
        "single_samples": int(source_root_counts.get("single", 0)),
        "real_train_samples": int(source_root_counts.get("real_train", 0)),
        "group_counts": dict(sorted(group_counts.items())),
        "label_counts": dict(sorted(label_counts.items())),
        "ratio_counts": dict(sorted(ratio_counts.items())),
        "warnings": warnings_list,
    }


def print_summary(summary: dict[str, Any]) -> None:
    print(f"class_names={summary['class_names']}")
    print(f"total_real_samples = {summary['total_real_samples']}")
    print(f"single samples={summary['single_samples']}")
    print(f"real_train samples={summary['real_train_samples']}")

    print("group counts:")
    if not summary["group_counts"]:
        print("  (none)")
    for group, count in summary["group_counts"].items():
        print(f"  {group}: {count}")

    print("label counts:")
    if not summary["label_counts"]:
        print("  (none)")
    for label, count in summary["label_counts"].items():
        print(f"  {label}: {count}")

    if summary["ratio_counts"]:
        print("ratio counts:")
        for ratio, count in summary["ratio_counts"].items():
            print(f"  {ratio}: {count}")

    if summary["warnings"]:
        print("warnings:")
        for message in summary["warnings"]:
            print(f"  warning: {message}")


def write_summary(summary: dict[str, Any], output: str | Path) -> None:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def build_real_index(
    single_dir: str | Path = DEFAULT_SINGLE_DIR,
    real_train_dir: str | Path = DEFAULT_REAL_TRAIN_DIR,
    output: str | Path = DEFAULT_OUTPUT,
    summary_output: str | Path | None = None,
    class_names: list[str] | None = None,
    include_single: bool = True,
    include_real_train: bool = True,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    discovered_class_names = class_names or discover_class_names(single_dir)
    if not discovered_class_names:
        raise ValueError("No class names found. Create source directories under data/single or pass --class-names.")

    warnings_list: list[str] = []
    rows: list[dict[str, str]] = []
    if include_single:
        rows.extend(scan_single(single_dir, discovered_class_names, warnings_list))
    if include_real_train:
        rows.extend(scan_real_train(real_train_dir, discovered_class_names, warnings_list))

    rows = sorted(rows, key=lambda row: (row["source_root"], row["group"], row["condition_path"], row["file"]))
    write_index(rows, output, discovered_class_names)
    summary = summarize_rows(rows, discovered_class_names, warnings_list)
    if summary_output is None:
        summary_output = Path(output).with_name("real_dataset_summary.json")
    write_summary(summary, summary_output)
    print(f"index={output} samples={len(rows)}")
    print(f"summary={summary_output}")
    print_summary(summary)
    return rows, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a unified recursive real dataset index from data/single and data/real_train.")
    parser.add_argument("--single-dir", type=Path, default=DEFAULT_SINGLE_DIR, help="Single-source root directory.")
    parser.add_argument("--real-train-dir", type=Path, default=DEFAULT_REAL_TRAIN_DIR, help="Real combo training root directory.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output unified index CSV path.")
    parser.add_argument("--summary-output", type=Path, default=None, help="Output summary JSON path.")
    parser.add_argument("--class-names", nargs="+", help="Optional ordered class names; defaults to sorted data/single dirs.")
    parser.add_argument("--no-single", action="store_true", help="Do not include data/single rows.")
    parser.add_argument("--no-real-train", action="store_true", help="Do not include data/real_train rows.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_real_index(
        single_dir=args.single_dir,
        real_train_dir=args.real_train_dir,
        output=args.output,
        summary_output=args.summary_output,
        class_names=args.class_names,
        include_single=not args.no_single,
        include_real_train=not args.no_real_train,
    )


if __name__ == "__main__":
    main()
