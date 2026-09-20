from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def _parse_counts(text: str) -> list[int]:
    values = sorted({int(x.strip()) for x in text.split(",") if x.strip()})
    if not values or any(x <= 0 for x in values):
        raise argparse.ArgumentTypeError("counts must be positive comma-separated integers")
    return values


def _read_split(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Split CSV has no header: {path}")
        rows = list(reader)
        fields = list(reader.fieldnames)
    required = {"file", "group", "split"}
    missing = required.difference(fields)
    if missing:
        raise ValueError(f"Split CSV missing fields: {sorted(missing)}")
    return rows, fields


def build_ablation_splits(
    input_path: Path,
    output_dir: Path,
    counts: list[int],
    seed: int,
    groups: list[str] | None,
) -> dict:
    rows, fields = _read_split(input_path)
    train_by_group: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("split") == "train":
            train_by_group[row["group"]].append(row)

    target_groups = groups or sorted(train_by_group)
    unknown = sorted(set(target_groups).difference(train_by_group))
    if unknown:
        raise ValueError(f"Requested group(s) not present in train split: {unknown}")

    # One deterministic shuffle per group. Smaller experiments are strict subsets
    # of larger experiments, so sample-count is the only intended variable.
    shuffled: dict[str, list[dict[str, str]]] = {}
    for index, group in enumerate(target_groups):
        bucket = list(train_by_group[group])
        random.Random(seed + index * 1009).shuffle(bucket)
        shuffled[group] = bucket

    available = {group: len(shuffled[group]) for group in target_groups}
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "input": str(input_path),
        "seed": seed,
        "target_groups": target_groups,
        "available_train_per_group": available,
        "experiments": {},
    }

    out_fields = [f for f in fields if f != "selected_for_train"] + ["selected_for_train"]
    for count in counts:
        selected_files = set()
        actual = {}
        for group in target_groups:
            chosen = shuffled[group][: min(count, len(shuffled[group]))]
            selected_files.update(row["file"] for row in chosen)
            actual[group] = len(chosen)

        out_rows = []
        for row in rows:
            out = dict(row)
            if row.get("split") == "train":
                if row["group"] in target_groups:
                    out["selected_for_train"] = "true" if row["file"] in selected_files else "false"
                else:
                    # Non-target train groups are disabled to keep this a clean
                    # single-source/group sample-count experiment.
                    out["selected_for_train"] = "false"
            else:
                # RealCsvDataset ignores selected_for_train for val/test; keep an
                # explicit marker here for easier human inspection.
                out["selected_for_train"] = "true"
            out_rows.append(out)

        output = output_dir / f"real_dataset_split_ablation_{count}.csv"
        with output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=out_fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(out_rows)

        split_counts = Counter(row["split"] for row in out_rows)
        selected_count = sum(
            row["split"] == "train" and row["selected_for_train"] == "true"
            for row in out_rows
        )
        summary["experiments"][str(count)] = {
            "file": str(output),
            "requested_per_group": count,
            "actual_per_group": actual,
            "selected_train_total": selected_count,
            "val_total_fixed": split_counts.get("val", 0),
            "test_total_fixed": split_counts.get("test", 0),
            "limited_groups": [g for g, n in actual.items() if n < count],
        }

    summary_path = output_dir / "ablation_manifest.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("========== COPY THIS SUMMARY ==========")
    print("groups=" + ",".join(target_groups))
    print("available_train_per_group=" + json.dumps(available, ensure_ascii=False))
    for count in counts:
        item = summary["experiments"][str(count)]
        print(
            f"N={count} selected_train={item['selected_train_total']} "
            f"actual_per_group={json.dumps(item['actual_per_group'], ensure_ascii=False)} "
            f"val_fixed={item['val_total_fixed']} test_fixed={item['test_total_fixed']}"
        )
    print(f"manifest={summary_path.resolve()}")
    print("=======================================")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create nested train-size ablation split CSVs while keeping val/test fixed."
    )
    parser.add_argument("--input", type=Path, default=Path("outputs/reports/real_dataset_split.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/reports/ablation"))
    parser.add_argument("--counts", type=_parse_counts, default=_parse_counts("100,200,500,1000,1500"))
    parser.add_argument(
        "--groups",
        default="",
        help="Optional comma-separated group names. Default: every train group.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    groups = [x.strip() for x in args.groups.split(",") if x.strip()] or None
    build_ablation_splits(args.input, args.output_dir, args.counts, args.seed, groups)


if __name__ == "__main__":
    main()
