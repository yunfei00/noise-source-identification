from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path


def _parse_counts(text: str) -> list[int]:
    values = sorted({int(x.strip()) for x in text.split(",") if x.strip()})
    if not values or any(x <= 0 for x in values):
        raise argparse.ArgumentTypeError("counts must be positive comma-separated integers")
    return values


def _discover_groups(root: Path) -> dict[str, list[Path]]:
    if not root.exists():
        raise FileNotFoundError(f"Raw data root not found: {root}")
    direct = {}
    for child in sorted(p for p in root.iterdir() if p.is_dir()):
        files = sorted(p for p in child.rglob("*.csv") if p.is_file())
        if files:
            direct[child.name] = files
    if direct:
        return direct
    files = sorted(p for p in root.rglob("*.csv") if p.is_file())
    if files:
        return {root.name: files}
    raise ValueError(f"No CSV files found under: {root}")


def _parse_group_map(text: str) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise argparse.ArgumentTypeError("group-label-map entries must use GROUP=INDEX")
        group, index_text = item.split("=", 1)
        group = group.strip()
        try:
            index = int(index_text.strip())
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Invalid label index in {item!r}") from exc
        if not group or index < 0 or group in mapping:
            raise argparse.ArgumentTypeError(f"Invalid group-label-map entry: {item!r}")
        mapping[group] = index
    if not mapping:
        raise argparse.ArgumentTypeError("group-label-map cannot be empty")
    indices = sorted(mapping.values())
    if indices != list(range(len(mapping))):
        raise argparse.ArgumentTypeError("group-label-map indices must be unique contiguous values starting at 0")
    return mapping


def _split_group(files: list[Path], seed: int, val_ratio: float, test_ratio: float):
    if val_ratio < 0 or test_ratio < 0 or val_ratio + test_ratio >= 1:
        raise ValueError("val_ratio/test_ratio must be >=0 and sum to <1")
    items = list(files)
    random.Random(seed).shuffle(items)
    n = len(items)
    n_val = int(round(n * val_ratio))
    n_test = int(round(n * test_ratio))
    if n >= 3:
        n_val = max(1, n_val)
        n_test = max(1, n_test)
    while n_val + n_test >= n and (n_val > 0 or n_test > 0):
        if n_val >= n_test and n_val > 0:
            n_val -= 1
        elif n_test > 0:
            n_test -= 1
    val = items[:n_val]
    test = items[n_val:n_val + n_test]
    train = items[n_val + n_test:]
    return train, val, test


def build_from_raw(
    raw_root: Path,
    output_dir: Path,
    counts: list[int],
    seed: int,
    val_ratio: float,
    test_ratio: float,
    group_label_map: dict[str, int],
) -> dict:
    groups = _discover_groups(raw_root)
    if set(groups) != set(group_label_map):
        missing = sorted(set(groups) - set(group_label_map))
        extra = sorted(set(group_label_map) - set(groups))
        raise ValueError(
            "group-label-map must exactly match discovered raw folders; "
            f"missing={missing} extra={extra} discovered={sorted(groups)}"
        )
    split = {}
    for i, (group, files) in enumerate(groups.items()):
        split[group] = _split_group(files, seed + i * 1009, val_ratio, test_ratio)

    output_dir.mkdir(parents=True, exist_ok=True)
    available = {g: len(parts[0]) for g, parts in split.items()}
    total = {g: sum(len(x) for x in parts) for g, parts in split.items()}
    summary = {
        "raw_root": str(raw_root.resolve()),
        "seed": seed,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "total_per_group": total,
        "available_train_per_group": available,
        "group_label_map": group_label_map,
        "experiments": {},
    }
    fields = ["file", "source_root", "group", "condition_path", "label", "split", "selected_for_train"]

    group_names = list(groups)
    for count in counts:
        selected = {}
        for group in group_names:
            train = split[group][0]
            selected[group] = {str(p.resolve()) for p in train[:min(count, len(train))]}

        out_path = output_dir / f"real_dataset_split_ablation_{count}.csv"
        actual = {g: len(selected[g]) for g in group_names}
        with out_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for group in group_names:
                group_index = group_label_map[group]
                label = "[" + ",".join("1" if j == group_index else "0" for j in range(len(group_names))) + "]"
                train, val, test = split[group]
                for split_name, files in (("train", train), ("val", val), ("test", test)):
                    for path in files:
                        resolved = str(path.resolve())
                        writer.writerow({
                            "file": resolved,
                            "source_root": str(raw_root.resolve()),
                            "group": group,
                            "condition_path": "",
                            "label": label,
                            "split": split_name,
                            "selected_for_train": "true" if split_name != "train" or resolved in selected[group] else "false",
                        })
        summary["experiments"][str(count)] = {
            "file": str(out_path.resolve()),
            "requested_per_group": count,
            "actual_per_group": actual,
            "selected_train_total": sum(actual.values()),
            "val_total_fixed": sum(len(v[1]) for v in split.values()),
            "test_total_fixed": sum(len(v[2]) for v in split.values()),
            "limited_groups": [g for g, n in actual.items() if n < count],
        }

    manifest = output_dir / "ablation_manifest.json"
    manifest.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("========== COPY THIS SUMMARY ==========")
    print("groups=" + ",".join(group_names))
    print("group_label_map=" + json.dumps(group_label_map, ensure_ascii=False, sort_keys=True))
    print("total_per_group=" + json.dumps(total, ensure_ascii=False))
    print("available_train_per_group=" + json.dumps(available, ensure_ascii=False))
    for count in counts:
        item = summary["experiments"][str(count)]
        print(
            f"N={count} selected_train={item['selected_train_total']} "
            f"actual_per_group={json.dumps(item['actual_per_group'], ensure_ascii=False)} "
            f"val_fixed={item['val_total_fixed']} test_fixed={item['test_total_fixed']}"
        )
    print(f"manifest={manifest.resolve()}")
    print("=======================================")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build nested sample-count ablation splits directly from raw single-source CSV folders."
    )
    parser.add_argument("--raw-root", type=Path, required=True, help="Folder containing one subfolder per source/group.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/reports/ablation"))
    parser.add_argument("--counts", type=_parse_counts, default=_parse_counts("100,200,500,1000,1500"))
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--group-label-map",
        type=_parse_group_map,
        required=True,
        help="Explicit raw-folder to model-label-index mapping, e.g. source_1=0,source_3=1,source_5=2.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_from_raw(
        args.raw_root, args.output_dir, args.counts, args.seed,
        args.val_ratio, args.test_ratio, args.group_label_map,
    )


if __name__ == "__main__":
    main()
