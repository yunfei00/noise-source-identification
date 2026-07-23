from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from src.build_real_index import (
    _ratio_from_condition,
    discover_class_names,
    scan_real_dataset,
    scan_single,
)
from src.features import read_signal_csv_info


DEFAULT_OUTPUT = Path("outputs/reports/real_data_audit.json")
_METRIC_KEYS = (
    "median_db",
    "std_db",
    "peak_to_peak_db",
    "diff_p99_db",
    "max_jump_db",
)


def _round(value: float, digits: int = 4) -> float:
    return round(float(value), digits)


def _csv_count(root: Path) -> int:
    if not root.exists() or not root.is_dir():
        return 0
    return sum(1 for path in root.rglob("*.csv") if path.is_file())


def _first_level_directories(root: Path) -> list[str]:
    if not root.exists() or not root.is_dir():
        return []
    return sorted(path.name for path in root.iterdir() if path.is_dir())


def _distribution(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {}
    return {
        "min": _round(np.min(array)),
        "p05": _round(np.percentile(array, 5)),
        "median": _round(np.median(array)),
        "p95": _round(np.percentile(array, 95)),
        "max": _round(np.max(array)),
    }


def compute_db_statistics(values: np.ndarray, jump_threshold_db: float) -> dict[str, Any]:
    """Compute per-file dB and jump statistics without modifying the samples."""
    signal = np.asarray(values, dtype=np.float32).reshape(-1)
    if signal.size == 0:
        raise ValueError("empty signal")
    if not np.all(np.isfinite(signal)):
        raise ValueError("signal contains non-finite values after parsing")

    differences = np.abs(np.diff(signal).astype(np.float64, copy=False))
    minimum = float(np.min(signal))
    maximum = float(np.max(signal))
    boundary_hits = max(int(np.count_nonzero(signal == minimum)), int(np.count_nonzero(signal == maximum)))
    return {
        "num_samples": int(signal.size),
        "min_db": _round(minimum),
        "p01_db": _round(np.percentile(signal, 1)),
        "p05_db": _round(np.percentile(signal, 5)),
        "mean_db": _round(np.mean(signal)),
        "median_db": _round(np.median(signal)),
        "p95_db": _round(np.percentile(signal, 95)),
        "p99_db": _round(np.percentile(signal, 99)),
        "max_db": _round(maximum),
        "std_db": _round(np.std(signal)),
        "peak_to_peak_db": _round(maximum - minimum),
        "diff_p95_db": _round(np.percentile(differences, 95)) if differences.size else 0.0,
        "diff_p99_db": _round(np.percentile(differences, 99)) if differences.size else 0.0,
        "max_jump_db": _round(np.max(differences)) if differences.size else 0.0,
        "jump_count": int(np.count_nonzero(differences >= jump_threshold_db)),
        "jump_ratio": _round(np.mean(differences >= jump_threshold_db)) if differences.size else 0.0,
        "nonnegative_ratio": _round(np.mean(signal >= 0.0)),
        "unique_value_count": int(np.unique(signal).size),
        "boundary_repeat_ratio": _round(boundary_hits / signal.size),
    }


def _metric_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"file_count": 0}
    return {
        "file_count": len(rows),
        "observed_db_min": _round(min(float(row["min_db"]) for row in rows)),
        "observed_db_max": _round(max(float(row["max_db"]) for row in rows)),
        "file_median_db": _distribution([float(row["median_db"]) for row in rows]),
        "file_std_db": _distribution([float(row["std_db"]) for row in rows]),
        "file_peak_to_peak_db": _distribution([float(row["peak_to_peak_db"]) for row in rows]),
        "file_max_jump_db": _distribution([float(row["max_jump_db"]) for row in rows]),
    }


def _summaries_by(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        value = str(row.get(key, "")) or "(none)"
        grouped[value].append(row)
    return {name: _metric_summary(group_rows) for name, group_rows in sorted(grouped.items())}


def _add_issue(
    issues: list[dict[str, str]],
    row: dict[str, Any],
    issue_type: str,
    severity: str,
    detail: str,
) -> None:
    issues.append(
        {
            "severity": severity,
            "type": issue_type,
            "file": str(row.get("file", "")),
            "group": str(row.get("group", "")),
            "condition_path": str(row.get("condition_path", "")),
            "detail": detail,
        }
    )


def _flag_condition_outliers(
    rows: list[dict[str, Any]],
    issues: list[dict[str, str]],
    robust_z_threshold: float,
) -> None:
    by_condition: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_condition[(str(row["group"]), str(row["condition_path"]))].append(row)

    for (group, condition), condition_rows in by_condition.items():
        if len(condition_rows) < 8:
            continue
        for metric in _METRIC_KEYS:
            values = np.asarray([float(row[metric]) for row in condition_rows], dtype=np.float64)
            center = float(np.median(values))
            mad = float(np.median(np.abs(values - center)))
            if mad < 1e-8:
                continue
            robust_scale = 1.4826 * mad
            scores = np.abs(values - center) / robust_scale
            for row, value, score in zip(condition_rows, values, scores):
                if score >= robust_z_threshold:
                    _add_issue(
                        issues,
                        row,
                        "condition_outlier",
                        "review",
                        (
                            f"{metric}={value:.4f} is a robust outlier within "
                            f"group={group}, condition={condition or '(none)'} "
                            f"(median={center:.4f}, robust_z={score:.2f})"
                        ),
                    )


def _expected_labels(class_names: list[str]) -> list[str]:
    if not class_names:
        return []
    return [
        "[" + ",".join(str((value >> (len(class_names) - 1 - index)) & 1) for index in range(len(class_names))) + "]"
        for value in range(1, 2 ** len(class_names))
    ]


def _recommended_preprocessing(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    medians = np.asarray([float(row["median_db"]) for row in rows], dtype=np.float64)
    standard_deviations = np.asarray([float(row["std_db"]) for row in rows], dtype=np.float64)
    length_counts = Counter(int(row["num_samples"]) for row in rows)
    modal_length = length_counts.most_common(1)[0][0]
    level_min = math.floor(float(np.percentile(medians, 1)) - 5.0)
    level_max = math.ceil(float(np.percentile(medians, 99)) + 5.0)
    variation_scale = max(1.0, math.ceil(float(np.percentile(standard_deviations, 99))))
    return {
        "method": "db_trace",
        "signal_length": modal_length,
        "padding": "per-file median dB",
        "center_before_stft": "per-file median dB",
        "db_level_range": [level_min, level_max],
        "db_variation_scale": variation_scale,
        "note": "Review flagged files before applying these values to configs/train.yaml.",
    }


def _build_copy_paste_summary(report: dict[str, Any]) -> str:
    counts = report["counts"]
    signal = report["signal_summary"]
    recommendation = report["recommended_preprocessing"]
    issue_counts = report["issues"]["counts_by_type"]
    lines = [
        f"class_names={report['class_names']}",
        (
            f"files: raw={counts['raw_csv_files']} accepted={counts['discovered_files']} "
            f"skipped={counts['skipped_or_unrecognized_csv_files']} "
            f"parsed={counts['parsed_files']} failed={counts['parse_failed_files']}"
        ),
        f"label_counts={counts['label_counts']}",
        f"ratio_counts={counts['ratio_counts']}",
        f"length_counts={counts['length_counts']}",
        f"observed_db_range=[{signal.get('observed_db_min')}, {signal.get('observed_db_max')}]",
        f"file_median_db={signal.get('file_median_db', {})}",
        f"file_std_db={signal.get('file_std_db', {})}",
        f"issue_counts={issue_counts}",
        f"recommended_preprocessing={recommendation}",
    ]
    return "\n".join(lines)


def audit_real_data(
    single_dir: str | Path = "data/single",
    combo_dir: str | Path = "data/real_dataset",
    output: str | Path = DEFAULT_OUTPUT,
    class_names: list[str] | None = None,
    expected_length: int | None = None,
    no_signal_threshold_db: float | None = None,
    jump_threshold_db: float = 12.0,
    flat_std_threshold_db: float = 0.05,
    robust_z_threshold: float = 5.0,
) -> dict[str, Any]:
    """Audit real dB CSV data and write all summaries/issues to one JSON file."""
    classes = class_names or discover_class_names(single_dir)
    single_root = Path(single_dir)
    combo_root = Path(combo_dir)
    raw_single_file_count = _csv_count(single_root)
    raw_combo_file_count = _csv_count(combo_root)
    scan_warnings: list[str] = []
    indexed_rows = [
        *scan_single(single_dir, classes, scan_warnings),
        *scan_real_dataset(combo_dir, classes, scan_warnings),
    ]
    indexed_rows.sort(key=lambda row: (row["source_root"], row["group"], row["condition_path"], row["file"]))

    parsed_rows: list[dict[str, Any]] = []
    issues: list[dict[str, str]] = []
    fingerprints: dict[str, list[dict[str, Any]]] = defaultdict(list)
    data_marker_counts: Counter[str] = Counter()

    for indexed_row in indexed_rows:
        path = Path(indexed_row["file"])
        try:
            info = read_signal_csv_info(path)
            statistics = compute_db_statistics(info.values, jump_threshold_db)
        except Exception as exc:
            _add_issue(issues, indexed_row, "parse_failure", "error", str(exc))
            continue

        ratio = _ratio_from_condition(indexed_row.get("condition_path", "")) or "(none)"
        row: dict[str, Any] = {
            **indexed_row,
            "ratio": ratio,
            "has_data_marker": bool(info.found_data_line),
            "numeric_value_count": int(info.numeric_value_count),
            "discarded_nonfinite_count": int(info.discarded_nonfinite_count),
            **statistics,
        }
        parsed_rows.append(row)
        data_marker_counts["with_DATA_marker" if info.found_data_line else "legacy_without_DATA_marker"] += 1
        fingerprint = hashlib.sha1(info.values.astype(np.float32, copy=False).tobytes()).hexdigest()
        fingerprints[fingerprint].append(row)

        if info.discarded_nonfinite_count:
            _add_issue(
                issues,
                row,
                "nonfinite_values",
                "error",
                (
                    f"discarded_nonfinite_count={info.discarded_nonfinite_count}, "
                    f"numeric_value_count={info.numeric_value_count}"
                ),
            )

        if float(row["std_db"]) <= flat_std_threshold_db or float(row["peak_to_peak_db"]) <= 0.5:
            _add_issue(
                issues,
                row,
                "flat_or_nearly_flat",
                "error",
                f"std_db={row['std_db']}, peak_to_peak_db={row['peak_to_peak_db']}",
            )
        if float(row["min_db"]) < -160.0 or float(row["max_db"]) > 10.0:
            _add_issue(
                issues,
                row,
                "implausible_db_range",
                "error",
                f"observed range=[{row['min_db']}, {row['max_db']}] dB",
            )
        if float(row["nonnegative_ratio"]) > 0.0:
            _add_issue(
                issues,
                row,
                "nonnegative_db_values",
                "error",
                f"nonnegative_ratio={row['nonnegative_ratio']}",
            )
        if float(row["boundary_repeat_ratio"]) >= 0.10 and int(row["unique_value_count"]) > 1:
            _add_issue(
                issues,
                row,
                "possible_clipping",
                "review",
                f"boundary_repeat_ratio={row['boundary_repeat_ratio']}",
            )
        if float(row["max_jump_db"]) >= jump_threshold_db:
            _add_issue(
                issues,
                row,
                "large_db_jump",
                "review",
                f"max_jump_db={row['max_jump_db']}, jump_count={row['jump_count']}",
            )
        if no_signal_threshold_db is not None and float(row["p95_db"]) <= no_signal_threshold_db:
            _add_issue(
                issues,
                row,
                "possible_no_signal",
                "review",
                f"p95_db={row['p95_db']} <= threshold={no_signal_threshold_db}",
            )

    length_counts = Counter(int(row["num_samples"]) for row in parsed_rows)
    inferred_length = length_counts.most_common(1)[0][0] if length_counts else None
    target_length = expected_length if expected_length is not None else inferred_length
    if target_length is not None:
        for row in parsed_rows:
            if int(row["num_samples"]) != target_length:
                _add_issue(
                    issues,
                    row,
                    "length_mismatch",
                    "error",
                    f"num_samples={row['num_samples']}, expected={target_length}",
                )

    for fingerprint, duplicate_rows in fingerprints.items():
        labels = {str(row["label"]) for row in duplicate_rows}
        if len(duplicate_rows) > 1 and len(labels) > 1:
            for row in duplicate_rows:
                _add_issue(
                    issues,
                    row,
                    "duplicate_across_labels",
                    "error",
                    (
                        f"identical signal appears under labels={sorted(labels)} "
                        f"duplicate_count={len(duplicate_rows)} fingerprint={fingerprint}"
                    ),
                )

    _flag_condition_outliers(parsed_rows, issues, robust_z_threshold)
    issues.sort(key=lambda item: (item["severity"], item["type"], item["group"], item["file"]))

    label_counts = Counter(str(row["label"]) for row in indexed_rows)
    group_counts = Counter(str(row["group"]) for row in indexed_rows)
    ratio_counts = Counter(
        ratio
        for row in indexed_rows
        if (ratio := _ratio_from_condition(str(row.get("condition_path", "")))) is not None
    )
    condition_counts = Counter(
        f"{row['group']} | {row.get('condition_path', '') or '(none)'}" for row in indexed_rows
    )
    group_ratio_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in indexed_rows:
        ratio = _ratio_from_condition(str(row.get("condition_path", ""))) or "(none)"
        group_ratio_counts[str(row["group"])][ratio] += 1
    issue_type_counts = Counter(issue["type"] for issue in issues)
    issue_severity_counts = Counter(issue["severity"] for issue in issues)
    expected_labels = _expected_labels(classes) if len(classes) == 3 else []
    missing_labels = [label for label in expected_labels if label_counts.get(label, 0) == 0]

    report: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "roots": {
            "single_dir": str(single_root.resolve()),
            "combo_dir": str(combo_root.resolve()),
            "single_first_level_directories": _first_level_directories(single_root),
            "combo_first_level_directories": _first_level_directories(combo_root),
        },
        "class_names": classes,
        "audit_settings": {
            "expected_length": target_length,
            "expected_length_source": "argument" if expected_length is not None else "modal parsed length",
            "no_signal_threshold_db": no_signal_threshold_db,
            "jump_threshold_db": jump_threshold_db,
            "flat_std_threshold_db": flat_std_threshold_db,
            "robust_z_threshold": robust_z_threshold,
        },
        "counts": {
            "discovered_files": len(indexed_rows),
            "raw_csv_files": raw_single_file_count + raw_combo_file_count,
            "raw_single_csv_files": raw_single_file_count,
            "raw_combo_csv_files": raw_combo_file_count,
            "skipped_or_unrecognized_csv_files": (
                raw_single_file_count + raw_combo_file_count - len(indexed_rows)
            ),
            "parsed_files": len(parsed_rows),
            "parse_failed_files": len(indexed_rows) - len(parsed_rows),
            "single_files": sum(row["source_root"] == "single" for row in indexed_rows),
            "combo_files": sum(row["source_root"] == "real_dataset" for row in indexed_rows),
            "group_counts": dict(sorted(group_counts.items())),
            "label_counts": dict(sorted(label_counts.items())),
            "ratio_counts": dict(sorted(ratio_counts.items())),
            "group_ratio_counts": {
                group: dict(sorted(counts.items()))
                for group, counts in sorted(group_ratio_counts.items())
            },
            "condition_counts": dict(sorted(condition_counts.items())),
            "length_counts": {str(key): value for key, value in sorted(length_counts.items())},
            "csv_layout_counts": dict(sorted(data_marker_counts.items())),
            "missing_expected_labels": missing_labels,
        },
        "signal_summary": _metric_summary(parsed_rows),
        "by_group": _summaries_by(parsed_rows, "group"),
        "by_ratio": _summaries_by(parsed_rows, "ratio"),
        "recommended_preprocessing": _recommended_preprocessing(parsed_rows),
        "issues": {
            "total_issue_records": len(issues),
            "affected_file_count": len({issue["file"] for issue in issues}),
            "counts_by_severity": dict(sorted(issue_severity_counts.items())),
            "counts_by_type": dict(sorted(issue_type_counts.items())),
            "files": issues,
        },
        "scan_warnings": scan_warnings,
    }
    report["copy_paste_summary"] = _build_copy_paste_summary(report)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(report["copy_paste_summary"])
    print(f"report={output_path.resolve()}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit real source CSV counts, dB ranges, preprocessing parameters, and suspicious files."
    )
    parser.add_argument("--single-dir", type=Path, default=Path("data/single"))
    parser.add_argument("--combo-dir", type=Path, default=Path("data/real_dataset"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--class-names",
        nargs="+",
        help="Expected sources, for example: source_1 source_3 source_4. Defaults to data/single folders.",
    )
    parser.add_argument("--expected-length", type=int, default=None)
    parser.add_argument(
        "--no-signal-threshold-db",
        type=float,
        default=None,
        help="Optional: flag a file when its 95th percentile does not exceed this dB threshold.",
    )
    parser.add_argument("--jump-threshold-db", type=float, default=12.0)
    parser.add_argument("--flat-std-threshold-db", type=float, default=0.05)
    parser.add_argument("--robust-z-threshold", type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit_real_data(
        single_dir=args.single_dir,
        combo_dir=args.combo_dir,
        output=args.output,
        class_names=args.class_names,
        expected_length=args.expected_length,
        no_signal_threshold_db=args.no_signal_threshold_db,
        jump_threshold_db=args.jump_threshold_db,
        flat_std_threshold_db=args.flat_std_threshold_db,
        robust_z_threshold=args.robust_z_threshold,
    )


if __name__ == "__main__":
    main()
