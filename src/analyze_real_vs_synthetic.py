from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.features import compute_stft_feature, fix_length, normalize_signal, read_signal_csv
from src.infer import load_checkpoint


def _stats(signal: np.ndarray, stft_feature: np.ndarray) -> dict[str, float]:
    signal = np.asarray(signal, dtype=np.float32).reshape(-1)
    stft_feature = np.asarray(stft_feature, dtype=np.float32)
    return {
        "signal_mean": float(np.mean(signal)),
        "signal_std": float(np.std(signal)),
        "signal_rms": float(np.sqrt(np.mean(np.square(signal)))) if signal.size else 0.0,
        "signal_peak_to_peak": float(np.ptp(signal)) if signal.size else 0.0,
        "stft_mean": float(np.mean(stft_feature)),
        "stft_std": float(np.std(stft_feature)),
        "stft_max": float(np.max(stft_feature)) if stft_feature.size else 0.0,
        "stft_energy": float(np.mean(np.square(stft_feature))) if stft_feature.size else 0.0,
    }


def _feature(signal: np.ndarray, config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    data_config = config.get("data", {})
    stft_config = config.get("stft", {})
    fixed = fix_length(signal, int(data_config.get("signal_length", 4096)))
    normalized = normalize_signal(fixed)
    feature = compute_stft_feature(
        normalized,
        sample_rate=int(data_config.get("sample_rate", 1_000_000)),
        nperseg=int(stft_config.get("nperseg", 256)),
        noverlap=int(stft_config.get("noverlap", 128)),
        target_freq_bins=int(stft_config.get("target_freq_bins", 128)),
        target_time_bins=int(stft_config.get("target_time_bins", 64)),
    )
    return fixed.astype(np.float32, copy=False), feature


def _mean_stats(records: list[dict[str, float]]) -> dict[str, float]:
    if not records:
        return {}
    keys = records[0].keys()
    return {key: float(np.mean([record[key] for record in records])) for key in keys}


def collect_real_stats(real_dir: str | Path, config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    root = Path(real_dir)
    if not root.exists():
        raise FileNotFoundError(f"Real directory not found: {root}")
    groups: dict[str, dict[str, Any]] = {}
    for group_dir in sorted((path for path in root.iterdir() if path.is_dir()), key=lambda path: path.name):
        csv_files = sorted(path for path in group_dir.rglob("*.csv") if path.is_file())
        if not csv_files:
            print(f"warning: no CSV files found recursively under real group: {group_dir}")
            continue
        records = []
        for csv_path in csv_files:
            signal, stft_feature = _feature(read_signal_csv(csv_path), config)
            records.append(_stats(signal, stft_feature))
        groups[group_dir.name] = {"num_samples": len(records), "features": _mean_stats(records)}
    if not groups:
        raise ValueError(f"No recursive real CSV files found under: {root}")
    return groups


def collect_synthetic_stats(split: str, config: dict[str, Any]) -> dict[str, Any]:
    data_config = config.get("data", {})
    x_dir = Path(data_config.get("mixed_dir", "data/mixed")) / split / "x"
    if not x_dir.exists():
        raise FileNotFoundError(f"Synthetic split x directory not found: {x_dir}")
    x_files = sorted(x_dir.glob("*.npy"))
    if not x_files:
        raise ValueError(f"No synthetic .npy files found under: {x_dir}")
    records = []
    for x_file in x_files:
        signal, stft_feature = _feature(np.load(x_file), config)
        records.append(_stats(signal, stft_feature))
    return {"num_samples": len(records), "features": _mean_stats(records)}


def build_differences(
    real_groups: dict[str, dict[str, Any]], synthetic: dict[str, Any]
) -> dict[str, dict[str, dict[str, float]]]:
    synthetic_features = synthetic["features"]
    differences = {}
    for group, payload in real_groups.items():
        group_diffs = {}
        for key, real_value in payload["features"].items():
            synthetic_value = synthetic_features[key]
            absolute = float(real_value - synthetic_value)
            relative = float(absolute / (abs(synthetic_value) + 1e-8))
            group_diffs[key] = {
                "real_mean": float(real_value),
                "synthetic_mean": float(synthetic_value),
                "absolute_difference": absolute,
                "relative_difference": relative,
            }
        differences[group] = group_diffs
    return differences


def write_features_csv(
    real_groups: dict[str, dict[str, Any]], synthetic: dict[str, Any], output_path: str | Path
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"domain": "synthetic", "group": "synthetic_test", "num_samples": synthetic["num_samples"], **synthetic["features"]}]
    rows.extend(
        {"domain": "real", "group": group, "num_samples": payload["num_samples"], **payload["features"]}
        for group, payload in real_groups.items()
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def analyze(model_path: str | Path, real_dir: str | Path, synthetic_split: str, output: str | Path) -> dict[str, Any]:
    checkpoint = load_checkpoint(model_path, map_location=torch.device("cpu"))
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise ValueError("Checkpoint is missing config")

    real_groups = collect_real_stats(real_dir, config)
    synthetic = collect_synthetic_stats(synthetic_split, config)
    differences = build_differences(real_groups, synthetic)
    output_path = Path(output)
    features_path = output_path.with_name("domain_gap_features.csv")
    report = {
        "model": str(model_path),
        "real_dir": str(real_dir),
        "synthetic_split": synthetic_split,
        "synthetic": synthetic,
        "real_groups": real_groups,
        "differences_vs_synthetic": differences,
        "features_csv": str(features_path),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    write_features_csv(real_groups, synthetic, features_path)
    print(f"domain_gap_report={output_path}")
    print(f"domain_gap_features={features_path}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze real vs synthetic domain-gap feature statistics.")
    parser.add_argument("--model", type=Path, default=Path("outputs/checkpoints/best.pt"))
    parser.add_argument("--real-dir", type=Path, default=Path("data/real_test"))
    parser.add_argument("--synthetic-split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--output", type=Path, default=Path("outputs/reports/domain_gap_report.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analyze(args.model, args.real_dir, args.synthetic_split, args.output)


if __name__ == "__main__":
    main()
