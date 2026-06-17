from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from scipy.signal import stft

from src.features import read_signal_csv


def label_to_text(label: str) -> str:
    return label.replace(" ", "")


def load_config(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data if isinstance(data, dict) else {}


def resolve_sample_path(file_value: str, split_path: Path, config: dict) -> Path:
    path = Path(file_value)
    if path.exists():
        return path
    candidates = [
        Path(config.get("real_data", {}).get("dataset_root", "")) / path,
        Path(config.get("real_data", {}).get("combo_dir", "")) / path,
        split_path.parent / path,
        Path.cwd() / path,
    ]
    for candidate in candidates:
        if str(candidate) and candidate.exists():
            return candidate
    return candidates[0]


def compute_stats(signal: np.ndarray, sample_rate: int, nperseg: int, noverlap: int) -> dict[str, float]:
    x = np.asarray(signal, dtype=np.float32).reshape(-1)
    if x.size == 0:
        raise ValueError("empty signal")
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    signal_energy = float(np.sum(np.square(x, dtype=np.float64)))

    fft = np.fft.rfft(x)
    fft_amp = np.abs(fft).astype(np.float64, copy=False)
    freqs = np.fft.rfftfreq(x.size, d=1.0 / sample_rate)
    fft_energy = np.square(fft_amp)
    total_fft_energy = float(np.sum(fft_energy))
    peak_index = int(np.argmax(fft_amp)) if fft_amp.size else 0
    centroid = float(np.sum(freqs * fft_amp) / np.sum(fft_amp)) if np.sum(fft_amp) > 0 else 0.0
    bandwidth = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * fft_amp) / np.sum(fft_amp))) if np.sum(fft_amp) > 0 else 0.0

    effective_nperseg = min(max(1, int(nperseg)), int(x.size))
    effective_noverlap = min(max(0, int(noverlap)), effective_nperseg - 1)
    _, _, zxx = stft(x, fs=sample_rate, nperseg=effective_nperseg, noverlap=effective_noverlap, boundary=None, padded=False)
    stft_mag = np.abs(zxx).astype(np.float64, copy=False)

    return {
        "signal_mean": float(np.mean(x)),
        "signal_std": float(np.std(x)),
        "signal_rms": float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))),
        "signal_peak": float(np.max(np.abs(x))),
        "signal_peak_to_peak": float(np.ptp(x)),
        "signal_energy": signal_energy,
        "fft_total_energy": total_fft_energy,
        "fft_peak_freq": float(freqs[peak_index]) if freqs.size else 0.0,
        "fft_peak_amp": float(fft_amp[peak_index]) if fft_amp.size else 0.0,
        "fft_centroid": centroid,
        "fft_bandwidth": bandwidth,
        "stft_mean": float(np.mean(stft_mag)) if stft_mag.size else 0.0,
        "stft_std": float(np.std(stft_mag)) if stft_mag.size else 0.0,
        "stft_max": float(np.max(stft_mag)) if stft_mag.size else 0.0,
        "stft_energy": float(np.sum(np.square(stft_mag))) if stft_mag.size else 0.0,
    }


def aggregate(rows: list[dict[str, str]]) -> dict:
    numeric_keys = [
        "signal_rms",
        "signal_energy",
        "fft_total_energy",
        "stft_energy",
    ]
    by_group: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_group[row["group"]].append(row)

    groups = {}
    for group, group_rows in sorted(by_group.items()):
        summary = {"sample_count": len(group_rows)}
        for key in numeric_keys:
            values = np.asarray([float(row[key]) for row in group_rows], dtype=np.float64)
            summary[f"{key}_mean"] = float(values.mean()) if values.size else 0.0
            if key == "signal_rms":
                summary[f"{key}_std"] = float(values.std(ddof=1)) if values.size > 1 else 0.0
        groups[group] = summary

    source_names = ("source_1", "source_3", "source_5")
    return {
        "groups": groups,
        "source_energy_comparison": {f"{name}_only": groups.get(name, {}) for name in source_names},
    }


def run(split: Path, split_name: str, output: Path, max_samples_per_group: int, config_path: Path) -> dict:
    config = load_config(config_path)
    sample_rate = int(config.get("data", {}).get("sample_rate", 1_000_000))
    nperseg = int(config.get("stft", {}).get("nperseg", 256))
    noverlap = int(config.get("stft", {}).get("noverlap", 128))
    output.parent.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = defaultdict(int)
    rows: list[dict[str, str]] = []
    with split.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("split") != split_name:
                continue
            group = row.get("group", "")
            if counts[group] >= max_samples_per_group:
                continue
            sample_path = resolve_sample_path(row["file"], split, config)
            signal = read_signal_csv(sample_path)
            stats = compute_stats(signal, sample_rate, nperseg, noverlap)
            counts[group] += 1
            rows.append({
                "file": row.get("file", ""),
                "group": group,
                "condition_path": row.get("condition_path", ""),
                "label": label_to_text(row.get("label", "")),
                **{key: f"{value:.10g}" for key, value in stats.items()},
            })

    fieldnames = [
        "file", "group", "condition_path", "label",
        "signal_mean", "signal_std", "signal_rms", "signal_peak", "signal_peak_to_peak", "signal_energy",
        "fft_total_energy", "fft_peak_freq", "fft_peak_amp", "fft_centroid", "fft_bandwidth",
        "stft_mean", "stft_std", "stft_max", "stft_energy",
    ]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    summary = aggregate(rows)
    summary_path = output.with_name("feature_statistics_summary.json")
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"samples={len(rows)} output={output} summary={summary_path}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute raw signal and spectral statistics by real-data group.")
    parser.add_argument("--split", type=Path, required=True, help="real_dataset_split.csv path")
    parser.add_argument("--split-name", default="train", choices=("train", "val", "test"))
    parser.add_argument("--output", type=Path, default=Path("outputs/reports/feature_statistics.csv"))
    parser.add_argument("--max-samples-per-group", type=int, default=500)
    parser.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.split, args.split_name, args.output, args.max_samples_per_group, args.config)


if __name__ == "__main__":
    main()
