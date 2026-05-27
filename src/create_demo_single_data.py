from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt

from src.features import normalize_signal


def make_motor_signal(t: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    base_freq = float(rng.uniform(650.0, 1200.0))
    phase = float(rng.uniform(0.0, 2.0 * np.pi))
    signal = (
        np.sin(2.0 * np.pi * base_freq * t + phase)
        + 0.45 * np.sin(2.0 * np.pi * 2.0 * base_freq * t + 0.5 * phase)
        + 0.20 * np.sin(2.0 * np.pi * 3.0 * base_freq * t + 0.25 * phase)
    )
    signal += 0.05 * rng.normal(size=t.shape[0])
    return normalize_signal(signal)


def make_fan_signal(t: np.ndarray, sample_rate: int, rng: np.random.Generator) -> np.ndarray:
    white = rng.normal(size=t.shape[0])
    sos = butter(4, [8_000.0, 80_000.0], btype="bandpass", fs=sample_rate, output="sos")
    broadband = sosfiltfilt(sos, white)
    blade_freq = float(rng.uniform(900.0, 2200.0))
    carrier = float(rng.uniform(16_000.0, 32_000.0))
    periodic = 0.18 * np.sin(2.0 * np.pi * blade_freq * t) + 0.10 * np.sin(
        2.0 * np.pi * carrier * t
    )
    return normalize_signal(broadband + periodic)


def make_switch_power_signal(t: np.ndarray, sample_rate: int, rng: np.random.Generator) -> np.ndarray:
    switching_freq = float(rng.uniform(80_000.0, 160_000.0))
    phase = float(rng.uniform(0.0, 2.0 * np.pi))
    harmonic = (
        np.sin(2.0 * np.pi * switching_freq * t + phase)
        + 0.55 * np.sin(2.0 * np.pi * 2.0 * switching_freq * t + 0.3 * phase)
        + 0.25 * np.sin(2.0 * np.pi * 3.0 * switching_freq * t + 0.1 * phase)
    )

    spikes = np.zeros_like(t)
    period = max(2, int(sample_rate / switching_freq))
    offset = int(rng.integers(0, period))
    spike_indices = np.arange(offset, t.shape[0], period)
    spikes[spike_indices] = rng.uniform(1.5, 3.0, size=spike_indices.shape[0])
    decay = np.exp(-np.arange(16) / 3.0)
    spikes = np.convolve(spikes, decay, mode="same")
    signal = harmonic + spikes + 0.04 * rng.normal(size=t.shape[0])
    return normalize_signal(signal)


def write_csv(path: Path, t: np.ndarray, signal: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.column_stack([t, signal.astype(np.float32, copy=False)])
    np.savetxt(path, data, delimiter=",", header="time,value", comments="", fmt="%.9g")


def create_demo_data(
    output_dir: str | Path = Path("data/single"),
    num_files: int = 10,
    sample_rate: int = 1_000_000,
    signal_length: int = 4096,
    seed: int = 42,
) -> None:
    if num_files <= 0:
        raise ValueError(f"num_files must be positive, got {num_files}")
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}")
    if signal_length <= 0:
        raise ValueError(f"signal_length must be positive, got {signal_length}")

    root = Path(output_dir)
    rng = np.random.default_rng(seed)
    t = np.arange(signal_length, dtype=np.float32) / float(sample_rate)

    generators = {
        "motor": lambda: make_motor_signal(t, rng),
        "fan": lambda: make_fan_signal(t, sample_rate, rng),
        "switch_power": lambda: make_switch_power_signal(t, sample_rate, rng),
    }

    for class_name, generator in generators.items():
        for index in range(num_files):
            signal = generator()
            write_csv(root / class_name / f"{class_name}_{index:03d}.csv", t, signal)

    print(f"Wrote demo single-source CSV files to {root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create demo single-source noise CSV data.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/single"))
    parser.add_argument("--num-files", type=int, default=10)
    parser.add_argument("--sample-rate", type=int, default=1_000_000)
    parser.add_argument("--signal-length", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    create_demo_data(
        output_dir=args.output_dir,
        num_files=args.num_files,
        sample_rate=args.sample_rate,
        signal_length=args.signal_length,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
