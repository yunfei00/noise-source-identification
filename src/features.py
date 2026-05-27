from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import stft


def read_signal_csv(path: str | Path) -> np.ndarray:
    """Read a one-dimensional signal from a CSV file.

    Supported layouts:
    - value
    - time,value
    - either layout with or without a header row
    """
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    if not csv_path.is_file():
        raise ValueError(f"Expected a CSV file, got: {csv_path}")

    try:
        frame = pd.read_csv(csv_path, header=None)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"CSV file is empty: {csv_path}") from exc
    except Exception as exc:
        raise ValueError(f"Failed to read CSV file {csv_path}: {exc}") from exc

    if frame.empty or frame.shape[1] == 0:
        raise ValueError(f"CSV file has no signal values: {csv_path}")

    if frame.shape[1] == 1:
        value_column = frame.iloc[:, 0]
    elif frame.shape[1] == 2:
        value_column = frame.iloc[:, 1]
    else:
        raise ValueError(
            f"CSV file must have one column (value) or two columns (time,value): {csv_path}"
        )

    values = pd.to_numeric(value_column, errors="coerce").to_numpy(dtype=np.float64)
    values = values[np.isfinite(values)]
    return values.astype(np.float32, copy=False)


def fix_length(
    signal: np.ndarray,
    target_length: int,
    *,
    random_crop: bool = False,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Pad or crop a signal to a fixed length."""
    if target_length <= 0:
        raise ValueError(f"target_length must be positive, got {target_length}")

    signal = np.asarray(signal, dtype=np.float32).reshape(-1)
    current_length = signal.shape[0]

    if current_length == target_length:
        return signal.astype(np.float32, copy=False)

    if current_length < target_length:
        output = np.zeros(target_length, dtype=np.float32)
        output[:current_length] = signal
        return output

    if random_crop:
        generator = rng if rng is not None else np.random.default_rng()
        start = int(generator.integers(0, current_length - target_length + 1))
    else:
        start = (current_length - target_length) // 2

    return signal[start : start + target_length].astype(np.float32, copy=False)


def normalize_signal(signal: np.ndarray) -> np.ndarray:
    """Zero-center and standardize a signal."""
    signal = np.asarray(signal, dtype=np.float32).reshape(-1)
    if signal.size == 0:
        return signal

    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
    mean = float(np.mean(signal))
    std = float(np.std(signal))
    centered = signal - mean
    if std < 1e-8:
        return centered.astype(np.float32, copy=False)
    return (centered / std).astype(np.float32, copy=False)


def _pad_or_crop_2d(feature: np.ndarray, target_freq_bins: int, target_time_bins: int) -> np.ndarray:
    if target_freq_bins <= 0 or target_time_bins <= 0:
        raise ValueError(
            "target_freq_bins and target_time_bins must be positive, "
            f"got {target_freq_bins}, {target_time_bins}"
        )

    output = np.zeros((target_freq_bins, target_time_bins), dtype=np.float32)
    freq_bins = min(target_freq_bins, feature.shape[0])
    time_bins = min(target_time_bins, feature.shape[1])
    output[:freq_bins, :time_bins] = feature[:freq_bins, :time_bins]
    return output


def compute_stft_feature(
    signal: np.ndarray,
    sample_rate: int,
    nperseg: int,
    noverlap: int,
    target_freq_bins: int,
    target_time_bins: int,
) -> np.ndarray:
    """Compute a fixed-shape log-magnitude STFT feature."""
    signal = np.asarray(signal, dtype=np.float32).reshape(-1)
    if signal.size == 0:
        raise ValueError("Cannot compute STFT for an empty signal")
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}")
    if nperseg <= 0:
        raise ValueError(f"nperseg must be positive, got {nperseg}")

    effective_nperseg = min(int(nperseg), int(signal.size))
    effective_noverlap = min(max(int(noverlap), 0), effective_nperseg - 1)

    _, _, zxx = stft(
        signal,
        fs=sample_rate,
        nperseg=effective_nperseg,
        noverlap=effective_noverlap,
        boundary=None,
        padded=False,
    )
    magnitude = np.abs(zxx)
    feature = np.log1p(magnitude).astype(np.float32, copy=False)
    return _pad_or_crop_2d(feature, target_freq_bins, target_time_bins)
