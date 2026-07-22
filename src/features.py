from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
from scipy.signal import stft


_TOKEN_SPLIT_RE = re.compile(r"[,\s]+")
_ABSOLUTE_MAGNITUDE_ALIASES = {"absolute", "linear", "magnitude", "abs"}
_LOG_MAGNITUDE_ALIASES = {"log1p", "log"}
_DB_MAGNITUDE_ALIASES = {"db", "dB", "decibel", "decibels"}
_NO_NORMALIZATION_ALIASES = {"none", "raw", "identity", "absolute"}
_STANDARDIZE_ALIASES = {"standardize", "zscore", "z-score", "normalize"}


@dataclass(frozen=True)
class SignalCsvInfo:
    """Parsed signal values plus CSV layout metadata for diagnostics."""

    values: np.ndarray
    found_data_line: bool
    data_line_number: int | None


def _split_numeric_tokens(line: str) -> list[str]:
    return [token for token in _TOKEN_SPLIT_RE.split(line.strip()) if token]


def _parse_float(token: str) -> float | None:
    try:
        return float(token)
    except ValueError:
        return None


def _finite_float32_values(values: list[float], csv_path: Path) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.size:
        array = array[np.isfinite(array)]
    if array.size == 0:
        raise ValueError(f"No valid numeric samples found in {csv_path}")
    return array.astype(np.float32, copy=False)


def _find_data_line(lines: list[str]) -> int | None:
    for index, line in enumerate(lines):
        if line.strip().lower() == "data":
            return index
    return None


def _parse_data_section(lines: list[str], start_index: int) -> list[float]:
    values: list[float] = []
    for line in lines[start_index:]:
        if not line.strip():
            continue
        tokens = _split_numeric_tokens(line)
        if not tokens:
            continue
        value_token = tokens[1] if len(tokens) >= 2 else tokens[0]
        value = _parse_float(value_token)
        if value is None:
            continue
        values.append(value)
    return values


def _parse_legacy_rows(lines: list[str]) -> list[float]:
    values: list[float] = []
    for line in lines:
        if not line.strip():
            continue
        tokens = _split_numeric_tokens(line)
        if not tokens:
            continue
        value = _parse_float(tokens[-1])
        if value is None:
            continue
        values.append(value)
    return values


def read_signal_csv_info(path: str | Path) -> SignalCsvInfo:
    """Read a one-dimensional signal and return parser diagnostics.

    The preferred real-device layout contains arbitrary metadata followed by a
    line whose stripped contents equal ``DATA`` (case-insensitive). Numeric rows
    after that marker are parsed with comma, whitespace, or tab delimiters; the
    second column is the signal value when present, otherwise the only column is
    used.

    Files without a ``DATA`` marker fall back to legacy single-column ``value``
    or two-column ``time,value`` parsing, with optional headers skipped and the
    last numeric column used as the signal value.
    """
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    if not csv_path.is_file():
        raise ValueError(f"Expected a CSV file, got: {csv_path}")

    try:
        lines = csv_path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    except Exception as exc:
        raise ValueError(f"Failed to read CSV file {csv_path}: {exc}") from exc

    data_line_index = _find_data_line(lines)
    if data_line_index is not None:
        values = _parse_data_section(lines, data_line_index + 1)
        array = _finite_float32_values(values, csv_path)
        return SignalCsvInfo(
            values=array,
            found_data_line=True,
            data_line_number=data_line_index + 1,
        )

    values = _parse_legacy_rows(lines)
    array = _finite_float32_values(values, csv_path)
    return SignalCsvInfo(values=array, found_data_line=False, data_line_number=None)


def read_signal_csv(path: str | Path) -> np.ndarray:
    """Read a one-dimensional signal from a CSV file.

    Preferred layout:
    - metadata lines
    - DATA marker line
    - numeric samples without a header; time,value rows use the second column

    Legacy layout:
    - value
    - time,value
    - either layout with or without a header row
    """
    return read_signal_csv_info(path).values


def fix_length(
    signal: np.ndarray,
    target_length: int,
    *,
    random_crop: bool = False,
    rng: np.random.Generator | None = None,
    pad_value: float = 0.0,
) -> np.ndarray:
    """Pad or crop a signal to a fixed length."""
    if target_length <= 0:
        raise ValueError(f"target_length must be positive, got {target_length}")

    signal = np.asarray(signal, dtype=np.float32).reshape(-1)
    current_length = signal.shape[0]

    if current_length == target_length:
        return signal.astype(np.float32, copy=False)

    if current_length < target_length:
        output = np.full(target_length, float(pad_value), dtype=np.float32)
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


def apply_signal_normalization(signal: np.ndarray, mode: str = "standardize") -> np.ndarray:
    """Apply configured time-domain normalization before feature extraction."""
    normalized_mode = str(mode).strip().lower()
    signal = np.asarray(signal, dtype=np.float32).reshape(-1)
    if normalized_mode in _NO_NORMALIZATION_ALIASES:
        return np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    if normalized_mode in _STANDARDIZE_ALIASES:
        return normalize_signal(signal)
    raise ValueError(
        "Unsupported signal_normalization mode: "
        f"{mode}. Use 'none' for raw absolute values or 'standardize' for z-score normalization."
    )


def scale_stft_magnitude(magnitude: np.ndarray, magnitude_scale: str = "log1p") -> np.ndarray:
    """Scale an STFT magnitude matrix according to the configured feature scale."""
    normalized_scale = str(magnitude_scale).strip().lower()
    magnitude = np.asarray(magnitude, dtype=np.float32)
    if normalized_scale in _ABSOLUTE_MAGNITUDE_ALIASES:
        return magnitude.astype(np.float32, copy=False)
    if normalized_scale in _LOG_MAGNITUDE_ALIASES:
        return np.log1p(magnitude).astype(np.float32, copy=False)
    if normalized_scale in _DB_MAGNITUDE_ALIASES:
        raise ValueError("dB STFT scaling is disabled for training; set stft.magnitude_scale=absolute.")
    raise ValueError(
        "Unsupported stft magnitude_scale: "
        f"{magnitude_scale}. Use 'absolute' for raw magnitude or 'log1p' for legacy log magnitude."
    )


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
    magnitude_scale: str = "log1p",
) -> np.ndarray:
    """Compute a fixed-shape STFT feature."""
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
    magnitude = np.abs(zxx).astype(np.float32, copy=False)
    feature = scale_stft_magnitude(magnitude, magnitude_scale)
    return _pad_or_crop_2d(feature, target_freq_bins, target_time_bins)


def prepare_stft_channels(
    feature: np.ndarray,
    representation: str = "single",
) -> np.ndarray:
    """Build model input channels from one absolute STFT magnitude feature.

    ``absolute_relative`` keeps the original amplitude channel and adds an
    energy-normalized log channel.  The latter exposes weak spectral structure
    that can otherwise be dominated by a much stronger source in a mixture.
    """
    feature = np.asarray(feature, dtype=np.float32)
    if feature.ndim != 2:
        raise ValueError(f"Expected a 2D STFT feature, got shape {feature.shape}")
    normalized_representation = str(representation).strip().lower()
    if normalized_representation in {"single", "absolute", "legacy"}:
        return feature[np.newaxis, ...].astype(np.float32, copy=False)
    if normalized_representation != "absolute_relative":
        raise ValueError(
            f"Unsupported stft.input_representation: {representation}. "
            "Use 'single' or 'absolute_relative'."
        )

    rms = float(np.sqrt(np.mean(np.square(feature, dtype=np.float64))))
    scale = max(rms, 1e-12)
    relative = np.log1p(feature / scale).astype(np.float32, copy=False)
    return np.stack((feature, relative), axis=0).astype(np.float32, copy=False)


def fix_model_signal_length(
    signal: np.ndarray,
    target_length: int,
    input_representation: str = "single",
) -> np.ndarray:
    """Fix signal length without introducing a false 0 dB tail for dB traces."""
    signal = np.asarray(signal, dtype=np.float32).reshape(-1)
    representation = str(input_representation).strip().lower()
    if representation != "db_trace":
        return fix_length(signal, target_length)

    finite = signal[np.isfinite(signal)]
    pad_value = float(np.median(finite)) if finite.size else 0.0
    return fix_length(signal, target_length, pad_value=pad_value)


def prepare_db_trace_channels(
    signal: np.ndarray,
    sample_rate: int,
    nperseg: int,
    noverlap: int,
    target_freq_bins: int,
    target_time_bins: int,
    *,
    db_level_range: tuple[float, float] = (-110.0, -50.0),
    db_variation_scale: float = 15.0,
) -> np.ndarray:
    """Build four model channels from a spectrum-analyzer dB-vs-time trace.

    The large absolute dB baseline is removed before STFT so it cannot dominate
    the fluctuation spectrum. Two constant metadata channels retain the absolute
    median level and the trace standard deviation, both scaled to roughly [0, 1].
    """
    signal = np.asarray(signal, dtype=np.float32).reshape(-1)
    if signal.size == 0:
        raise ValueError("Cannot prepare dB trace features from an empty signal")

    finite = signal[np.isfinite(signal)]
    if finite.size == 0:
        raise ValueError("dB trace contains no finite samples")
    median_db = float(np.median(finite))
    clean_signal = np.nan_to_num(
        signal,
        nan=median_db,
        posinf=median_db,
        neginf=median_db,
    ).astype(np.float32, copy=False)
    centered = (clean_signal - median_db).astype(np.float32, copy=False)

    absolute = compute_stft_feature(
        centered,
        sample_rate=sample_rate,
        nperseg=nperseg,
        noverlap=noverlap,
        target_freq_bins=target_freq_bins,
        target_time_bins=target_time_bins,
        magnitude_scale="absolute",
    )
    spectral_channels = prepare_stft_channels(absolute, "absolute_relative")

    level_min, level_max = float(db_level_range[0]), float(db_level_range[1])
    if level_max <= level_min:
        raise ValueError("stft.db_level_range maximum must be greater than minimum")
    if db_variation_scale <= 0.0:
        raise ValueError("stft.db_variation_scale must be positive")
    level_value = float(np.clip((median_db - level_min) / (level_max - level_min), 0.0, 1.0))
    variation_value = float(np.clip(np.std(centered) / db_variation_scale, 0.0, 1.0))
    metadata_shape = (1, target_freq_bins, target_time_bins)
    level_channel = np.full(metadata_shape, level_value, dtype=np.float32)
    variation_channel = np.full(metadata_shape, variation_value, dtype=np.float32)
    return np.concatenate(
        (spectral_channels, level_channel, variation_channel),
        axis=0,
    ).astype(np.float32, copy=False)


def compute_model_feature(
    signal: np.ndarray,
    sample_rate: int,
    nperseg: int,
    noverlap: int,
    target_freq_bins: int,
    target_time_bins: int,
    *,
    magnitude_scale: str = "log1p",
    input_representation: str = "single",
    signal_normalization: str = "standardize",
    db_level_range: tuple[float, float] = (-110.0, -50.0),
    db_variation_scale: float = 15.0,
) -> np.ndarray:
    """Compute the configured model input through one shared feature path."""
    representation = str(input_representation).strip().lower()
    if representation == "db_trace":
        return prepare_db_trace_channels(
            signal,
            sample_rate=sample_rate,
            nperseg=nperseg,
            noverlap=noverlap,
            target_freq_bins=target_freq_bins,
            target_time_bins=target_time_bins,
            db_level_range=db_level_range,
            db_variation_scale=db_variation_scale,
        )

    normalized = apply_signal_normalization(signal, signal_normalization)
    feature = compute_stft_feature(
        normalized,
        sample_rate=sample_rate,
        nperseg=nperseg,
        noverlap=noverlap,
        target_freq_bins=target_freq_bins,
        target_time_bins=target_time_bins,
        magnitude_scale=magnitude_scale,
    )
    return prepare_stft_channels(feature, representation)
