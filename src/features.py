from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import numpy as np
import torch
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
    numeric_value_count: int
    discarded_nonfinite_count: int


@dataclass(frozen=True)
class CsvSignal:
    """Signal samples and the CSV decisions that produced them.

    ``load_csv_signal`` is the common parser used by strict single-file
    inference and by the training/validation compatibility wrapper.  Strict
    inference requires an explicit DATA marker; legacy project datasets can
    opt in to the historical header-based layout.
    """

    csv_path: Path
    metadata_rows: list[list[str]]
    data_rows: list[list[str]]
    raw_signal: np.ndarray
    data_start_line: int
    selected_columns: list[int]
    encoding: str
    delimiter: str
    found_data_line: bool
    data_marker_line: int | None
    numeric_value_count: int
    discarded_nonfinite_count: int
    skipped_empty_rows: int
    skipped_invalid_rows: int
    parser_mode: str


@dataclass(frozen=True)
class PreprocessConfig:
    """All deterministic parameters needed to reproduce a model input."""

    signal_length: int = 4096
    sample_rate: int = 1_000_000
    nperseg: int = 256
    noverlap: int = 128
    target_freq_bins: int = 128
    target_time_bins: int = 64
    magnitude_scale: str = "log1p"
    input_representation: str = "single"
    signal_normalization: str = "standardize"
    db_level_range: tuple[float, float] = (-110.0, -50.0)
    db_variation_scale: float = 15.0

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "PreprocessConfig":
        data_config = config.get("data", {})
        stft_config = config.get("stft", {})
        preprocessing_config = config.get("preprocessing", {})
        db_level_range = stft_config.get("db_level_range", [-110.0, -50.0])
        if not isinstance(db_level_range, (list, tuple)) or len(db_level_range) != 2:
            raise ValueError("stft.db_level_range must contain exactly two values")
        return cls(
            signal_length=int(data_config.get("signal_length", 4096)),
            sample_rate=int(data_config.get("sample_rate", 1_000_000)),
            nperseg=int(stft_config.get("nperseg", 256)),
            noverlap=int(stft_config.get("noverlap", 128)),
            target_freq_bins=int(stft_config.get("target_freq_bins", 128)),
            target_time_bins=int(stft_config.get("target_time_bins", 64)),
            magnitude_scale=str(stft_config.get("magnitude_scale", "log1p")),
            input_representation=str(stft_config.get("input_representation", "single")),
            signal_normalization=str(preprocessing_config.get("signal_normalization", "standardize")),
            db_level_range=(float(db_level_range[0]), float(db_level_range[1])),
            db_variation_scale=float(stft_config.get("db_variation_scale", 15.0)),
        )


@dataclass(frozen=True)
class PreprocessResult:
    """Deterministic preprocessing output plus contract-report statistics."""

    raw_signal: np.ndarray
    linear_signal: np.ndarray | None
    resized_signal: np.ndarray
    normalized_signal: np.ndarray
    input_tensor: torch.Tensor
    statistics: dict[str, Any]


def _split_numeric_tokens(line: str) -> list[str]:
    return [token for token in _TOKEN_SPLIT_RE.split(line.strip()) if token]


def _parse_float(token: str) -> float | None:
    try:
        return float(token)
    except ValueError:
        return None


def _read_text_lines(csv_path: Path) -> tuple[list[str], str]:
    last_error: UnicodeDecodeError | None = None
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            return csv_path.read_text(encoding=encoding).splitlines(), encoding
        except UnicodeDecodeError as exc:
            last_error = exc
    assert last_error is not None
    raise ValueError(
        f"Unable to decode CSV as UTF-8 or GB18030: {csv_path}: {last_error}"
    ) from last_error


def _detected_delimiter(line: str) -> str:
    if "," in line:
        return "comma"
    if "\t" in line:
        return "tab"
    return "whitespace"


def _find_data_line(lines: list[str]) -> int | None:
    for index, line in enumerate(lines):
        if line.strip().lower() == "data":
            return index
    return None


def load_csv_signal(
    path: str | Path,
    *,
    require_data_marker: bool = True,
    invalid_row_policy: str = "error",
) -> CsvSignal:
    """Parse the model signal column from a CSV-like instrument export.

    DATA matching follows the historical project rule: the complete stripped
    line must equal ``DATA`` case-insensitively.  Data starts on the next line,
    uses column index 1, skips empty lines, and continues to EOF.  In strict
    mode malformed, short, missing, NaN, or infinite rows fail immediately so
    an anomalous value cannot silently change the sample length.

    ``require_data_marker=False`` is the explicit compatibility mode for the
    repository's older ``time,value`` files.  It retains their historical
    last-column parsing behavior.
    """
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    if not csv_path.is_file():
        raise ValueError(f"Expected a CSV file, got: {csv_path}")
    if invalid_row_policy not in {"error", "skip"}:
        raise ValueError(
            f"invalid_row_policy must be 'error' or 'skip', got {invalid_row_policy}"
        )

    try:
        lines, encoding = _read_text_lines(csv_path)
    except Exception as exc:
        if isinstance(exc, ValueError):
            raise
        raise ValueError(f"Failed to read CSV file {csv_path}: {exc}") from exc

    data_line_index = _find_data_line(lines)
    if data_line_index is None and require_data_marker:
        raise ValueError(
            "未找到 DATA 数据段，无法执行推理。\n"
            f"文件：{csv_path}"
        )

    values: list[float] = []
    data_rows: list[list[str]] = []
    skipped_empty_rows = 0
    skipped_invalid_rows = 0
    discarded_nonfinite_count = 0

    if data_line_index is not None:
        metadata_lines = lines[:data_line_index]
        candidate_lines = lines[data_line_index + 1 :]
        data_start_line = data_line_index + 2
        parser_mode = "data_section"
        selected_columns = [1]
        metadata_rows = [_split_numeric_tokens(line) for line in metadata_lines]
        delimiter = next(
            (_detected_delimiter(line) for line in candidate_lines if line.strip()),
            "unknown",
        )
        for zero_based_offset, line in enumerate(candidate_lines):
            line_number = data_start_line + zero_based_offset
            if not line.strip():
                skipped_empty_rows += 1
                continue
            tokens = _split_numeric_tokens(line)
            if len(tokens) < 2:
                message = (
                    f"DATA row {line_number} has {len(tokens)} column(s); "
                    f"column index 1 is required: {csv_path}"
                )
                if invalid_row_policy == "error":
                    raise ValueError(message)
                skipped_invalid_rows += 1
                continue
            value = _parse_float(tokens[1])
            if value is None:
                message = (
                    f"DATA row {line_number} column 1 is not numeric "
                    f"({tokens[1]!r}): {csv_path}"
                )
                if invalid_row_policy == "error":
                    raise ValueError(message)
                skipped_invalid_rows += 1
                continue
            if not np.isfinite(value):
                discarded_nonfinite_count += 1
                message = (
                    f"DATA row {line_number} column 1 is not finite "
                    f"({tokens[1]!r}): {csv_path}"
                )
                if invalid_row_policy == "error":
                    raise ValueError(message)
                continue
            data_rows.append(tokens)
            values.append(value)
    else:
        # Historical repository data has an optional header and no DATA marker.
        metadata_rows = []
        data_start_line = 1
        parser_mode = "legacy_last_column"
        selected_columns = [-1]
        delimiter = next(
            (_detected_delimiter(line) for line in lines if line.strip()),
            "unknown",
        )
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                skipped_empty_rows += 1
                continue
            tokens = _split_numeric_tokens(line)
            if not tokens:
                skipped_empty_rows += 1
                continue
            value = _parse_float(tokens[-1])
            if value is None:
                metadata_rows.append(tokens)
                skipped_invalid_rows += 1
                continue
            if not np.isfinite(value):
                discarded_nonfinite_count += 1
                continue
            data_rows.append(tokens)
            values.append(value)

    if not values:
        raise ValueError(f"No valid numeric samples found in {csv_path}")
    raw_signal = np.asarray(values, dtype=np.float32)
    return CsvSignal(
        csv_path=csv_path,
        metadata_rows=metadata_rows,
        data_rows=data_rows,
        raw_signal=raw_signal,
        data_start_line=data_start_line,
        selected_columns=selected_columns,
        encoding=encoding,
        delimiter=delimiter,
        found_data_line=data_line_index is not None,
        data_marker_line=(data_line_index + 1) if data_line_index is not None else None,
        numeric_value_count=len(values),
        discarded_nonfinite_count=discarded_nonfinite_count,
        skipped_empty_rows=skipped_empty_rows,
        skipped_invalid_rows=skipped_invalid_rows,
        parser_mode=parser_mode,
    )


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
    parsed = load_csv_signal(
        path,
        require_data_marker=False,
        invalid_row_policy="skip",
    )
    return SignalCsvInfo(
        values=parsed.raw_signal,
        found_data_line=parsed.found_data_line,
        data_line_number=parsed.data_marker_line,
        numeric_value_count=parsed.numeric_value_count + parsed.discarded_nonfinite_count,
        discarded_nonfinite_count=parsed.discarded_nonfinite_count,
    )


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
    return load_csv_signal(
        path,
        require_data_marker=False,
        invalid_row_policy="error",
    ).raw_signal


def _finite_range(values: np.ndarray) -> list[float] | None:
    finite = np.asarray(values)[np.isfinite(values)]
    if finite.size == 0:
        return None
    return [float(np.min(finite)), float(np.max(finite))]


def preprocess_signal(
    raw_signal: np.ndarray,
    config: PreprocessConfig,
) -> PreprocessResult:
    """Apply the exact deterministic feature path shared by datasets/inference."""
    raw = np.asarray(raw_signal, dtype=np.float32).reshape(-1)
    if raw.size == 0:
        raise ValueError("Cannot preprocess an empty signal")
    representation = config.input_representation.strip().lower()
    finite_raw = raw[np.isfinite(raw)]
    if finite_raw.size == 0:
        raise ValueError("Signal contains no finite samples")

    original_length = int(raw.size)
    target_length = int(config.signal_length)
    if original_length < target_length:
        length_method = "right_pad"
        crop_start = None
        pad_value = float(np.median(finite_raw)) if representation == "db_trace" else 0.0
    elif original_length > target_length:
        length_method = "center_crop"
        crop_start = (original_length - target_length) // 2
        pad_value = None
    else:
        length_method = "unchanged"
        crop_start = None
        pad_value = None

    resized = fix_model_signal_length(raw, target_length, representation)
    if representation == "db_trace":
        # db_trace intentionally stays in dB. prepare_db_trace_channels removes
        # the per-trace median before STFT and encodes level/variation separately.
        normalized = resized.astype(np.float32, copy=False)
        normalization_method = "none (dB trace median-centering occurs inside feature extraction)"
        normalization_parameters: dict[str, float] = {
            "median_db": float(np.median(resized[np.isfinite(resized)])),
            "db_level_min": float(config.db_level_range[0]),
            "db_level_max": float(config.db_level_range[1]),
            "db_variation_scale": float(config.db_variation_scale),
        }
        uses_training_statistics = False
    else:
        normalized = apply_signal_normalization(resized, config.signal_normalization)
        normalization_method = config.signal_normalization
        normalization_parameters = {}
        if config.signal_normalization.strip().lower() in _STANDARDIZE_ALIASES:
            clean = np.nan_to_num(resized, nan=0.0, posinf=0.0, neginf=0.0)
            normalization_parameters = {
                "sample_mean": float(np.mean(clean)),
                "sample_std": float(np.std(clean)),
                "epsilon": 1e-8,
            }
        uses_training_statistics = False

    feature = compute_model_feature(
        resized,
        sample_rate=config.sample_rate,
        nperseg=config.nperseg,
        noverlap=config.noverlap,
        target_freq_bins=config.target_freq_bins,
        target_time_bins=config.target_time_bins,
        magnitude_scale=config.magnitude_scale,
        input_representation=config.input_representation,
        signal_normalization=config.signal_normalization,
        db_level_range=config.db_level_range,
        db_variation_scale=config.db_variation_scale,
    )
    sample_tensor = torch.from_numpy(feature).float()
    statistics: dict[str, Any] = {
        "raw_shape": list(raw.shape),
        "raw_range": _finite_range(raw),
        "linear_conversion_applied": False,
        "linear_conversion_formula": "not applicable; training keeps the values in their configured representation",
        "linear_shape": None,
        "linear_range": None,
        "original_length": original_length,
        "target_length": target_length,
        "length_method": length_method,
        "crop_start": crop_start,
        "crop_end": (crop_start + target_length) if crop_start is not None else None,
        "pad_value": pad_value,
        "resized_shape": list(resized.shape),
        "resized_range": _finite_range(resized),
        "normalization_method": normalization_method,
        "normalization_parameters": normalization_parameters,
        "normalization_parameter_source": "current sample" if normalization_parameters else "not applicable",
        "uses_training_statistics": uses_training_statistics,
        "normalization_before_range": _finite_range(resized),
        "normalization_after_range": _finite_range(normalized),
        "normalized_shape": list(normalized.shape),
        "feature_shape": list(feature.shape),
        "sample_tensor_shape": list(sample_tensor.shape),
        "sample_tensor_dtype": str(sample_tensor.dtype),
        "input_representation": config.input_representation,
        "magnitude_scale": config.magnitude_scale,
        "nan_count_before": int(np.isnan(raw).sum()),
        "positive_inf_count_before": int(np.isposinf(raw).sum()),
        "negative_inf_count_before": int(np.isneginf(raw).sum()),
    }
    return PreprocessResult(
        raw_signal=raw,
        linear_signal=None,
        resized_signal=resized,
        normalized_signal=normalized,
        input_tensor=sample_tensor,
        statistics=statistics,
    )


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
