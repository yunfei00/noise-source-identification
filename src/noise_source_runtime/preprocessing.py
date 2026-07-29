from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from src.features import (
    CsvSignal,
    PreprocessConfig,
    PreprocessResult,
    load_csv_signal,
    preprocess_signal,
)


def prepare_file_input(
    csv_path: str | Path,
    config: Mapping[str, Any],
    *,
    require_data_marker: bool = True,
) -> tuple[CsvSignal, PreprocessResult]:
    parsed = load_csv_signal(
        csv_path,
        require_data_marker=require_data_marker,
        invalid_row_policy="error" if require_data_marker else "skip",
    )
    processed = preprocess_signal(
        parsed.raw_signal,
        PreprocessConfig.from_config(dict(config)),
    )
    return parsed, processed


def prepare_array_input(
    signal: np.ndarray,
    config: Mapping[str, Any],
) -> PreprocessResult:
    return preprocess_signal(
        signal,
        PreprocessConfig.from_config(dict(config)),
    )


def preprocessing_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    resolved = PreprocessConfig.from_config(dict(config))
    return {
        "signal_length": resolved.signal_length,
        "sample_rate": resolved.sample_rate,
        "nperseg": resolved.nperseg,
        "noverlap": resolved.noverlap,
        "target_freq_bins": resolved.target_freq_bins,
        "target_time_bins": resolved.target_time_bins,
        "magnitude_scale": resolved.magnitude_scale,
        "input_representation": resolved.input_representation,
        "signal_normalization": resolved.signal_normalization,
        "db_level_range": list(resolved.db_level_range),
        "db_variation_scale": resolved.db_variation_scale,
        "tensor_layout": "NCHW",
        "csv": {
            "strict_data_marker": True,
            "data_marker_match": "whole stripped line, case-insensitive",
            "data_starts": "line after DATA",
            "signal_column_zero_based": 1,
            "empty_rows": "skip",
            "invalid_or_nonfinite_rows": "error",
        },
    }


__all__ = [
    "CsvSignal",
    "PreprocessConfig",
    "PreprocessResult",
    "prepare_array_input",
    "prepare_file_input",
    "preprocess_signal",
    "preprocessing_contract",
]
