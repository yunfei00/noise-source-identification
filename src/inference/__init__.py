"""Backward-compatible inference interfaces.

The stable public API is now ``noise_source_runtime``.
"""

from src.inference.single_csv_predictor import (
    SinglePredictionResult,
    InferenceSession,
    inspect_checkpoint,
    predict_single_csv,
    prepare_single_csv_input,
)

__all__ = [
    "SinglePredictionResult",
    "InferenceSession",
    "inspect_checkpoint",
    "predict_single_csv",
    "prepare_single_csv_input",
]
