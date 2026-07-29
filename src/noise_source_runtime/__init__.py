from __future__ import annotations

__version__ = "1.0.0"
RUNTIME_VERSION = __version__

from src.noise_source_runtime.exceptions import (  # noqa: E402
    ModelPackageError,
    RuntimeInferenceError,
    SessionClosedError,
)
from src.noise_source_runtime.result import PredictionResult  # noqa: E402

__all__ = [
    "InferenceSession",
    "ModelPackageError",
    "PredictionResult",
    "RUNTIME_VERSION",
    "RuntimeInferenceError",
    "SessionClosedError",
    "__version__",
    "export_prediction_result",
    "verify_model_package",
    "write_inference_contract",
    "write_prediction_json",
]


def __getattr__(name: str):
    if name == "InferenceSession":
        from src.noise_source_runtime.session import InferenceSession

        return InferenceSession
    if name in {
        "export_prediction_result",
        "write_inference_contract",
        "write_prediction_json",
    }:
        from src.noise_source_runtime import reporting

        return getattr(reporting, name)
    if name == "verify_model_package":
        from src.noise_source_runtime.package import verify_model_package

        return verify_model_package
    raise AttributeError(name)
