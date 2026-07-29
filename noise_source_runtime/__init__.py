"""Public import alias for the runtime implemented under ``src/``."""

from __future__ import annotations

import sys
from importlib import import_module

_IMPLEMENTATION_PACKAGE = "src.noise_source_runtime"
_SUBMODULES = (
    "checkpoint",
    "csv_parser",
    "device",
    "exceptions",
    "model",
    "package",
    "preprocessing",
    "reporting",
    "result",
    "session",
)

_implementation = import_module(_IMPLEMENTATION_PACKAGE)
for _name in _SUBMODULES:
    sys.modules[f"{__name__}.{_name}"] = import_module(
        f"{_IMPLEMENTATION_PACKAGE}.{_name}"
    )

__version__ = _implementation.__version__
RUNTIME_VERSION = _implementation.RUNTIME_VERSION

from src.noise_source_runtime import (  # noqa: E402,F401
    InferenceSession,
    ModelPackageError,
    PredictionResult,
    RuntimeInferenceError,
    SessionClosedError,
    export_prediction_result,
    verify_model_package,
    write_inference_contract,
    write_prediction_json,
)

__all__ = _implementation.__all__
