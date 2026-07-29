from __future__ import annotations


class RuntimeInferenceError(RuntimeError):
    """Base error raised by the reusable inference runtime."""


class SessionClosedError(RuntimeInferenceError):
    """Raised when prediction is attempted after a session is closed."""


class ModelPackageError(RuntimeInferenceError):
    """Raised when a model package cannot be built or verified."""
