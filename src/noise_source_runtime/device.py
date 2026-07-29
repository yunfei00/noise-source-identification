from __future__ import annotations

import torch


def resolve_device(device_name: str = "auto") -> torch.device:
    """Resolve an inference/training device without importing training code."""
    normalized = str(device_name).strip().lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        device = torch.device(normalized)
    except Exception as exc:
        raise ValueError(
            f"Invalid device {device_name!r}; expected auto, cpu, cuda, or cuda:N"
        ) from exc
    if device.type not in {"cpu", "cuda"}:
        raise ValueError(
            f"Unsupported device type {device.type!r}; expected cpu or cuda"
        )
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but torch.cuda.is_available() is False"
            )
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise RuntimeError(
                f"CUDA device index {device.index} is unavailable; "
                f"device_count={torch.cuda.device_count()}"
            )
    return device
