from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

_STATE_DICT_KEYS = ("model_state", "model_state_dict", "state_dict")


@dataclass(frozen=True)
class LoadedCheckpoint:
    path: Path
    payload: Any
    state_dict: OrderedDict[str, torch.Tensor]
    state_dict_key: str
    module_prefix_removed: bool
    inspection: dict[str, Any]


def safe_metadata(value: Any) -> Any:
    """Convert metadata to torch weights-only compatible primitive values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): safe_metadata(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_metadata(item) for item in value]
    raise TypeError(
        f"Checkpoint metadata must contain only safe primitive values, got {type(value).__name__}"
    )


def _safe_torch_load(
    path: Path,
    map_location: str | torch.device,
) -> tuple[Any, bool]:
    try:
        return torch.load(path, map_location=map_location, weights_only=True), True
    except TypeError:
        # Compatibility with PyTorch versions that predate weights_only.
        return torch.load(path, map_location=map_location), False
    except Exception as exc:
        raise ValueError(
            "Checkpoint could not be loaded with torch.load(weights_only=True). "
            f"Only state_dict checkpoints with safe primitive metadata are supported: {path}. "
            f"Original error: {exc}"
        ) from exc


def _is_raw_state_dict(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and bool(value)
        and all(isinstance(key, str) for key in value)
        and all(
            isinstance(item, (torch.Tensor, nn.Parameter)) for item in value.values()
        )
    )


def _extract_state_dict(payload: Any) -> tuple[Mapping[str, torch.Tensor], str]:
    if _is_raw_state_dict(payload):
        return payload, "<root>"
    if isinstance(payload, Mapping):
        for key in _STATE_DICT_KEYS:
            value = payload.get(key)
            if _is_raw_state_dict(value):
                return value, key
    raise ValueError(
        "Checkpoint does not contain a supported state dict. Expected one of "
        f"{', '.join(_STATE_DICT_KEYS)} or a pure state_dict."
    )


def _strip_module_prefix(
    state_dict: Mapping[str, torch.Tensor],
) -> tuple[OrderedDict[str, torch.Tensor], bool]:
    keys = list(state_dict)
    has_prefix = any(key.startswith("module.") for key in keys)
    if not has_prefix:
        return OrderedDict((key, value) for key, value in state_dict.items()), False
    if not all(key.startswith("module.") for key in keys):
        raise ValueError(
            "Checkpoint state_dict mixes keys with and without the 'module.' prefix; "
            "refusing an ambiguous rewrite."
        )
    stripped = OrderedDict(
        (key[len("module.") :], value) for key, value in state_dict.items()
    )
    if len(stripped) != len(state_dict):
        raise ValueError(
            "Removing the 'module.' prefix produced duplicate state_dict keys"
        )
    return stripped, True


def _metadata_value(payload: Any, *keys: str) -> Any:
    if not isinstance(payload, Mapping):
        return None
    for key in keys:
        if key in payload:
            return payload[key]
    return None


def _top_level_keys(payload: Any) -> list[str]:
    if not isinstance(payload, Mapping):
        return []
    return [str(key) for key in payload.keys()]


def load_checkpoint_artifact(
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> LoadedCheckpoint:
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {path}")
    if not path.is_file():
        raise ValueError(f"Expected a checkpoint file, got: {path}")

    payload, used_weights_only = _safe_torch_load(path, map_location)
    raw_state_dict, state_dict_key = _extract_state_dict(payload)
    state_dict, module_prefix_removed = _strip_module_prefix(raw_state_dict)
    sample_entries = [
        {
            "name": name,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
        }
        for name, tensor in list(state_dict.items())[:12]
    ]
    config = _metadata_value(payload, "config", "hparams", "hyper_parameters")
    labels = _metadata_value(payload, "class_names", "labels", "classes")
    thresholds = _metadata_value(payload, "thresholds", "threshold")
    preprocessing = _metadata_value(
        payload,
        "preprocessing_contract",
        "preprocessing",
        "preprocess_config",
    )
    if preprocessing is None and isinstance(config, Mapping):
        preprocessing = {
            "data": config.get("data"),
            "preprocessing": config.get("preprocessing"),
            "stft": config.get("stft"),
        }
    inspection: dict[str, Any] = {
        "checkpoint_path": str(path),
        "checkpoint_schema_version": _metadata_value(
            payload, "checkpoint_schema_version"
        ),
        "runtime_version": _metadata_value(payload, "runtime_version"),
        "created_at": _metadata_value(payload, "created_at"),
        "training_git_commit": _metadata_value(payload, "training_git_commit"),
        "prediction_mode": _metadata_value(payload, "prediction_mode"),
        "monitor_name": _metadata_value(payload, "monitor_name"),
        "file_size_bytes": path.stat().st_size,
        "top_level_type": f"{type(payload).__module__}.{type(payload).__name__}",
        "top_level_keys": _top_level_keys(payload),
        "state_dict_key": state_dict_key,
        "epoch": _metadata_value(payload, "epoch"),
        "best_metric": _metadata_value(payload, "best_metric", "best_score", "metric"),
        "labels": labels,
        "thresholds": thresholds,
        "config_present": isinstance(config, Mapping),
        "config_fields": [str(key) for key in config]
        if isinstance(config, Mapping)
        else [],
        "model_config": config.get("model") if isinstance(config, Mapping) else None,
        "preprocessing_config": preprocessing,
        "has_optimizer": any(
            key in _top_level_keys(payload)
            for key in ("optimizer", "optimizer_state", "optimizer_state_dict")
        ),
        "has_scheduler": any(
            key in _top_level_keys(payload)
            for key in ("scheduler", "scheduler_state", "scheduler_state_dict")
        ),
        "module_prefix_present": any(
            key.startswith("module.") for key in raw_state_dict
        ),
        "module_prefix_removed": module_prefix_removed,
        "strict_loading": True,
        "state_dict_tensor_count": len(state_dict),
        "state_dict_scalar_count": int(
            sum(int(tensor.numel()) for tensor in state_dict.values())
        ),
        "state_dict_sample": sample_entries,
        "torch_load_weights_only": used_weights_only,
    }
    return LoadedCheckpoint(
        path=path,
        payload=payload,
        state_dict=state_dict,
        state_dict_key=state_dict_key,
        module_prefix_removed=module_prefix_removed,
        inspection=inspection,
    )


def inspect_checkpoint(checkpoint_path: str | Path) -> dict[str, Any]:
    return load_checkpoint_artifact(checkpoint_path, map_location="cpu").inspection


def checkpoint_mapping(loaded: LoadedCheckpoint) -> Mapping[str, Any]:
    if _is_raw_state_dict(loaded.payload):
        return {}
    if isinstance(loaded.payload, Mapping):
        return loaded.payload
    return {}


def load_state_dict_strict(
    model: nn.Module,
    loaded: LoadedCheckpoint,
) -> None:
    expected = model.state_dict()
    supplied = loaded.state_dict
    missing_keys = sorted(set(expected).difference(supplied))
    unexpected_keys = sorted(set(supplied).difference(expected))
    shape_mismatches = [
        {
            "key": key,
            "checkpoint_shape": list(supplied[key].shape),
            "model_shape": list(expected[key].shape),
        }
        for key in sorted(set(expected).intersection(supplied))
        if tuple(expected[key].shape) != tuple(supplied[key].shape)
    ]
    if missing_keys or unexpected_keys or shape_mismatches:
        model_class = f"{model.__class__.__module__}.{model.__class__.__name__}"
        raise RuntimeError(
            "Strict checkpoint loading failed.\n"
            f"model_class: {model_class}\n"
            f"checkpoint: {loaded.path}\n"
            f"missing_keys: {missing_keys}\n"
            f"unexpected_keys: {unexpected_keys}\n"
            f"shape_mismatches: {shape_mismatches}"
        )
    try:
        model.load_state_dict(supplied, strict=True)
    except Exception as exc:
        raise RuntimeError(
            "Strict checkpoint loading failed after key/shape validation.\n"
            f"model_class: {model.__class__.__module__}.{model.__class__.__name__}\n"
            f"checkpoint: {loaded.path}\n"
            f"original_error: {exc}"
        ) from exc


__all__ = [
    "LoadedCheckpoint",
    "checkpoint_mapping",
    "inspect_checkpoint",
    "load_checkpoint_artifact",
    "load_state_dict_strict",
    "safe_metadata",
]
