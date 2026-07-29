from __future__ import annotations

import inspect
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from threading import RLock
from typing import Any

import numpy as np
import torch
import yaml

from src.noise_source_runtime import RUNTIME_VERSION
from src.noise_source_runtime.checkpoint import (
    LoadedCheckpoint,
    checkpoint_mapping,
    load_checkpoint_artifact,
    load_state_dict_strict,
)
from src.noise_source_runtime.device import resolve_device
from src.noise_source_runtime.exceptions import (
    RuntimeInferenceError,
    SessionClosedError,
)
from src.noise_source_runtime.model import NoiseCNN, build_model
from src.noise_source_runtime.preprocessing import (
    PreprocessResult,
    prepare_array_input,
    prepare_file_input,
    preprocessing_contract,
)
from src.noise_source_runtime.result import PredictionResult

DEFAULT_THRESHOLD = 0.5


def _load_config_file(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
    else:
        with path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return config


def _deep_merge(
    base: Mapping[str, Any],
    authoritative: Mapping[str, Any],
) -> dict[str, Any]:
    merged: dict[str, Any] = dict(base)
    for key, value in authoritative.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve_config(
    payload: Mapping[str, Any],
    config_path: str | Path | None,
) -> tuple[dict[str, Any], str]:
    external_config = (
        _load_config_file(config_path) if config_path is not None else None
    )
    checkpoint_config = payload.get("config")
    if checkpoint_config is None:
        checkpoint_config = payload.get(
            "model_config",
            payload.get("hparams", payload.get("hyper_parameters")),
        )
    if checkpoint_config is not None and not isinstance(checkpoint_config, Mapping):
        raise ValueError("Checkpoint config metadata is not a mapping")
    if checkpoint_config is None and external_config is None:
        raise ValueError(
            "Checkpoint is missing model/preprocessing config; pass the exact "
            "training YAML/JSON with config_path."
        )
    if checkpoint_config is not None:
        return (
            _deep_merge(external_config or {}, checkpoint_config),
            (
                "checkpoint config (authoritative; external config fills missing fields)"
                if external_config is not None
                else "checkpoint config"
            ),
        )
    return dict(external_config or {}), f"config file: {Path(config_path)}"


def _valid_labels(value: Any) -> list[str] | None:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        labels = list(value)
        if labels and all(isinstance(label, str) and label.strip() for label in labels):
            return [label.strip() for label in labels]
    return None


def _resolve_labels(
    payload: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[list[str], str]:
    for key in ("class_names", "labels", "classes"):
        labels = _valid_labels(payload.get(key))
        if labels is not None:
            return labels, f"checkpoint.{key}"
    labels = _valid_labels(config.get("class_names"))
    if labels is not None:
        return labels, "config.class_names"
    data_config = config.get("data", {})
    if isinstance(data_config, Mapping):
        labels = _valid_labels(data_config.get("class_names"))
        if labels is not None:
            return labels, "config.data.class_names"
    raise ValueError(
        "No explicit label order was found in checkpoint class_names/labels or "
        "config data.class_names; directory-name inference is intentionally disabled."
    )


def _normalize_thresholds(
    value: Any,
    labels: list[str],
) -> list[float]:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        value = float(value) if value.ndim == 0 else value.tolist()
    if isinstance(value, Mapping):
        missing = [label for label in labels if label not in value]
        if missing:
            raise ValueError(f"Per-label thresholds are missing: {missing}")
        thresholds = [float(value[label]) for label in labels]
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        thresholds = [float(item) for item in value]
        if len(thresholds) != len(labels):
            raise ValueError(
                f"Expected {len(labels)} threshold values, got {len(thresholds)}"
            )
    else:
        thresholds = [float(value)] * len(labels)
    if not all(np.isfinite(item) and 0.0 <= item <= 1.0 for item in thresholds):
        raise ValueError(
            f"Thresholds must be finite values in [0, 1], got {thresholds}"
        )
    return thresholds


def _resolve_thresholds(
    explicit_threshold: float | list[float] | None,
    payload: Mapping[str, Any],
    config: Mapping[str, Any],
    labels: list[str],
    config_source: str,
) -> tuple[list[float], str]:
    if explicit_threshold is not None:
        return _normalize_thresholds(explicit_threshold, labels), "explicit argument"
    for key in ("thresholds", "threshold"):
        if key in payload and payload[key] is not None:
            return _normalize_thresholds(payload[key], labels), f"checkpoint.{key}"
    train_config = config.get("train", {})
    if isinstance(train_config, Mapping):
        for key in ("thresholds", "threshold"):
            if key in train_config and train_config[key] is not None:
                return (
                    _normalize_thresholds(train_config[key], labels),
                    f"{config_source}: train.{key}",
                )
    return [DEFAULT_THRESHOLD] * len(labels), "runtime default"


def _csv_contract(parsed: Any) -> dict[str, Any]:
    return {
        "csv_path": str(parsed.csv_path),
        "encoding": parsed.encoding,
        "delimiter": parsed.delimiter,
        "parser_mode": parsed.parser_mode,
        "found_data_line": parsed.found_data_line,
        "data_marker_line": parsed.data_marker_line,
        "data_start_line": parsed.data_start_line,
        "selected_columns": parsed.selected_columns,
        "metadata_row_count": len(parsed.metadata_rows),
        "data_row_count": len(parsed.data_rows),
        "numeric_value_count": parsed.numeric_value_count,
        "discarded_nonfinite_count": parsed.discarded_nonfinite_count,
        "skipped_empty_rows": parsed.skipped_empty_rows,
        "skipped_invalid_rows": parsed.skipped_invalid_rows,
    }


class InferenceSession:
    """Reusable, thread-safe model session with no GUI or network dependency."""

    def __init__(
        self,
        *,
        checkpoint: LoadedCheckpoint,
        model: NoiseCNN,
        config: dict[str, Any],
        config_source: str,
        labels: list[str],
        labels_source: str,
        thresholds: list[float],
        threshold_source: str,
        device: torch.device,
    ) -> None:
        self._checkpoint = checkpoint
        self._model: NoiseCNN | None = model
        self._config = config
        self._config_source = config_source
        self._labels = labels
        self._labels_source = labels_source
        self._thresholds = thresholds
        self._threshold_source = threshold_source
        self._device = device
        self._lock = RLock()
        self._closed = False
        self._model_load_count = 1

    @classmethod
    def load_model(
        cls,
        checkpoint_path: str | Path,
        *,
        config_path: str | Path | None = None,
        threshold: float | list[float] | None = None,
        device: str = "auto",
    ) -> "InferenceSession":
        try:
            resolved_device = resolve_device(device)
            loaded = load_checkpoint_artifact(
                checkpoint_path,
                map_location=resolved_device,
            )
            payload = checkpoint_mapping(loaded)
            config, config_source = _resolve_config(payload, config_path)
            labels, labels_source = _resolve_labels(payload, config)
            thresholds, threshold_source = _resolve_thresholds(
                threshold,
                payload,
                config,
                labels,
                config_source,
            )
            # Validate the full preprocessing contract before model construction.
            preprocessing_contract(config)
            model = build_model(num_classes=len(labels), config=config).to(
                resolved_device
            )
            load_state_dict_strict(model, loaded)
            model.eval()
            return cls(
                checkpoint=loaded,
                model=model,
                config=config,
                config_source=config_source,
                labels=labels,
                labels_source=labels_source,
                thresholds=thresholds,
                threshold_source=threshold_source,
                device=resolved_device,
            )
        except RuntimeInferenceError:
            raise
        except Exception as exc:
            raise RuntimeInferenceError(
                "Model session loading failed\n"
                f"checkpoint_path: {Path(checkpoint_path)}\n"
                f"config_path: {Path(config_path) if config_path is not None else None}\n"
                f"device: {device}\n"
                f"original_error: {type(exc).__name__}: {exc}"
            ) from exc

    def __enter__(self) -> "InferenceSession":
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    @property
    def labels(self) -> list[str]:
        return list(self._labels)

    @property
    def config(self) -> dict[str, Any]:
        return _deep_merge({}, self._config)

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def model_load_count(self) -> int:
        return self._model_load_count

    @property
    def checkpoint_inspection(self) -> dict[str, Any]:
        return dict(self._checkpoint.inspection)

    @property
    def model(self) -> NoiseCNN:
        self._ensure_open()
        assert self._model is not None
        return self._model

    def _ensure_open(self) -> None:
        if self._closed or self._model is None:
            raise SessionClosedError("InferenceSession is closed")

    def inspect_model(self) -> dict[str, Any]:
        model = self.model
        total_parameters = sum(parameter.numel() for parameter in model.parameters())
        trainable_parameters = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        return {
            "runtime_version": RUNTIME_VERSION,
            "checkpoint": self.checkpoint_inspection,
            "model_class": f"{model.__class__.__module__}.{model.__class__.__name__}",
            "model_source": inspect.getsourcefile(model.__class__),
            "model_summary": str(model),
            "parameter_count": total_parameters,
            "trainable_parameter_count": trainable_parameters,
            "labels": self.labels,
            "labels_source": self._labels_source,
            "prediction_mode": model.prediction_mode,
            "thresholds": list(self._thresholds),
            "threshold_source": self._threshold_source,
            "thresholds_applicable": model.prediction_mode == "multilabel",
            "device": str(self._device),
            "config_source": self._config_source,
            "preprocessing_contract": preprocessing_contract(self._config),
        }

    def predict_file(
        self,
        csv_path: str | Path,
        *,
        require_data_marker: bool = True,
    ) -> PredictionResult:
        self._ensure_open()
        try:
            parsed, processed = prepare_file_input(
                csv_path,
                self._config,
                require_data_marker=require_data_marker,
            )
            return self._predict_processed(
                processed,
                csv_path=str(Path(csv_path)),
                csv_contract=_csv_contract(parsed),
            )
        except RuntimeInferenceError:
            raise
        except Exception as exc:
            raise RuntimeInferenceError(
                "File inference failed\n"
                f"csv_path: {Path(csv_path)}\n"
                f"checkpoint_path: {self._checkpoint.path}\n"
                f"original_error: {type(exc).__name__}: {exc}"
            ) from exc

    def predict_array(
        self,
        signal: np.ndarray,
        *,
        source_name: str | None = None,
    ) -> PredictionResult:
        self._ensure_open()
        try:
            processed = prepare_array_input(signal, self._config)
            return self._predict_processed(
                processed,
                csv_path=source_name,
                csv_contract=None,
            )
        except RuntimeInferenceError:
            raise
        except Exception as exc:
            raise RuntimeInferenceError(
                "Array inference failed\n"
                f"source_name: {source_name}\n"
                f"checkpoint_path: {self._checkpoint.path}\n"
                f"original_error: {type(exc).__name__}: {exc}"
            ) from exc

    def _predict_processed(
        self,
        processed: PreprocessResult,
        *,
        csv_path: str | None,
        csv_contract: dict[str, Any] | None,
    ) -> PredictionResult:
        model = self.model
        model_input = processed.input_tensor.unsqueeze(0).to(self._device)
        with self._lock, torch.inference_mode():
            if model.auxiliary_heads and model.prediction_mode == "structured":
                outputs = model.forward_with_auxiliary(model_input)
                multilabel_logits_tensor = outputs[0]
                multilabel_probabilities_tensor = (
                    model.multilabel_probabilities_from_outputs(outputs)
                )
                combination_probabilities_tensor = model.structured_combo_probabilities(
                    outputs
                )
                combo_labels_tensor = model.combo_labels.to(
                    device=combination_probabilities_tensor.device,
                    dtype=combination_probabilities_tensor.dtype,
                )
                label_marginals_tensor = (
                    combination_probabilities_tensor @ combo_labels_tensor
                )
                decoded_index = combination_probabilities_tensor.argmax(dim=1)
                decoded_tensor = combo_labels_tensor[decoded_index]
                combination_labels = [
                    "".join(str(int(value)) for value in row)
                    for row in combo_labels_tensor.detach().cpu().tolist()
                ]
                combination_probabilities = (
                    combination_probabilities_tensor.squeeze(0)
                    .detach()
                    .cpu()
                    .to(torch.float64)
                    .tolist()
                )
                auxiliary_logits = {
                    "combination_logits": outputs[1]
                    .squeeze(0)
                    .detach()
                    .cpu()
                    .to(torch.float64)
                    .tolist(),
                    "count_logits": outputs[2]
                    .squeeze(0)
                    .detach()
                    .cpu()
                    .to(torch.float64)
                    .tolist(),
                }
                thresholds_applicable = False
                decision_mode = "structured"
            else:
                multilabel_logits_tensor = model(model_input)
                multilabel_probabilities_tensor = torch.sigmoid(
                    multilabel_logits_tensor
                )
                label_marginals_tensor = multilabel_probabilities_tensor
                threshold_tensor = torch.tensor(
                    self._thresholds,
                    device=model_input.device,
                    dtype=multilabel_probabilities_tensor.dtype,
                ).view(1, -1)
                decoded_tensor = (
                    multilabel_probabilities_tensor >= threshold_tensor
                ).to(dtype=torch.float32)
                combination_labels = []
                combination_probabilities = None
                auxiliary_logits = None
                thresholds_applicable = True
                decision_mode = "multilabel"

        multilabel_logits = (
            multilabel_logits_tensor.squeeze(0)
            .detach()
            .cpu()
            .to(torch.float64)
            .tolist()
        )
        multilabel_probabilities = (
            multilabel_probabilities_tensor.squeeze(0)
            .detach()
            .cpu()
            .to(torch.float64)
            .tolist()
        )
        label_marginal_probabilities = (
            label_marginals_tensor.squeeze(0).detach().cpu().to(torch.float64).tolist()
        )
        decoded_label_vector = [
            int(value) for value in decoded_tensor.squeeze(0).detach().cpu().tolist()
        ]
        if len(decoded_label_vector) != len(self._labels):
            raise RuntimeInferenceError(
                "Decoded label count does not match checkpoint class_names: "
                f"{len(decoded_label_vector)} vs {len(self._labels)}"
            )
        predicted_combination = "".join(str(value) for value in decoded_label_vector)
        predicted_sources = [
            label
            for label, present in zip(self._labels, decoded_label_vector)
            if present == 1
        ]
        return PredictionResult(
            runtime_version=RUNTIME_VERSION,
            csv_path=csv_path,
            checkpoint_path=str(self._checkpoint.path),
            model_class=f"{model.__class__.__module__}.{model.__class__.__name__}",
            device=str(self._device),
            labels=list(self._labels),
            decision_mode=decision_mode,
            thresholds=list(self._thresholds),
            thresholds_applicable=thresholds_applicable,
            threshold_source=self._threshold_source,
            multilabel_logits=[float(value) for value in multilabel_logits],
            multilabel_probabilities=[
                float(value) for value in multilabel_probabilities
            ],
            combination_labels=combination_labels,
            combination_probabilities=(
                [float(value) for value in combination_probabilities]
                if combination_probabilities is not None
                else None
            ),
            label_marginal_probabilities=[
                float(value) for value in label_marginal_probabilities
            ],
            decoded_label_vector=decoded_label_vector,
            predicted_combination=predicted_combination,
            predicted_sources=predicted_sources,
            input_shape=list(model_input.shape),
            sample_tensor_shape=list(processed.input_tensor.shape),
            auxiliary_logits=auxiliary_logits,
            preprocessing_statistics=dict(processed.statistics),
            csv_contract=csv_contract,
        )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._model = None
            self._closed = True
            if self._device.type == "cuda":
                torch.cuda.empty_cache()


__all__ = ["InferenceSession"]
