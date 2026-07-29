from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal

DecisionMode = Literal["multilabel", "structured"]


@dataclass(frozen=True)
class PredictionResult:
    """Stable, GUI-friendly result returned by ``InferenceSession``.

    In structured mode, ``label_marginal_probabilities`` are the default
    per-source display values and ``decoded_label_vector`` is produced solely
    by combination argmax. Thresholds are metadata only in that mode.
    """

    runtime_version: str
    csv_path: str | None
    checkpoint_path: str
    model_class: str
    device: str
    labels: list[str]
    decision_mode: DecisionMode
    thresholds: list[float]
    thresholds_applicable: bool
    threshold_source: str
    multilabel_logits: list[float]
    multilabel_probabilities: list[float]
    combination_labels: list[str]
    combination_probabilities: list[float] | None
    label_marginal_probabilities: list[float]
    decoded_label_vector: list[int]
    predicted_combination: str
    predicted_sources: list[str]
    input_shape: list[int]
    sample_tensor_shape: list[int]
    auxiliary_logits: dict[str, list[float]] | None
    preprocessing_statistics: dict[str, Any]
    csv_contract: dict[str, Any] | None
    prediction_json_path: str | None = None
    report_path: str | None = None

    @property
    def display_probabilities(self) -> list[float]:
        """Probabilities that the GUI should display per source."""
        return self.label_marginal_probabilities

    @property
    def display_percentages(self) -> list[float]:
        return [value * 100.0 for value in self.display_probabilities]

    # Compatibility aliases. In structured mode these now expose true marginal
    # probabilities, never the old hard decoded vector.
    @property
    def logits(self) -> list[float]:
        return self.multilabel_logits

    @property
    def probabilities(self) -> list[float]:
        return self.display_probabilities

    @property
    def percentages(self) -> list[float]:
        return self.display_percentages

    @property
    def binary_prediction(self) -> list[int]:
        return self.decoded_label_vector

    @property
    def binary_label(self) -> str:
        return self.predicted_combination

    def with_export_paths(
        self,
        *,
        prediction_json_path: str,
        report_path: str,
    ) -> "PredictionResult":
        return replace(
            self,
            prediction_json_path=prediction_json_path,
            report_path=report_path,
        )

    def to_dict(self, *, include_compatibility_aliases: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "runtime_version": self.runtime_version,
            "csv_path": self.csv_path,
            "checkpoint_path": self.checkpoint_path,
            "model_class": self.model_class,
            "device": self.device,
            "labels": self.labels,
            "decision_mode": self.decision_mode,
            "thresholds": self.thresholds,
            "thresholds_applicable": self.thresholds_applicable,
            "threshold_source": self.threshold_source,
            "multilabel_logits": self.multilabel_logits,
            "multilabel_probabilities": self.multilabel_probabilities,
            "combination_labels": self.combination_labels,
            "combination_probabilities": self.combination_probabilities,
            "label_marginal_probabilities": self.label_marginal_probabilities,
            "display_percentages": self.display_percentages,
            "decoded_label_vector": self.decoded_label_vector,
            "predicted_combination": self.predicted_combination,
            "predicted_sources": self.predicted_sources,
            "input_shape": self.input_shape,
            "sample_tensor_shape": self.sample_tensor_shape,
            "auxiliary_logits": self.auxiliary_logits,
            "preprocessing_statistics": self.preprocessing_statistics,
            "csv_contract": self.csv_contract,
            "prediction_json_path": self.prediction_json_path,
            "report_path": self.report_path,
        }
        if include_compatibility_aliases:
            payload.update(
                {
                    "logits": self.multilabel_logits,
                    "probabilities": self.display_probabilities,
                    "percentages": self.display_percentages,
                    "binary_prediction": self.decoded_label_vector,
                    "binary_label": self.predicted_combination,
                    "probability_semantics": "label_marginal_probabilities",
                }
            )
        return payload
