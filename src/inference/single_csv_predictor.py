"""Compatibility layer for the public ``noise_source_runtime`` package.

New GUI code should keep an ``InferenceSession`` alive and call
``predict_file``/``predict_array`` directly. This module preserves the earlier
one-shot API and its automatic artifact export for existing callers and CLI.
"""

from __future__ import annotations

from pathlib import Path

from src.noise_source_runtime.checkpoint import inspect_checkpoint
from src.noise_source_runtime.exceptions import RuntimeInferenceError
from src.noise_source_runtime.preprocessing import (
    prepare_file_input as prepare_single_csv_input,
)
from src.noise_source_runtime.reporting import export_prediction_result
from src.noise_source_runtime.result import PredictionResult
from src.noise_source_runtime.session import InferenceSession

SinglePredictionResult = PredictionResult
SingleCsvInferenceError = RuntimeInferenceError


def predict_single_csv(
    csv_path: str | Path,
    checkpoint_path: str | Path,
    *,
    config_path: str | Path | None = None,
    threshold: float | list[float] | None = None,
    device: str = "auto",
    report_dir: str | Path | None = "outputs/inference_contract",
) -> PredictionResult:
    """One-shot compatibility helper.

    Unlike ``InferenceSession.predict_file``, this helper intentionally loads a
    model for one call. Set ``report_dir=None`` for a pure in-memory result.
    """
    with InferenceSession.load_model(
        checkpoint_path,
        config_path=config_path,
        threshold=threshold,
        device=device,
    ) as session:
        result = session.predict_file(csv_path)
        if report_dir is not None:
            result = export_prediction_result(result, session, report_dir)
        return result


def print_prediction_result(result: PredictionResult) -> None:
    print(f"CSV 文件：{result.csv_path}")
    print(f"checkpoint：{result.checkpoint_path}")
    print(f"设备：{result.device}")
    print(f"模型类：{result.model_class}")
    print(f"decision mode：{result.decision_mode}")
    print(f"模型输入形状：{result.input_shape}")
    print(f"标签顺序：{result.labels}")
    if result.thresholds_applicable:
        print(f"阈值：{result.thresholds}")
    else:
        print("阈值：不参与 structured 最终判定")

    print("\n各标签预测：")
    for index, label in enumerate(result.labels):
        multilabel_probability = result.multilabel_probabilities[index]
        marginal_probability = result.label_marginal_probabilities[index]
        decoded = result.decoded_label_vector[index]
        print(
            f"{label}: multilabel_probability={multilabel_probability:.6f}, "
            f"label_marginal_probability={marginal_probability:.6f}, "
            f"percentage={marginal_probability * 100.0:.2f}% -> {decoded}"
        )
    if result.combination_probabilities is not None:
        print("\n合法组合概率：")
        for label, probability in zip(
            result.combination_labels,
            result.combination_probabilities,
        ):
            print(f"{label}: {probability:.6f}")
    print(f"\n预测标签向量：\n{result.predicted_combination}")
    print("\n识别到的干扰源：")
    if result.predicted_sources:
        for source in result.predicted_sources:
            print(f"- {source}")
    else:
        print("- 无")
    if result.prediction_json_path:
        print(f"\nJSON：{result.prediction_json_path}")
    if result.report_path:
        print(f"推理契约报告：{result.report_path}")


__all__ = [
    "InferenceSession",
    "SingleCsvInferenceError",
    "SinglePredictionResult",
    "inspect_checkpoint",
    "predict_single_csv",
    "prepare_single_csv_input",
    "print_prediction_result",
]
