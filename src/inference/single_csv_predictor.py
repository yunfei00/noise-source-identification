from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import inspect
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from src.features import (
    CsvSignal,
    PreprocessConfig,
    PreprocessResult,
    load_csv_signal,
    preprocess_signal,
)
from src.inference.checkpoint import (
    checkpoint_mapping,
    inspect_checkpoint,
    load_checkpoint_artifact,
    load_state_dict_strict,
)
from src.model_cnn import NoiseCNN, build_model
from src.train import resolve_device


DEFAULT_THRESHOLD = 0.5


@dataclass(frozen=True)
class SinglePredictionResult:
    csv_path: str
    checkpoint_path: str
    model_class: str
    device: str
    labels: list[str]
    logits: list[float]
    auxiliary_logits: dict[str, list[float]] | None
    probabilities: list[float]
    percentages: list[float]
    thresholds: list[float]
    binary_prediction: list[int]
    binary_label: str
    predicted_sources: list[str]
    input_shape: list[int]
    prediction_json_path: str
    report_path: str


class SingleCsvInferenceError(RuntimeError):
    """An inference failure with the input/checkpoint context preserved."""


def _raise_inference_error(
    stage: str,
    csv_path: str | Path,
    checkpoint_path: str | Path,
    exc: Exception,
    suggestion: str,
) -> None:
    raise SingleCsvInferenceError(
        "单文件推理失败\n"
        f"出错阶段：{stage}\n"
        f"CSV 路径：{Path(csv_path)}\n"
        f"checkpoint 路径：{Path(checkpoint_path)}\n"
        f"原始异常：{type(exc).__name__}: {exc}\n"
        f"建议检查项：{suggestion}"
    ) from exc


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
    """Merge mappings while keeping checkpoint values authoritative."""
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
) -> tuple[dict[str, Any], str, dict[str, Any] | None]:
    external_config = _load_config_file(config_path) if config_path is not None else None
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
            "Checkpoint is missing the model/preprocessing config. "
            "Pass the exact training config with --config."
        )
    if checkpoint_config is not None:
        effective = _deep_merge(external_config or {}, checkpoint_config)
        source = (
            "checkpoint config (authoritative; external config only fills missing fields)"
            if external_config is not None
            else "checkpoint config"
        )
    else:
        effective = dict(external_config or {})
        source = f"config file: {Path(config_path)}"
    return effective, source, external_config


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
        "No explicit label order was found in checkpoint class_names/labels "
        "or config data.class_names. Refusing to infer label order from directories."
    )


def _normalize_thresholds(
    value: float | Sequence[float] | Mapping[str, float] | np.ndarray | torch.Tensor,
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
        raise ValueError(f"Thresholds must be finite values in [0, 1], got {thresholds}")
    return thresholds


def _resolve_thresholds(
    explicit_threshold: float | list[float] | None,
    payload: Mapping[str, Any],
    config: Mapping[str, Any],
    labels: list[str],
    config_source: str,
) -> tuple[list[float], str, float]:
    configured_default = DEFAULT_THRESHOLD
    train_config = config.get("train", {})
    if isinstance(train_config, Mapping) and "threshold" in train_config:
        configured_value = train_config["threshold"]
        if not isinstance(configured_value, Mapping) and np.asarray(configured_value).ndim == 0:
            configured_default = float(configured_value)
    if explicit_threshold is not None:
        return _normalize_thresholds(explicit_threshold, labels), "CLI/Python explicit argument", configured_default
    for key in ("thresholds", "threshold"):
        if key in payload and payload[key] is not None:
            return _normalize_thresholds(payload[key], labels), f"checkpoint.{key}", configured_default
    if isinstance(train_config, Mapping) and "thresholds" in train_config:
        return (
            _normalize_thresholds(train_config["thresholds"], labels),
            f"{config_source}: train.thresholds",
            configured_default,
        )
    if isinstance(train_config, Mapping) and "threshold" in train_config:
        return (
            _normalize_thresholds(train_config["threshold"], labels),
            f"{config_source}: train.threshold",
            configured_default,
        )
    return [DEFAULT_THRESHOLD] * len(labels), "project default", DEFAULT_THRESHOLD


def _model_init_parameters(config: Mapping[str, Any], num_classes: int) -> dict[str, Any]:
    model_config = config.get("model", {})
    stft_config = config.get("stft", {})
    auxiliary_config = model_config.get("auxiliary_heads", {}) if isinstance(model_config, Mapping) else {}
    prediction_config = model_config.get("prediction", {}) if isinstance(model_config, Mapping) else {}
    representation = str(stft_config.get("input_representation", "single"))
    channels = {
        "single": 1,
        "absolute": 1,
        "legacy": 1,
        "absolute_relative": 2,
        "db_trace": 4,
    }[representation.strip().lower()]
    return {
        "num_classes": num_classes,
        "auxiliary_heads": bool(auxiliary_config.get("enabled", False)),
        "input_channels": channels,
        "architecture": str(model_config.get("architecture", "lightweight")),
        "base_channels": int(model_config.get("base_channels", 32)),
        "dropout": float(model_config.get("dropout", 0.0)),
        "prediction_mode": str(prediction_config.get("mode", "multilabel")),
        "combo_score_weight": float(prediction_config.get("combo_score_weight", 1.0)),
        "multilabel_score_weight": float(prediction_config.get("multilabel_score_weight", 0.3)),
        "count_score_weight": float(prediction_config.get("count_score_weight", 0.2)),
    }


def prepare_single_csv_input(
    csv_path: str | Path,
    config: Mapping[str, Any],
    *,
    require_data_marker: bool = True,
) -> tuple[CsvSignal, PreprocessResult]:
    """Build one model sample with the same deterministic path as Dataset."""
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


def _loss_description(config: Mapping[str, Any]) -> str:
    loss_config = config.get("loss", {})
    loss_type = str(loss_config.get("type", "bce")).strip().lower()
    if loss_type == "asymmetric_bce":
        base = "AsymmetricBCEWithLogitsLoss (binary_cross_entropy_with_logits)"
    elif loss_type == "bce":
        base = "BCEWithLogitsLoss"
    else:
        base = loss_type
    auxiliary = config.get("model", {}).get("auxiliary_heads", {})
    if isinstance(auxiliary, Mapping) and bool(auxiliary.get("enabled", False)):
        return f"MultiTaskLoss: {base} + combination CrossEntropyLoss + count CrossEntropyLoss"
    return base


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_json_safe(item) for item in value]
    return repr(value)


def _json_text(value: Any) -> str:
    return json.dumps(_json_safe(value), ensure_ascii=False, indent=2)


def _value_range_text(value: Any) -> str:
    if value is None:
        return "不适用"
    return str(value)


def _write_prediction_json(result: SinglePredictionResult, path: Path) -> None:
    payload = asdict(result)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _write_contract_report(
    path: Path,
    *,
    result: SinglePredictionResult,
    parsed: CsvSignal,
    processed: PreprocessResult,
    checkpoint_inspection: Mapping[str, Any],
    config: Mapping[str, Any],
    config_source: str,
    labels_source: str,
    threshold_source: str,
    configured_default_threshold: float,
    model: NoiseCNN,
    model_init: Mapping[str, Any],
    activation: str,
    output_description: str,
) -> None:
    stats = processed.statistics
    model_class = result.model_class
    model_source = inspect.getsourcefile(model.__class__) or "unknown"
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    label_row_values = []
    for index, label in enumerate(result.labels):
        safe_label = label.replace("|", r"\|")
        label_row_values.append(
            f"| {index} | {safe_label} | 干扰源 `{safe_label}` | {index} |"
        )
    label_rows = "\n".join(label_row_values)
    thresholds_are_independent = len(set(result.thresholds)) > 1
    representation = str(stats["input_representation"]).strip().lower()
    if representation == "db_trace":
        db_rule = (
            "不执行 dB→线性转换。训练代码保留 dB 轨迹，先减去逐轨迹中位数，"
            "再计算绝对/相对 STFT，并以两个常量通道编码中位 dB 与变化强度。"
        )
        db_step = "长度调整后，在 `prepare_db_trace_channels` 内进行中位数中心化"
    else:
        db_rule = "不执行；checkpoint 的训练配置未声明 dB→线性步骤。"
        db_step = "不适用"
    loss_description = _loss_description(config)
    if model.auxiliary_heads and model.prediction_mode == "structured":
        activation_reason = (
            "训练/验证使用结构化组合评分的 argmax，输出一个合法的非空标签组合；"
            "此处直接复用 `NoiseCNN.probabilities_from_outputs`。"
        )
    else:
        activation_reason = (
            "损失接收未激活 logits；多标签各输出相互独立，因此推理使用 sigmoid，"
            "与训练/验证代码一致，不能改用 softmax。"
        )

    report = f"""# 模型推理契约报告

本报告由单文件推理入口自动生成。checkpoint 内嵌配置优先于外部配置，所有权重以 `strict=True` 加载。

## 1. 模型类

- Python 模块路径：`{model.__class__.__module__}`
- 模型类名称：`{model.__class__.__name__}`
- 完整导入路径：`{model_class}`
- 初始化参数：

```json
{_json_text(model_init)}
```

- 输出类别数量：{len(result.labels)}
- 参数总量：{total_parameters}
- 可训练参数量：{trainable_parameters}
- 模型源代码文件：`{model_source}`
- 配置来源：{config_source}
- 模型结构摘要：

```text
{model}
```

## 2. checkpoint 结构

- checkpoint 文件路径：`{checkpoint_inspection["checkpoint_path"]}`
- 文件大小：{checkpoint_inspection["file_size_bytes"]} bytes
- 顶层对象类型：`{checkpoint_inspection["top_level_type"]}`
- 顶层键列表：`{checkpoint_inspection["top_level_keys"]}`
- state_dict 所在键：`{checkpoint_inspection["state_dict_key"]}`
- epoch：{checkpoint_inspection["epoch"]}
- best metric：{checkpoint_inspection["best_metric"]}
- 配置字段：`{checkpoint_inspection["config_fields"]}`
- 标签字段及值：`{_json_text(checkpoint_inspection["labels"])}`
- 阈值字段及值：`{_json_text(checkpoint_inspection["thresholds"])}`
- 预处理字段：`{_json_text(checkpoint_inspection["preprocessing_config"])}`
- 包含 optimizer：{checkpoint_inspection["has_optimizer"]}
- 包含 scheduler：{checkpoint_inspection["has_scheduler"]}
- 原始权重含 `module.` 前缀：{checkpoint_inspection["module_prefix_present"]}
- 是否移除 `module.` 前缀：{checkpoint_inspection["module_prefix_removed"]}
- 权重加载模式：`strict=True`
- state_dict 参数张量数：{checkpoint_inspection["state_dict_tensor_count"]}
- state_dict 标量总数：{checkpoint_inspection["state_dict_scalar_count"]}
- `torch.load`：`weights_only={checkpoint_inspection["torch_load_weights_only"]}`
- state_dict 部分名称和形状：

```json
{_json_text(checkpoint_inspection["state_dict_sample"])}
```

## 3. 输入张量形状

- CSV 有效数据长度：{parsed.numeric_value_count}
- CSV 原始数组形状：`{stats["raw_shape"]}`
- dB 转换后形状：`{stats["linear_shape"]}`（未转换）
- 长度调整后形状：`{stats["resized_shape"]}`
- 归一化后形状：`{stats["normalized_shape"]}`
- 预处理后 NumPy 特征形状：`{stats["feature_shape"]}`
- 单样本张量形状：`{stats["sample_tensor_shape"]}`
- batch 后模型输入形状：`{result.input_shape}`
- dtype：`{stats["sample_tensor_dtype"]}`
- device：`{result.device}`

## 4. CSV 读取规则

- 解析函数：`src.features.load_csv_signal`
- 文件编码：`{parsed.encoding}`（依次尝试 UTF-8 BOM 兼容格式、GB18030）
- 检测分隔符：`{parsed.delimiter}`；解析接受逗号、空格或 tab
- 表头处理：DATA 后不允许表头；第二列必须直接为数值
- 元数据处理：DATA 之前的 {len(parsed.metadata_rows)} 行只记录、不进入模型
- DATA 前内容：全部排除
- 空行处理：跳过（本次 {parsed.skipped_empty_rows} 行）
- 非法数据处理：严格模式立即报错，不跳过、不填充
- 使用的数据列：`{parsed.selected_columns}`（零基索引）
- 文件结束处理：从 DATA 下一行读取到 EOF

## 5. DATA 段提取规则

- DATA 标志匹配：整行 `strip()` 后忽略大小写精确匹配 `DATA`
- DATA 标志列：整行标志，逻辑列 0
- DATA 标志所在行：{parsed.data_marker_line}
- DATA 数据起始行：{parsed.data_start_line}
- DATA 标志行是否含数据：否
- DATA 数据结束规则：EOF
- 提取列：第二列（索引 1）
- 提取数据点数：{parsed.numeric_value_count}
- 缺失 DATA：立即失败并报告“未找到 DATA 数据段”

## 6. dB 转线性规则

- 是否转换：否
- 规则：{db_rule}
- 发生步骤：{db_step}
- 实际公式：不适用
- 输入值范围：`{_value_range_text(stats["raw_range"])}`
- 输出值范围：不适用（没有 dB→线性数组）
- 是否裁剪数值：否；仅长度可能中心截断
- NaN/Inf：严格 CSV 解析阶段立即报错
- epsilon：dB→线性不适用；相对 STFT 通道缩放下限为 `1e-12`

## 7. 归一化规则

- 方法：`{stats["normalization_method"]}`
- 顺序：CSV 解析 → 固定长度 → 归一化/表示特定中心化 → STFT
- 参数：`{_json_text(stats["normalization_parameters"])}`
- 参数来源：{stats["normalization_parameter_source"]}
- epsilon：`{stats["normalization_parameters"].get("epsilon", "不适用")}`
- 归一化前范围：`{stats["normalization_before_range"]}`
- 归一化后范围：`{stats["normalization_after_range"]}`
- 是否逐样本处理：是（若配置为 `standardize`）
- 是否使用训练集统计量：{stats["uses_training_statistics"]}
- 长度处理：{stats["length_method"]}
- 原始点数 / 目标点数：{stats["original_length"]} / {stats["target_length"]}
- 截断起止位置：{stats["crop_start"]} / {stats["crop_end"]}
- 填充值：{stats["pad_value"]}

## 8. 标签顺序

- 标签顺序来源：`{labels_source}`
- 二进制字符串方向：从左到右依次对应输出索引 0..{len(result.labels) - 1}

| 输出索引 | 标签名称 | 业务含义 | 二进制字符串位置 |
|---:|---|---|---:|
{label_rows}

## 9. 输出激活

- loss 函数：{loss_description}
- 模型直接输出：{output_description}
- 模型最后一层：`Linear`（`model.classifier`）
- 模型内部是否包含 sigmoid：否
- 推理实际激活/解码：`{activation}`
- 匹配原因：{activation_reason}
- 本次原始输出 logits：`{result.logits}`
- 本次辅助原始输出：`{result.auxiliary_logits}`
- 本次最终概率/验证输出：`{result.probabilities}`

## 10. 阈值

- 阈值来源：{threshold_source}
- 项目/配置默认阈值：{configured_default_threshold}
- 本次实际阈值：`{result.thresholds}`
- 每标签独立阈值：{thresholds_are_independent}
- 判定条件：`probability >= threshold`
- CLI 覆盖规则：显式 `--threshold` > checkpoint 顶层阈值 > checkpoint/外部配置 > 项目默认值 `{DEFAULT_THRESHOLD}`
- 结构化模式说明：结构化模型先按训练/验证解码规则产生合法组合，再保留相同的 `>=` 判定接口。

## 11. 单文件预测入口

- 脚本路径：`scripts/predict_single_csv.py`
- Python 接口：`src.inference.single_csv_predictor.predict_single_csv`
- 必填参数：`--csv`、`--checkpoint`
- 可选参数：`--threshold`、`--device`、`--report-dir`、`--config`
- Python 返回值：`SinglePredictionResult`
- JSON 输出：`{result.prediction_json_path}`
- 契约报告：`{result.report_path}`
- 错误处理：按 checkpoint、配置、CSV、预处理、严格权重加载、模型执行、产物写入阶段报告；保留原始异常和检查建议。

```bash
python scripts/predict_single_csv.py --csv "{result.csv_path}" --checkpoint "{result.checkpoint_path}"
```
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")


def predict_single_csv(
    csv_path: str | Path,
    checkpoint_path: str | Path,
    *,
    config_path: str | Path | None = None,
    threshold: float | list[float] | None = None,
    device: str = "auto",
    report_dir: str | Path = "outputs/inference_contract",
) -> SinglePredictionResult:
    """Run strict DATA-section inference and write JSON plus a contract report."""
    csv_path = Path(csv_path)
    checkpoint_path = Path(checkpoint_path)

    try:
        resolved_device = resolve_device(device)
    except Exception as exc:
        _raise_inference_error(
            "设备解析",
            csv_path,
            checkpoint_path,
            exc,
            "device 应为 auto、cpu、cuda 或可用的 cuda:N。",
        )

    try:
        loaded = load_checkpoint_artifact(
            checkpoint_path,
            map_location=resolved_device,
        )
    except Exception as exc:
        _raise_inference_error(
            "checkpoint 检查",
            csv_path,
            checkpoint_path,
            exc,
            "确认文件存在、来自 PyTorch，且包含 model_state/model_state_dict/state_dict 或纯 state_dict。",
        )
    payload = checkpoint_mapping(loaded)

    try:
        config, config_source, _ = _resolve_config(payload, config_path)
        labels, labels_source = _resolve_labels(payload, config)
        thresholds, threshold_source, configured_default = _resolve_thresholds(
            threshold,
            payload,
            config,
            labels,
            config_source,
        )
        PreprocessConfig.from_config(config)
        model_init = _model_init_parameters(config, len(labels))
    except Exception as exc:
        _raise_inference_error(
            "配置/标签/阈值恢复",
            csv_path,
            checkpoint_path,
            exc,
            "优先检查 checkpoint 的 config、class_names 与 threshold；纯 state_dict 必须配套 --config。",
        )

    try:
        parsed, processed = prepare_single_csv_input(
            csv_path,
            config,
            require_data_marker=True,
        )
    except Exception as exc:
        _raise_inference_error(
            "CSV 解析与预处理",
            csv_path,
            checkpoint_path,
            exc,
            "确认存在独占一行的 DATA 标志；其后每个非空行至少两列，第二列均为有限数值。",
        )

    try:
        model = build_model(num_classes=len(labels), config=config).to(resolved_device)
        load_state_dict_strict(model, loaded)
        model.eval()
    except Exception as exc:
        _raise_inference_error(
            "模型构建与严格权重加载",
            csv_path,
            checkpoint_path,
            exc,
            "确认使用训练 checkpoint 内嵌配置，核对模型类、输入通道、输出类别和全部权重形状。",
        )

    model_input = processed.input_tensor.unsqueeze(0).to(resolved_device)
    try:
        with torch.inference_mode():
            if model.auxiliary_heads and model.prediction_mode == "structured":
                outputs = model.forward_with_auxiliary(model_input)
                raw_logits = outputs[0]
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
                probability_tensor = model.probabilities_from_outputs(outputs)
                activation = "structured combination argmax (training/validation decoder)"
                output_description = (
                    "multilabel logits、combination logits、count logits；"
                    "JSON 的 logits 字段保存 multilabel logits"
                )
            else:
                raw_logits = model(model_input)
                auxiliary_logits = None
                probability_tensor = torch.sigmoid(raw_logits)
                activation = "sigmoid"
                output_description = "每标签未激活 logits"
        logits = raw_logits.squeeze(0).detach().cpu().to(torch.float64).tolist()
        probabilities = (
            probability_tensor.squeeze(0).detach().cpu().to(torch.float64).tolist()
        )
    except Exception as exc:
        _raise_inference_error(
            "模型前向推理",
            csv_path,
            checkpoint_path,
            exc,
            f"模型输入形状为 {list(model_input.shape)}；检查 checkpoint 配置与训练输入契约。",
        )

    if len(logits) != len(labels) or len(probabilities) != len(labels):
        exc = ValueError(
            f"Model outputs do not match labels: logits={len(logits)}, "
            f"probabilities={len(probabilities)}, labels={len(labels)}"
        )
        _raise_inference_error(
            "输出维度检查",
            csv_path,
            checkpoint_path,
            exc,
            "核对 checkpoint 标签顺序与模型 num_classes。",
        )

    binary_prediction = [
        int(probability >= class_threshold)
        for probability, class_threshold in zip(probabilities, thresholds)
    ]
    predicted_sources = [
        label
        for label, predicted in zip(labels, binary_prediction)
        if predicted == 1
    ]
    output_dir = Path(report_dir)
    json_path = output_dir / f"{csv_path.stem}_prediction.json"
    report_path = output_dir / "inference_contract.md"
    result = SinglePredictionResult(
        csv_path=str(csv_path),
        checkpoint_path=str(checkpoint_path),
        model_class=f"{model.__class__.__module__}.{model.__class__.__name__}",
        device=str(resolved_device),
        labels=labels,
        logits=[float(value) for value in logits],
        auxiliary_logits=auxiliary_logits,
        probabilities=[float(value) for value in probabilities],
        percentages=[float(value) * 100.0 for value in probabilities],
        thresholds=thresholds,
        binary_prediction=binary_prediction,
        binary_label="".join(str(value) for value in binary_prediction),
        predicted_sources=predicted_sources,
        input_shape=list(model_input.shape),
        prediction_json_path=str(json_path),
        report_path=str(report_path),
    )
    try:
        _write_prediction_json(result, json_path)
        _write_contract_report(
            report_path,
            result=result,
            parsed=parsed,
            processed=processed,
            checkpoint_inspection=loaded.inspection,
            config=config,
            config_source=config_source,
            labels_source=labels_source,
            threshold_source=threshold_source,
            configured_default_threshold=configured_default,
            model=model,
            model_init=model_init,
            activation=activation,
            output_description=output_description,
        )
    except Exception as exc:
        _raise_inference_error(
            "JSON/契约报告写入",
            csv_path,
            checkpoint_path,
            exc,
            f"确认输出目录可写：{output_dir}",
        )
    return result


def print_prediction_result(result: SinglePredictionResult) -> None:
    print(f"CSV 文件：{result.csv_path}")
    print(f"checkpoint：{result.checkpoint_path}")
    print(f"设备：{result.device}")
    print(f"模型类：{result.model_class}")
    print(f"模型输入形状：{result.input_shape}")
    print(f"标签顺序：{result.labels}")
    print(f"阈值：{result.thresholds}")
    print("\n各标签预测：")
    for label, logit, probability, percentage, predicted in zip(
        result.labels,
        result.logits,
        result.probabilities,
        result.percentages,
        result.binary_prediction,
    ):
        print(
            f"{label}: logit={logit:.6f}, probability={probability:.6f}, "
            f"percentage={percentage:.2f}% -> {predicted}"
        )
    print(f"\n预测标签向量：\n{result.binary_label}")
    print("\n识别到的干扰源：")
    if result.predicted_sources:
        for source in result.predicted_sources:
            print(f"- {source}")
    else:
        print("- 无")
    print(f"\nJSON：{result.prediction_json_path}")
    print(f"推理契约报告：{result.report_path}")
