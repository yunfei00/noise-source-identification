from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.noise_source_runtime.result import PredictionResult

if TYPE_CHECKING:
    from src.noise_source_runtime.session import InferenceSession


def write_prediction_json(
    result: PredictionResult,
    output_path: str | Path,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            result.to_dict(include_compatibility_aliases=True),
            handle,
            ensure_ascii=False,
            indent=2,
        )
        handle.write("\n")
    return path


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def _label_table(result: PredictionResult) -> str:
    rows = []
    for index, label in enumerate(result.labels):
        safe_label = label.replace("|", r"\|")
        rows.append(f"| {index} | {safe_label} | 干扰源 `{safe_label}` | {index} |")
    return "\n".join(rows)


def _combination_table(result: PredictionResult) -> str:
    if result.combination_probabilities is None:
        return "当前为 multilabel 模式，不使用合法组合 softmax。"
    rows = [
        "| 组合索引 | 标签向量 | 概率 |",
        "|---:|---|---:|",
    ]
    rows.extend(
        f"| {index} | `{label}` | {probability:.8f} |"
        for index, (label, probability) in enumerate(
            zip(result.combination_labels, result.combination_probabilities)
        )
    )
    return "\n".join(rows)


def write_inference_contract(
    result: PredictionResult,
    session: "InferenceSession",
    output_path: str | Path,
) -> Path:
    """Write a reproducible Markdown contract without mutating session state."""
    path = Path(output_path)
    model_info = session.inspect_model()
    checkpoint = model_info["checkpoint"]
    preprocessing = model_info["preprocessing_contract"]
    stats = result.preprocessing_statistics
    csv_contract = result.csv_contract or {}
    structured = result.decision_mode == "structured"
    activation_text = (
        "sigmoid(multilabel_logits)；softmax(structured_scores)；"
        "label marginal = combination probabilities @ combo_labels"
        if structured
        else "sigmoid(multilabel_logits)"
    )
    decision_text = (
        "七种合法组合概率的 argmax；阈值不参与最终标签判定"
        if structured
        else "逐标签执行 probability >= threshold"
    )
    db_representation = (
        str(preprocessing["input_representation"]).strip().lower() == "db_trace"
    )
    db_rule = (
        "不执行 dB→线性；保留 dB 轨迹，减去逐轨迹中位数后计算 STFT，"
        "并编码绝对电平和变化强度。"
        if db_representation
        else "checkpoint 训练配置未声明 dB→线性转换，因此不执行。"
    )

    report = f"""# 模型推理契约报告

- runtime version：`{result.runtime_version}`
- decision mode：`{result.decision_mode}`
- 纯推理接口默认不写磁盘；本报告由显式导出函数生成。

## 1. 模型类

- 完整类路径：`{result.model_class}`
- 源代码：`{model_info["model_source"]}`
- 输出标签数：{len(result.labels)}
- 参数总量：{model_info["parameter_count"]}
- 可训练参数量：{model_info["trainable_parameter_count"]}
- 配置来源：{model_info["config_source"]}
- 模型结构：

```text
{model_info["model_summary"]}
```

## 2. checkpoint 结构

- 路径：`{checkpoint["checkpoint_path"]}`
- checkpoint schema：`{checkpoint["checkpoint_schema_version"]}`
- checkpoint 创建时间：`{checkpoint["created_at"]}`
- checkpoint runtime version：`{checkpoint["runtime_version"]}`
- training git commit：`{checkpoint["training_git_commit"]}`
- 顶层类型：`{checkpoint["top_level_type"]}`
- 顶层键：`{checkpoint["top_level_keys"]}`
- state_dict 键：`{checkpoint["state_dict_key"]}`
- epoch：{checkpoint["epoch"]}
- monitor：`{checkpoint["monitor_name"]}`
- best metric：{checkpoint["best_metric"]}
- `module.` 前缀存在/移除：{checkpoint["module_prefix_present"]} / {checkpoint["module_prefix_removed"]}
- 加载模式：`strict=True`
- `torch.load(weights_only=...)`：{checkpoint["torch_load_weights_only"]}
- 权重张量数 / 标量数：{checkpoint["state_dict_tensor_count"]} / {checkpoint["state_dict_scalar_count"]}
- 部分权重名称与形状：

```json
{_json_text(checkpoint["state_dict_sample"])}
```

## 3. 输入张量形状

- CSV/数组原始形状：`{stats.get("raw_shape")}`
- 原始有效点数：{stats.get("original_length")}
- 固定长度后形状：`{stats.get("resized_shape")}`
- 归一化后形状：`{stats.get("normalized_shape")}`
- 单样本特征形状：`{result.sample_tensor_shape}`
- 模型输入形状：`{result.input_shape}`
- dtype：`{stats.get("sample_tensor_dtype")}`
- device：`{result.device}`

## 4. CSV 读取规则

- 解析器：`src.features.load_csv_signal`（Dataset 与 runtime 共用）
- 编码：`{csv_contract.get("encoding", "数组输入，不适用")}`
- 分隔符：`{csv_contract.get("delimiter", "数组输入，不适用")}`
- parser mode：`{csv_contract.get("parser_mode", "array")}`
- DATA 前内容：只作为元数据，不进入模型
- 元数据行数：{csv_contract.get("metadata_row_count", "不适用")}
- 空行：跳过
- 非法、缺列或非有限数值：严格 DATA 模式立即报错
- 数据列：`{csv_contract.get("selected_columns", "数组输入，不适用")}`
- 结束规则：读取到 EOF

## 5. DATA 段提取规则

- 匹配：整行 `strip()` 后忽略大小写精确匹配 `DATA`
- DATA 行：{csv_contract.get("data_marker_line", "数组输入，不适用")}
- 数据起始行：{csv_contract.get("data_start_line", "数组输入，不适用")}
- 有效点数：{csv_contract.get("numeric_value_count", stats.get("original_length"))}
- 缺失 DATA：严格 `predict_file` 默认立即失败；旧仓库数据仅可显式启用兼容模式

## 6. dB 转线性规则

- 是否转换：否
- 规则：{db_rule}
- 输入范围：`{stats.get("raw_range")}`
- dB→线性公式：不适用
- NaN/Inf：严格文件解析阶段拒绝

## 7. 归一化规则

- input representation：`{preprocessing["input_representation"]}`
- 方法：`{stats.get("normalization_method")}`
- 参数：`{_json_text(stats.get("normalization_parameters"))}`
- 参数来源：{stats.get("normalization_parameter_source")}
- 归一化前/后范围：`{stats.get("normalization_before_range")}` / `{stats.get("normalization_after_range")}`
- 长度处理：{stats.get("length_method")}
- 原始/目标点数：{stats.get("original_length")} / {stats.get("target_length")}
- 截断起止：{stats.get("crop_start")} / {stats.get("crop_end")}
- 填充值：{stats.get("pad_value")}

## 8. 标签与组合顺序

- 标签顺序来源：`{model_info["labels_source"]}`
- 二进制字符串从左到右对应输出索引 0..{len(result.labels) - 1}

| 输出索引 | 标签名称 | 业务含义 | 字符串位置 |
|---:|---|---|---:|
{_label_table(result)}

### 合法组合顺序

{_combination_table(result)}

## 9. 输出概率与解码语义

- multilabel probabilities：`sigmoid(multilabel_logits)`
- combination probabilities：`softmax(structured_scores)`（仅 structured）
- label marginal probabilities：`combination_probabilities @ combo_labels`（structured）；multilabel 模式等于 sigmoid 输出
- 实际激活链：{activation_text}
- GUI 默认每标签显示值：`label_marginal_probabilities`
- 最终判定：{decision_text}
- multilabel probabilities：`{result.multilabel_probabilities}`
- label marginal probabilities：`{result.label_marginal_probabilities}`
- decoded label vector：`{result.decoded_label_vector}`
- predicted combination：`{result.predicted_combination}`

## 10. 阈值

- 来源：{result.threshold_source}
- 数值：`{result.thresholds}`
- thresholds applicable：{result.thresholds_applicable}
- multilabel 判定条件：`>=`
- structured 规则：阈值只保留为兼容元数据，不参与最终组合 argmax

## 11. 运行时入口与导出

- 正式包：`noise_source_runtime`
- 会话接口：`InferenceSession.load_model`、`predict_file`、`predict_array`、`inspect_model`、`close`
- CLI：`scripts/predict_single_csv.py`
- JSON：`{result.prediction_json_path}`
- 报告：`{result.report_path}`
- 纯推理默认行为：只返回 `PredictionResult`，不写 JSON/Markdown
- 显式导出：`write_prediction_json`、`write_inference_contract`、`export_prediction_result`
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")
    return path


def export_prediction_result(
    result: PredictionResult,
    session: "InferenceSession",
    output_dir: str | Path,
    *,
    prediction_filename: str | None = None,
    report_filename: str = "inference_contract.md",
) -> PredictionResult:
    directory = Path(output_dir)
    source_stem = Path(result.csv_path).stem if result.csv_path else "array"
    json_path = directory / (prediction_filename or f"{source_stem}_prediction.json")
    report_path = directory / report_filename
    exported = result.with_export_paths(
        prediction_json_path=str(json_path),
        report_path=str(report_path),
    )
    write_prediction_json(exported, json_path)
    write_inference_contract(exported, session, report_path)
    return exported


__all__ = [
    "export_prediction_result",
    "write_inference_contract",
    "write_prediction_json",
]
