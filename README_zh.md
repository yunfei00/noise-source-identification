# Noise Source Identification 中文说明

本项目用于基于 CSV 时域信号识别噪声源。当前真实数据工作流只使用两个目录：

- `data/single`：单源真实数据。
- `data/real_dataset`：真实组合数据。

不要再使用旧的真实训练/测试目录。

## 数据目录

```text
data/
  single/
    source_1/
      600.000MHz/
        000001.csv
    source_3/
    source_5/

  real_dataset/
    source_1_source_3_mix/
      ratio_1_1/
      ratio_1_2/
      ratio_1_4/
    source_1_source_5_mix/
    source_3_source_5_mix/
    source_1_source_3_source_5_mix/
      ratio_1_1_1/
      ratio_1_2_1/
      ratio_1_2_4/
      ratio_4_2_1/
```

`data/single` 的一级目录是类别名。`data/real_dataset` 的一级目录是组合标签名。更深层目录可以表示频率、比例、批次或工况，CSV 会递归读取。

配比目录同时兼容 `ratio_*` 和现有数据中的 `radio_*` 写法；统计报告会统一输出为 `ratio_*`。

## 标签规则

默认类别顺序：

```text
["source_1", "source_3", "source_5"]
```

标签映射：

```text
data/single/source_1/**/*.csv -> [1,0,0]
data/single/source_3/**/*.csv -> [0,1,0]
data/single/source_5/**/*.csv -> [0,0,1]
data/real_dataset/source_1_source_3_mix/**/*.csv -> [1,1,0]
data/real_dataset/source_1_source_5_mix/**/*.csv -> [1,0,1]
data/real_dataset/source_3_source_5_mix/**/*.csv -> [0,1,1]
data/real_dataset/source_1_source_3_source_5_mix/**/*.csv -> [1,1,1]
```

## 当前推荐配置

当前默认只使用真实 CSV，不使用合成数据，也不启用 quota-balanced split。

```yaml
training_data:
  mode: real_only

preprocessing:
  signal_normalization: none

balanced_train:
  enabled: false

real_data:
  split_file: outputs/reports/real_dataset_split.csv

stft:
  magnitude_scale: absolute

early_stopping:
  monitor: exact_match
```

关键点：

- CSV 第一列是时间，第二列是频谱仪采集的负 dB 轨迹，不是线性时域波形。
- `stft.input_representation: db_trace`：先减去每条轨迹的中位 dB 基线，再对波动量做 STFT。
- 模型输入为四通道：波动绝对谱、波动相对谱、中位 dB 电平、dB 波动强度。
- 长度不足时用该轨迹的中位 dB 补齐，避免补 `0 dB` 造成伪跳变。
- `best.pt` 按验证集 `exact_match` 保存。

## 推荐执行流程

新数据采集完成后，先运行一次只读体检：

```bash
python -m src.audit_real_data \
  --class-names source_1 source_3 source_4 \
  --output outputs/reports/source_134_data_audit.json
```

所有数量、七类/比例分布、dB 范围、长度分布、异常文件和推荐预处理参数都会写入这一个 JSON 文件。终端同时打印其中的 `copy_paste_summary`，可以直接把这段结果发回来。脚本不会修改原始 CSV。

如果已经通过空载采集确定“无信号”的 dB 上限，还可以增加例如 `--no-signal-threshold-db -90`；没有可靠空载阈值时不要填写，避免把正常的弱信号误判为坏数据。

检查路径：

```bash
python -m src.check_paths --config configs/train.yaml
```

构建真实数据索引：

```bash
python -m src.build_real_index \
  --single-dir data/single \
  --combo-dir data/real_dataset \
  --output outputs/reports/real_dataset_index.csv
```

划分训练、验证、测试集：

```bash
python -m src.split_real_dataset \
  --index outputs/reports/real_dataset_index.csv \
  --output outputs/reports/real_dataset_split.csv \
  --train-ratio 0.7 \
  --val-ratio 0.15 \
  --test-ratio 0.15 \
  --seed 42
```

分析标签分布：

```bash
python -m src.analyze_label_distribution \
  --split outputs/reports/real_dataset_split.csv \
  --output outputs/reports/label_distribution.json
```

训练：

```bash
python -m src.train --config configs/train.yaml
```

当前 dB 轨迹特征把模型输入从两通道改为四通道，必须从头训练。不要添加 `--init-model` 加载旧检查点；当前配置也关闭了此前已证明会欠拟合的频谱增强。

评估结构化组合模型：

```bash
python -m src.evaluate \
  --model outputs/checkpoints/best.pt \
  --real-split test
```

当前推荐模型直接从七种有效噪声源组合中选择一种，因此不再搜索三个独立阈值。`src.search_thresholds` 仅保留给旧的 `multilabel` 检查点使用。

对单个 CSV 严格推理，并同时生成结构化 JSON 与模型推理契约报告：

```bash
python scripts/predict_single_csv.py \
  --csv path/to/instrument_export.csv \
  --checkpoint outputs/checkpoints/best.pt \
  --device auto \
  --report-dir outputs/inference_contract
```

单文件入口要求 CSV 中存在独占一行的 `DATA` 标志，且其后第二列全部为有限数值。模型结构、标签顺序、预处理配置和默认阈值优先从 checkpoint 恢复；只有纯 `state_dict` 没有内嵌训练契约时才必须传 `--config`。`--threshold` 可以是一个统一阈值，也可以是按标签顺序排列的逗号分隔阈值。输出目录包含 `<样本名>_prediction.json` 和 `inference_contract.md`。

未来 GUI 可以直接调用同一个 Python 接口，无需启动外部进程：

```python
from src.inference import predict_single_csv

result = predict_single_csv(
    csv_path="path/to/instrument_export.csv",
    checkpoint_path="outputs/checkpoints/best.pt",
)
```

对一个没有真实标签、目录层级任意的文件夹递归推理并统计分布：

```bash
python -m src.infer_metadata_folder \
  --model outputs/checkpoints/best.pt \
  --input-dir data/metadata \
  --output outputs/reports/metadata_inference.csv \
  --confidence-threshold 0.6
```

命令递归处理所有 CSV。逐文件预测写入 `metadata_inference.csv`，组合分布、各信号源出现率、平均概率、置信度、第二候选、低置信度文件和失败文件写入 `metadata_inference.summary.json`。类别名称直接读取模型检查点，因此同时适用于 `source_1/source_3/source_4` 等后续模型。该命令不依赖文件夹名称提供真实标签；需要计算真实准确率时仍使用 `src.infer_folder` 或 `src.evaluate`。

物理频谱模板与神经网络融合评估：

```bash
python -m src.template_ensemble \
  --model outputs/checkpoints/best.pt \
  --output outputs/reports/template_ensemble_report.json
```

该命令从训练集单源数据建立稳健功率谱模板，执行非负模板系数分解，校准七种组合，并在验证集搜索 CNN/模板融合权重。无需重新训练 CNN，报告会分别给出神经网络、NNLS 模板和融合模型的测试指标。

## CSV 批量转图片查看数据

把输入目录下所有 CSV 时域数据递归转换成 PNG，输出目录结构和输入目录结构一致，只把 `.csv` 改成 `.png`。

```bash
python -m src.csv_to_images \
  --input data/single \
  --output outputs/csv_images \
  --max-files 20
```

脚本会自动查找 `DATA` 行，只读取其后的两列数值：第一列作为时间横轴，第二列作为幅值纵轴。支持逗号、空格和 tab 分隔；单个 CSV 失败时只打印 warning，不中断整体转换。

## 如何确认训练数据

当前训练文件来自：

```text
outputs/reports/real_dataset_split.csv 中 split == train 的所有行
```

打印实际训练文件和标签组合数量：

```bash
python - <<'PY'
import csv
from collections import Counter

split_file = "outputs/reports/real_dataset_split.csv"
counts = Counter()
with open(split_file, newline="", encoding="utf-8") as handle:
    for row in csv.DictReader(handle):
        if row.get("split") == "train":
            counts[row["label"]] += 1
            print(row["file"])

print("train label counts:")
for label, count in sorted(counts.items()):
    print(label, count)
PY
```

## source5 过预测诊断

如果评估显示 `source_5` 经常被误报，重点查看：

- `source5_false_positive_rate`
- `source5_over_prediction_rate`
- `double_to_triple_rate`
- `combo_confusion`
- `[1,1,0] -> [1,1,1]` 的比例

诊断特征能量：

```bash
python -m src.feature_statistics \
  --split outputs/reports/real_dataset_split.csv \
  --split-name train \
  --output outputs/reports/feature_statistics.csv \
  --max-samples-per-group 500
```

## 可选 balanced split

当前默认不推荐 quota-balanced split。只有复现实验或明确需要配额筛选时，才启用 `balanced_train.enabled: true` 并生成：

```bash
python -m src.create_balanced_split \
  --input outputs/reports/real_dataset_split.csv \
  --output outputs/reports/real_dataset_split_balanced.csv \
  --quota 100=2100,010=2100,001=2100,110=5000,101=3500,011=3500,111=3000 \
  --seed 42
```

## 输出文件

- `outputs/reports/real_dataset_index.csv`：真实数据索引。
- `outputs/reports/real_dataset_summary.json`：真实数据基础统计。
- `outputs/reports/real_dataset_split.csv`：训练、验证、测试划分。
- `outputs/reports/label_distribution.json`：标签、source、group、ratio 分布诊断。
- `outputs/reports/eval_report.json`：评估报告。
- `outputs/reports/error_analysis.csv`：逐样本错误分析。
- `outputs/reports/combo_confusion.csv`：组合混淆矩阵。
- `outputs/reports/threshold_search.csv`：阈值搜索结果。
- `outputs/reports/best_thresholds.json`：最佳 per-class 阈值。
- `outputs/reports/feature_statistics.csv`：特征统计明细。
- `outputs/reports/feature_statistics_summary.json`：特征统计聚合摘要。
- `outputs/csv_images`：CSV 时域数据批量转图片输出目录。
