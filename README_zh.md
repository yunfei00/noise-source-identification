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

- `preprocessing.signal_normalization: none`：保留原始时域幅值，不做单样本 z-score。
- `stft.magnitude_scale: absolute`：使用 STFT 线性绝对幅值，不使用 dB，也不使用 `log1p` 对数压缩。
- `best.pt` 按验证集 `exact_match` 保存。

## 推荐执行流程

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

统一阈值评估：

```bash
python -m src.evaluate \
  --model outputs/checkpoints/best.pt \
  --real-split test \
  --threshold 0.5
```

per-class 阈值搜索：

```bash
python -m src.search_thresholds \
  --model outputs/checkpoints/best.pt \
  --real-split test \
  --metric exact_match \
  --start 0.3 \
  --end 0.95 \
  --step 0.05 \
  --min-source5-threshold 0.6 \
  --output outputs/reports/threshold_search.csv
```

使用最佳阈值重新评估：

```bash
python -m src.evaluate \
  --model outputs/checkpoints/best.pt \
  --real-split test \
  --thresholds-json outputs/reports/best_thresholds.json
```

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
