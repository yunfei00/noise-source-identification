# Noise Source Identification

本项目用于基于 CSV 信号识别噪声源，当前真实数据工作流已经统一为两个目录：

- `data/single`：单源真实数据。
- `data/real_dataset`：真实组合数据。

不再使用旧目录；请不要创建或配置旧的真实训练 / 测试目录。

## 真实数据目录标准

```text
data/
  single/
    source_1/
      600.000MHz/
        000001.csv
    source_3/
      600.000MHz/
        000001.csv
    source_5/
      600.000MHz/
        000001.csv

  real_dataset/
    source_1_source_3_mix/
      ratio_1_1/
        600.000MHz/
          000001.csv
      ratio_1_2/
        600.000MHz/
          000001.csv
      ratio_1_4/
        600.000MHz/
          000001.csv
    source_1_source_5_mix/
      ratio_1_1/
      ratio_1_2/
      ratio_1_4/
    source_3_source_5_mix/
      ratio_1_1/
      ratio_1_2/
      ratio_1_4/
    source_1_source_3_source_5_mix/
      ratio_1_1_1/
        600.000MHz/
          000001.csv
      ratio_1_2_1/
        600.000MHz/
          000001.csv
      ratio_1_2_4/
        600.000MHz/
          000001.csv
      ratio_4_2_1/
        600.000MHz/
          000001.csv
```

规则：

1. `data/single` 的下一级目录是类别名。
2. `data/real_dataset` 的下一级目录是组合标签名。
3. 下层目录可以是频率、比例、批次、工况等任意条件目录。
4. `ratio_1_1`、`ratio_1_2_4`、`600.000MHz` 等下层目录都只是采集工况，不是类别。
5. 所有 CSV 都会递归读取。

## 标签规则

`class_names` 只来自 `data/single` 的下一级目录，并按名称排序。例如：

```text
["source_1", "source_3", "source_5"]
```

单源标签：

```text
data/single/source_1/**/*.csv -> [1,0,0]
data/single/source_3/**/*.csv -> [0,1,0]
data/single/source_5/**/*.csv -> [0,0,1]
```

组合标签：

```text
data/real_dataset/source_1_source_3_mix/**/*.csv -> [1,1,0]
data/real_dataset/source_1_source_5_mix/**/*.csv -> [1,0,1]
data/real_dataset/source_3_source_5_mix/**/*.csv -> [0,1,1]
data/real_dataset/source_1_source_3_source_5_mix/**/*.csv -> [1,1,1]
```

其中 `source_1_source_3_source_5_mix` 是三源真实组合目录，标签固定为 `[1,1,1]`；其下的 `ratio_1_1_1`、`ratio_1_2_1`、`ratio_1_2_4`、`ratio_4_2_1` 仅表示三源强弱比例工况，不会新增类别维度。

## 推荐完整流程

先检查路径：

```bash
python -m src.check_paths --config configs/train.yaml
```

新增或替换真实数据后，需要重新构建真实数据索引：

```bash
python -m src.build_real_index \
  --single-dir data/single \
  --combo-dir data/real_dataset \
  --output outputs/reports/real_dataset_index.csv
```

然后重新划分训练、验证、测试集：

```bash
python -m src.split_real_dataset \
  --index outputs/reports/real_dataset_index.csv \
  --output outputs/reports/real_dataset_split.csv
```

训练：

```bash
python -m src.train --config configs/train.yaml
```

评估测试划分：

```bash
python -m src.evaluate \
  --model outputs/checkpoints/best.pt \
  --real-split test
```

对文件夹递归推理：

```bash
python -m src.infer_folder \
  --model outputs/checkpoints/best.pt \
  --input-dir data/real_dataset \
  --output outputs/reports/infer_folder_report.csv \
  --threshold 0.5 \
  --unknown-threshold 0.35
```

## 配置

`configs/train.yaml` 默认使用真实数据：

```yaml
training_data:
  mode: real_only

real_data:
  single_dir: data/single
  combo_dir: data/real_dataset
  index_file: outputs/reports/real_dataset_index.csv
  split_file: outputs/reports/real_dataset_split.csv

real_split:
  enabled: true
  train_ratio: 0.7
  val_ratio: 0.15
  test_ratio: 0.15
  seed: 42
  split_by_group: true
```

在 `real_only` 模式下：

- 不使用 `data/mixed`。
- 不使用 `data.num_samples`。
- 样本数量由 `data/single` 和 `data/real_dataset` 中实际存在的 CSV 决定。
- 如果 `real_dataset_split.csv` 为空，训练会给出清晰错误。

## 输出文件

- `outputs/reports/real_dataset_index.csv`：真实数据索引，包含 `file`、`source_root`、`group`、`condition_path`、`label` 以及每个 source 列。
- `outputs/reports/real_dataset_summary.json`：真实数据统计摘要，包含类别、样本数、group 统计、label 统计、ratio 统计和无效 group。
- `outputs/reports/real_dataset_split.csv`：真实数据划分文件，在索引基础上新增 `split` 字段。
- `outputs/reports/eval_report.json`：评估报告，包含每个 source 的 precision / recall / f1、每个 group / label / ratio 的准确率。
- `outputs/reports/infer_folder_report.csv`：文件夹递归推理报告。
