# Noise Source Identification 中文说明

本项目用于基于 CSV 信号识别噪声源。当前真实数据工作流统一使用两个目录：

- `data/single`：单源真实数据。
- `data/real_dataset`：真实组合数据。

请不要再创建或配置旧的真实训练/测试目录。

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
      ratio_1_4/
    source_1_source_5_mix/
    source_3_source_5_mix/
    source_1_source_3_source_5_mix/
      ratio_1_1_1/
      ratio_1_2_1/
      ratio_1_2_4/
      ratio_4_2_1/
```

`data/single` 的下一级目录是类别名。`data/real_dataset` 的下一级目录是组合标签名。更深层目录可以是频率、比例、批次或工况，所有 CSV 会递归读取。

## 标签规则

`class_names` 默认来自 `data/single` 的下一级目录并按名称排序，例如：

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

## 模型偏置问题说明

多标签模型如果长期倾向预测某个 source 存在，就会出现类别预测偏置。当前重点观察到的问题是 `source_5` 过度预测：很多真实不包含 `source_5` 的样本也被预测成包含 `source_5`。

这类问题通常不能只看整体 F1，需要同时看：

- 每个 source 的 precision、recall、false positive rate、false negative rate。
- 每个组合标签的 exact match accuracy。
- 组合混淆矩阵 `combo_confusion`。
- 双源被预测成三源的比例。

## source_5 过预测问题

`source_5 recall` 很高只说明真实包含 `source_5` 的样本大多被找到了，不代表模型没有误报。如果模型几乎总是把 `source_5` 置为 1，recall 会很高，但 precision 和 false positive rate 会变差。

新增评估指标：

```text
source5_over_prediction_rate
```

它表示：真实 `source_5 = 0` 的样本中，被预测为 `source_5 = 1` 的比例。这个值越高，说明 source_5 误报越严重。

## 为什么 recall 高不代表模型好

recall 只关心真实为 1 的样本有没有被预测出来。对于 `source_5`：

```text
recall = TP / (TP + FN)
```

如果模型把大量样本都预测成包含 `source_5`，FN 会很少，recall 可能接近 1。但这会制造大量 FP，也就是把不含 `source_5` 的样本误判为含 `source_5`。

因此需要同时看：

```text
precision = TP / (TP + FP)
false_positive_rate = FP / (FP + TN)
```

## false positive 和 false negative 区别

对某个 source 来说：

- false positive：真实不存在，但模型预测存在。比如真实 `[1,1,0]` 被预测成 `[1,1,1]`，就是 `source_5` false positive。
- false negative：真实存在，但模型预测不存在。比如真实 `[1,1,0]` 被预测成 `[0,1,0]`，就是 `source_1` false negative。

当前 source_5 的核心问题是 false positive 偏高。

## WeightedRandomSampler 的作用

训练配置支持：

```yaml
sampler:
  enabled: true
  strategy: label_combo
```

`label_combo` 会按标签组合均衡采样，让 `[1,0,0]`、`[0,1,0]`、`[0,0,1]`、`[1,1,0]`、`[1,0,1]`、`[0,1,1]`、`[1,1,1]` 在训练中被看到的频率更接近。样本少的组合权重大，样本多的组合权重小。

`source_balance` 会按每个 source 的正负样本稀缺程度计算样本权重。

`none` 表示普通 shuffle。

训练启动时会打印每种 label combo 的原始数量和采样权重。

## asymmetric_bce 的作用

默认损失函数改为：

```yaml
loss:
  type: asymmetric_bce
  use_pos_weight: false
  fp_penalty: 2.0
  fn_penalty: 1.0
  class_fp_penalty:
    source_1: 1.5
    source_3: 1.5
    source_5: 3.0
```

`asymmetric_bce` 会先计算 `BCEWithLogitsLoss(reduction="none")`，再按真实标签加权：

- `target == 0` 的位置乘以 `fp_penalty`。
- `target == 1` 的位置乘以 `fn_penalty`。
- 如果设置了 `class_fp_penalty`，对应类别的 `target == 0` 损失会再乘以该类别的惩罚。

这里默认不使用 `pos_weight`，因为当前问题是过预测 1。盲目增加正样本权重可能进一步鼓励模型预测 source 存在。

## 为什么 source_5 需要更高 fp_penalty

如果 `source_5` 经常被误报，就需要让模型在 `source_5` 的负样本上犯错时付出更高代价：

```yaml
class_fp_penalty:
  source_5: 3.0
```

这会重点惩罚真实不含 `source_5` 但预测含 `source_5` 的情况，帮助降低 `source5_over_prediction_rate`。

## 为什么 EarlyStopping 改为 exact_match

当前任务最终关心的是组合是否完全识别正确，而不是某个 source 单独是否命中。因此 early stopping 默认监控：

```yaml
early_stopping:
  enabled: true
  monitor: exact_match
  mode: max
  patience: 15
  min_delta: 0.001
```

`best.pt` 会按验证集 `exact_match` 保存。训练历史 `training_history.csv` 会记录：

```text
epoch, train_loss, val_loss, micro_f1, macro_f1, exact_match,
double_to_triple_rate, source5_over_prediction_rate, lr
```

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
  --output outputs/reports/real_dataset_split.csv
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

## 如何执行阈值搜索

对 `source_1`、`source_3`、`source_5` 分别搜索独立阈值：

```bash
python -m src.search_thresholds \
  --model outputs/checkpoints/best.pt \
  --real-split test \
  --metric exact_match \
  --start 0.3 \
  --end 0.9 \
  --step 0.05 \
  --output outputs/reports/threshold_search.csv
```

输出：

- `outputs/reports/threshold_search.csv`
- `outputs/reports/best_thresholds.json`

使用最佳阈值重新评估：

```bash
python -m src.evaluate \
  --model outputs/checkpoints/best.pt \
  --real-split test \
  --thresholds source_1=0.55,source_3=0.45,source_5=0.75
```

也可以从 JSON 读取：

```bash
python -m src.evaluate \
  --model outputs/checkpoints/best.pt \
  --real-split test \
  --thresholds-json outputs/reports/best_thresholds.json
```

## 如何解读关键输出

`source5_over_prediction_rate`：

真实不含 `source_5` 的样本中，被预测成含 `source_5` 的比例。越高说明 source_5 误报越严重。

`double_source_predict_as_triple_rate`：

真实标签 source 数量为 2，但预测 source 数量为 3 的比例。这个值高说明双源经常被扩张成三源。

`combo_confusion`：

组合级混淆矩阵。例如真实 `[1,1,0]` 被预测成 `[1,1,1]` 很多，说明 source_1 + source_3 组合里 source_5 被频繁误加。对应 CSV 文件是：

```text
outputs/reports/combo_confusion.csv
```

## 输出文件

- `outputs/reports/real_dataset_index.csv`：真实数据索引。
- `outputs/reports/real_dataset_summary.json`：真实数据基础统计。
- `outputs/reports/real_dataset_split.csv`：训练、验证、测试划分。
- `outputs/reports/label_distribution.json`：标签、source、group、ratio 分布诊断。
- `outputs/reports/label_distribution.csv`：分布诊断表格版。
- `outputs/reports/eval_report.json`：评估报告。
- `outputs/reports/error_analysis.csv`：逐样本错误分析。
- `outputs/reports/combo_confusion.csv`：组合混淆矩阵。
- `outputs/reports/threshold_search.csv`：阈值搜索结果。
- `outputs/reports/best_thresholds.json`：最佳 per-class 阈值。

## 推荐训练配比

当前推荐训练配额：

- `100`：2100
- `010`：2100
- `001`：2100
- `110`：5000
- `101`：3500
- `011`：3500
- `111`：3000

说明：

- 训练集可以不使用全部数据。
- 验证集和测试集保持真实分布。
- 当某个 source 过预测严重时，应降低包含该 source 的组合占比，提高不包含该 source 的组合权重。

### 如何确保训练数据就是指定配额

训练数据不是靠 sampler 随机“近似”成上述比例，而是在训练前先生成一个带标记的 balanced split 文件：

```bash
python -m src.create_balanced_split \
  --input outputs/reports/real_dataset_split.csv \
  --output outputs/reports/real_dataset_split_balanced.csv \
  --quota 100=2100,010=2100,001=2100,110=5000,101=3500,011=3500,111=3000 \
  --seed 42
```

实现方式：

1. `outputs/reports/real_dataset_split.csv` 仍然保存原始真实 train/val/test 划分。
2. `create_balanced_split` 只在 `split=train` 的样本里按 quota 抽样。
3. 输出文件 `outputs/reports/real_dataset_split_balanced.csv` 会新增 `selected_for_train` 字段。
4. 训练时 `configs/train.yaml` 中的 `real_data.split_file` 指向 `outputs/reports/real_dataset_split_balanced.csv`。
5. `RealCsvDataset` 读取 `split=train` 时只加载 `selected_for_train=true` 的样本。
6. `split=val` 和 `split=test` 不检查 `selected_for_train`，因此验证集和测试集仍保持真实分布。

如果使用 `python -m src.train --config configs/train.yaml`，且 `balanced_train.enabled: true`，训练脚本会根据 `balanced_train.input_split_file`、`balanced_train.output_split_file` 和 `balanced_train.quota` 自动重新生成 balanced split，再用该文件训练。

可以通过以下文件确认实际训练配额：

- `outputs/reports/real_dataset_split_balanced.csv`：检查 `split=train` 且 `selected_for_train=true` 的行数。
- `outputs/reports/balanced_train_summary.json`：检查 `after_train_combo_counts` 是否等于或接近推荐配额，并检查 `source5_positive_ratio_after` 是否下降。
