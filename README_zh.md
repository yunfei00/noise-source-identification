# 噪声源 / 干扰源多标签识别系统

本项目用于在已知若干个单独干扰源的情况下，识别一个采集信号中包含哪些干扰源。它不是单纯的 demo，而是一套面向真实时域 CSV 数据的多标签识别 baseline：支持仪器 CSV 读取、真实数据递归索引、真实/合成/混合训练、测试集评估、外部文件夹批量推理，以及 unknown / uncertain 判断。

## 1. 项目目标

本项目用于已知多个单独干扰源的情况下，识别一个采集信号中包含哪些干扰源。

输入：

- 时域 CSV
- CSV 前面可能有仪器元信息
- 从 `DATA` 行之后才是真正数据
- `DATA` 下面直接是两列数值
- 第 1 列是时间
- 第 2 列是幅值

输出：

- 每个 source 的存在概率
- 多标签判断结果
- `unknown` / `uncertain` 判断

整体流程是：读取 CSV 幅值序列 -> 补齐或裁剪到固定长度 -> 计算 STFT 特征 -> CNN 输出每个 source 的概率 -> 根据阈值得到多标签结果和拒识/不确定判断。

## 2. 当前问题定义

当前任务是**多标签识别**，不是普通多分类。

普通多分类假设一个样本只能属于一个类别，例如“只能是 source_1 或 source_3 或 source_5”。但干扰源识别中，一个采集信号可能同时包含多个 source，因此需要为每个 source 独立判断是否存在。

例如：

```python
class_names = ["source_1", "source_3", "source_5"]
```

对应标签如下：

```text
source_1_only                  -> [1,0,0]
source_3_only                  -> [0,1,0]
source_5_only                  -> [0,0,1]
source_1_source_3_mix          -> [1,1,0]
source_1_source_5_mix          -> [1,0,1]
source_3_source_5_mix          -> [0,1,1]
source_1_source_3_source_5_mix -> [1,1,1]
unknown_source_x               -> [0,0,0]
```

含义：

- 向量长度等于 `class_names` 的长度。
- 每一位对应一个已知 source。
- `1` 表示该 source 存在。
- `0` 表示该 source 不存在。
- `[0,0,0]` 可用于 unknown / background / 未知干扰源拒识测试。

## 3. 数据目录结构

不要把 `data/`、`outputs/`、模型 checkpoint 或其他生成文件写进 git。仓库中只维护代码、配置和说明文档。

### 3.1 单源数据目录

```text
data/single/
    source_1/
        600.000MHz/
            000001.csv
    source_3/
        600.000MHz/
            000001.csv
    source_5/
        600.000MHz/
            000001.csv
```

说明：

- `data/single` 下一级目录就是类别名。
- 类别名会生成 `class_names`。
- 频率目录只是工况，不是类别。
- 所有 CSV 会递归读取。

例如 `data/single/source_1/600.000MHz/000001.csv` 表示：类别是 `source_1`，`600.000MHz` 只是采集频率或工况信息，不会成为模型类别。

### 3.2 真实组合训练数据目录

```text
data/real_train/
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
```

说明：

- `data/real_train` 下一级目录决定标签。
- `ratio_1_1`、`ratio_1_2`、`ratio_1_4` 表示强弱比例。
- `600.000MHz` 表示频率工况。
- 这些子目录都不会作为类别。
- 程序递归读取所有 CSV。

例如 `data/real_train/source_1_source_3_mix/ratio_1_2/600.000MHz/000001.csv` 的标签由第一层目录 `source_1_source_3_mix` 决定。如果 `class_names = ["source_1", "source_3", "source_5"]`，该文件标签为 `[1,1,0]`。

### 3.3 真实测试数据目录

```text
data/real_test/
    source_1_only/
    source_3_only/
    source_5_only/
    source_1_source_3_mix/
    source_1_source_5_mix/
    source_3_source_5_mix/
    unknown_source_x/
```

说明：

- `real_test` 也支持多级子目录。
- `infer_folder` 会递归读取所有 CSV。
- 第一层目录作为 group，用于解析真实标签。
- `unknown_source_x` 这类目录用于未知干扰源拒识测试，其真实标签为全 0。

## 4. CSV 格式说明

真实 CSV 可能是：

```text
仪器信息
采样参数
...
DATA
0.000000000,-0.0123
0.000000001,-0.0118
0.000000002,-0.0120
```

读取逻辑：

- 自动查找 `DATA` 行。
- `DATA` 之前全部忽略。
- `DATA` 之后如果有两列，取第二列作为幅值。
- `DATA` 之后如果只有一列，取该列作为幅值。
- 如果没有 `DATA`，则兼容普通单列 / 双列 CSV。
- 对于没有 `DATA` 的双列 CSV，会使用最后一个可解析的数值列作为幅值列，常见的 `time,value` 或无表头两列数据都可以读取。

因此，真实仪器导出的“元信息 + DATA + 两列数值”格式可以直接用于训练和推理，不需要手工删除 `DATA` 之前的内容。

## 5. 训练模式

当前支持三种训练模式，由 `configs/train.yaml` 中的 `training_data.mode` 控制。

### 5.1 synthetic_only

```yaml
training_data:
  mode: synthetic_only
```

只使用合成数据训练。合成数据通常来自 `data/single` 中的单源 CSV，通过线性叠加、随机增益、随机平移、噪声增强等方式生成混合样本。

适用场景：

- 真实组合数据还没有采集完成。
- 需要先验证端到端流程是否能跑通。
- 希望用单源数据快速构建 baseline。

### 5.2 hybrid

```yaml
training_data:
  mode: hybrid
```

同时使用合成数据和真实 CSV 数据训练。该模式用于缓解 synthetic 与 real 之间的 domain gap。

适用场景：

- 已经有一部分真实组合数据，但数量不多。
- 希望保留合成数据覆盖面，同时让模型看到真实采集分布。

### 5.3 real_only

```yaml
training_data:
  mode: real_only
```

只使用真实 CSV 数据训练、验证和测试。

重点说明：

- `real_only` 模式下不使用 `data.num_samples`。
- 真实 CSV 扫描到多少就用多少。
- `data.num_samples` 只用于 synthetic 数据生成。
- 如果真实数据索引为空，`real_only` 训练会失败，因为没有可训练样本。

## 6. 推荐真实数据训练流程

下面流程适用于以真实数据为主的训练，尤其是 `training_data.mode: real_only`。

### 第一步：检查路径

```bash
python -m src.check_paths --config configs/train.yaml
```

该命令用于检查配置中引用的数据目录、输出目录和关键路径是否符合预期。

### 第二步：建立真实数据索引

```bash
python -m src.build_real_index \
  --single-dir data/single \
  --real-train-dir data/real_train \
  --output outputs/reports/real_dataset_index.csv
```

该命令会递归扫描：

- `data/single` 中的单源 CSV；
- `data/real_train` 中的真实组合 CSV。

输出文件：

```text
outputs/reports/real_dataset_index.csv
```

### 第三步：划分 train / val / test

```bash
python -m src.split_real_dataset \
  --index outputs/reports/real_dataset_index.csv \
  --output outputs/reports/real_dataset_split.csv \
  --train-ratio 0.7 \
  --val-ratio 0.15 \
  --test-ratio 0.15 \
  --seed 42
```

该命令会把真实数据索引划分为训练集、验证集和测试集。

输出文件：

```text
outputs/reports/real_dataset_split.csv
```

### 第四步：训练

```bash
python -m src.train --config configs/train.yaml
```

训练会读取 `configs/train.yaml`，根据 `training_data.mode` 选择 synthetic、hybrid 或 real_only 数据源。模型 checkpoint 默认保存到：

```text
outputs/checkpoints/best.pt
outputs/checkpoints/last.pt
```

### 第五步：评估真实 test split

```bash
python -m src.evaluate \
  --model outputs/checkpoints/best.pt \
  --real-split test
```

该命令用于评估第三步划分出的真实 test split。

### 第六步：评估外部 real_test

```bash
python -m src.infer_folder \
  --model outputs/checkpoints/best.pt \
  --input-dir data/real_test \
  --output outputs/reports/real_test_report.csv \
  --threshold 0.5 \
  --unknown-threshold 0.35
```

该命令会递归读取 `data/real_test`，输出逐文件推理结果和汇总统计。

## 7. 配置文件说明

`configs/train.yaml` 中关键项说明如下。

### data.signal_length

固定输入信号长度。CSV 读取出的幅值序列会被补零或裁剪到该长度。

如果采集点数明显大于或小于该值，需要根据真实数据调整。

### stft.nperseg

STFT 每个窗口的长度。该值会影响频率分辨率和时间分辨率。

### stft.noverlap

STFT 相邻窗口的重叠长度。通常需要小于 `stft.nperseg`。

### training_data.mode

训练数据模式，可选：

- `synthetic_only`
- `hybrid`
- `real_only`

### real_data.index_file

真实数据索引文件路径，默认类似：

```text
outputs/reports/real_dataset_index.csv
```

### real_data.split_file

真实数据划分文件路径，默认类似：

```text
outputs/reports/real_dataset_split.csv
```

### train.batch_size

训练 batch size。显存不足或内存不足时可以减小。

### train.epochs

最大训练 epoch 数。实际训练可能会被 early stopping 提前停止。

### train.threshold

默认多标签判定阈值。例如 `0.5` 表示某个 source 概率大于等于 0.5 时判为存在。

### early_stopping.patience

早停耐心值。如果验证集指标连续若干个 epoch 没有改善，则提前停止训练。

### cache.enabled

是否启用特征缓存。启用后可以减少重复计算 STFT 的开销，但会占用额外磁盘空间，并需要注意配置改变后缓存是否需要重建。

## 8. 阈值说明

```text
threshold = 0.5
```

用于判断某个 source 是否存在。

例如某个样本输出：

```text
source_1_prob = 0.82
source_3_prob = 0.64
source_5_prob = 0.12
```

当 `threshold = 0.5` 时，预测结果为：

```text
[source_1, source_3]
```

对应多标签向量：

```text
[1,1,0]
```

```text
unknown_threshold = 0.35
```

如果所有 source 概率都低于 `unknown_threshold`，则判为 `unknown`。

如果最大概率在 `unknown_threshold` 和 `threshold` 之间，则判为 `uncertain`。

示例：

```text
[0.12, 0.20, 0.31] -> unknown
[0.12, 0.41, 0.22] -> uncertain
[0.12, 0.51, 0.22] -> known，source_3 存在
```

建议在真实验证集上调节 `threshold` 和 `unknown_threshold`，不要只依赖默认值。

## 9. 输出报告说明

### outputs/reports/real_dataset_index.csv

真实数据索引文件。记录递归扫描到的真实 CSV、来源目录、group、标签等信息。它是后续划分 train / val / test 的输入。

### outputs/reports/real_dataset_split.csv

真实数据划分文件。基于 `real_dataset_index.csv` 生成，记录每个样本属于 `train`、`val` 还是 `test`。

### outputs/reports/training_history.csv

训练历史文件。记录每个 epoch 的训练损失、验证损失、micro F1、macro F1 等训练过程指标，便于观察是否过拟合或欠拟合。

### outputs/reports/eval_report.json

评估报告。通常包含不同阈值下的 precision、recall、F1、support，以及 micro / macro / sample F1 等汇总指标。

### outputs/reports/real_test_report.csv

外部 `real_test` 文件夹的逐样本推理报告。通常包含：

- 文件路径
- group 名称
- 真实标签 `true_label`
- 预测标签 `pred_label`
- 每个 source 的概率
- 最大概率 `max_prob`
- 结果类型 `result_type`
- 是否完全匹配 `correct`

### outputs/reports/real_test_summary.json

外部 `real_test` 文件夹的汇总报告。通常包含：

- 总样本数
- 完全匹配准确率
- 每个 group 的准确率
- 每个 source 的 precision / recall / F1
- unknown 样本拒识统计
- 各 group 的平均概率
- 常见误判统计

## 10. 常见问题

### 1. 为什么 source_5 没参与训练时会全错？

因为模型没有 `source_5` 这个类别，只能把它作为 unknown 测试。

如果训练时的 `class_names` 只有：

```python
["source_1", "source_3"]
```

那么模型输出层只会有 `source_1` 和 `source_3` 两个概率，不可能正确输出 `source_5`。此时 `source_5_only` 应该被当作未知源，理想结果是所有已知 source 概率都较低，从而判为 `unknown`。

### 2. 为什么合成数据准确率高，真实数据差？

主要原因是 domain gap。合成数据通常过于理想，而真实环境中可能存在：

- 采集噪声
- 背景干扰
- 设备耦合
- 非线性叠加
- 幅度差异
- 传感器增益变化
- 相位、极性、触发位置变化
- 频率工况差异

因此，如果真实测试表现差，需要补充真实 `real_train` 数据，或使用更贴近真实采集的增强策略。

### 3. 是否必须知道混合信号中各 source 占比？

不需要。

多标签识别只需要知道哪些 source 开启，不要求知道每个 source 的贡献占比。目录名中的 `ratio_1_1`、`ratio_1_2`、`ratio_1_4` 可以作为工况信息保留，但不会作为类别标签。

### 4. ratio_1_1 / ratio_1_2 / ratio_1_4 是否是类别？

不是。

它们只是强弱比例或工况目录。真正的标签由 `data/real_train` 或 `data/real_test` 下的第一层目录决定，例如 `source_1_source_3_mix`。

### 5. 600.000MHz 是否是类别？

不是。

`600.000MHz` 只是频率工况目录，不会成为类别。类别来自 `data/single` 的第一层 source 目录，或真实组合数据中的第一层 group 目录解析结果。

### 6. real_only 模式为什么不看 num_samples？

因为真实数据训练扫描到多少用多少，`num_samples` 只用于合成数据。

在 `real_only` 模式下，样本数量由 `data/single` 和 `data/real_train` 中实际存在的 CSV 文件数量决定。`data.num_samples` 不会截断真实数据，也不会凭空生成真实数据。

## 11. 当前限制

- 当前主要做存在性识别，不估计贡献占比。
- unknown 拒识依赖负样本和阈值。
- 如果真实环境变化大，需要补充 `real_train`。
- STFT 参数需要根据采样点数和采样率调整。
- 模型是 baseline，后续可升级为更深 CNN 或时域 + 频域双分支模型。

## 12. 后续规划

- 增加更多真实组合。
- 增加三源真实组合。
- 增加 unknown 负样本。
- 增加贡献度估计。
- 增加可视化报告。
- 增加模型对比实验。
