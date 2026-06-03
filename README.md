# Noise Source Identification

This project is a first runnable pipeline for multi-label noise source identification:

single-source noise CSV -> synthesized mixed noise samples -> STFT features -> CNN multi-label training -> mixed-signal inference.

## Data Layout

Real single-source CSV files can use either of the following layouts under `data/single`.

### Format A: simple source-class directories

```text
data/single/
  motor/
    001.csv
  fan/
    001.csv
```

### Format B: source-class directories with frequency conditions

```text
data/single/
  source_1/
    600.000MHz/
      000001.csv
    700.000MHz/
      000001.csv
  source_2/
    600.000MHz/
      000001.csv
```

The two formats can coexist. In Format B, `source_1` and `source_2` are the classification labels, `600.000MHz` and `700.000MHz` are frequency-condition directory names, and `000001.csv` is one time-domain sample. Frequency conditions are not labels. Class order is stable and sorted by source-directory name. The synthesis step writes it to `data/mixed/class_names.json`.

For each synthesized source contribution, the synthesis script randomly selects a frequency condition and then randomly selects one CSV from that condition. CSV files from Format A are handled as a `default` condition. Generated mixed data is written to:

```text
data/mixed/train/x/*.npy
data/mixed/train/y/*.npy
data/mixed/train/metadata/*.json
data/mixed/val/x/*.npy
data/mixed/val/y/*.npy
data/mixed/val/metadata/*.json
data/mixed/test/x/*.npy
data/mixed/test/y/*.npy
data/mixed/test/metadata/*.json
```

Each `x` file is a fixed-length time-domain signal. Each `y` file contains the multi-label source-class vector. Each `metadata` JSON file records the selected source class, frequency condition, relative CSV path, random gain, random shift, and label vector. STFT features are computed dynamically in the PyTorch dataset.

## CSV Format

CSV input can be either:

```text
value
0.12
0.08
```

or:

```text
time,value
0.000000,0.12
0.000001,0.08
```

Headerless one-column and two-column CSV files are also supported.

## Install

Python 3.10 or newer is recommended.

```bash
pip install -e .
```

## Demo Data

When no real data is available, generate three simulated source classes:

```bash
python -m src.create_demo_single_data
```

This creates 10 CSV files each for `motor`, `fan`, and `switch_power`. To generate demo data with frequency-condition directories instead, run:

```bash
python -m src.create_demo_single_data --with-frequency-dirs
```

This creates `source_1`, `source_2`, and `source_3` classes with `600.000MHz` and `700.000MHz` conditions, including paths such as `data/single/source_1/600.000MHz/000001.csv`.

## Synthesize Mixed Dataset

```bash
python -m src.synthesize_mixed_dataset --config configs/train.yaml
```

The default config synthesizes `num_samples: 300` examples and allows up to `epochs: 200` training epochs. Training monitors validation `micro_f1`, reduces the learning rate when it plateaus, and stops early when it no longer improves.

When `balanced_generation: true` and exactly two source classes are detected, synthesis creates an approximately equal number of `[1, 0]`, `[0, 1]`, and `[1, 1]` labels (exactly one third each when `num_samples` is divisible by three). With any other number of classes, synthesis warns and falls back to random source selection.

## Train

```bash
python -m src.train --config configs/train.yaml
```

Checkpoints are saved to:

```text
outputs/checkpoints/best.pt
outputs/checkpoints/last.pt
```

Per-epoch training metrics are saved to:

```text
outputs/reports/training_history.csv
```

## Evaluate

Evaluate the test split at thresholds `0.3`, `0.4`, `0.5`, `0.6`, and `0.7`:

```bash
python -m src.evaluate --model outputs/checkpoints/best.pt --split test
```

The command prints per-class precision, recall, F1, and support along with overall micro, macro, and sample F1 scores. It saves the complete threshold report to `outputs/reports/eval_report.json`.

## Infer

```bash
python -m src.infer --model outputs/checkpoints/best.pt --input path/to/mixed.csv
```

Inference still accepts one CSV directly and does not need a frequency argument. The output lists each source class by probability:

```text
switch_power  0.923  存在
motor         0.811  存在
fan           0.231  不存在
```

Threshold rules:

- `>= 0.7`: 存在
- `0.4 ~ 0.7`: 疑似
- `< 0.4`: 不存在

## 真实复合验证

真实采集验证集按标签组合放在 `data/real_test` 的子目录中。脚本会递归读取所有 CSV，并从目录名解析真实标签：

```text
data/real_test/
  source_1_only/
    000001.csv      # true_label = [1,0]
  source_3_only/
    000001.csv      # true_label = [0,1]
  source_1_source_3_mix/
    000001.csv      # true_label = [1,1]
  unknown_source_5/
    000001.csv      # true_label = [0,0]
```

目录名只要以 `unknown` 开头（例如 `unknown_source_5`），就会被当作未知干扰源，真实标签强制设为全 0。以当前模型 `class_names = ["source_1", "source_3"]` 为例，`unknown_source_5` 的 `true_label = [0,0]`。评估时仍会输出每个已知 source 的概率，`pred_label` 由 `threshold=0.5`（或命令行指定阈值）决定；如果未知样本的 `pred_label` 也是 `[0,0]`，则 `correct=true`，如果被判成 `source_1` 或 `source_3`，则 `correct=false`。

如果 `source_5` 没有参与训练，它不能作为 `source_5` 分类测试，只能作为未知干扰源测试。理想结果是模型对所有已知 source 都输出低概率，从而完全拒识该未知干扰源。

批量推理命令：

```bash
python -m src.infer_folder \
  --model outputs/checkpoints/best.pt \
  --input-dir data/real_test \
  --output outputs/reports/real_test_report.csv \
  --threshold 0.5
```

脚本会自动读取 checkpoint 中的 `class_names`，复用训练配置中的信号长度、采样率和 STFT 参数，对每个 CSV 输出概率和阈值化后的多标签预测。逐样本报告保存为：

```text
outputs/reports/real_test_report.csv
```

字段包括 `file`、`group`、`true_label`、`pred_label`、`source_1_prob`、`source_3_prob`、`correct`。只有 `pred_label` 与 `true_label` 完全一致时，`correct` 才为 `true`。

汇总指标保存为：

```text
outputs/reports/real_test_summary.json
```

汇总内容包括总样本数、完全匹配准确率、每个 group 的准确率、每个 source 的 precision / recall / f1，以及 unknown 统计：unknown 样本数、unknown 完全拒识准确率、unknown 被误判为各 source 的次数、unknown 平均最大概率 `max_prob_mean`、unknown 最大概率 `max_prob_max`。

## Current Limitations

- The demo signals are synthetic and only prove the pipeline, not real-world accuracy.
- Mixed samples use linear summation, random gain, random roll shift, and small Gaussian noise.
- STFT shape is fixed by padding/cropping, so extreme signal lengths or sampling rates may need config tuning.
- The CNN is intentionally lightweight and should be treated as a baseline model.
