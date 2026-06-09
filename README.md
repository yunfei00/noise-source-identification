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

## Real Data Domain-Gap Workflow

### Recursive real-test directory support

`src.infer_folder` treats each first-level directory under `data/real_test` as the true group label and now scans CSV files recursively inside that group with `group_dir.rglob("*.csv")`. Nested directories such as `600.000MHz`, `700.000MHz`, or `group_001` are only operating-condition/frequency folders; they are not used as labels.

Example supported layout:

```text
data/real_test/
  source_1_only/
    600.000MHz/000001.csv
    700.000MHz/000001.csv
  source_3_only/
    600.000MHz/000001.csv
  source_5_only/
    600.000MHz/000001.csv
  source_1_source_3_mix/
    group_001/000001.csv
```

The report stores `file` paths relative to `--input-dir`, for example `source_1_only/600.000MHz/000001.csv`, while `group` remains `source_1_only`. Known labels are parsed only from that first-level group name:

- `source_1_only` -> `[1,0,0]`
- `source_3_only` -> `[0,1,0]`
- `source_5_only` -> `[0,0,1]`
- `source_1_source_3_mix` -> `[1,1,0]`
- `source_1_source_5_mix` -> `[1,0,1]`
- `source_3_source_5_mix` -> `[0,1,1]`
- `source_1_source_3_source_5_mix` -> `[1,1,1]`
- `unknown_xxx` and `background_xxx` -> all-zero unknown labels

If a first-level group directory contains no recursive CSV files, the script prints a warning and skips that group.

### Real-test error analysis outputs

Running real-data inference writes:

- `outputs/reports/real_test_report.csv`
- `outputs/reports/real_test_summary.json`

The CSV report includes `file`, `group`, `true_label`, `pred_label`, per-source probabilities, `max_prob`, `result_type`, and `correct`. `result_type` is:

- `known`: at least one source probability is above `--threshold`
- `unknown`: all probabilities are below `--unknown-threshold`
- `uncertain`: the sample is not confidently unknown, but no source reached `--threshold`

The JSON summary includes overall exact-match accuracy, per-group accuracy, per-source precision/recall/F1, per-group mean probabilities, and misclassification counts for `source_1_only`, `source_3_only`, `source_5_only`, and mix groups.

### Why synthetic and real data can differ

A model trained only on synthesized mixtures can perform well on synthetic test data but poorly on real captures when the synthetic pipeline is too idealized. Common causes include:

- real sensors have different gain, amplitude scale, clipping, and baseline/DC offset;
- real captures include background emissions and acquisition noise that are absent from synthetic data;
- real source timing, phase/polarity, and frequency conditions can differ from the synthetic mixing assumptions;
- normalizing every synthetic mix to the same distribution can erase amplitude and energy differences that still exist in real data.

### Reducing domain gap with augmentation

`configs/train.yaml` contains an `augmentation` section used by `src.synthesize_mixed_dataset`:

```yaml
augmentation:
  enabled: true
  gain_range: [0.1, 2.0]
  noise_snr_db_range: [10, 40]
  random_dc_offset: true
  dc_offset_range: [-0.05, 0.05]
  random_polarity_flip: true
  random_time_shift: true
  random_crop: true
  background_noise_enabled: true
  background_noise_scale_range: [0.01, 0.2]
  mix_normalization: rms
  source_dropout_prob: 0.0
```

This makes synthesized training examples less ideal by varying source gains, adding random SNR noise, adding DC offsets, randomly flipping polarity, randomly shifting/cropping signals, and optionally mixing in recursive background CSV files from `data/unknown/background_noise` without changing the source label. `mix_normalization: rms` avoids forcing every mixture into exactly the same z-score distribution; supported values are `none`, `zscore`, and `rms`.

### Mixing a small real training set

If real performance is still poor, collect a small labeled real training set under `data/real_train` using the same first-level group naming rules as `data/real_test`. Multi-level subdirectories are supported recursively:

```text
data/real_train/
  source_1_only/condition_a/*.csv
  source_3_only/condition_a/*.csv
  source_5_only/condition_a/*.csv
  source_1_source_3_mix/condition_a/*.csv
```

Enable hybrid real-data mixing in `configs/train.yaml`:

```yaml
training_data:
  mode: hybrid
  synthetic_ratio: 0.8
  real_ratio: 0.2
real_train:
  dir: data/real_train
```

The training loader uses `synthetic_ratio` and `real_ratio` as weighted sampling targets. If `real_split.enabled: true`, real `split=train` samples are mixed into training and real `split=val` samples can participate in validation.

### Recommended workflow

1. Run recursive real-test inference:

```bash
python -m src.infer_folder --model outputs/checkpoints/best.pt --input-dir data/real_test --threshold 0.5 --unknown-threshold 0.35
```

2. Diagnose the synthetic/real feature gap:

```bash
python -m src.analyze_real_vs_synthetic --model outputs/checkpoints/best.pt --real-dir data/real_test --synthetic-split test
```

3. Enable augmentation, regenerate synthetic data, train, and evaluate:

```bash
rm -rf data/mixed
python -m src.synthesize_mixed_dataset --config configs/train.yaml --mode balanced_multilabel
python -m src.train --config configs/train.yaml
python -m src.evaluate --model outputs/checkpoints/best.pt --split test
python -m src.infer_folder --model outputs/checkpoints/best.pt --input-dir data/real_test --threshold 0.5 --unknown-threshold 0.35
```

4. If real performance is still poor, collect a small `data/real_train` set and set:

```yaml
training_data:
  mode: hybrid
  synthetic_ratio: 0.8
  real_ratio: 0.2
real_train:
  dir: data/real_train
```

## Large-Scale Real Training Data

The training pipeline supports three data modes through `configs/train.yaml`:

```yaml
training_data:
  mode: hybrid
  synthetic_ratio: 0.3
  real_ratio: 0.7
```

- `synthetic_only`: train and validate only with `data/mixed/train` and `data/mixed/val`.
- `hybrid`: keep synthetic `data/mixed/train` / `data/mixed/val` and mix in recursive samples from `data/real_train`. The training loader uses `synthetic_ratio` / `real_ratio` as sampling weights. If `data/real_train` is empty, training prints a warning and falls back to synthetic-only data.
- `real_only`: train, validate, and test only with the real split built from recursive CSV scans of `data/single` and `data/real_train`. This mode does not depend on `data/mixed` and does not use `data.num_samples`; every real CSV discovered by the index is eligible for the train/validation/test split. If no real samples are found it fails with `real_only mode requires non-empty real dataset`.

`data.num_samples` is only used to decide how many synthetic examples are generated for `synthetic_only` and for the synthetic portion of `hybrid`; it does not cap, truncate, or otherwise affect `real_only` training.

When real data is large enough, switch to:

```yaml
training_data:
  mode: real_only
  synthetic_ratio: 0.0
  real_ratio: 1.0
```

### Real data directory structure

`data/real_train` is organized by first-level label groups. All directories under a group are operating-condition metadata, such as frequency, source strength, and collection batch. They are scanned recursively but are not used for label parsing.

```text
data/real_train/
  source_1_only/
    600.000MHz/
      strong/
        000001.csv
      weak/
        000002.csv

  source_1_source_3_mix/
    600.000MHz/
      source1_strong_source3_weak/
        000001.csv
      source1_weak_source3_strong/
        000002.csv
      same_level/
        000003.csv

  source_1_source_3_source_5_mix/
    700.000MHz/
      batch_001/
        000001.csv
```

For a file such as:

```text
data/real_train/source_1_source_3_mix/600.000MHz/source1_strong_source3_weak/000001.csv
```

- `group = source_1_source_3_mix`
- `label = [1,1,0]` when `class_names = [source_1, source_3, source_5]`
- `condition_path = 600.000MHz/source1_strong_source3_weak`
- `file = source_1_source_3_mix/600.000MHz/source1_strong_source3_weak/000001.csv`

Strong/weak relationships should be placed under the first-level group, for example:

```text
source_1_source_3_mix/600.000MHz/source1_strong_source3_weak/
source_1_source_3_mix/600.000MHz/source1_weak_source3_strong/
source_1_source_3_mix/600.000MHz/same_level/
```

### STFT cache for large CSV datasets

Real CSV samples are loaded lazily during training, so tens of thousands of files are not loaded into memory at once. To avoid recomputing STFT features every epoch, enable the optional cache:

```yaml
cache:
  enabled: true
  dir: data/cache_stft
  rebuild: false
```

With caching enabled, the first read of each CSV computes its STFT feature and stores a `.npy` cache file. Later reads reuse that cache. The cache key is based on the real file path plus the signal/STFT configuration hash.

## Unified Real Single + Combo Training

This project supports a real-data-only training flow that combines true single-source captures with true combined-source captures. In this mode, labels are parsed only from source directories, while intermediate operating-condition folders are preserved as metadata.

### Recommended real-data layout

Single-source captures stay under `data/single`. The first-level directory names define the model classes and are sorted to form `class_names`:

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

True combo captures go under `data/real_train`. The first-level directory decides the multi-label target. Ratio and frequency folders are operating conditions only and are not classes:

```text
data/real_train/
  source_1_source_3_mix/
    ratio_1_1/600.000MHz/000001.csv
    ratio_1_2/600.000MHz/000001.csv
    ratio_1_4/600.000MHz/000001.csv
  source_1_source_5_mix/
    ratio_1_1/600.000MHz/000001.csv
  source_3_source_5_mix/
    ratio_1_1/600.000MHz/000001.csv
```

Recommended ratio folder names are:

- `ratio_1_1`
- `ratio_1_2`
- `ratio_1_4`

All CSV files are discovered recursively, so new frequency folders or additional nested condition folders do not require code changes.

### Label rules

With `data/single` classes sorted as `["source_1", "source_3", "source_5"]`:

- `data/single/source_1/**/*.csv` -> `[1,0,0]`
- `data/single/source_3/**/*.csv` -> `[0,1,0]`
- `data/single/source_5/**/*.csv` -> `[0,0,1]`
- `data/real_train/source_1_source_3_mix/**/*.csv` -> `[1,1,0]`
- `data/real_train/source_1_source_5_mix/**/*.csv` -> `[1,0,1]`
- `data/real_train/source_3_source_5_mix/**/*.csv` -> `[0,1,1]`
- Future `source_1_source_3_source_5_mix` data is parsed as `[1,1,1]`.

Do not use condition folders as labels: `ratio_1_1`, `ratio_1_2`, `ratio_1_4`, and `600.000MHz` are recorded in `condition_path` only.

### Build the unified real-data index

```bash
python -m src.build_real_index \
  --single-dir data/single \
  --real-train-dir data/real_train \
  --output outputs/reports/real_dataset_index.csv
```

The command writes:

- `outputs/reports/real_dataset_index.csv`
- `outputs/reports/real_dataset_summary.json`

The index includes `file`, `source_root`, `group`, `condition_path`, `label`, and one column per source class. It also prints class names, single/combo sample totals, group counts, label-combination counts, ratio counts when present, and warnings for empty or unparseable directories.

### Split the unified real dataset

```bash
python -m src.split_real_dataset \
  --index outputs/reports/real_dataset_index.csv \
  --output outputs/reports/real_dataset_split.csv \
  --train-ratio 0.7 \
  --val-ratio 0.15 \
  --test-ratio 0.15 \
  --seed 42
```

Splitting is stratified by `group` and additionally by ratio condition when a `ratio_` directory exists, helping each split keep similar `ratio_1_1`, `ratio_1_2`, and `ratio_1_4` coverage.

### Train on real single + real combo data

`configs/train.yaml` defaults to:

```yaml
training_data:
  mode: real_only
real_data:
  enabled: true
  index_file: outputs/reports/real_dataset_index.csv
  split_file: outputs/reports/real_dataset_split.csv
  include_single: true
  include_real_train: true
cache:
  enabled: false
```

Run training with:

```bash
python -m src.train --config configs/train.yaml
```

When `training_data.mode: real_only`, training no longer depends on `data/mixed` and ignores `data.num_samples`; the real dataset size is fully determined by the recursive CSV scan of `data/single` plus `data/real_train`, so if the scan finds 25,200 CSV files, all 25,200 are split and used according to the configured real split ratios. In `real_only`, training rebuilds the real index and split from the current CSV files at startup, then reads samples lazily from CSV during training and transforms them with the existing CSV reader, fixed-length preprocessing, and STFT feature extraction; 30,000+ CSV files are not loaded into memory at once. STFT caching remains optional and is disabled by default.

`data.num_samples` applies only to synthetic data generation (`synthetic_only` or the synthetic portion of `hybrid`) and never caps real-only training.

### Evaluate a real split

```bash
python -m src.evaluate --model outputs/checkpoints/best.pt --real-split test
```

Real-split evaluation reads `outputs/reports/real_dataset_split.csv`, filters `split=test`, and reports:

- per-group exact-match accuracy
- per-label-combination exact-match accuracy
- per-source precision / recall / F1
- per-ratio accuracy when `condition_path` contains a `ratio_` directory

### Adding more real data later

When new real data is collected:

1. Put each CSV under the appropriate `data/single` or `data/real_train` directory.
2. Re-run `python -m src.build_real_index --single-dir data/single --real-train-dir data/real_train --output outputs/reports/real_dataset_index.csv`.
3. Re-run `python -m src.split_real_dataset --index outputs/reports/real_dataset_index.csv --output outputs/reports/real_dataset_split.csv`.
4. Re-run `python -m src.train --config configs/train.yaml`.
