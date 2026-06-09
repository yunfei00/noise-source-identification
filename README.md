# Noise Source Identification

中文文档请见：[README_zh.md](README_zh.md)

Noise Source Identification is a baseline multi-label system for identifying which known interference/noise sources are present in a captured time-domain signal. It is intended for scenarios where separate examples of known sources are available, and a later capture may contain one source, multiple sources, or an unknown/unrecognized source.

The pipeline supports real instrument CSV files, recursive real-data indexing, synthetic mixture generation, hybrid training, real-only training, model evaluation, and folder-level inference reports.

## Project overview

Given a time-domain CSV capture, the system:

1. reads the signal amplitude from the CSV;
2. pads or crops the signal to a fixed length;
3. converts the signal to an STFT feature map;
4. predicts one probability for each known source;
5. thresholds the probabilities into a multi-label result;
6. optionally marks samples as `unknown` or `uncertain` based on confidence thresholds.

This is a **multi-label** task, not a standard single-label multi-class task. A sample can contain more than one source at the same time. For example, with:

```python
class_names = ["source_1", "source_3", "source_5"]
```

labels are interpreted as:

```text
source_1_only                  -> [1, 0, 0]
source_3_only                  -> [0, 1, 0]
source_5_only                  -> [0, 0, 1]
source_1_source_3_mix          -> [1, 1, 0]
source_1_source_5_mix          -> [1, 0, 1]
source_3_source_5_mix          -> [0, 1, 1]
source_1_source_3_source_5_mix -> [1, 1, 1]
unknown_source_x               -> [0, 0, 0]
```

## Data layout

Do not commit `data/`, `outputs/`, generated checkpoints, or generated model files to git.

### Single-source data

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

Rules:

- the first-level directories under `data/single` are source class names;
- these class names become `class_names`;
- frequency directories such as `600.000MHz` are operating conditions, not labels;
- all CSV files are discovered recursively.

### Real mixed training data

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

Rules:

- the first-level directories under `data/real_train` determine labels;
- `ratio_1_1`, `ratio_1_2`, `ratio_1_4`, and frequency directories are conditions, not labels;
- all CSV files are discovered recursively.

### Real test data

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

`src.infer_folder` supports nested directories and recursively reads every CSV under each first-level group.

## CSV format

Instrument CSV files may contain metadata before the real numeric samples:

```text
instrument information
sampling parameters
...
DATA
0.000000000,-0.0123
0.000000001,-0.0118
0.000000002,-0.0120
```

CSV parsing rules:

- the reader automatically finds a line whose stripped text is `DATA`;
- all lines before `DATA` are ignored;
- after `DATA`, if a row has two or more columns, the second column is used as the amplitude;
- after `DATA`, if a row has one column, that column is used as the amplitude;
- if no `DATA` marker exists, legacy one-column or two-column CSV files are still supported.

## Training modes

The `training_data.mode` field in `configs/train.yaml` supports three modes:

```yaml
training_data:
  mode: synthetic_only
```

Uses generated synthetic mixtures only.

```yaml
training_data:
  mode: hybrid
```

Uses both synthetic mixtures and real CSV samples.

```yaml
training_data:
  mode: real_only
```

Uses only real CSV samples from the indexed real dataset.

In `real_only` mode, `data.num_samples` is **not** used. The number of training examples is determined by how many real CSV files are found in `data/single` and `data/real_train`. `data.num_samples` only controls synthetic data generation for synthetic modes.

## Recommended real-only workflow

### 1. Check configured paths

```bash
python -m src.check_paths --config configs/train.yaml
```

### 2. Build the real dataset index

```bash
python -m src.build_real_index \
  --single-dir data/single \
  --real-train-dir data/real_train \
  --output outputs/reports/real_dataset_index.csv
```

### 3. Split train / validation / test

```bash
python -m src.split_real_dataset \
  --index outputs/reports/real_dataset_index.csv \
  --output outputs/reports/real_dataset_split.csv \
  --train-ratio 0.7 \
  --val-ratio 0.15 \
  --test-ratio 0.15 \
  --seed 42
```

### 4. Train

```bash
python -m src.train --config configs/train.yaml
```

### 5. Evaluate the real test split

```bash
python -m src.evaluate \
  --model outputs/checkpoints/best.pt \
  --real-split test
```

### 6. Evaluate an external `data/real_test` folder

```bash
python -m src.infer_folder \
  --model outputs/checkpoints/best.pt \
  --input-dir data/real_test \
  --output outputs/reports/real_test_report.csv \
  --threshold 0.5 \
  --unknown-threshold 0.35
```

## Evaluation and inference

Model checkpoints are written to:

```text
outputs/checkpoints/best.pt
outputs/checkpoints/last.pt
```

Training and evaluation reports are written under `outputs/reports/`, including:

- `training_history.csv`: per-epoch training and validation metrics;
- `eval_report.json`: evaluation metrics across thresholds or real splits;
- `real_test_report.csv`: per-file folder inference probabilities and labels;
- `real_test_summary.json`: aggregate folder inference metrics.

Threshold behavior:

- `threshold = 0.5`: a source is predicted present when its probability is at least this value;
- `unknown_threshold = 0.35`: if every source probability is below this value, the sample is treated as unknown;
- if the maximum probability is between `unknown_threshold` and `threshold`, the sample is treated as uncertain.

## Configuration highlights

Important `configs/train.yaml` fields:

- `data.signal_length`: fixed time-domain length after padding or cropping;
- `stft.nperseg`: STFT window length;
- `stft.noverlap`: STFT overlap;
- `training_data.mode`: `synthetic_only`, `hybrid`, or `real_only`;
- `real_data.index_file`: path to the real dataset index CSV;
- `real_data.split_file`: path to the real dataset split CSV;
- `train.batch_size`: batch size;
- `train.epochs`: maximum number of epochs;
- `train.threshold`: default probability threshold;
- `early_stopping.patience`: early-stopping patience;
- `cache.enabled`: whether STFT feature caching is enabled.

## Limitations

- The current model focuses on source presence/absence, not contribution-ratio estimation.
- Unknown rejection depends on negative examples and threshold calibration.
- If the real environment changes significantly, additional `real_train` data is needed.
- STFT parameters should be adjusted for the actual sample count and sample rate.
- The CNN is a baseline and can be replaced by a deeper CNN or a time-domain + frequency-domain dual-branch model.
