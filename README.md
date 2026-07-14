# Noise Source Identification

This project identifies noise sources from CSV signals. The real-data workflow now uses one canonical layout only:

- `data/single` for real single-source data.
- `data/real_dataset` for real combined-source data.

Legacy real-data directories are no longer used or supported.

## Canonical real-data layout

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
```

Rules:

1. First-level directories under `data/single` are class names.
2. First-level directories under `data/real_dataset` are combined-source group labels.
3. Deeper directories can represent frequency, ratio, batch, condition, or other metadata.
4. CSV files are read recursively.

## Labels

`class_names` are discovered only from first-level directories under `data/single`, sorted by name. For example:

```text
["source_1", "source_3", "source_5"]
```

Single-source labels:

```text
data/single/source_1/**/*.csv -> [1,0,0]
data/single/source_3/**/*.csv -> [0,1,0]
data/single/source_5/**/*.csv -> [0,0,1]
```

Combined-source labels:

```text
data/real_dataset/source_1_source_3_mix/**/*.csv -> [1,1,0]
data/real_dataset/source_1_source_5_mix/**/*.csv -> [1,0,1]
data/real_dataset/source_3_source_5_mix/**/*.csv -> [0,1,1]
data/real_dataset/source_1_source_3_source_5_mix/**/*.csv -> [1,1,1]
```

## Recommended workflow

Run the commands in this order.

### Step 1: Check paths

```bash
python -m src.check_paths --config configs/train.yaml
```

### Step 2: Build the real-data index

```bash
python -m src.build_real_index \
  --single-dir data/single \
  --combo-dir data/real_dataset \
  --output outputs/reports/real_dataset_index.csv
```

### Step 3: Split train / val / test

```bash
python -m src.split_real_dataset \
  --index outputs/reports/real_dataset_index.csv \
  --output outputs/reports/real_dataset_split.csv \
  --train-ratio 0.7 \
  --val-ratio 0.15 \
  --test-ratio 0.15 \
  --seed 42
```

### Step 4: Train

```bash
python -m src.train --config configs/train.yaml
```

### Step 5: Evaluate with one global threshold

```bash
python -m src.evaluate \
  --model outputs/checkpoints/best.pt \
  --real-split test \
  --threshold 0.5
```

### Step 6: Search per-class thresholds after training

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

### Step 7: Re-evaluate with `best_thresholds.json`

```bash
python -m src.evaluate \
  --model outputs/checkpoints/best.pt \
  --real-split test \
  --thresholds-json outputs/reports/best_thresholds.json
```

### Step 8: Diagnose source5 over-prediction

```bash
python -m src.feature_statistics \
  --split outputs/reports/real_dataset_split.csv \
  --split-name train \
  --output outputs/reports/feature_statistics.csv \
  --max-samples-per-group 500
```

`feature_statistics` does not depend on the model checkpoint, so it can be run either before or after training.

## Command dependencies

- `check_paths`: no dependency on generated outputs.
- `build_real_index`: depends on `data/single` and `data/real_dataset`.
- `split_real_dataset`: depends on `outputs/reports/real_dataset_index.csv`.
- `train`: depends on `outputs/reports/real_dataset_split.csv`.
- `evaluate`: depends on `outputs/checkpoints/best.pt` and `outputs/reports/real_dataset_split.csv`.
- `search_thresholds`: depends on `outputs/checkpoints/best.pt` and `outputs/reports/real_dataset_split.csv`.
- `feature_statistics`: depends on `outputs/reports/real_dataset_split.csv`; it does not depend on `outputs/checkpoints/best.pt`.

Important ordering notes:

- `outputs/reports/best_thresholds.json` can only be generated after model training finishes.
- `src.search_thresholds` depends on `outputs/checkpoints/best.pt`.
- `src.evaluate` depends on `outputs/checkpoints/best.pt`.
- `src.train` depends on `outputs/reports/real_dataset_split.csv`.

Folder inference, after training:

```bash
python -m src.infer_folder \
  --model outputs/checkpoints/best.pt \
  --input-dir data/real_dataset \
  --output outputs/reports/infer_folder_report.csv \
  --threshold 0.5 \
  --unknown-threshold 0.35
```

## Configuration

`configs/train.yaml` is configured for real-only training without quota-selected balanced training:

```yaml
training_data:
  mode: real_only

balanced_train:
  enabled: false

real_data:
  single_dir: data/single
  combo_dir: data/real_dataset
  index_file: outputs/reports/real_dataset_index.csv
  split_file: outputs/reports/real_dataset_split.csv

preprocessing:
  signal_normalization: none

loss:
  type: asymmetric_bce
  gamma_neg: 4
  gamma_pos: 1
  label_smoothing: 0.05

early_stopping:
  monitor: exact_match

stft:
  magnitude_scale: absolute
```

In `real_only` mode:

- `data/mixed` is not used.
- `data.num_samples` is not used.
- Samples are loaded lazily from CSV files listed in `real_data.split_file`.
- An empty split file raises a clear training error.

The current feature path preserves absolute numeric scale:

- `preprocessing.signal_normalization: none` keeps raw time-domain amplitude values.
- `stft.magnitude_scale: absolute` uses linear STFT magnitude, not dB or log-compressed features.

## How training samples are fixed

Training data selection is controlled by the generated CSV split file, not by the README text:

1. `src.train` reads `training_data.mode`. With `real_only`, it sets the synthetic ratio to `0` and the real ratio to `1`.
2. `src.train` rebuilds `outputs/reports/real_dataset_index.csv` from `real_data.single_dir` and `real_data.combo_dir`.
3. `outputs/reports/real_dataset_split.csv` assigns each indexed CSV to `train`, `val`, or `test`.
4. During training, `RealCsvDataset` loads rows for the requested split from `outputs/reports/real_dataset_split.csv`.

Therefore the exact training set is:

```text
outputs/reports/real_dataset_split.csv
where split == train
```

To inspect the exact files that will be trained on:

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

## Optional balanced split

Quota-selected balanced training is not the recommended workflow for the current configuration. If you intentionally enable `balanced_train.enabled: true`, generate the quota-selected training split first:

```bash
python -m src.create_balanced_split \
  --input outputs/reports/real_dataset_split.csv \
  --output outputs/reports/real_dataset_split_balanced.csv \
  --quota 100=2100,010=2100,001=2100,110=5000,101=3500,011=3500,111=3000 \
  --seed 42
```

## Outputs

- `outputs/reports/real_dataset_index.csv`: unified real-data index.
- `outputs/reports/real_dataset_summary.json`: class, sample, group, label, ratio, and invalid-group summary.
- `outputs/reports/real_dataset_split.csv`: train / val / test split with a `split` column.
- `outputs/reports/eval_report.json`: source metrics plus group, label-combination, and ratio accuracies.
- `outputs/reports/threshold_search.csv`: per-class threshold search results.
- `outputs/reports/best_thresholds.json`: best per-class thresholds generated by `src.search_thresholds` after training.
- `outputs/reports/feature_statistics.csv`: feature statistics used to diagnose source over-prediction.
- `outputs/reports/infer_folder_report.csv`: recursive folder inference report.
