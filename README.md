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

```bash
python -m src.check_paths --config configs/train.yaml
```

```bash
python -m src.build_real_index \
  --single-dir data/single \
  --combo-dir data/real_dataset \
  --output outputs/reports/real_dataset_index.csv
```

```bash
python -m src.split_real_dataset \
  --index outputs/reports/real_dataset_index.csv \
  --output outputs/reports/real_dataset_split.csv
```

```bash
python -m src.train --config configs/train.yaml
```

```bash
python -m src.evaluate \
  --model outputs/checkpoints/best.pt \
  --real-split test
```

Folder inference:

```bash
python -m src.infer_folder \
  --model outputs/checkpoints/best.pt \
  --input-dir data/real_dataset \
  --output outputs/reports/infer_folder_report.csv \
  --threshold 0.5 \
  --unknown-threshold 0.35
```

## Configuration

`configs/train.yaml` defaults to real-only training:

```yaml
training_data:
  mode: real_only

real_data:
  single_dir: data/single
  combo_dir: data/real_dataset
  index_file: outputs/reports/real_dataset_index.csv
  split_file: outputs/reports/real_dataset_split.csv
```

In `real_only` mode:

- `data/mixed` is not used.
- `data.num_samples` is not used.
- Samples are loaded lazily from CSV files listed in `real_dataset_split.csv`.
- An empty split file raises a clear training error.

## Outputs

- `outputs/reports/real_dataset_index.csv`: unified real-data index.
- `outputs/reports/real_dataset_summary.json`: class, sample, group, label, ratio, and invalid-group summary.
- `outputs/reports/real_dataset_split.csv`: train / val / test split with a `split` column.
- `outputs/reports/eval_report.json`: source metrics plus group, label-combination, and ratio accuracies.
- `outputs/reports/infer_folder_report.csv`: recursive folder inference report.
