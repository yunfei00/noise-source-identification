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

Before indexing or training a newly collected `source_1/source_3/source_4` dataset, run the read-only audit:

```bash
python -m src.audit_real_data \
  --class-names source_1 source_3 source_4 \
  --output outputs/reports/source_134_data_audit.json
```

The single JSON report contains counts, label/ratio coverage, dB and length distributions, suspicious files, and recommended preprocessing parameters. The command also prints a compact `copy_paste_summary`. Add `--no-signal-threshold-db` only after measuring a reliable no-signal/background threshold.

```bash
python -m src.train --config configs/train.yaml
```

The CSV second column is a negative dB-vs-time trace from the spectrum analyzer, not a linear waveform. The `db_trace` representation removes each trace's median baseline before STFT and then supplies four channels: absolute fluctuation spectrum, relative fluctuation spectrum, median dB level, and dB variation strength. Short traces are padded with their median dB value rather than a false `0 dB` tail.

This changes the model input from two channels to four, so train from scratch and do not pass `--init-model` for this experiment. The current profile also disables the spectrum augmentation that previously caused underfitting.

### Step 5: Evaluate the structured combination model

```bash
python -m src.evaluate \
  --model outputs/checkpoints/best.pt \
  --real-split test
```

The recommended model decodes one of the seven valid source combinations directly, so per-source threshold search is not used. `src.search_thresholds` remains available for legacy checkpoints whose prediction mode is `multilabel`.

### Step 6: Evaluate the physics-guided spectral-template ensemble

```bash
python -m src.template_ensemble \
  --model outputs/checkpoints/best.pt \
  --output outputs/reports/template_ensemble_report.json
```

This builds robust single-source power-spectrum templates from the training split, fits non-negative template coefficients, calibrates the seven combinations, and searches the CNN/template blend weight on validation data. It reports separate neural, NNLS-template, and ensemble test metrics without retraining the CNN.

### Step 7: Diagnose source5 errors

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

Strict single-file inference with a machine-readable result and a model
inference-contract report:

```bash
python scripts/predict_single_csv.py \
  --csv path/to/instrument_export.csv \
  --checkpoint outputs/checkpoints/best.pt \
  --device auto \
  --report-dir outputs/inference_contract
```

The single-file entry requires a standalone `DATA` marker and finite numeric
values in the second column after it. It restores labels, model structure,
preprocessing, and the default threshold from the checkpoint; `--config` is
required only when a pure state dict has no embedded training contract.
`--threshold` accepts either one value or one comma-separated value per label.
The output directory receives `<sample>_prediction.json` and
`inference_contract.md`.

The stable GUI/runtime API is the installable `noise_source_runtime` package.
Keep one session alive to avoid reloading the checkpoint for every file:

```python
from noise_source_runtime import (
    InferenceSession,
    export_prediction_result,
)

with InferenceSession.load_model(
    "outputs/checkpoints/best.pt",
    device="auto",
) as session:
    first = session.predict_file("path/to/first.csv")
    second = session.predict_file("path/to/second.csv")
    memory_result = session.predict_array(signal_array)

    # Pure prediction writes nothing. Export only when explicitly requested.
    exported = export_prediction_result(
        first,
        session,
        "outputs/inference_contract",
    )
```

`structured` results keep four distinct concepts:

- `multilabel_probabilities`: sigmoid of the independent source logits.
- `combination_probabilities`: softmax over the seven legal combinations,
  ordered `001, 010, 011, 100, 101, 110, 111`.
- `label_marginal_probabilities`: combination probabilities multiplied by the
  combination-label matrix; this is the GUI's default per-source percentage.
- `decoded_label_vector`: the combination argmax. Thresholds are not used for
  the final structured decision (`thresholds_applicable=false`).

`multilabel` checkpoints continue to use sigmoid plus explicit per-label
thresholds. The old `src.inference` API remains as a compatibility wrapper.

Build and verify a versioned model package:

```bash
python scripts/build_model_package.py \
  --checkpoint outputs/checkpoints/best.pt \
  --version 1.0.0 \
  --model-name noise-source-production \
  --output-dir outputs/model-packages/noise-source-production-1.0.0

python scripts/build_model_package.py \
  --verify outputs/model-packages/noise-source-production-1.0.0
```

The package contains `model.pt`, `manifest.json`, `metrics.json`,
`preprocess.json`, `labels.json`, `README.md`, and `sha256.txt`. New training
checkpoints include schema/runtime versions, preprocessing contract, creation
time, prediction mode, training commit, monitor, best metric, and label order;
legacy checkpoints remain loadable.

For recursively inferring an arbitrarily nested, unlabeled metadata folder and reporting prediction distributions:

```bash
python -m src.infer_metadata_folder \
  --model outputs/checkpoints/best.pt \
  --input-dir data/metadata \
  --output outputs/reports/metadata_inference.csv \
  --confidence-threshold 0.6
```

The CSV contains per-file predictions. The adjacent `.summary.json` contains combination/source distributions, confidence and margin statistics, second choices, low-confidence files, failures, and first-level-folder breakdowns. Class names come from the checkpoint. Use `src.infer_folder` or `src.evaluate` instead when folder names provide ground truth and accuracy metrics are required.

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
  input_representation: db_trace
  db_level_range: [-110.0, -50.0]
  db_variation_scale: 15.0
```

In `real_only` mode:

- `data/mixed` is not used.
- `data.num_samples` is not used.
- Samples are loaded lazily from CSV files listed in `real_data.split_file`.
- An empty split file raises a clear training error.

The current feature path is specialized for spectrum-analyzer dB traces:

- The per-file median dB baseline is removed before computing the fluctuation STFT.
- Absolute level and variation are retained as two separate scalar feature channels.
- Legacy `single` and `absolute_relative` representations remain available for old checkpoints.

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
