# Project Status

Updated: 2026-07-20

## Current Goal

The project identifies three noise sources from CSV time-domain signals:

- `source_1`
- `source_3`
- `source_5`

The current business target is to push real-data exact match accuracy toward 95%. This has not been proven yet. The latest measured failure mode is broad under-prediction, led by very low `source_5` recall; `[1,1,0]` to `[1,1,1]` errors still matter but are no longer the dominant diagnosis.

## Latest Relayed Baseline

The latest company-data test result was relayed manually because the dataset is not available in this workspace:

- thresholds: `source_1=0.55`, `source_3=0.55`, `source_5=0.60`
- micro F1: `0.7837`
- exact match: `0.4447`
- over-prediction rate: `0.2136`
- under-prediction rate: `0.4780`
- source_5 precision / recall / F1: `0.9560 / 0.4117 / 0.5756`

This changes the diagnosis: source_5 under-detection is the dominant problem. The old source_5 false-positive penalty and the forced `source_5 >= 0.60` threshold search constraint were too aggressive.

Removing that constraint and using thresholds `0.55 / 0.55 / 0.50` produced the following test result with the legacy checkpoint:

- micro F1: `0.7800`
- exact match: `0.4781`
- source_5 precision / recall / F1: `0.8307 / 0.6600 / 0.7360`
- over-prediction / under-prediction: `0.2360 / 0.3990`

## Current Recommended Training Mode

The current configuration is real-only training with absolute numeric features:

```yaml
training_data:
  mode: real_only

preprocessing:
  signal_normalization: none

balanced_train:
  enabled: false

sampler:
  enabled: true
  strategy: label_combo

stft:
  magnitude_scale: absolute
  input_representation: absolute_relative

model:
  architecture: residual
  auxiliary_heads:
    enabled: true
  prediction:
    mode: structured
```

Important details:

- Time-domain CSV values are kept at their original numeric scale.
- STFT features use linear absolute magnitude.
- dB features are intentionally not used.
- `log1p` STFT compression is kept only as a legacy option, not the current default.
- Real feature cache keys now include `signal_normalization` and `magnitude_scale`, so old and new cached features should not collide.

## Implemented So Far

- Real-data indexing and split workflow:
  - `src.build_real_index`
  - `src.split_real_dataset`
  - `outputs/reports/real_dataset_split.csv`

- Bias and error diagnostics:
  - label distribution analysis
  - per-source precision/recall/F1/FPR/FNR
  - combo confusion matrix
  - source5 over-prediction rate
  - double-source predicted as triple-source rate
  - `[1,1,0]` focused error analysis

- Training-side bias controls:
  - `WeightedRandomSampler`
  - `sampler.strategy: label_combo`
  - asymmetric BCE style loss
  - exact-match early stopping
  - separate class-specific false-negative penalties
  - auxiliary 7-combination and 1/2/3-source-count heads

- Threshold search:
  - per-class threshold grid search
  - source5 minimum threshold support
  - `best_thresholds.json`

- Visualization and diagnostics:
  - `src.csv_to_images` converts recursive CSV folders into PNG plots.
  - `src.feature_statistics` analyzes raw signal / FFT / STFT energy statistics.

- Current feature update:
  - `preprocessing.signal_normalization: none`
  - `stft.magnitude_scale: absolute`
  - training, evaluation, single-file inference, folder inference, and real-vs-synthetic analysis now use the same configurable feature path.

- Structured-model update:
  - residual spectral backbone with coarse frequency position preserved
  - dual absolute/relative-log STFT channels for weak-source visibility
  - seven-combination classification as the primary objective
  - multilabel and source-count objectives as auxiliary supervision
  - fused structured decoding that always returns one valid combination

## Current Known Risks

- The 95% exact-match target has not been validated because full company data is not available in this environment.
- If source_5 is physically stronger or spectrally overlaps with source_1/source_3 combinations, model changes alone may not reach 95%.
- If old cached STFT files were created before the absolute-magnitude change, keep cache disabled or rebuild cache before training.
- `balanced_train` exists but is currently not the default recommendation. Default training uses `outputs/reports/real_dataset_split.csv`.

## Recommended Next Experiment

Run this sequence on the machine that has the company CSV data:

```bash
python -m src.check_paths --config configs/train.yaml

python -m src.build_real_index \
  --single-dir data/single \
  --combo-dir data/real_dataset \
  --output outputs/reports/real_dataset_index.csv

python -m src.split_real_dataset \
  --index outputs/reports/real_dataset_index.csv \
  --output outputs/reports/real_dataset_split.csv \
  --train-ratio 0.7 \
  --val-ratio 0.15 \
  --test-ratio 0.15 \
  --seed 42

python -m src.analyze_label_distribution \
  --split outputs/reports/real_dataset_split.csv \
  --output outputs/reports/label_distribution.json

python -m src.train --config configs/train.yaml

python -m src.evaluate \
  --model outputs/checkpoints/best.pt \
  --real-split test
```

## Reports To Inspect First

After training/evaluation, inspect these files first:

- `outputs/reports/eval_report.json`
- `outputs/reports/error_analysis.csv`
- `outputs/reports/combo_confusion.csv`
- `outputs/reports/threshold_search.csv`
- `outputs/reports/best_thresholds.json`
- `outputs/reports/label_distribution.json`

Priority questions:

1. Is `[1,1,0]` still mostly predicted as `[1,1,1]`?
2. Is `source5_false_positive_rate` still high?
3. Is `double_to_triple_rate` still high?
4. Are errors concentrated in specific ratio or frequency folders?
5. Does increasing only the source_5 threshold improve exact match?

## If Accuracy Still Cannot Reach 95%

Do not jump directly to a larger CNN. Use this order:

1. Build an NNLS / spectrum-template baseline.
   - Average source_1/source_3/source_5 FFT or STFT absolute-magnitude templates.
   - Fit each mixed sample as a non-negative combination of templates.
   - Use the fitted coefficients as an interpretable source-presence baseline.

2. If NNLS separates `[1,1,0]` and `[1,1,1]`, use template coefficients as extra features or as an ensemble signal.

3. If NNLS is weak, upgrade the model:
   - first choice: ResNet-style 2D CNN on absolute STFT magnitude
   - second choice: 1D ResNet / InceptionTime on raw waveform
   - strongest option: dual-branch model with raw waveform + STFT magnitude + scalar energy features

4. Add multi-task heads:
   - 3-source multilabel head
   - 7-combo classification head
   - source-count head for 1/2/3 sources

The source-count head is especially useful when double-source samples are often predicted as triple-source.

## Local Validation Commands

These commands should pass without company data:

```bash
python -m compileall src tests
python -m unittest discover -s tests -v
```

Full training and final accuracy validation require the internal company CSV dataset.
