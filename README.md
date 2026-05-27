# Noise Source Identification

This project is a first runnable pipeline for multi-label noise source identification:

single-source noise CSV -> synthesized mixed noise samples -> STFT features -> CNN multi-label training -> mixed-signal inference.

## Data Layout

Put real single-source CSV files under `data/single`, with one subdirectory per source class:

```text
data/single/
  fan/
    fan_000.csv
  motor/
    motor_000.csv
  switch_power/
    switch_power_000.csv
```

Class order is stable and sorted by directory name. The synthesis step writes it to `data/mixed/class_names.json`.

Generated mixed data is written to:

```text
data/mixed/train/x/*.npy
data/mixed/train/y/*.npy
data/mixed/val/x/*.npy
data/mixed/val/y/*.npy
data/mixed/test/x/*.npy
data/mixed/test/y/*.npy
```

Each `x` file is a fixed-length time-domain signal. STFT features are computed dynamically in the PyTorch dataset.

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

This creates 10 CSV files each for `motor`, `fan`, and `switch_power`.

## Synthesize Mixed Dataset

```bash
python -m src.synthesize_mixed_dataset --config configs/train.yaml
```

The default config is intentionally small (`num_samples: 300`, `epochs: 3`) so the complete workflow can be verified quickly on CPU. Increase these values for real experiments.

## Train

```bash
python -m src.train --config configs/train.yaml
```

Checkpoints are saved to:

```text
outputs/checkpoints/best.pt
outputs/checkpoints/last.pt
```

## Infer

```bash
python -m src.infer --model outputs/checkpoints/best.pt --input path/to/mixed.csv
```

The output lists each source class by probability:

```text
switch_power  0.923  存在
motor         0.811  存在
fan           0.231  不存在
```

Threshold rules:

- `>= 0.7`: 存在
- `0.4 ~ 0.7`: 疑似
- `< 0.4`: 不存在

## Current Limitations

- The demo signals are synthetic and only prove the pipeline, not real-world accuracy.
- Mixed samples use linear summation, random gain, random roll shift, and small Gaussian noise.
- STFT shape is fixed by padding/cropping, so extreme signal lengths or sampling rates may need config tuning.
- The CNN is intentionally lightweight and should be treated as a baseline model.
