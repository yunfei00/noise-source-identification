from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

from src.features import read_signal_csv_info


def load_signal_length(config_path: Path) -> int:
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a YAML mapping: {config_path}")
    data_config = config.get("data", {})
    if not isinstance(data_config, dict):
        raise ValueError(f"Config data section must be a YAML mapping: {config_path}")
    return int(data_config.get("signal_length", 4096))


def format_values(values: np.ndarray) -> str:
    return np.array2string(values, precision=8, separator=", ", max_line_width=120)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug signal CSV parsing.")
    parser.add_argument("--input", type=Path, required=True, help="Path to the CSV file to inspect.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/train.yaml"),
        help="Training config containing data.signal_length (default: configs/train.yaml).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    info = read_signal_csv_info(args.input)
    values = info.values
    signal_length = load_signal_length(args.config)

    print(f"input: {args.input}")
    print(f"found DATA line: {info.found_data_line}")
    print(f"DATA line number: {info.data_line_number if info.data_line_number is not None else 'N/A'}")
    print(f"valid sample count: {values.size}")
    print(f"first 10 amplitudes: {format_values(values[:10])}")
    print(f"last 10 amplitudes: {format_values(values[-10:])}")
    print(f"min: {float(np.min(values)):.8g}")
    print(f"max: {float(np.max(values)):.8g}")
    print(f"mean: {float(np.mean(values)):.8g}")
    print(f"std: {float(np.std(values)):.8g}")
    print(f"signal_length: {signal_length}")
    if values.size < signal_length:
        print(f"length action: raw sample count is smaller than signal_length; zero-padding will be applied.")
    elif values.size > signal_length:
        print(f"length action: raw sample count is larger than signal_length; cropping will be applied.")
    else:
        print("length action: raw sample count equals signal_length; no padding or cropping needed.")


if __name__ == "__main__":
    main()
