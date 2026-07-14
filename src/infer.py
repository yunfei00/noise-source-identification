from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from src.features import apply_signal_normalization, compute_stft_feature, fix_length, read_signal_csv
from src.model_cnn import NoiseCNN
from src.train import resolve_device


def load_checkpoint(path: str | Path, map_location: torch.device) -> dict:
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
        raise ValueError(f"Invalid checkpoint format: {checkpoint_path}")
    return checkpoint


def status_from_probability(probability: float) -> str:
    if probability >= 0.7:
        return "存在"
    if probability >= 0.4:
        return "疑似"
    return "不存在"


def infer(model_path: str | Path, input_path: str | Path, device_name: str = "auto") -> list[tuple[str, float, str]]:
    device = resolve_device(device_name)
    checkpoint = load_checkpoint(model_path, map_location=device)
    class_names = checkpoint.get("class_names")
    config = checkpoint.get("config")
    if not isinstance(class_names, list) or not class_names:
        raise ValueError("Checkpoint is missing class_names")
    if not isinstance(config, dict):
        raise ValueError("Checkpoint is missing config")

    data_config = config.get("data", {})
    stft_config = config.get("stft", {})
    preprocessing_config = config.get("preprocessing", {})
    signal = read_signal_csv(input_path)
    signal = fix_length(signal, int(data_config.get("signal_length", 4096)))
    signal = apply_signal_normalization(signal, str(preprocessing_config.get("signal_normalization", "standardize")))
    feature = compute_stft_feature(
        signal,
        sample_rate=int(data_config.get("sample_rate", 1_000_000)),
        nperseg=int(stft_config.get("nperseg", 256)),
        noverlap=int(stft_config.get("noverlap", 128)),
        target_freq_bins=int(stft_config.get("target_freq_bins", 128)),
        target_time_bins=int(stft_config.get("target_time_bins", 64)),
        magnitude_scale=str(stft_config.get("magnitude_scale", "log1p")),
    )

    model = NoiseCNN(num_classes=len(class_names)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    x = torch.from_numpy(feature).unsqueeze(0).unsqueeze(0).float().to(device)
    with torch.no_grad():
        logits = model(x)
        probabilities = torch.sigmoid(logits).squeeze(0).cpu().numpy()

    results = [
        (class_name, float(probability), status_from_probability(float(probability)))
        for class_name, probability in zip(class_names, probabilities)
    ]
    return sorted(results, key=lambda item: item[1], reverse=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Infer noise sources from a mixed signal CSV.")
    parser.add_argument("--model", type=Path, required=True, help="Path to model checkpoint.")
    parser.add_argument("--input", type=Path, required=True, help="Path to input CSV.")
    parser.add_argument("--device", default="auto", help="Device: auto, cpu, or cuda.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = infer(args.model, args.input, args.device)
    name_width = max(len(name) for name, _, _ in results)
    for name, probability, status in results:
        print(f"{name:<{name_width}}  {probability:.3f}  {status}")


if __name__ == "__main__":
    main()
