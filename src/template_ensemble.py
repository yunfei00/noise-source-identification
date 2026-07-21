from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import nnls
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from src.dataset import RealCsvDataset
from src.infer import load_checkpoint
from src.model_cnn import build_model
from src.train import prepare_real_split, resolve_device


def combo_label_matrix(num_classes: int) -> np.ndarray:
    return np.asarray(
        [
            [(value >> (num_classes - 1 - index)) & 1 for index in range(num_classes)]
            for value in range(1, 2**num_classes)
        ],
        dtype=np.int32,
    )


def labels_to_combo_indices(labels: np.ndarray) -> np.ndarray:
    labels = (np.asarray(labels) > 0.5).astype(np.int64)
    bit_weights = 2 ** np.arange(labels.shape[1] - 1, -1, -1, dtype=np.int64)
    indices = labels @ bit_weights - 1
    if np.any(indices < 0):
        raise ValueError("Template ensemble requires at least one source in every sample")
    return indices.astype(np.int64, copy=False)


def normalized_power_spectrum(model_input: np.ndarray) -> np.ndarray:
    feature = np.asarray(model_input, dtype=np.float64)
    if feature.ndim != 3 or feature.shape[0] < 1:
        raise ValueError(f"Expected model input with shape [channels,freq,time], got {feature.shape}")
    absolute_magnitude = feature[0]
    power = np.mean(np.square(absolute_magnitude), axis=1)
    total = float(power.sum())
    if total <= 1e-20:
        return np.full_like(power, 1.0 / max(len(power), 1), dtype=np.float64)
    return (power / total).astype(np.float64, copy=False)


def nnls_features(power: np.ndarray, templates: np.ndarray) -> np.ndarray:
    power = np.asarray(power, dtype=np.float64)
    templates = np.asarray(templates, dtype=np.float64)
    if templates.ndim != 2 or templates.shape[1] != power.shape[0]:
        raise ValueError(f"Template/power mismatch: {templates.shape} vs {power.shape}")

    background = np.full(power.shape[0], 1.0 / power.shape[0], dtype=np.float64)
    design = np.column_stack([templates.T, background])
    coefficients, residual = nnls(design, power)
    source_coefficients = coefficients[: templates.shape[0]]
    coefficient_sum = max(float(source_coefficients.sum()), 1e-12)
    fractions = source_coefficients / coefficient_sum
    correlations = templates @ power
    correlation_norms = np.linalg.norm(templates, axis=1) * max(float(np.linalg.norm(power)), 1e-12)
    correlations = correlations / np.maximum(correlation_norms, 1e-12)
    relative_residual = float(residual / max(np.linalg.norm(power), 1e-12))
    return np.concatenate(
        [
            fractions,
            np.log1p(source_coefficients),
            correlations,
            np.asarray([relative_residual, coefficients[-1]], dtype=np.float64),
        ]
    ).astype(np.float32)


def _dataset(config: dict, class_names: list[str], split: str, split_path: Path) -> RealCsvDataset:
    return RealCsvDataset(
        config.get("real_data", {}).get("dataset_root", "."),
        class_names,
        config,
        split=split,
        index_path=split_path,
    )


def _sample_indices_by_combo(dataset: RealCsvDataset, limit: int | None, seed: int) -> list[int]:
    by_combo: dict[int, list[int]] = defaultdict(list)
    for index, sample in enumerate(dataset.samples):
        label = np.asarray(sample["label"], dtype=np.float32).reshape(1, -1)
        by_combo[int(labels_to_combo_indices(label)[0])].append(index)
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for combo_index in sorted(by_combo):
        indices = np.asarray(by_combo[combo_index], dtype=np.int64)
        if limit is not None and len(indices) > limit:
            indices = rng.choice(indices, size=limit, replace=False)
        selected.extend(int(value) for value in indices)
    return selected


def build_templates(
    dataset: RealCsvDataset,
    num_classes: int,
    max_samples_per_source: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, int]]:
    rng = np.random.default_rng(seed)
    templates: list[np.ndarray] = []
    counts: dict[str, int] = {}
    for class_index in range(num_classes):
        candidates = [
            index
            for index, sample in enumerate(dataset.samples)
            if int(np.asarray(sample["label"]).sum()) == 1
            and int(np.asarray(sample["label"])[class_index]) == 1
        ]
        if not candidates:
            raise ValueError(f"No single-source training samples found for class index {class_index}")
        if len(candidates) > max_samples_per_source:
            candidates = rng.choice(candidates, size=max_samples_per_source, replace=False).tolist()
        spectra = []
        for index in tqdm(candidates, desc=f"Template source {class_index + 1}", unit="sample"):
            x, _ = dataset[int(index)]
            spectra.append(normalized_power_spectrum(x.numpy()))
        template = np.median(np.vstack(spectra), axis=0)
        template /= max(float(template.sum()), 1e-12)
        templates.append(template)
        counts[str(class_index)] = len(candidates)
    return np.vstack(templates).astype(np.float64), counts


def extract_features_and_neural_probabilities(
    dataset: RealCsvDataset,
    indices: list[int],
    templates: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    description: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    all_template_features: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_neural_probabilities: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for x, y in tqdm(loader, desc=description, unit="batch"):
            for sample in x.numpy():
                power = normalized_power_spectrum(sample)
                all_template_features.append(nnls_features(power, templates))
            outputs = model.forward_with_auxiliary(x.to(device))
            neural_probabilities = model.structured_combo_probabilities(outputs)
            all_neural_probabilities.append(neural_probabilities.cpu().numpy())
            all_labels.append(y.numpy())
    labels = np.vstack(all_labels).astype(np.int32)
    return (
        np.vstack(all_template_features).astype(np.float32),
        labels_to_combo_indices(labels),
        np.vstack(all_neural_probabilities).astype(np.float32),
    )


def align_probabilities(probabilities: np.ndarray, classes: np.ndarray, num_combos: int) -> np.ndarray:
    aligned = np.zeros((probabilities.shape[0], num_combos), dtype=np.float64)
    aligned[:, classes.astype(int)] = probabilities
    return aligned


def classification_metrics(
    true_indices: np.ndarray,
    predicted_indices: np.ndarray,
    class_names: list[str],
) -> dict:
    label_matrix = combo_label_matrix(len(class_names))
    targets = label_matrix[true_indices]
    predictions = label_matrix[predicted_indices]
    precision, recall, f1, support = precision_recall_fscore_support(
        targets,
        predictions,
        average=None,
        zero_division=0,
    )
    exact = np.all(targets == predictions, axis=1)
    over = np.any(predictions > targets, axis=1)
    under = np.any(predictions < targets, axis=1)
    per_label = {}
    for combo_index, label in enumerate(label_matrix):
        mask = true_indices == combo_index
        key = "".join(str(int(value)) for value in label)
        per_label[key] = {
            "num_samples": int(mask.sum()),
            "accuracy": float((predicted_indices[mask] == combo_index).mean()) if np.any(mask) else 0.0,
        }
    return {
        "num_samples": int(len(true_indices)),
        "exact_match": float(exact.mean()),
        "micro_f1": float(f1_score(targets, predictions, average="micro", zero_division=0)),
        "over_prediction_rate": float(over.mean()),
        "under_prediction_rate": float(under.mean()),
        "per_source": {
            name: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, name in enumerate(class_names)
        },
        "label_accuracy": per_label,
        "prediction_counts": {
            str(index): int(count)
            for index, count in sorted(Counter(predicted_indices.tolist()).items())
        },
    }


def search_blend_weight(
    true_indices: np.ndarray,
    neural_probabilities: np.ndarray,
    template_probabilities: np.ndarray,
    step: float,
) -> tuple[float, float]:
    if step <= 0.0 or step > 1.0:
        raise ValueError("blend step must be in (0, 1]")
    best_alpha = 0.0
    best_accuracy = -1.0
    for alpha in np.arange(0.0, 1.0 + step / 2.0, step):
        blended = (1.0 - alpha) * neural_probabilities + alpha * template_probabilities
        accuracy = float((blended.argmax(axis=1) == true_indices).mean())
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_alpha = float(alpha)
    return best_alpha, best_accuracy


def run_template_ensemble(
    model_path: str | Path,
    output: str | Path = "outputs/reports/template_ensemble_report.json",
    device_name: str = "auto",
    max_template_samples: int = 1000,
    max_train_samples_per_combo: int = 2000,
    blend_step: float = 0.05,
    seed: int = 42,
) -> dict:
    device = resolve_device(device_name)
    checkpoint = load_checkpoint(model_path, map_location=device)
    config = checkpoint.get("config")
    class_names = checkpoint.get("class_names")
    if not isinstance(config, dict) or not isinstance(class_names, list):
        raise ValueError("Checkpoint is missing config or class_names")

    split_path = prepare_real_split(config, class_names)
    if split_path is None:
        raise ValueError("Template ensemble requires the real-data split")
    train_dataset = _dataset(config, class_names, "train", split_path)
    val_dataset = _dataset(config, class_names, "val", split_path)
    test_dataset = _dataset(config, class_names, "test", split_path)

    templates, template_counts = build_templates(
        train_dataset,
        len(class_names),
        max_template_samples,
        seed,
    )
    model = build_model(len(class_names), config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    if not model.auxiliary_heads:
        raise ValueError("Template ensemble requires a checkpoint with combination heads")

    train_indices = _sample_indices_by_combo(train_dataset, max_train_samples_per_combo, seed)
    val_indices = list(range(len(val_dataset)))
    test_indices = list(range(len(test_dataset)))
    batch_size = int(config.get("train", {}).get("batch_size", 32))
    num_workers = int(config.get("train", {}).get("num_workers", 0))

    train_x, train_y, _ = extract_features_and_neural_probabilities(
        train_dataset, train_indices, templates, model, device, batch_size, num_workers, "NNLS train"
    )
    val_x, val_y, val_neural = extract_features_and_neural_probabilities(
        val_dataset, val_indices, templates, model, device, batch_size, num_workers, "NNLS validation"
    )
    test_x, test_y, test_neural = extract_features_and_neural_probabilities(
        test_dataset, test_indices, templates, model, device, batch_size, num_workers, "NNLS test"
    )

    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
    )
    classifier.fit(train_x, train_y)
    logistic = classifier.named_steps["logisticregression"]
    num_combos = (2 ** len(class_names)) - 1
    val_template = align_probabilities(classifier.predict_proba(val_x), logistic.classes_, num_combos)
    test_template = align_probabilities(classifier.predict_proba(test_x), logistic.classes_, num_combos)
    best_alpha, best_val_exact = search_blend_weight(
        val_y,
        val_neural,
        val_template,
        blend_step,
    )

    neural_predictions = test_neural.argmax(axis=1)
    template_predictions = test_template.argmax(axis=1)
    ensemble_predictions = (
        (1.0 - best_alpha) * test_neural + best_alpha * test_template
    ).argmax(axis=1)
    report = {
        "model": str(model_path),
        "class_names": class_names,
        "template_samples": template_counts,
        "calibration_train_samples": int(len(train_y)),
        "best_template_weight": best_alpha,
        "best_validation_exact_match": best_val_exact,
        "test": {
            "neural": classification_metrics(test_y, neural_predictions, class_names),
            "template_nnls": classification_metrics(test_y, template_predictions, class_names),
            "ensemble": classification_metrics(test_y, ensemble_predictions, class_names),
        },
    }
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"best_template_weight={best_alpha:.4f}")
    print(f"best_validation_exact_match={best_val_exact:.4f}")
    for name, metrics in report["test"].items():
        print(
            f"test_{name}: exact_match={metrics['exact_match']:.4f} "
            f"micro_f1={metrics['micro_f1']:.4f}"
        )
    print(f"report={output_path}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an NNLS spectral-template and CNN ensemble.")
    parser.add_argument("--model", type=Path, required=True, help="Structured model checkpoint.")
    parser.add_argument("--output", type=Path, default=Path("outputs/reports/template_ensemble_report.json"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-template-samples", type=int, default=1000)
    parser.add_argument("--max-train-samples-per-combo", type=int, default=2000)
    parser.add_argument("--blend-step", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_template_ensemble(
        model_path=args.model,
        output=args.output,
        device_name=args.device,
        max_template_samples=args.max_template_samples,
        max_train_samples_per_combo=args.max_train_samples_per_combo,
        blend_step=args.blend_step,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
