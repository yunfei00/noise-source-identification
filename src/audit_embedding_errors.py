from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.dataset import RealCsvDataset
from src.evaluate import collect_probabilities, normalize_thresholds, predictions_at_threshold
from src.infer import load_checkpoint
from src.model_cnn import build_model
from src.noise_source_runtime.device import resolve_device


def label_text(x: np.ndarray) -> str:
    return "[" + ",".join(str(int(v)) for v in x.tolist()) + "]"


def collect_embeddings(model, loader, device):
    values = []
    model.eval()
    with torch.no_grad():
        for x, _ in loader:
            values.append(model.encode(x.to(device)).cpu().numpy())
    return np.vstack(values)


def knn(emb: np.ndarray, labels: np.ndarray, k: int):
    z = emb.astype(np.float32)
    z /= np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-8)
    sim = z @ z.T
    np.fill_diagonal(sim, -np.inf)
    idx = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]
    return (labels[idx] == labels[:, None]).mean(axis=1)


def expected_errors_from_report(path: Path, expected_samples: int) -> int:
    payload = json.loads(path.read_text(encoding="utf-8"))
    report_samples = int(payload.get("num_samples", -1))
    if report_samples != expected_samples:
        raise RuntimeError(
            f"Reference evaluate report has {report_samples} samples, audit has {expected_samples}. "
            "Use the report produced for the same model and fixed Test split."
        )
    overall = payload.get("selected_metrics", {}).get("overall", {})
    exact = overall.get("exact_match")
    if exact is None:
        exact = payload.get("overall", {}).get("exact_match")
    if exact is None:
        raise RuntimeError("Cannot find selected exact_match in reference evaluate report.")
    return int(round(expected_samples * (1.0 - float(exact))))


def main():
    p = argparse.ArgumentParser(
        description="Audit fixed-Test errors in the CNN embedding space using evaluator-identical decoding."
    )
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--split-file", type=Path, required=True)
    p.add_argument("--eval-report", type=Path, required=True,
                   help="Existing src.evaluate JSON report for the same model and fixed Test split.")
    p.add_argument("--output-dir", type=Path, default=Path("outputs/reports/s2_s3_embedding_audit"))
    p.add_argument("--neighbors", type=int, default=30)
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    ckpt = load_checkpoint(args.model, map_location=device)
    config = ckpt["config"]
    names = ckpt["class_names"]
    config = dict(config)
    config["real_data"] = dict(config.get("real_data", {}))
    config["real_data"]["split_file"] = str(args.split_file)

    root = config["real_data"].get("dataset_root", ".")
    ds = RealCsvDataset(root, names, config, split="test", index_path=args.split_file)
    loader = DataLoader(
        ds,
        batch_size=int(config.get("train", {}).get("batch_size", 32)),
        shuffle=False,
        num_workers=0,
    )

    model = build_model(len(names), config).to(device)
    model.load_state_dict(ckpt["model_state"])

    # This is the exact probability/structured-decode path used by src.evaluate.
    probs, targets = collect_probabilities(model, loader, device)
    threshold = float(config.get("train", {}).get("threshold", 0.5))
    threshold_values = normalize_thresholds(threshold, len(names))
    preds = predictions_at_threshold(probs, names, threshold_values)

    # Embeddings are collected in a second deterministic pass over the same non-shuffled loader.
    emb = collect_embeddings(model, loader, device)
    if len(emb) != len(targets) or len(ds.samples) != len(targets):
        raise RuntimeError("Embedding/prediction/sample count mismatch; audit aborted.")

    # Current ablation Test is single-source one-hot data.
    mask = targets.sum(axis=1) == 1
    emb = emb[mask]
    targets = targets[mask]
    preds = preds[mask]
    samples = [s for s, keep in zip(ds.samples, mask) if keep]

    exact = np.all(preds == targets, axis=1)
    actual_errors = int((~exact).sum())
    expected_errors = expected_errors_from_report(args.eval_report, len(targets))
    if actual_errors != expected_errors:
        raise RuntimeError(
            "SAFETY CHECK FAILED: embedding audit and official evaluate report disagree. "
            f"audit_errors={actual_errors}, evaluate_report_errors={expected_errors}. "
            "No embedding interpretation should be made until prediction inputs/thresholds match."
        )

    class_idx = targets.argmax(axis=1)
    k = min(args.neighbors, len(emb) - 1)
    own = knn(emb, class_idx, k)

    rows = []
    for i, sample in enumerate(samples):
        category = "typical" if own[i] >= 0.8 else ("boundary" if own[i] >= 0.5 else "other_dominated")
        rows.append({
            "file": str(sample.get("file", "")),
            "group": str(sample.get("group", "")),
            "true_label": label_text(targets[i]),
            "pred_label": label_text(preds[i]),
            "exact_match": bool(exact[i]),
            "own_neighbor_fraction": float(own[i]),
            "embedding_category": category,
        })

    out = args.output_dir / "embedding_test_audit.csv"
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    cats = ("typical", "boundary", "other_dominated")
    per = {}
    for ci, name in enumerate(names):
        indices = np.where(class_idx == ci)[0]
        per[name] = {"samples": int(len(indices))}
        for cat in cats:
            per[name][cat] = int(sum(rows[j]["embedding_category"] == cat for j in indices))
        per[name]["errors"] = int((~exact[indices]).sum())

    error_categories = {
        cat: int(sum((not row["exact_match"]) and row["embedding_category"] == cat for row in rows))
        for cat in cats
    }
    confusions = {}
    for row in rows:
        if not row["exact_match"]:
            key = f"{row['true_label']} -> {row['pred_label']}"
            confusions[key] = confusions.get(key, 0) + 1

    summary = {
        "model": str(args.model),
        "reference_eval_report": str(args.eval_report),
        "safety_check": "PASS",
        "test_samples": len(rows),
        "neighbors": k,
        "per_source": per,
        "total_errors": actual_errors,
        "error_categories": error_categories,
        "confusions": confusions,
    }
    summary_path = args.output_dir / "embedding_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    error_rows = [row for row in rows if not row["exact_match"]]
    error_review = args.output_dir / "embedding_error_review.csv"
    with error_review.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(error_rows)

    print("\n========== READ THIS EMBEDDING AUDIT SUMMARY ==========")
    print(f"SAFETY_CHECK=PASS evaluate_errors={expected_errors} audit_errors={actual_errors}")
    print(f"model={args.model.name} test_samples={len(rows)} total_errors={actual_errors}")
    for name in names:
        d = per[name]
        print(
            f"{name}: samples={d['samples']} typical={d['typical']} "
            f"boundary={d['boundary']} other_dominated={d['other_dominated']} errors={d['errors']}"
        )
    print(
        f"error_locations: typical={error_categories['typical']} "
        f"boundary={error_categories['boundary']} "
        f"other_dominated={error_categories['other_dominated']}"
    )
    print("confusions: " + (
        ", ".join(f"{key}={value}" for key, value in sorted(confusions.items()))
        if confusions else "none"
    ))
    print(f"k_neighbors={k}")
    print("========================================================")
    print(f"audit_csv={out.resolve()}")
    print(f"summary={summary_path.resolve()}")
    print(f"error_review_csv={error_review.resolve()}")


if __name__ == "__main__":
    main()
