from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

_TOKEN_SPLIT_RE = re.compile(r"[,\\s]+")

def _read_signal_csv(path: Path) -> np.ndarray:
    """Standalone NumPy-only parser; intentionally does not import src.features."""
    last_error = None
    lines = None
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            lines = path.read_text(encoding=encoding).splitlines()
            break
        except UnicodeDecodeError as exc:
            last_error = exc
    if lines is None:
        raise ValueError(f"Unable to decode CSV: {path}: {last_error}")

    data_index = next(
        (i for i, line in enumerate(lines) if line.strip().lower() == "data"),
        None,
    )
    candidates = lines[data_index + 1:] if data_index is not None else lines
    values = []
    for line in candidates:
        tokens = [t for t in _TOKEN_SPLIT_RE.split(line.strip()) if t]
        if not tokens:
            continue
        token = tokens[1] if data_index is not None and len(tokens) >= 2 else tokens[-1]
        try:
            value = float(token)
        except ValueError:
            continue
        if np.isfinite(value):
            values.append(value)
    if not values:
        raise ValueError(f"No valid numeric samples found in {path}")
    return np.asarray(values, dtype=np.float32)


def _fix_length(x: np.ndarray, length: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if x.size >= length:
        return x[:length]
    fill = float(np.median(x)) if x.size else 0.0
    return np.pad(x, (0, length - x.size), constant_values=fill).astype(np.float32)


def _standardize_rows(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64, copy=False)
    x = x - x.mean(axis=1, keepdims=True)
    scale = x.std(axis=1, keepdims=True)
    scale[scale < 1e-12] = 1.0
    return x / scale


def _feature_matrix(signals: np.ndarray, bins: int = 128) -> np.ndarray:
    # Compact NumPy-only feature representation: downsampled standardized trace
    # plus normalized rFFT magnitude. This intentionally avoids torch/scipy.
    z = _standardize_rows(signals)
    n, length = z.shape
    edges = np.linspace(0, length, bins + 1, dtype=int)
    trace = np.stack(
        [np.stack([row[edges[i]:edges[i + 1]].mean() for i in range(bins)]) for row in z]
    )
    spec = np.abs(np.fft.rfft(z, axis=1))
    spec = np.log1p(spec)
    spec_bins = min(bins, spec.shape[1])
    spec = spec[:, :spec_bins]
    spec /= np.linalg.norm(spec, axis=1, keepdims=True) + 1e-12
    trace /= np.linalg.norm(trace, axis=1, keepdims=True) + 1e-12
    return np.concatenate([trace, spec], axis=1).astype(np.float32)


def _nearest_similarity(features: np.ndarray, block: int = 256) -> np.ndarray:
    f = features.astype(np.float64, copy=False)
    f /= np.linalg.norm(f, axis=1, keepdims=True) + 1e-12
    result = np.full(len(f), -1.0, dtype=np.float64)
    for start in range(0, len(f), block):
        stop = min(start + block, len(f))
        sims = f[start:stop] @ f.T
        rows = np.arange(stop - start)
        sims[rows, np.arange(start, stop)] = -1.0
        result[start:stop] = np.max(sims, axis=1)
    return result


def _pca_dimensions(features: np.ndarray) -> dict[str, int]:
    x = features.astype(np.float64, copy=False)
    x -= x.mean(axis=0, keepdims=True)
    if len(x) < 2:
        return {"pca_dim_90": 0, "pca_dim_95": 0, "pca_dim_99": 0}
    singular = np.linalg.svd(x, full_matrices=False, compute_uv=False)
    variance = singular * singular
    total = float(variance.sum())
    if total <= 1e-20:
        return {"pca_dim_90": 0, "pca_dim_95": 0, "pca_dim_99": 0}
    cumulative = np.cumsum(variance) / total
    return {
        "pca_dim_90": int(np.searchsorted(cumulative, 0.90) + 1),
        "pca_dim_95": int(np.searchsorted(cumulative, 0.95) + 1),
        "pca_dim_99": int(np.searchsorted(cumulative, 0.99) + 1),
    }


def _coverage_curve(features: np.ndarray, seed: int) -> dict[str, float]:
    n = len(features)
    if n < 2:
        return {}
    f = features.astype(np.float64, copy=False)
    f /= np.linalg.norm(f, axis=1, keepdims=True) + 1e-12
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    checkpoints = sorted(set(min(n, x) for x in (100, 200, 500, 1000, 1500, 2000, n)))
    result: dict[str, float] = {}
    for count in checkpoints:
        reference = f[order[:count]]
        best = np.full(n, -1.0)
        for start in range(0, count, 256):
            best = np.maximum(best, np.max(f @ reference[start:start + 256].T, axis=1))
        result[f"coverage_{count}"] = round(float(np.mean(best)), 6)
    return result


def analyze(input_dir: Path, output_dir: Path, seed: int = 42, feature_bins: int = 128) -> dict[str, Any]:
    files = sorted(p for p in input_dir.rglob("*.csv") if p.is_file())
    if not files:
        raise ValueError(f"No CSV files found under: {input_dir}")

    parsed: list[tuple[Path, np.ndarray]] = []
    failed: list[dict[str, str]] = []
    hashes: dict[str, list[str]] = {}
    lengths: list[int] = []
    for path in files:
        try:
            values = _read_signal_csv(path).reshape(-1)
            if values.size == 0 or not np.all(np.isfinite(values)):
                raise ValueError("empty or non-finite signal")
            parsed.append((path, values))
            lengths.append(int(values.size))
            digest = hashlib.sha1(values.tobytes()).hexdigest()
            hashes.setdefault(digest, []).append(str(path))
        except Exception as exc:
            failed.append({"file": str(path), "error": str(exc)})

    if not parsed:
        raise ValueError("All CSV files failed to parse")

    target_length = int(np.median(lengths))
    signals = np.stack([_fix_length(x, target_length) for _, x in parsed])
    features = _feature_matrix(signals, bins=feature_bins)
    nearest = _nearest_similarity(features)

    duplicate_groups = [v for v in hashes.values() if len(v) > 1]
    exact_duplicate_files = sum(len(v) - 1 for v in duplicate_groups)

    report: dict[str, Any] = {
        "input": str(input_dir.resolve()),
        "files_found": len(files),
        "files_parsed": len(parsed),
        "parse_failed": len(failed),
        "signal_length_used": target_length,
        "exact_duplicate_groups": len(duplicate_groups),
        "exact_duplicate_extra_files": exact_duplicate_files,
        "exact_duplicate_ratio": round(exact_duplicate_files / len(parsed), 6),
        "nearest_similarity": {
            "mean": round(float(np.mean(nearest)), 6),
            "median": round(float(np.median(nearest)), 6),
            "p95": round(float(np.percentile(nearest, 95)), 6),
            "p99": round(float(np.percentile(nearest, 99)), 6),
            "ratio_ge_0_99": round(float(np.mean(nearest >= 0.99)), 6),
            "ratio_ge_0_995": round(float(np.mean(nearest >= 0.995)), 6),
            "ratio_ge_0_999": round(float(np.mean(nearest >= 0.999)), 6),
        },
        **_pca_dimensions(features),
        "coverage_curve": _coverage_curve(features, seed),
        "failed_files": failed,
        "duplicate_groups": duplicate_groups,
        "method_note": (
            "NumPy-only diagnostic. Similarity uses compact standardized trace + rFFT magnitude features; "
            "coverage is descriptive, not an automatic sample-count decision."
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "redundancy_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("========== COPY THIS SUMMARY ==========")
    print(f"input={report['input']}")
    print(f"files={report['files_found']} parsed={report['files_parsed']} failed={report['parse_failed']}")
    print(
        f"exact_duplicates={report['exact_duplicate_extra_files']} "
        f"ratio={report['exact_duplicate_ratio']}"
    )
    ns = report["nearest_similarity"]
    print(
        f"nearest_similarity: mean={ns['mean']} median={ns['median']} "
        f"p95={ns['p95']} p99={ns['p99']}"
    )
    print(
        f"similarity_ratio: >=0.99={ns['ratio_ge_0_99']} "
        f">=0.995={ns['ratio_ge_0_995']} >=0.999={ns['ratio_ge_0_999']}"
    )
    print(
        f"pca_dims: 90%={report['pca_dim_90']} "
        f"95%={report['pca_dim_95']} 99%={report['pca_dim_99']}"
    )
    print("coverage=" + json.dumps(report["coverage_curve"], ensure_ascii=False))
    print(f"report={report_path.resolve()}")
    print("=======================================")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze redundancy/diversity of CSV captures using NumPy only (no torch/scipy)."
    )
    parser.add_argument("--input", type=Path, required=True, help="Directory containing CSV captures.")
    parser.add_argument("--output", type=Path, default=Path("outputs/redundancy_analysis"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--feature-bins", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analyze(args.input, args.output, seed=args.seed, feature_bins=args.feature_bins)


if __name__ == "__main__":
    main()
