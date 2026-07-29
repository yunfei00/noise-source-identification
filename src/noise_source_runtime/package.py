from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from src.noise_source_runtime import RUNTIME_VERSION
from src.noise_source_runtime.checkpoint import (
    checkpoint_mapping,
    load_checkpoint_artifact,
)
from src.noise_source_runtime.exceptions import ModelPackageError
from src.noise_source_runtime.preprocessing import preprocessing_contract

PACKAGE_SCHEMA_VERSION = "1.0"
REQUIRED_PACKAGE_FILES = (
    "model.pt",
    "manifest.json",
    "metrics.json",
    "preprocess.json",
    "labels.json",
    "README.md",
    "sha256.txt",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _labels(payload: Mapping[str, Any], config: Mapping[str, Any]) -> list[str]:
    for key in ("class_names", "labels", "classes"):
        value = payload.get(key)
        if (
            isinstance(value, list)
            and value
            and all(isinstance(item, str) for item in value)
        ):
            return list(value)
    configured = config.get("data", {}).get("class_names")
    if (
        isinstance(configured, list)
        and configured
        and all(isinstance(item, str) for item in configured)
    ):
        return list(configured)
    raise ModelPackageError(
        "Cannot build a model package without an explicit checkpoint/config label order"
    )


def build_model_package(
    checkpoint_path: str | Path,
    model_version: str,
    *,
    output_dir: str | Path,
    model_name: str | None = None,
    metrics_path: str | Path | None = None,
    config: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Build a portable, hash-verifiable inference model package."""
    checkpoint = load_checkpoint_artifact(checkpoint_path, map_location="cpu")
    payload = checkpoint_mapping(checkpoint)
    checkpoint_config = payload.get("config")
    if checkpoint_config is None:
        checkpoint_config = config
    if not isinstance(checkpoint_config, Mapping):
        raise ModelPackageError(
            "Checkpoint is missing config; pass the exact training config"
        )
    labels = _labels(payload, checkpoint_config)
    package_dir = Path(output_dir)
    if package_dir.exists() and any(package_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Model package directory is not empty: {package_dir}; use overwrite=True"
        )
    package_dir.mkdir(parents=True, exist_ok=True)

    model_path = package_dir / "model.pt"
    shutil.copy2(checkpoint.path, model_path)
    model_hash = sha256_file(model_path)
    prediction_mode = str(
        payload.get(
            "prediction_mode",
            checkpoint_config.get("model", {})
            .get("prediction", {})
            .get("mode", "multilabel"),
        )
    )
    monitor_name = str(payload.get("monitor_name") or "unknown")
    best_metric = payload.get("best_metric")
    metrics: dict[str, Any] = {
        "best_epoch": payload.get("epoch"),
        "best_metric_name": monitor_name,
        "best_metric_value": best_metric,
    }
    if metrics_path is not None:
        with Path(metrics_path).open("r", encoding="utf-8") as handle:
            supplied_metrics = json.load(handle)
        if not isinstance(supplied_metrics, dict):
            raise ModelPackageError("metrics JSON must contain an object")
        metrics["training_metrics"] = supplied_metrics

    preprocess = preprocessing_contract(checkpoint_config)
    manifest = {
        "package_schema_version": PACKAGE_SCHEMA_VERSION,
        "model_name": model_name or checkpoint.path.stem,
        "model_version": str(model_version),
        "runtime_version": RUNTIME_VERSION,
        "framework": "PyTorch",
        "framework_version": torch.__version__,
        "task_type": "multi_label_noise_source_identification",
        "prediction_mode": prediction_mode,
        "labels": labels,
        "input_contract": preprocess,
        "checkpoint_sha256": model_hash,
        "training_git_commit": payload.get("training_git_commit")
        or "unknown (legacy checkpoint)",
        "created_at": _utc_now(),
        "best_epoch": payload.get("epoch"),
        "best_metric_name": monitor_name,
        "best_metric_value": best_metric,
    }
    _write_json(package_dir / "manifest.json", manifest)
    _write_json(package_dir / "metrics.json", metrics)
    _write_json(package_dir / "preprocess.json", preprocess)
    _write_json(package_dir / "labels.json", {"labels": labels})
    (package_dir / "README.md").write_text(
        f"""# {manifest["model_name"]} {manifest["model_version"]}

This directory is a noise-source runtime model package.

- Runtime version: `{RUNTIME_VERSION}`
- Prediction mode: `{prediction_mode}`
- Labels: `{labels}`
- Model file: `model.pt`
- Integrity: run `python scripts/build_model_package.py --verify {package_dir}`

Load it from Python:

```python
from noise_source_runtime import InferenceSession

session = InferenceSession.load_model("model.pt")
```
""",
        encoding="utf-8",
    )

    hashed_names = [name for name in REQUIRED_PACKAGE_FILES if name != "sha256.txt"]
    checksum_lines = [
        f"{sha256_file(package_dir / name)}  {name}" for name in hashed_names
    ]
    (package_dir / "sha256.txt").write_text(
        "\n".join(checksum_lines) + "\n",
        encoding="utf-8",
    )
    verification = verify_model_package(package_dir)
    return {
        "package_dir": str(package_dir),
        "manifest": manifest,
        "verification": verification,
    }


def verify_model_package(package_dir: str | Path) -> dict[str, Any]:
    directory = Path(package_dir)
    missing = [
        name for name in REQUIRED_PACKAGE_FILES if not (directory / name).is_file()
    ]
    if missing:
        raise ModelPackageError(f"Model package is missing required files: {missing}")
    checksums: dict[str, str] = {}
    for line_number, line in enumerate(
        (directory / "sha256.txt").read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            raise ModelPackageError(f"Invalid sha256.txt line {line_number}: {line!r}")
        expected_hash, filename = parts
        filename = filename.strip()
        if Path(filename).name != filename or filename == "sha256.txt":
            raise ModelPackageError(
                f"Unsafe or unsupported checksum filename: {filename!r}"
            )
        target = directory / filename
        if not target.is_file():
            raise ModelPackageError(f"Checksummed file is missing: {filename}")
        actual_hash = sha256_file(target)
        if actual_hash != expected_hash:
            raise ModelPackageError(
                f"SHA256 mismatch for {filename}: expected {expected_hash}, got {actual_hash}"
            )
        checksums[filename] = actual_hash

    expected_hashed = {name for name in REQUIRED_PACKAGE_FILES if name != "sha256.txt"}
    if set(checksums) != expected_hashed:
        raise ModelPackageError(
            "sha256.txt coverage mismatch: "
            f"expected={sorted(expected_hashed)} actual={sorted(checksums)}"
        )
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("package_schema_version") != PACKAGE_SCHEMA_VERSION:
        raise ModelPackageError(
            f"Unsupported package_schema_version: {manifest.get('package_schema_version')}"
        )
    if manifest.get("checkpoint_sha256") != checksums["model.pt"]:
        raise ModelPackageError("manifest.checkpoint_sha256 does not match model.pt")
    labels_payload = json.loads((directory / "labels.json").read_text(encoding="utf-8"))
    if labels_payload.get("labels") != manifest.get("labels"):
        raise ModelPackageError("labels.json does not match manifest.labels")
    return {
        "valid": True,
        "package_dir": str(directory),
        "package_schema_version": PACKAGE_SCHEMA_VERSION,
        "verified_files": sorted(checksums),
        "checkpoint_sha256": checksums["model.pt"],
    }


__all__ = [
    "PACKAGE_SCHEMA_VERSION",
    "REQUIRED_PACKAGE_FILES",
    "build_model_package",
    "sha256_file",
    "verify_model_package",
]
