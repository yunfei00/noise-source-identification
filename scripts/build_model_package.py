from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml  # noqa: E402

from noise_source_runtime.package import (  # noqa: E402
    build_model_package,
    verify_model_package,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build or verify a versioned noise-source model package."
    )
    parser.add_argument(
        "--checkpoint", type=Path, help="Training checkpoint, e.g. best.pt."
    )
    parser.add_argument("--version", help="Semantic/application model version.")
    parser.add_argument("--output-dir", type=Path, help="Package output directory.")
    parser.add_argument(
        "--model-name", help="Stable model name (default: checkpoint stem)."
    )
    parser.add_argument(
        "--metrics", type=Path, help="Optional additional metrics JSON."
    )
    parser.add_argument(
        "--config", type=Path, help="Exact YAML/JSON for a legacy checkpoint."
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite known package files."
    )
    parser.add_argument(
        "--verify", type=Path, help="Verify an existing package and exit."
    )
    return parser.parse_args()


def _load_config(path: Path | None) -> dict | None:
    if path is None:
        return None
    with path.open("r", encoding="utf-8") as handle:
        if path.suffix.lower() == ".json":
            payload = json.load(handle)
        else:
            payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return payload


def main() -> None:
    args = parse_args()
    if args.verify is not None:
        print(
            json.dumps(verify_model_package(args.verify), ensure_ascii=False, indent=2)
        )
        return
    missing = [
        name
        for name, value in (
            ("--checkpoint", args.checkpoint),
            ("--version", args.version),
            ("--output-dir", args.output_dir),
        )
        if value is None
    ]
    if missing:
        raise SystemExit(f"Building a package requires: {', '.join(missing)}")
    result = build_model_package(
        args.checkpoint,
        args.version,
        output_dir=args.output_dir,
        model_name=args.model_name,
        metrics_path=args.metrics,
        config=_load_config(args.config),
        overwrite=args.overwrite,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
