from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from noise_source_runtime import (  # noqa: E402
    InferenceSession,
    RuntimeInferenceError,
    export_prediction_result,
)
from src.inference.single_csv_predictor import print_prediction_result  # noqa: E402


def parse_threshold(text: str) -> float | list[float]:
    values = [item.strip() for item in text.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("--threshold cannot be empty")
    try:
        parsed = [float(value) for value in values]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--threshold must be one number or a comma-separated list"
        ) from exc
    return parsed[0] if len(parsed) == 1 else parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run strict single-CSV noise-source inference and write a contract report."
    )
    parser.add_argument(
        "--csv", type=Path, required=True, help="CSV containing a DATA section."
    )
    parser.add_argument(
        "--checkpoint", type=Path, required=True, help="PyTorch checkpoint path."
    )
    parser.add_argument(
        "--threshold",
        type=parse_threshold,
        help="One threshold or a comma-separated threshold per label.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Inference device: auto, cpu, cuda, cuda:0, ...",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("outputs/inference_contract"),
        help="Directory for prediction JSON and inference_contract.md.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Exact training YAML/JSON; required for pure state_dict checkpoints.",
    )
    parser.add_argument(
        "--allow-legacy-csv",
        action="store_true",
        help="Explicitly allow legacy CSV files without a DATA marker.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        with InferenceSession.load_model(
            args.checkpoint,
            config_path=args.config,
            threshold=args.threshold,
            device=args.device,
        ) as session:
            result = session.predict_file(
                args.csv,
                require_data_marker=not args.allow_legacy_csv,
            )
            result = export_prediction_result(
                result,
                session,
                args.report_dir,
            )
    except RuntimeInferenceError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
    print_prediction_result(result)


if __name__ == "__main__":
    main()
