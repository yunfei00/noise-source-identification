from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

TOKEN_SPLIT_RE = re.compile(r"[,\s]+")


def _is_data_line(line: str) -> bool:
    return line.strip().lower() == "data"


def _split_tokens(line: str) -> list[str]:
    return [token for token in TOKEN_SPLIT_RE.split(line.strip()) if token]


def _parse_float(token: str) -> float | None:
    try:
        value = float(token)
    except ValueError:
        return None
    if not math.isfinite(value):
        return None
    return value


def read_time_amplitude_csv(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    if not csv_path.is_file():
        raise ValueError(f"Expected a CSV file, got: {csv_path}")

    try:
        lines = csv_path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    except Exception as exc:
        raise ValueError(f"Failed to read CSV file {csv_path}: {exc}") from exc

    data_line_index = next((index for index, line in enumerate(lines) if _is_data_line(line)), None)
    if data_line_index is None:
        raise ValueError("DATA line not found")

    times: list[float] = []
    amplitudes: list[float] = []
    for line in lines[data_line_index + 1 :]:
        if not line.strip():
            continue
        tokens = _split_tokens(line)
        if len(tokens) < 2:
            continue
        time_value = _parse_float(tokens[0])
        amplitude_value = _parse_float(tokens[1])
        if time_value is None or amplitude_value is None:
            continue
        times.append(time_value)
        amplitudes.append(amplitude_value)

    if not times:
        raise ValueError("No valid two-column numeric rows found below DATA")

    return (
        np.asarray(times, dtype=np.float64),
        np.asarray(amplitudes, dtype=np.float64),
    )


def iter_csv_files(input_dir: str | Path, max_files: int | None = None) -> list[Path]:
    root = Path(input_dir)
    if not root.exists():
        raise FileNotFoundError(f"Input directory not found: {root}")
    if not root.is_dir():
        raise ValueError(f"Input path must be a directory: {root}")
    files = sorted(path for path in root.rglob("*.csv") if path.is_file())
    if max_files is not None:
        if max_files < 0:
            raise ValueError("--max-files must be >= 0")
        files = files[:max_files]
    return files


def save_csv_plot(
    csv_path: Path,
    input_dir: Path,
    output_dir: Path,
    *,
    dpi: int,
    image_format: str,
) -> Path:
    times, amplitudes = read_time_amplitude_csv(csv_path)
    relative_path = csv_path.relative_to(input_dir)
    output_path = output_dir / relative_path.with_suffix(f".{image_format}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(times, amplitudes, linewidth=0.8)
    ax.set_title(relative_path.as_posix())
    ax.set_xlabel("Time")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, format=image_format)
    plt.close(fig)
    return output_path


def convert_csv_folder(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    max_files: int | None = None,
    dpi: int = 120,
    image_format: str = "png",
) -> dict[str, int | str]:
    input_root = Path(input_dir)
    output_root = Path(output_dir)
    image_format = image_format.lower().lstrip(".")
    if not image_format:
        raise ValueError("--format cannot be empty")
    if dpi <= 0:
        raise ValueError("--dpi must be positive")

    csv_files = iter_csv_files(input_root, max_files)
    success_count = 0
    failed_count = 0

    for csv_path in csv_files:
        try:
            save_csv_plot(
                csv_path,
                input_root,
                output_root,
                dpi=dpi,
                image_format=image_format,
            )
            success_count += 1
        except Exception as exc:
            failed_count += 1
            print(f"warning: failed to convert {csv_path}: {exc}")

    report = {
        "total_csv": len(csv_files),
        "success_count": success_count,
        "failed_count": failed_count,
        "output_dir": str(output_root),
    }
    print(f"total_csv={report['total_csv']}")
    print(f"success_count={report['success_count']}")
    print(f"failed_count={report['failed_count']}")
    print(f"output_dir={report['output_dir']}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert recursive CSV time-domain data into images.")
    parser.add_argument("--input", type=Path, default=Path("data/single"), help="Input CSV directory.")
    parser.add_argument("--output", type=Path, default=Path("outputs/csv_images"), help="Output image directory.")
    parser.add_argument("--max-files", type=int, help="Optional maximum number of CSV files to convert.")
    parser.add_argument("--dpi", type=int, default=120, help="Output image DPI.")
    parser.add_argument("--format", default="png", help="Output image format, default: png.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert_csv_folder(
        input_dir=args.input,
        output_dir=args.output,
        max_files=args.max_files,
        dpi=args.dpi,
        image_format=args.format,
    )


if __name__ == "__main__":
    main()
