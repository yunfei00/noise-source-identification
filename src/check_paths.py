from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a YAML mapping: {config_path}")
    return config


def _csv_count(path: Path) -> int:
    if not path.exists() or not path.is_dir():
        return 0
    return sum(1 for csv_path in path.rglob("*.csv") if csv_path.is_file())


def _children(path: Path) -> list[str]:
    if not path.exists() or not path.is_dir():
        return []
    return sorted(child.name for child in path.iterdir() if child.is_dir())


def check_paths(config_path: str | Path) -> None:
    config = load_config(config_path)
    real_data_config = config.get("real_data", {})
    paths_config = config.get("paths", {})
    report_dir = Path(paths_config.get("report_dir", "outputs/reports"))

    single_dir = Path(real_data_config.get("single_dir", "data/single"))
    combo_dir = Path(real_data_config.get("combo_dir", "data/real_dataset"))
    index_file = Path(real_data_config.get("index_file", report_dir / "real_dataset_index.csv"))
    split_file = Path(real_data_config.get("split_file", report_dir / "real_dataset_split.csv"))

    cwd = Path.cwd()
    print(f"cwd={cwd}")
    print(f"single_dir={single_dir.resolve()}")
    print(f"combo_dir={combo_dir.resolve()}")
    print(f"index_file={index_file}")
    print(f"split_file={split_file}")

    single_exists = single_dir.exists() and single_dir.is_dir()
    combo_exists = combo_dir.exists() and combo_dir.is_dir()
    print(f"single_dir_exists={single_exists}")
    if not single_exists:
        print(f"warning: single_dir does not exist or is not a directory: {single_dir}")
    print(f"combo_dir_exists={combo_exists}")
    if not combo_exists:
        print(f"warning: combo_dir does not exist or is not a directory: {combo_dir}")

    print(f"single_csv_count={_csv_count(single_dir)}")
    print(f"combo_csv_count={_csv_count(combo_dir)}")
    print(f"single_children={_children(single_dir)}")
    print(f"combo_children={_children(combo_dir)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check configured real dataset paths without failing on missing data directories.")
    parser.add_argument("--config", type=Path, default=Path("configs/train.yaml"), help="Training config path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    check_paths(args.config)


if __name__ == "__main__":
    main()
