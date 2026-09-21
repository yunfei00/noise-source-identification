from __future__ import annotations

import argparse
import copy
import csv
import json
import subprocess
import sys
from pathlib import Path

from src.train import load_config


def _counts(text: str) -> list[int]:
    values = sorted({int(x.strip()) for x in text.split(",") if x.strip()})
    if not values:
        raise argparse.ArgumentTypeError("counts cannot be empty")
    return values


def main() -> None:
    p = argparse.ArgumentParser(description="Run sample-count ablation with fixed val/test splits.")
    p.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    p.add_argument("--split-dir", type=Path, default=Path("outputs/reports/ablation_800M"))
    p.add_argument("--counts", type=_counts, default=_counts("100,200,500,1000,1500"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/ablation_runs/800M"))
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    import yaml
    base = load_config(args.config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    generated = []
    for count in args.counts:
        split_file = args.split_dir / f"real_dataset_split_ablation_{count}.csv"
        if not split_file.exists():
            raise FileNotFoundError(f"Missing ablation split: {split_file}")
        cfg = copy.deepcopy(base)
        cfg.setdefault("training_data", {})["mode"] = "real_only"
        cfg.setdefault("real_data", {})["split_file"] = str(split_file.resolve())
        cfg["real_data"]["dataset_root"] = "."
        cfg.setdefault("paths", {})["checkpoint_dir"] = str((args.output_dir / f"n{count}" / "checkpoints").resolve())
        cfg["paths"]["report_dir"] = str((args.output_dir / f"n{count}" / "reports").resolve())
        cfg.setdefault("cache", {})["enabled"] = False
        cfg_path = args.output_dir / f"train_n{count}.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
        generated.append(str(cfg_path))
        cmd = [sys.executable, "-m", "src.train", "--config", str(cfg_path)]
        print(f"\n===== N={count} =====")
        print("command=" + subprocess.list2cmdline(cmd))
        if not args.dry_run:
            subprocess.run(cmd, check=True)

    manifest = args.output_dir / "run_manifest.json"
    manifest.write_text(json.dumps({"configs": generated, "counts": args.counts}, indent=2) + "\n", encoding="utf-8")
    print(f"manifest={manifest.resolve()}")


if __name__ == "__main__":
    main()
