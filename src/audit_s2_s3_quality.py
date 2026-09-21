from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def read_signal(path: Path) -> np.ndarray:
    rows = None
    for enc in ("utf-8-sig", "gb18030"):
        try:
            with path.open("r", encoding=enc, newline="") as f:
                rows = list(csv.reader(f))
            break
        except UnicodeDecodeError:
            pass
    if rows is None:
        raise ValueError(f"cannot decode {path}")
    start = 0
    for i, row in enumerate(rows):
        if any(str(x).strip().upper() == "DATA" for x in row):
            start = i + 1
            break
    vals = []
    for row in rows[start:]:
        if not row:
            continue
        candidates = [row[1]] if len(row) >= 2 and start else reversed(row)
        for cell in candidates:
            try:
                v = float(str(cell).strip())
                if np.isfinite(v):
                    vals.append(v)
                    break
            except ValueError:
                continue
    if not vals:
        raise ValueError(f"no numeric signal in {path}")
    return np.asarray(vals, dtype=np.float32)


def resample(x: np.ndarray, n: int = 256) -> np.ndarray:
    if len(x) == n:
        return x.astype(np.float32)
    old = np.linspace(0.0, 1.0, len(x))
    new = np.linspace(0.0, 1.0, n)
    return np.interp(new, old, x).astype(np.float32)


def feature(x: np.ndarray) -> np.ndarray:
    y = resample(x)
    # Preserve absolute/raw-scale information while also capturing shape and spectrum.
    raw = y
    centered = y - y.mean()
    scale = float(np.ptp(y))
    shape = centered / max(scale, 1e-6)
    spec = np.log1p(np.abs(np.fft.rfft(centered))).astype(np.float32)
    spec = resample(spec, 96)
    stats = np.asarray([
        y.mean(), y.std(), y.min(), y.max(), np.ptp(y),
        np.percentile(y, 5), np.percentile(y, 50), np.percentile(y, 95)
    ], dtype=np.float32)
    return np.concatenate([raw, shape, spec, stats])


def robust_standardize(train: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    med = np.median(train, axis=0)
    mad = np.median(np.abs(train - med), axis=0)
    scale = np.where(mad > 1e-6, 1.4826 * mad, np.std(train, axis=0) + 1e-6)
    return (train - med) / scale, (x - med) / scale


def load_group(root: Path) -> tuple[list[Path], list[np.ndarray], list[str]]:
    files, feats, errors = [], [], []
    for p in sorted(root.rglob("*.csv")):
        try:
            files.append(p)
            feats.append(feature(read_signal(p)))
        except Exception as exc:
            errors.append(f"{p}: {exc}")
    if not feats:
        raise ValueError(f"no valid csv under {root}")
    return files, feats, errors


def distances(ref: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((x[:, None, :] - ref[None, :, :]) ** 2, axis=2))


def main() -> None:
    p = argparse.ArgumentParser(description="Audit S2/S3 quality and class overlap without torch.")
    p.add_argument("--s2-dir", type=Path, required=True)
    p.add_argument("--s3-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/reports/s2_s3_audit"))
    p.add_argument("--top", type=int, default=100, help="Number of highest-risk samples to highlight.")
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    f2, x2_list, e2 = load_group(args.s2_dir)
    f3, x3_list, e3 = load_group(args.s3_dir)
    x2, x3 = np.vstack(x2_list), np.vstack(x3_list)
    allx = np.vstack([x2, x3])
    zall, _ = robust_standardize(allx, allx)
    z2, z3 = zall[:len(x2)], zall[len(x2):]

    # Robust class centers and within-class radii.
    c2, c3 = np.median(z2, axis=0), np.median(z3, axis=0)
    d22 = np.sqrt(np.mean((z2-c2)**2, axis=1)); d23 = np.sqrt(np.mean((z2-c3)**2, axis=1))
    d33 = np.sqrt(np.mean((z3-c3)**2, axis=1)); d32 = np.sqrt(np.mean((z3-c2)**2, axis=1))
    q2 = float(np.percentile(d22, 95)); q3 = float(np.percentile(d33, 95))

    rows = []
    for label, files, own, other, own95 in [
        ("S2", f2, d22, d23, q2), ("S3", f3, d33, d32, q3)
    ]:
        for path, od, xd in zip(files, own, other):
            margin = float(xd-od)  # positive = closer to own class
            if od > own95 and xd < od:
                risk = "high"
                reason = "own_outlier_and_closer_to_other"
            elif xd < od:
                risk = "high"
                reason = "closer_to_other_class"
            elif od > own95:
                risk = "medium"
                reason = "within_class_outlier"
            elif margin < 0.15 * max(float(od), 1e-6):
                risk = "medium"
                reason = "class_boundary"
            else:
                risk = "low"
                reason = "typical"
            rows.append({"file":str(path.resolve()),"label":label,"own_distance":float(od),
                         "other_distance":float(xd),"margin_other_minus_own":margin,
                         "risk":risk,"reason":reason})

    order={"high":0,"medium":1,"low":2}
    rows.sort(key=lambda r:(order[r["risk"]], r["margin_other_minus_own"]))
    out_csv=args.output_dir/"s2_s3_audit.csv"
    with out_csv.open("w",encoding="utf-8",newline="") as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    counts={}
    for label in ("S2","S3"):
        rr=[r for r in rows if r["label"]==label]
        counts[label]={k:sum(r["risk"]==k for r in rr) for k in ("high","medium","low")}
    top=[r for r in rows if r["risk"]!="low"][:args.top]
    summary={"s2_files":len(f2),"s3_files":len(f3),"parse_errors":e2+e3,
             "risk_counts":counts,"s2_own_distance_p95":q2,"s3_own_distance_p95":q3,
             "top_candidates":top,"method_note":"Candidates are for manual review only; do not auto-delete. Distances combine raw dB scale, shape and spectrum."}
    (args.output_dir/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(f"S2 files={len(f2)} risk={counts['S2']}")
    print(f"S3 files={len(f3)} risk={counts['S3']}")
    print(f"audit_csv={out_csv.resolve()}")
    print(f"summary={(args.output_dir/'summary.json').resolve()}")
    print("IMPORTANT: high/medium are review candidates, not proof of bad acquisition and must not be auto-deleted.")


if __name__ == "__main__":
    main()
