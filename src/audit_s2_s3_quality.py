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


def resample(x: np.ndarray, n: int) -> np.ndarray:
    if len(x) == n:
        return x.astype(np.float32)
    return np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(x)), x).astype(np.float32)


def feature(x: np.ndarray) -> np.ndarray:
    y = resample(x, 256)
    centered = y - y.mean()
    dynamic = max(float(np.ptp(y)), 1e-6)
    shape = centered / dynamic
    spec = resample(np.log1p(np.abs(np.fft.rfft(centered))).astype(np.float32), 96)
    stats = np.asarray([y.mean(), y.std(), y.min(), y.max(), np.ptp(y),
                        np.percentile(y, 5), np.median(y), np.percentile(y, 95)], dtype=np.float32)
    return np.concatenate([y, shape, spec, stats])


def load_group(root: Path):
    files, feats, errors = [], [], []
    for p in sorted(root.rglob("*.csv")):
        try:
            files.append(p)
            feats.append(feature(read_signal(p)))
        except Exception as exc:
            errors.append(f"{p}: {exc}")
    if not feats:
        raise ValueError(f"no valid csv under {root}")
    return files, np.vstack(feats), errors


def robust_z(x: np.ndarray) -> np.ndarray:
    med = np.median(x, axis=0)
    mad = np.median(np.abs(x-med), axis=0)
    scale = np.where(mad > 1e-6, 1.4826*mad, np.std(x, axis=0)+1e-6)
    z = (x-med)/scale
    # Prevent a handful of almost-constant dimensions dominating Euclidean distance.
    return np.clip(z, -8.0, 8.0).astype(np.float32)


def knn_profiles(z: np.ndarray, labels: np.ndarray, k: int, block: int = 128):
    n = len(z)
    own_fraction = np.empty(n, dtype=np.float32)
    kth_distance = np.empty(n, dtype=np.float32)
    nearest_distance = np.empty(n, dtype=np.float32)
    for start in range(0, n, block):
        stop = min(start+block, n)
        q = z[start:stop]
        # squared RMS distance; ordering is identical to RMS and avoids a huge 3-D tensor
        q2 = np.sum(q*q, axis=1, keepdims=True)
        z2 = np.sum(z*z, axis=1)[None, :]
        d2 = np.maximum(q2 + z2 - 2.0*(q @ z.T), 0.0) / z.shape[1]
        for local, global_i in enumerate(range(start, stop)):
            d2[local, global_i] = np.inf
        idx = np.argpartition(d2, kth=k-1, axis=1)[:, :k]
        selected = np.take_along_axis(d2, idx, axis=1)
        order = np.argsort(selected, axis=1)
        idx = np.take_along_axis(idx, order, axis=1)
        selected = np.take_along_axis(selected, order, axis=1)
        own_fraction[start:stop] = (labels[idx] == labels[start:stop, None]).mean(axis=1)
        nearest_distance[start:stop] = np.sqrt(selected[:, 0])
        kth_distance[start:stop] = np.sqrt(selected[:, -1])
    return own_fraction, nearest_distance, kth_distance


def classify(own_fraction: np.ndarray, kth_distance: np.ndarray, labels: np.ndarray):
    # Outlier threshold is learned within each class from local-neighborhood radius.
    thresholds = {}
    for label in (0, 1):
        values = kth_distance[labels == label]
        q1, q3 = np.percentile(values, [25, 75])
        thresholds[label] = float(q3 + 3.0*(q3-q1))  # conservative Tukey outer fence
    categories = []
    for frac, radius, label in zip(own_fraction, kth_distance, labels):
        if radius > thresholds[int(label)]:
            categories.append("isolated_outlier")
        elif frac < 0.5:
            categories.append("other_class_dominated")
        elif frac < 0.8:
            categories.append("class_boundary")
        else:
            categories.append("typical")
    return categories, thresholds


def main():
    p=argparse.ArgumentParser(description="Local-neighborhood S2/S3 overlap audit; NumPy only.")
    p.add_argument("--s2-dir",type=Path,required=True)
    p.add_argument("--s3-dir",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,default=Path("outputs/reports/s2_s3_audit_v2"))
    p.add_argument("--neighbors",type=int,default=30)
    args=p.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)

    f2,x2,e2=load_group(args.s2_dir); f3,x3,e3=load_group(args.s3_dir)
    x=np.vstack([x2,x3]); labels=np.r_[np.zeros(len(x2),dtype=int),np.ones(len(x3),dtype=int)]
    z=robust_z(x)
    k=min(args.neighbors,len(z)-1)
    own,nearest,radius=knn_profiles(z,labels,k)
    categories,thresholds=classify(own,radius,labels)

    files=f2+f3
    rows=[]
    for i,(path,label) in enumerate(zip(files,labels)):
        rows.append({"file":str(path.resolve()),"label":"S2" if label==0 else "S3",
                     "category":categories[i],"own_neighbor_fraction":float(own[i]),
                     "nearest_distance":float(nearest[i]),"kth_neighbor_distance":float(radius[i])})
    priority={"isolated_outlier":0,"other_class_dominated":1,"class_boundary":2,"typical":3}
    rows.sort(key=lambda r:(priority[r["category"]],r["own_neighbor_fraction"]))
    out=args.output_dir/"s2_s3_local_audit.csv"
    with out.open("w",encoding="utf-8",newline="") as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    cats=("typical","class_boundary","other_class_dominated","isolated_outlier")
    counts={}
    for label in ("S2","S3"):
        rr=[r for r in rows if r["label"]==label]
        counts[label]={cat:sum(r["category"]==cat for r in rr) for cat in cats}
    summary={"s2_files":len(f2),"s3_files":len(f3),"neighbors":k,"counts":counts,
             "outlier_radius_thresholds":{"S2":thresholds[0],"S3":thresholds[1]},
             "parse_errors":e2+e3,
             "method_note":"Local kNN diagnostic. Categories are review aids only; never auto-delete or relabel."}
    summary_path=args.output_dir/"summary_v2.json"
    summary_path.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

    print("\n========== READ THIS S2/S3 AUDIT SUMMARY ==========")
    print(f"S2 files={len(f2)} typical={counts['S2']['typical']} boundary={counts['S2']['class_boundary']} other_dominated={counts['S2']['other_class_dominated']} isolated={counts['S2']['isolated_outlier']}")
    print(f"S3 files={len(f3)} typical={counts['S3']['typical']} boundary={counts['S3']['class_boundary']} other_dominated={counts['S3']['other_class_dominated']} isolated={counts['S3']['isolated_outlier']}")
    print(f"k_neighbors={k} parse_errors={len(e2)+len(e3)}")
    print("IMPORTANT: boundary/other_dominated/isolated are diagnostic candidates, NOT files to delete.")
    print("====================================================")
    print(f"audit_csv={out.resolve()}")
    print(f"summary={summary_path.resolve()}")


if __name__=="__main__":
    main()
