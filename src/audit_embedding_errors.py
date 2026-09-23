from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.dataset import RealCsvDataset
from src.infer import load_checkpoint
from src.model_cnn import build_model
from src.noise_source_runtime.device import resolve_device


def label_text(x: np.ndarray) -> str:
    return "[" + ",".join(str(int(v)) for v in x.tolist()) + "]"


def collect(model, loader, device):
    embeddings=[]; probs=[]; targets=[]
    model.eval()
    with torch.no_grad():
        for x,y in loader:
            x=x.to(device)
            emb=model.encode(x)
            logits=model.classifier(emb)
            embeddings.append(emb.cpu().numpy())
            probs.append(torch.sigmoid(logits).cpu().numpy())
            targets.append(y.numpy().astype(np.int32))
    return np.vstack(embeddings),np.vstack(probs),np.vstack(targets)


def knn(emb: np.ndarray, labels: np.ndarray, k: int):
    # L2-normalize learned embeddings; cosine distance is stable for representation geometry.
    z=emb.astype(np.float32)
    z/=np.maximum(np.linalg.norm(z,axis=1,keepdims=True),1e-8)
    sim=z@z.T
    np.fill_diagonal(sim,-np.inf)
    idx=np.argpartition(-sim,kth=k-1,axis=1)[:,:k]
    own=(labels[idx]==labels[:,None]).mean(axis=1)
    return own


def main():
    p=argparse.ArgumentParser(description="Audit S2/S3 in CNN embedding space and cross-check model errors.")
    p.add_argument("--model",type=Path,required=True)
    p.add_argument("--split-file",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,default=Path("outputs/reports/s2_s3_embedding_audit"))
    p.add_argument("--neighbors",type=int,default=30)
    p.add_argument("--device",default="auto")
    args=p.parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True)

    device=resolve_device(args.device)
    ckpt=load_checkpoint(args.model,map_location=device)
    config=ckpt["config"]; names=ckpt["class_names"]
    # Force the exact split supplied by the experiment instead of rebuilding anything.
    config=dict(config); config["real_data"]=dict(config.get("real_data",{}))
    config["real_data"]["split_file"]=str(args.split_file)
    root=config["real_data"].get("dataset_root",".")
    ds=RealCsvDataset(root,names,config,split="test",index_path=args.split_file)
    loader=DataLoader(ds,batch_size=int(config.get("train",{}).get("batch_size",32)),shuffle=False,num_workers=0)
    emb,probs,targets=collect(build_model(len(names),config).to(device),loader,device) if False else (None,None,None)

    model=build_model(len(names),config).to(device); model.load_state_dict(ckpt["model_state"])
    emb,probs,targets=collect(model,loader,device)
    threshold=float(config.get("train",{}).get("threshold",0.5))
    preds=(probs>=threshold).astype(np.int32)

    # The current real single-source test contains the three one-hot classes.
    sums=targets.sum(axis=1)
    mask=sums==1
    emb=emb[mask]; probs=probs[mask]; targets=targets[mask]; preds=preds[mask]
    samples=[s for s,m in zip(ds.samples,mask) if m]
    class_idx=targets.argmax(axis=1)
    k=min(args.neighbors,len(emb)-1)
    own=knn(emb,class_idx,k)
    exact=np.all(preds==targets,axis=1)

    rows=[]
    for i,s in enumerate(samples):
        category="typical" if own[i]>=0.8 else ("boundary" if own[i]>=0.5 else "other_dominated")
        rows.append({"file":str(s.get("file","")),"group":str(s.get("group","")),
                     "true_label":label_text(targets[i]),"pred_label":label_text(preds[i]),
                     "exact_match":bool(exact[i]),"own_neighbor_fraction":float(own[i]),
                     "embedding_category":category})
    out=args.output_dir/"embedding_test_audit.csv"
    with out.open("w",encoding="utf-8",newline="") as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    cats=("typical","boundary","other_dominated")
    per={}
    for ci,name in enumerate(names):
        ii=np.where(class_idx==ci)[0]
        per[name]={"samples":int(len(ii))}
        for cat in cats: per[name][cat]=int(sum(rows[j]["embedding_category"]==cat for j in ii))
        per[name]["errors"]=int((~exact[ii]).sum())

    error_categories={cat:int(sum((not r["exact_match"]) and r["embedding_category"]==cat for r in rows)) for cat in cats}
    conf={}
    for r in rows:
        if not r["exact_match"]:
            key=f"{r['true_label']} -> {r['pred_label']}"
            conf[key]=conf.get(key,0)+1
    summary={"model":str(args.model),"test_samples":len(rows),"neighbors":k,"per_source":per,
             "total_errors":int((~exact).sum()),"error_categories":error_categories,"confusions":conf}
    sp=args.output_dir/"embedding_summary.json"
    sp.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

    print("\n========== READ THIS EMBEDDING AUDIT SUMMARY ==========")
    print(f"model={args.model.name} test_samples={len(rows)} total_errors={summary['total_errors']}")
    for name in names:
        d=per[name]; print(f"{name}: samples={d['samples']} typical={d['typical']} boundary={d['boundary']} other_dominated={d['other_dominated']} errors={d['errors']}")
    print(f"error_locations: typical={error_categories['typical']} boundary={error_categories['boundary']} other_dominated={error_categories['other_dominated']}")
    if conf:
        print("confusions: "+", ".join(f"{k}={v}" for k,v in sorted(conf.items())))
    else: print("confusions: none")
    print(f"k_neighbors={k}")
    print("========================================================")
    print(f"audit_csv={out.resolve()}")
    print(f"summary={sp.resolve()}")


if __name__=="__main__":
    main()
