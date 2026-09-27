"""FIT-patient-only episodic selection for the frozen-support teacher study."""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

import teacher_core as tc


def context_key(ctx):
    s=f"{ctx['fold']}|{ctx['epoch']}|{np.float64(ctx['tau']).hex()}"
    return hashlib.sha256(s.encode()).hexdigest()[:20]


def episodes(ctx, repetitions=3):
    out=[]
    for sid in sorted(ctx["payload"]["fit"]):
        for rep in range(repetitions):
            ep=tc.fit_episode(ctx,sid,rep)
            if len(ep["candidate"])>=8 and len(np.unique(ep["y"][ep["query"]]))==2:
                out.append(ep)
    if not out:
        raise RuntimeError("No estimable FIT B8 episodes")
    return out


def patient_equal_ap(eps,basis,lam_w,lam_b,full_pool):
    scores={}
    for ep in eps:
        prediction,_,_=tc.prediction(ep,basis,lam_w,lam_b,full_pool)
        y=ep["y"][ep["query"]]
        scores.setdefault(ep["sid"],[]).append(float(average_precision_score(y,prediction)))
    return float(np.mean([np.mean(v) for v in scores.values()])),len(scores)


def select_variant(eps,basis,grid_w,full_pool):
    records=[]
    for w in grid_w:
        for b in tc.GRID_B:
            ap,n=patient_equal_ap(eps,basis,w,b,full_pool)
            records.append({"lambda_w":w,"lambda_b":b,"fit_patient_equal_ap":ap,"estimable_fit_patients":n})
    best_ap=max(r["fit_patient_equal_ap"] for r in records)
    best=max((r for r in records if best_ap-r["fit_patient_equal_ap"]<=1e-6),
             key=lambda r:(r["lambda_w"],r["lambda_b"]))
    return best,records


def run_context(ctx,variant=None,mode=None):
    key=context_key(ctx)
    path=tc.RUNTIME/"private"/"fit_selection"/f"{key}.pkl"
    if path.exists():
        with path.open("rb") as f: result=pickle.load(f)
        if result["lock_sha"]!=tc.LOCK_SHA or result["fold"]!=ctx["fold"] or result["epoch"]!=ctx["epoch"] or result["tau"]!=ctx["tau"]:
            raise RuntimeError("FIT selection resume provenance mismatch")
        return result
    if variant not in ("R64","PCA4","PCA8","PCA16") or mode not in ("b8","fullpool"):
        raise RuntimeError("One FIT variant/mode per process is required")
    partial=path.with_name(f"{key}_{variant}_{mode}.pkl")
    if not partial.exists():
        eps=episodes(ctx)
        basis=np.eye(64) if variant=="R64" else tc.pca_basis(ctx,int(variant[3:]))
        grid=tc.B8_GRID_W if variant=="R64" else tc.LOW_GRID_W
        best,records=select_variant(eps,basis,grid,mode=="fullpool")
        obj={"lock_sha":tc.LOCK_SHA,"fold":ctx["fold"],"epoch":ctx["epoch"],"tau":ctx["tau"],
             "variant":variant,"mode":mode,"best":best,"grid":records,"basis":basis,
             "fit_episodes":len(eps),"fit_patients":len({e['sid'] for e in eps})}
        tc.write_private(partial,obj)
        print(f"[FIT] f{ctx['fold']} e{ctx['epoch']} {variant} {mode} AP={best['fit_patient_equal_ap']:.4f}",flush=True)
    else:
        with partial.open("rb") as f:obj=pickle.load(f)
        if obj["lock_sha"]!=tc.LOCK_SHA:raise RuntimeError("FIT partial resume mismatch")
    all_parts={}
    for name in ("R64","PCA4","PCA8","PCA16"):
        for kind in ("b8","fullpool"):
            q=path.with_name(f"{key}_{name}_{kind}.pkl")
            if not q.exists():return obj
            with q.open("rb") as f:part=pickle.load(f)
            if part["lock_sha"]!=tc.LOCK_SHA or part["fold"]!=ctx["fold"] or part["epoch"]!=ctx["epoch"] or part["tau"]!=ctx["tau"]:
                raise RuntimeError("FIT partial aggregation mismatch")
            all_parts[(name,kind)]=part
    variants={}
    for name in ("R64","PCA4","PCA8","PCA16"):
        a=all_parts[(name,"b8")];b=all_parts[(name,"fullpool")]
        if not np.array_equal(a["basis"],b["basis"]):raise RuntimeError("FIT basis changed between modes")
        variants[name]={"b8":a["best"],"fullpool":b["best"],"b8_grid":a["grid"],
                        "fullpool_grid":b["grid"],"basis":a["basis"]}
    first=all_parts[("R64","b8")]
    result={"lock_sha":tc.LOCK_SHA,"fold":ctx["fold"],"epoch":ctx["epoch"],"tau":ctx["tau"],
            "key":key,"fit_episodes":first["fit_episodes"],"fit_patients":first["fit_patients"],"variants":variants}
    tc.write_private(path,result)
    print(f"[FIT_CONTEXT_COMPLETE] fold={ctx['fold']} epoch={ctx['epoch']} key={key}",flush=True)
    return result


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--fold",type=int,required=True)
    ap.add_argument("--context",type=int,default=-1)
    ap.add_argument("--variant",choices=("R64","PCA4","PCA8","PCA16"),required=True)
    ap.add_argument("--mode",choices=("b8","fullpool"),required=True)
    args=ap.parse_args()
    tc.preflight()
    if args.context>=0:
        rows=tc.selected_rows(args.fold)
        keys=list(dict.fromkeys((int(r["selected_epoch"]),float(r["selected_threshold"])) for r in rows))
        if args.context>=len(keys):raise RuntimeError("Context index out of range")
        key=keys[args.context]
        row=next(r for r in rows if (int(r["selected_epoch"]),float(r["selected_threshold"]))==key)
        contexts=[tc.context_for_row(args.fold,row)]
        contexts[0]["target_ids"]=[r["subject_id"] for r in rows if (int(r["selected_epoch"]),float(r["selected_threshold"]))==key]
    else:
        contexts=tc.fold_contexts(args.fold)
    for i,ctx in enumerate(contexts):
        run_context(ctx,args.variant,args.mode)
        print(f"[DONE] fold={args.fold} context={args.context if args.context>=0 else i} {args.variant} {args.mode} key={context_key(ctx)}",flush=True)


if __name__=="__main__":main()
