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


def run_context(ctx):
    key=context_key(ctx)
    path=tc.RUNTIME/"private"/"fit_selection"/f"{key}.pkl"
    if path.exists():
        with path.open("rb") as f: result=pickle.load(f)
        if result["lock_sha"]!=tc.LOCK_SHA or result["fold"]!=ctx["fold"] or result["epoch"]!=ctx["epoch"] or result["tau"]!=ctx["tau"]:
            raise RuntimeError("FIT selection resume provenance mismatch")
        return result
    eps=episodes(ctx)
    variants={}
    for name,basis,grid in [("R64",np.eye(64),tc.B8_GRID_W)]+[(f"PCA{d}",tc.pca_basis(ctx,d),tc.LOW_GRID_W) for d in tc.DIMS]:
        b8,b8_grid=select_variant(eps,basis,grid,False)
        full,full_grid=select_variant(eps,basis,grid,True)
        variants[name]={"b8":b8,"fullpool":full,"b8_grid":b8_grid,"fullpool_grid":full_grid,"basis":basis}
        print(f"[FIT] f{ctx['fold']} e{ctx['epoch']} {name} B8={b8['fit_patient_equal_ap']:.4f} FULL={full['fit_patient_equal_ap']:.4f}",flush=True)
    result={"lock_sha":tc.LOCK_SHA,"fold":ctx["fold"],"epoch":ctx["epoch"],"tau":ctx["tau"],
            "key":key,"fit_episodes":len(eps),"fit_patients":len({e['sid'] for e in eps}),"variants":variants}
    tc.write_private(path,result)
    return result


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--fold",type=int,required=True)
    ap.add_argument("--context",type=int,default=-1)
    args=ap.parse_args()
    tc.preflight()
    contexts=tc.fold_contexts(args.fold)
    if args.context>=0:
        if args.context>=len(contexts):raise RuntimeError("Context index out of range")
        contexts=[contexts[args.context]]
    for i,ctx in enumerate(contexts):
        run_context(ctx)
        print(f"[DONE] fold={args.fold} context={args.context if args.context>=0 else i} key={context_key(ctx)}",flush=True)


if __name__=="__main__":main()
