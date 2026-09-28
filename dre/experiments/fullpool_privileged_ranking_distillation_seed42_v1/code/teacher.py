"""Strict leave-patient-out lambda selection and fivefold OOF FULLPOOL Teacher."""
from __future__ import annotations

import argparse
import json
import pickle

import numpy as np

import protocol as p


def grid_path(ctx,sid):
    return p.RUNTIME/"private"/"lambda_grid"/f"fold_{ctx['fold']}"/ctx["context_id"]/(p.stem(sid)+".pkl")


def oof_path(ctx,sid):
    return p.RUNTIME/"private"/"teacher_oof"/f"fold_{ctx['fold']}"/ctx["context_id"]/(p.stem(sid)+".pkl")


def one_grid(ctx,sid):
    path=grid_path(ctx,sid)
    if path.is_file():
        with path.open("rb") as f:prior=pickle.load(f)
        if prior["lock_sha"]!=p.LOCK_SHA or prior["sid"]!=sid or len(prior["ap"])!=len(p.GRID):
            raise RuntimeError("Teacher lambda grid resume mismatch")
        return
    z,m0,y=p.patient_data(ctx,sid)
    episodes=[]
    for rep in range(3):
        support,query=p.afc.split_indices(len(y),42,ctx["fold"],sid,rep,"privileged_lambda_meta")
        episodes.append((support,query))
    per_lambda=[]
    for lw,lb in p.GRID:
        values=[]
        for support,query in episodes:
            b,w=p.tc.fit_head(m0[support],z[support],y[support],lw,lb)
            score=m0[query]+b+z[query]@w
            ap=p.afc.rank_metrics(y[query],score)["ap"]
            if np.isfinite(ap):values.append(float(ap))
        per_lambda.append(float(np.mean(values)) if values else float("nan"))
    p.tc.write_private(path,dict(lock_sha=p.LOCK_SHA,fold=ctx["fold"],context_id=ctx["context_id"],
                                 sid=sid,n_channels=len(y),ap=np.asarray(per_lambda,dtype=np.float64),
                                 meta_repetitions=3))
    print(f"[GRID_PATIENT] fold={ctx['fold']} {ctx['context_id']} done",flush=True)


def run_grid(ctx):
    p.preflight()
    for sid in sorted(p.payload(ctx["fold"],ctx["epoch"])["fit"]):one_grid(ctx,sid)
    print(f"[GRID_CONTEXT_COMPLETE] fold={ctx['fold']} {ctx['context_id']}",flush=True)


def leave_patient_out_lambda(ctx,sid,fit_ids):
    scores=[]
    for other in fit_ids:
        if other==sid:continue
        with grid_path(ctx,other).open("rb") as f:row=pickle.load(f)
        if row["sid"]!=other or row["context_id"]!=ctx["context_id"] or row["lock_sha"]!=p.LOCK_SHA:
            raise RuntimeError("Other-FIT-patient lambda grid changed")
        scores.append(row["ap"])
    matrix=np.asarray(scores,dtype=np.float64)
    with np.errstate(invalid="ignore"):
        means=np.nanmean(matrix,axis=0)
    if not np.isfinite(means).all():raise RuntimeError("No estimable other-patient lambda AP")
    best=float(np.max(means))
    eligible=[i for i,value in enumerate(means) if best-value<=1e-6]
    selected=max(eligible,key=lambda i:p.GRID[i])
    return selected,float(means[selected]),len(scores)


def one_oof(ctx,sid,fit_ids):
    path=oof_path(ctx,sid)
    if path.is_file():
        with path.open("rb") as f:prior=pickle.load(f)
        if (prior["lock_sha"]!=p.LOCK_SHA or prior["sid"]!=sid or
                prior["context_id"]!=ctx["context_id"] or len(prior["teacher_score"])!=prior["n_channels"]):
            raise RuntimeError("OOF Teacher resume mismatch")
        return
    choice,other_ap,n_other=leave_patient_out_lambda(ctx,sid,fit_ids)
    lw,lb=p.GRID[choice]
    z,m0,y=p.patient_data(ctx,sid)
    perm=np.random.default_rng(p.afc.stable_seed(42,ctx["fold"],ctx["context_id"],sid,"teacher_channel_folds")).permutation(len(y))
    groups=np.array_split(perm,5)
    if any(len(group)==0 for group in groups):raise RuntimeError("Empty OOF channel fold")
    teacher=np.full(len(y),np.nan,dtype=np.float64)
    heldout=[]
    for group in groups:
        support=np.setdiff1d(np.arange(len(y)),group,assume_unique=False)
        if set(support)&set(group) or len(support)+len(group)!=len(y):
            raise RuntimeError("Teacher support includes held-out channel")
        b,w=p.tc.fit_head(m0[support],z[support],y[support],lw,lb)
        teacher[group]=m0[group]+b+z[group]@w
        heldout.append(len(group))
    if not np.isfinite(teacher).all():raise RuntimeError("OOF Teacher did not cover all channels")
    # Private y and scores are intentionally never copied to the repository.
    p.tc.write_private(path,dict(lock_sha=p.LOCK_SHA,fold=ctx["fold"],context_id=ctx["context_id"],
                                 sid=sid,n_channels=len(y),lambda_index=choice,lambda_w=lw,lambda_b=lb,
                                 selected_other_patient_meta_ap=other_ap,n_other_fit_patients=n_other,
                                 heldout_fold_sizes=heldout,teacher_score=teacher,a1_score=m0,y=y,
                                 no_scored_channel_label_in_teacher_fit=True,
                                 no_scored_patient_label_in_lambda_selection=True))
    print(f"[OOF_PATIENT] fold={ctx['fold']} {ctx['context_id']} done",flush=True)


def run_oof(ctx):
    p.preflight()
    fit_ids=sorted(p.payload(ctx["fold"],ctx["epoch"])["fit"])
    if not all(grid_path(ctx,sid).is_file() for sid in fit_ids):
        raise RuntimeError("All other-patient lambda grids must be frozen before OOF")
    for sid in fit_ids:one_oof(ctx,sid,fit_ids)
    print(f"[OOF_CONTEXT_COMPLETE] fold={ctx['fold']} {ctx['context_id']}",flush=True)


def main():
    arg=argparse.ArgumentParser()
    arg.add_argument("--stage",choices=("grid","oof"),required=True)
    arg.add_argument("--context-index",type=int,choices=range(17),required=True)
    a=arg.parse_args()
    ctx=p.all_contexts()[a.context_index]
    if a.stage=="grid":run_grid(ctx)
    else:run_oof(ctx)


if __name__=="__main__":main()
