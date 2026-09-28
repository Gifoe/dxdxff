"""Freeze every B=0 target score/control before label-using metrics."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle

import numpy as np
import torch

import protocol as p
from relational import RelationalReadout


def selection():
    path=p.ROOT/"FIT_SELECTION_LOCK.json"
    row=json.loads(path.read_text(encoding="utf-8"))
    if row["lock_sha"]!=p.LOCK_SHA or not row["fit_grid_complete"]:
        raise RuntimeError("FIT-only method selection missing")
    frozen=p.RUNTIME/"FIT_SELECTION_LOCK_SHA.json"
    if json.loads(frozen.read_text(encoding="utf-8"))["selection_sha"]!=p.sha(path):
        raise RuntimeError("FIT-only selection changed after full-FIT training")
    return row


def full_checkpoint(fold,ctx,arch,arm,selection_rows):
    choice=next(r for r in selection_rows if r["fold"]==fold and r["arch"]==arch)
    index=0 if arm=="BCE_ONLY" else choice["positive_index"]
    folder=p.RUNTIME/"full"/f"fold_{fold}"/ctx["context_id"]/arch/arm
    summary=json.loads((folder/"summary.json").read_text(encoding="utf-8"))
    if (summary["lock_sha"]!=p.LOCK_SHA or summary["lambda_index"]!=index or
            summary["context_id"]!=ctx["context_id"] or summary["arm"]!=arm or
            summary["checkpoint_sha"]!=p.sha(folder/"selected.pt")):
        raise RuntimeError("Full-FIT selected readout identity mismatch")
    ckpt=torch.load(folder/"selected.pt",map_location="cpu",weights_only=False)
    if (ckpt["lock_sha"]!=p.LOCK_SHA or ckpt["fold"]!=fold or
            ckpt["context_id"]!=ctx["context_id"] or ckpt["arch"]!=arch or ckpt["arm"]!=arm):
        raise RuntimeError("Full-FIT readout checkpoint provenance mismatch")
    model=RelationalReadout(arch)
    model.load_state_dict(ckpt["model"],strict=True);model.eval()
    return model,ckpt,folder/"selected.pt"


def features(row,ckpt,tau):
    raw=np.asarray(row["R4"],dtype=np.float64)
    mean=np.asarray(ckpt["scaler_mean"],dtype=np.float64)
    scale=np.asarray(ckpt["scaler_scale"],dtype=np.float64)
    z=((raw-mean)/scale).astype(np.float32)
    m=(math.log(tau/(1-tau))+np.asarray(row["source_ez"],dtype=np.float64)).astype(np.float32)
    if z.shape!=(len(m),64) or not np.isfinite(z).all() or not np.isfinite(m).all():
        raise RuntimeError("Malformed target or donor unlabeled A1 features")
    return torch.from_numpy(z),torch.from_numpy(m)


def donor(fit,target_n,fold,sid,rep):
    # One deterministic wrong-patient context, closest size to avoid a trivial
    # channel-count confound. Never inspect donor or target labels here.
    choices=[]
    for other,row in fit.items():
        n=len(row["R4"])
        tie=p.afc.stable_seed(42,fold,sid,rep,other,"wrong_context")
        choices.append((abs(n-target_n),tie,other))
    if not choices:raise RuntimeError("No FIT donor patient")
    return fit[min(choices)[2]],min(choices)[2]


def score_cell(fold,ctx,sid,arch,arm,model,ckpt):
    source=p.payload(fold,ctx["epoch"])
    row=source["val"][sid]
    z,m0=features(row,ckpt,ctx["threshold"])
    n=len(z)
    if n<4:raise RuntimeError("Too few target channels")
    all_index=torch.arange(n)
    with torch.no_grad():
        full_all=model(z,m0,self_index=all_index).numpy()
    reps=[]
    for rep in range(20):
        candidate,query=p.afc.split_indices(n,42,fold,sid,rep)
        if set(candidate)&set(query) or len(candidate)+len(query)!=n:
            raise RuntimeError("Fixed query partition changed")
        q=torch.as_tensor(query,dtype=torch.long)
        zq,mq=z[q],m0[q]
        wrong_row,donor_sid=donor(source["fit"],n,fold,sid,rep)
        dz,dm=features(wrong_row,ckpt,ctx["threshold"])
        perm=np.random.default_rng(p.afc.stable_seed(42,fold,sid,rep,"shuffle_relation")).permutation(n)
        if np.array_equal(perm,np.arange(n)):perm=np.roll(perm,1)
        with torch.no_grad():
            query_only=model(zq,mq,self_index=torch.arange(len(q))).numpy()
            wrong=model(zq,mq,dz,dm).numpy()
            shuffled=model(zq,mq,z,m0[torch.as_tensor(perm)] ,q).numpy()
        reps.append(dict(rep=rep,query=query.astype(np.int32),
                         full=np.asarray(full_all[query],dtype=np.float32),
                         query_only=np.asarray(query_only,dtype=np.float32),
                         wrong=np.asarray(wrong,dtype=np.float32),
                         shuffled=np.asarray(shuffled,dtype=np.float32),
                         a1=np.asarray(m0[query],dtype=np.float32),
                         donor_hash=hashlib.sha256(donor_sid.encode()).hexdigest()[:12]))
    return dict(lock_sha=p.LOCK_SHA,fold=fold,context_id=ctx["context_id"],
                epoch=ctx["epoch"],threshold=ctx["threshold"],sid=sid,
                arch=arch,arm=arm,n_channels=n,
                full_all=np.asarray(full_all,dtype=np.float32),
                a1_all=np.asarray(m0,dtype=np.float32),reps=reps)


def run_scores():
    p.preflight();torch.set_num_threads(1)
    locked=selection()
    active=(p.R1,p.R2,p.R3) if locked["r3_gate"]["run"] else (p.R1,p.R2)
    for ctx in p.all_contexts():
        fold=ctx["fold"]
        for arch in active:
            for arm in p.ARMS:
                model,ckpt,_=full_checkpoint(fold,ctx,arch,arm,locked["selections"])
                for sid in ctx["target_ids"]:
                    stem=hashlib.sha256(sid.encode()).hexdigest()[:16]
                    folder=p.RUNTIME/"private"/"scores"/f"fold_{fold}"/ctx["context_id"]
                    path=folder/f"{stem}_{arch}_{arm}.pkl"
                    if path.is_file():
                        with path.open("rb") as f:prior=pickle.load(f)
                        if (prior["lock_sha"]!=p.LOCK_SHA or prior["sid"]!=sid or
                                len(prior["reps"])!=20 or len(prior["full_all"])!=prior["n_channels"]):
                            raise RuntimeError("Frozen score resume identity mismatch")
                        continue
                    result=score_cell(fold,ctx,sid,arch,arm,model,ckpt)
                    p.atomic_pickle(path,result)
                print(f"[SCORES] fold={fold} {ctx['context_id']} {arch} {arm} cells={len(ctx['target_ids'])}",flush=True)
    print("[ALL_RELATIONAL_SCORES_READY_FOR_HASH]",flush=True)


def hash_freeze():
    p.preflight();locked=selection()
    active=(p.R1,p.R2,p.R3) if locked["r3_gate"]["run"] else (p.R1,p.R2)
    files={};checkpoints={}
    for ctx in p.all_contexts():
        fold=ctx["fold"]
        for arch in active:
            for arm in p.ARMS:
                _,_,ckpt=full_checkpoint(fold,ctx,arch,arm,locked["selections"])
                checkpoints[str(ckpt.relative_to(p.RUNTIME))]=p.sha(ckpt)
                for sid in ctx["target_ids"]:
                    stem=hashlib.sha256(sid.encode()).hexdigest()[:16]
                    path=p.RUNTIME/"private"/"scores"/f"fold_{fold}"/ctx["context_id"]/f"{stem}_{arch}_{arm}.pkl"
                    with path.open("rb") as f:row=pickle.load(f)
                    if row["sid"]!=sid or row["fold"]!=fold or len(row["reps"])!=20:
                        raise RuntimeError("Incomplete relational score/control grid")
                    files[str(path.relative_to(p.RUNTIME))]=p.sha(path)
    expected=65*len(active)*2
    if len(files)!=expected or len(checkpoints)!=17*len(active)*2:
        raise RuntimeError("Relational score/checkpoint freeze coverage incomplete")
    manifest=dict(lock_sha=p.LOCK_SHA,selection_sha=p.sha(p.ROOT/"FIT_SELECTION_LOCK.json"),
                  score_files=len(files),checkpoint_files=len(checkpoints),target_cells=65,
                  context_repetitions=20,active_architectures=list(active),
                  new_target_labels_not_indexed_in_score_freeze=True,
                  files=files,checkpoints=checkpoints)
    path=p.RUNTIME/"ALL_SCORES_FROZEN_BEFORE_NEW_TARGET_LABEL_USE.json"
    if path.is_file() and json.loads(path.read_text(encoding="utf-8"))!=manifest:
        raise RuntimeError("Relational score freeze changed")
    p.atomic_json(path,manifest)
    print(f"[RELATIONAL_SCORE_FREEZE_PASS] files={len(files)} checkpoints={len(checkpoints)}",flush=True)


def main():
    q=argparse.ArgumentParser();q.add_argument("--stage",choices=("scores","freeze"),required=True)
    a=q.parse_args()
    if a.stage=="scores":run_scores()
    else:hash_freeze()


if __name__=="__main__":main()
