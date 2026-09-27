"""FIT-only dimension/gamma selection for the prototype relation control."""
from __future__ import annotations

import argparse
import pickle

import numpy as np
from sklearn.metrics import average_precision_score

import teacher_core as tc
from meta_train import prepare


def prototype_score(ep,p,gamma):
    s=ep["support"];q=ep["query"]
    ys=ep["y"][s]
    if len(np.unique(ys))<2:
        return ep["m0"][q].copy(),True
    r=ep["z"]@p.T
    ez=r[s[ys==1]].mean(axis=0);nez=r[s[ys==0]].mean(axis=0)
    ez/=max(np.linalg.norm(ez),1e-12);nez/=max(np.linalg.norm(nez),1e-12)
    query=r[q];query=query/np.maximum(np.linalg.norm(query,axis=1,keepdims=True),1e-12)
    return ep["m0"][q]+gamma*(query@ez-query@nez),False


def select(fold):
    tc.preflight()
    path=tc.RUNTIME/"private"/tc.META_FOLDER/f"fold_{fold}_prototype.pkl"
    if path.exists():
        with path.open("rb") as f: obj=pickle.load(f)
        if obj["lock_sha"]!=tc.LOCK_SHA:raise RuntimeError("Prototype resume mismatch")
        return obj
    prepared=prepare(fold)
    models={}
    for d in tc.DIMS:
        with (tc.RUNTIME/"private"/tc.META_FOLDER/f"fold_{fold}_d{d}.pkl").open("rb") as f:models[d]=pickle.load(f)
    best_d=min(tc.DIMS,key=lambda d:(-models[d]["fit_val_ap"],d))
    p=models[best_d]["projection"].astype(np.float64)
    records=[]
    for gamma in (.1,.3,1.):
        by_patient={};oneclass=0;total=0
        for ep in prepared["val"]:
            score,fallback=prototype_score(ep,p,gamma);oneclass+=int(fallback);total+=1
            y=ep["y"][ep["query"]]
            if len(np.unique(y))<2:continue
            by_patient.setdefault(ep["sid"],[]).append(float(average_precision_score(y,score)))
        ap=float(np.mean([np.mean(v) for v in by_patient.values()]))
        records.append({"gamma":gamma,"fit_val_patient_equal_ap":ap,"fit_val_patients":len(by_patient),
                        "oneclass_fit_val_episodes":oneclass,"fit_val_episodes":total})
    best=max(records,key=lambda r:(r["fit_val_patient_equal_ap"],-r["gamma"]))
    obj={"lock_sha":tc.LOCK_SHA,"fold":fold,"dimension":best_d,"gamma":best["gamma"],
         "projection":p,"fit_val_ap":best["fit_val_patient_equal_ap"],"gamma_grid":records}
    tc.write_private(path,obj)
    print(f"[PROTO] fold={fold} d={best_d} gamma={best['gamma']} FIT-val AP={best['fit_val_patient_equal_ap']:.5f}",flush=True)
    return obj


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--fold",type=int,required=True)
    select(ap.parse_args().fold)


if __name__=="__main__":main()
