"""Exact A1/B8 inputs and fixed-support teacher primitives; private data never enter Git."""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.decomposition import PCA

ROOT=Path(__file__).resolve().parents[1]
PROJECT=ROOT.parent
ACTIVE=PROJECT/"active_fewshot_patient_calibration_seed42_v1"
sys.path.insert(0,str(ACTIVE/"code"))
afc=importlib.import_module("common")
afr=importlib.import_module("run")

RUNTIME=Path(os.environ.get("B8_RUNTIME",""))
ACTIVE_RUNTIME=Path(os.environ.get("AFPC_RUNTIME",""))
LOCK_SHA="6e9194da70cd850f29deceb1321e525ef08a18e61292f003221394b1210a20e2"
ACTIVE_LOCK_SHA="0a8ef24d36f6e942aa0587565270ddf72740f1bacbed687967395e06226e8821"
DIMS=(4,8,16)
B8_GRID_W=(.001,.01,.1,1.,10.,100.,1000.)
LOW_GRID_W=(.01,.1,1.,10.,100.)
GRID_B=(.1,1.,10.)


def preflight():
    afc.preflight()
    if not os.environ.get("B8_RUNTIME") or not RUNTIME.is_absolute():
        raise RuntimeError("B8_RUNTIME private absolute path required")
    if not os.environ.get("AFPC_RUNTIME") or not ACTIVE_RUNTIME.is_absolute():
        raise RuntimeError("AFPC_RUNTIME private absolute path required")
    if afc.sha(ROOT/"PROTOCOL_LOCK.json")!=LOCK_SHA or afc.sha(ACTIVE/"PROTOCOL_LOCK.json")!=ACTIVE_LOCK_SHA:
        raise RuntimeError("Frozen B8/active protocol lock mismatch")
    prior=json.loads((ACTIVE/"SOURCE_REPRODUCTION.json").read_text(encoding="utf-8"))
    if prior["checkpoints"]!=150 or prior["max_grid_error"]>1e-6 or prior["R4_max_logit_replay_error"]>1e-6:
        raise RuntimeError("SOURCE_A1_REPRODUCTION_FAILED")


def private_active_cell(fold,sid):
    stem=hashlib.sha256(sid.encode()).hexdigest()[:16]
    with (ACTIVE_RUNTIME/"private"/f"fold_{fold}"/(stem+".pkl")).open("rb") as f:
        cell=pickle.load(f)
    if cell["lock_sha"]!=ACTIVE_LOCK_SHA or cell["fold"]!=fold or cell["subject_id"]!=sid or len(cell["split_audit"])!=20:
        raise RuntimeError("Prior active target-cell provenance invalid")
    return cell


def selected_rows(fold):
    rows=afc.read_csv(afc.PRIOR_RUNTIME/"private"/f"fold_{fold}"/"A1_VLOO_PRIVATE.csv")
    if len(rows)!=13 or {int(r["fold"]) for r in rows}!={fold}:
        raise RuntimeError("65-cell VLOO source changed")
    return rows


def fold_contexts(fold):
    """Distinct selected checkpoint/threshold coordinates, with their original FIT lambda."""
    contexts={}
    for row in selected_rows(fold):
        sid=row["subject_id"]
        epoch=int(row["selected_epoch"])
        tau=float(row["selected_threshold"])
        cell=private_active_cell(fold,sid)
        if cell["epoch"]!=epoch or cell["threshold"]!=tau:
            raise RuntimeError("Source epoch/threshold and current B8 provenance disagree")
        key=(epoch,tau)
        old=contexts.get(key)
        if old is None:
            contexts[key]={"fold":fold,"epoch":epoch,"tau":tau,
                           "original_lambda":float(cell["lambda_result"]["selected_lambda"]),
                           "payload":afc.load_representation(fold,epoch),"target_ids":[]}
        elif old["original_lambda"]!=float(cell["lambda_result"]["selected_lambda"]):
            raise RuntimeError("Shared context prior lambda changed")
        contexts[key]["target_ids"].append(sid)
    return list(contexts.values())


def context_for_row(fold,row):
    sid=row["subject_id"]
    epoch=int(row["selected_epoch"]);tau=float(row["selected_threshold"])
    cell=private_active_cell(fold,sid)
    if cell["epoch"]!=epoch or cell["threshold"]!=tau:
        raise RuntimeError("Source row and prior active cell mismatch")
    return {"fold":fold,"epoch":epoch,"tau":tau,
            "original_lambda":float(cell["lambda_result"]["selected_lambda"]),
            "payload":afc.load_representation(fold,epoch),"target_ids":[sid]}


def fixed_support(candidate,z,m0,y,original_lambda,fold,sid,rep,tag):
    """Same 64D uncertainty policy as the prior teacher, stopped at eight labels."""
    support=[]
    b=0.0; w=np.zeros(64,dtype=np.float64)
    for _ in range(8):
        rest=np.asarray([int(c) for c in candidate if int(c) not in support],dtype=int)
        if len(rest)==0: raise RuntimeError("Candidate pool shorter than eight")
        margin=m0[rest]+b+z[rest]@w
        ties=np.asarray([afc.stable_seed(42,fold,sid,rep,tag,int(c)) for c in rest])
        chosen=int(rest[np.lexsort((ties,np.abs(margin)))[0]])
        support.append(chosen)
        b,w=afc.fit_residual(m0[support],z[support],y[support],original_lambda)
    return np.asarray(support,dtype=int)


def target_episode(context,sid,rep):
    fold=context["fold"]
    payload=context["payload"]
    row=payload["val"][sid]
    scaler=afr.scaler_for(payload)
    z=scaler.transform(row["R4"])
    m0=afr.margin_for(row,context["tau"])
    afr.verify_b0(row,context["tau"],m0)
    candidate,query=afc.split_indices(len(m0),42,fold,sid,rep)
    support=fixed_support(candidate,z,m0,row["y"],context["original_lambda"],fold,sid,rep,"UNCERTAINTY")
    return {"fold":fold,"sid":sid,"epoch":context["epoch"],"tau":context["tau"],"rep":rep,
            "z":z.astype(np.float64),"m0":m0,"candidate":candidate,"query":query,
            "support":support,"y":row["y"],"r4":row["R4"],"scaler_scale":scaler.scale_}


def fit_episode(context,sid,rep,split_tag="teacher_fit"):
    fold=context["fold"]
    payload=context["payload"]
    row=payload["fit"][sid]
    scaler=afr.scaler_for(payload)
    z=scaler.transform(row["R4"])
    m0=afr.margin_for(row,context["tau"])
    candidate,query=afc.split_indices(len(m0),42,fold,sid,rep,split_tag)
    support=fixed_support(candidate,z,m0,row["y"],context["original_lambda"],fold,sid,rep,split_tag)
    return {"fold":fold,"sid":sid,"epoch":context["epoch"],"tau":context["tau"],"rep":rep,
            "z":z.astype(np.float64),"m0":m0,"candidate":candidate,"query":query,
            "support":support,"y":row["y"]}


def pca_basis(context,d):
    payload=context["payload"]
    scaler=afr.scaler_for(payload)
    z=np.concatenate([scaler.transform(payload["fit"][sid]["R4"]) for sid in sorted(payload["fit"])])
    basis=PCA(n_components=d,svd_solver="full").fit(z).components_.astype(np.float64)
    return basis


def fit_head(m0,r,y,lambda_w,lambda_b):
    """Convex support-only residual head for any projection dimension."""
    m0=np.asarray(m0,dtype=np.float64); r=np.asarray(r,dtype=np.float64); y=np.asarray(y,dtype=np.float64)
    if len(y)==0: return 0.0,np.zeros(r.shape[1],dtype=np.float64)
    t=2*y-1; n=len(y)
    def objective(theta):
        b=theta[0]; w=theta[1:]
        m=m0+b+r@w
        e=-t*expit(-t*m)/n
        loss=float(np.logaddexp(0,-t*m).mean()+lambda_b*b*b+lambda_w*(w@w))
        grad=np.r_[e.sum()+2*lambda_b*b,r.T@e+2*lambda_w*w]
        return loss,grad
    result=minimize(objective,np.zeros(r.shape[1]+1),jac=True,method="L-BFGS-B",
                    options={"maxiter":200,"ftol":1e-12,"gtol":1e-9})
    if not np.isfinite(result.fun) or not np.isfinite(result.x).all() or np.linalg.norm(result.jac)>1e-3:
        raise RuntimeError(f"Teacher head convex optimizer failed: {result.message}")
    return float(result.x[0]),result.x[1:].copy()


def prediction(ep,basis,lambda_w,lambda_b,full_pool=False):
    support=ep["candidate"] if full_pool else ep["support"]
    r=ep["z"]@basis.T
    b,w=fit_head(ep["m0"][support],r[support],ep["y"][support],lambda_w,lambda_b)
    return ep["m0"][ep["query"]]+b+r[ep["query"]]@w,b,basis.T@w


def paired_ap(ep,score):
    y=ep["y"][ep["query"]]
    return afc.rank_metrics(y,score)["ap"]


def write_private(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+".tmp")
    with temp.open("wb") as f:pickle.dump(data,f,protocol=5)
    temp.replace(path)
