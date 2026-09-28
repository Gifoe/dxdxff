"""Frozen A1/B8 provenance and FIT-only patient-disjoint Teacher contexts."""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
PROJECT=ROOT.parent
sys.path.insert(0,str(PROJECT/"active_fewshot_patient_calibration_seed42_v1"/"code"))
afc=importlib.import_module("common")
afr=importlib.import_module("run")
sys.path.insert(0,str(PROJECT/"b8_teacher_ceiling_meta_readout_seed42_v1"/"code"))
tc=importlib.import_module("teacher_core")

RUNTIME=Path(os.environ.get("FPPD_RUNTIME",""))
LOCK_SHA="43b7680413d360163eb9428bc03f0816c46e8a83fbc93465a10abd131aafbcb9"
A1_AP=.5767434626151353
FULLPOOL_AP=.6924644868
B8_CURRENT=.5996322681940147
B8_BEST=.6052912572363056
GRID_W=(.001,.01,.1,1.,10.,100.,1000.)
GRID_B=(.1,1.,10.)
GRID=tuple((w,b) for w in GRID_W for b in GRID_B)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def preflight():
    if not os.environ.get("FPPD_RUNTIME") or not RUNTIME.is_absolute():
        raise RuntimeError("FPPD_RUNTIME must be an absolute private directory")
    if sha(ROOT/"PROTOCOL_LOCK.json")!=LOCK_SHA:raise RuntimeError("Distillation protocol lock changed")
    afc.preflight()
    source=json.loads((PROJECT/"zeroshot_geometry_canonicalization_metadg_seed42_v1"/"SOURCE_REPRODUCTION.json").read_text(encoding="utf-8"))
    b0=json.loads((PROJECT/"zeroshot_geometry_canonicalization_metadg_seed42_v1"/"B0_IDENTITY_AUDIT.json").read_text(encoding="utf-8"))
    if (source["checkpoints"]!=150 or source["max_grid_error"]>1e-6 or
            source["R4_max_logit_replay_error"]>1e-6 or not b0["pass"] or
            abs(b0["source_ap"]-A1_AP)>1e-6):
        raise RuntimeError("A1 exact replay failed")
    matrix=afc.read_csv(PROJECT/"b8_teacher_ceiling_meta_readout_seed42_v1"/"TEACHER_VARIANT_MATRIX.csv")
    ref=next(x for x in matrix if x["variant"]=="RETUNED_64D_FULLPOOL")
    if abs(float(ref["mean_ap"])-FULLPOOL_AP)>1e-6 or int(ref["n_cells"])!=65:
        raise RuntimeError("Retuned 64D FULLPOOL reference replay failed")


@lru_cache(maxsize=5)
def contexts(fold):
    rows=afc.read_csv(afc.PRIOR_RUNTIME/"private"/f"fold_{fold}"/"A1_VLOO_PRIVATE.csv")
    if len(rows)!=13 or {int(r["fold"]) for r in rows}!={fold}:
        raise RuntimeError("A1 VLOO context grid changed")
    selected={}
    for row in rows:
        epoch=int(row["selected_epoch"]);tau_text=row["selected_threshold"]
        tau=float(tau_text)
        if epoch not in range(1,31) or not 0<tau<1:raise RuntimeError("Invalid A1 context")
        key=(epoch,tau_text)
        if key not in selected:
            digest=hashlib.sha256(f"{fold}|{epoch}|{tau_text}".encode()).hexdigest()[:10]
            selected[key]=dict(fold=fold,epoch=epoch,tau=tau,tau_text=tau_text,
                               context_id=f"e{epoch:02d}_{digest}",target_ids=[])
        selected[key]["target_ids"].append(row["subject_id"])
    return tuple(sorted(selected.values(),key=lambda x:(x["epoch"],x["tau"])))


def all_contexts():
    out=[ctx for fold in range(1,6) for ctx in contexts(fold)]
    if len(out)!=17 or sum(len(c["target_ids"]) for c in out)!=65:
        raise RuntimeError("Expected 17 source contexts for 65 cells")
    return out


@lru_cache(maxsize=3)
def payload(fold,epoch):return afc.load_representation(fold,epoch)


def patient_data(ctx,sid):
    source=payload(ctx["fold"],ctx["epoch"])
    fit=source["fit"]
    if sid not in fit:raise RuntimeError("Teacher can read only FIT patients")
    scaler=afr.scaler_for(source)
    row=fit[sid]
    z=np.asarray(scaler.transform(row["R4"]),dtype=np.float64)
    m0=np.asarray(afr.margin_for(row,ctx["tau"]),dtype=np.float64)
    y=np.asarray(row["y"],dtype=np.int8)
    if z.shape!=(len(y),64) or m0.shape!=y.shape or not np.isin(y,[0,1]).all():
        raise RuntimeError("FIT patient R4/margin/labels malformed")
    return z,m0,y


def stem(sid):return hashlib.sha256(sid.encode()).hexdigest()[:16]
