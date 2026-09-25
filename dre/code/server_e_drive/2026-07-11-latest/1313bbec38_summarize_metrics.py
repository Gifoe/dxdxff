#!/usr/bin/env python3
"""Combine five outer-fold channel OOF files into paper-ready metric rows."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score)


def safe(fn, default=float("nan")):
    try: return fn()
    except ValueError: return default


def main() -> int:
    p = argparse.ArgumentParser(); p.add_argument("--root", type=Path, required=True); p.add_argument("--models", default="omni_seegnet,omni_timeconv_cnn,omni_clap"); p.add_argument("--seed", type=int, default=42); p.add_argument("--output", type=Path, required=True); a=p.parse_args()
    rows = []
    for model in [x.strip() for x in a.models.split(",") if x.strip()]:
        paths = sorted(a.root.glob(f"*fold_*/oof_ledgers/{model}/seed_{a.seed}_channel_oof.csv"))
        if len(paths) != 5:
            print(f"SKIP {model}: expected five OOF files, found {len(paths)}")
            continue
        x = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
        if x.duplicated(["subject_id", "channel_name"]).any(): raise ValueError(f"duplicate OOF channel for {model}")
        y=x.label_nez.astype(int).to_numpy(); pred=x.predicted_nez.astype(int).to_numpy(); score=x.score_nez_probability.to_numpy(); tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
        patient=[]
        for _, g in x.groupby("subject_id"):
            gy=g.label_nez.astype(int); gp=g.predicted_nez.astype(int)
            patient.append((f1_score(gy,gp,average="macro",zero_division=0),f1_score(gy,gp,pos_label=0,zero_division=0),f1_score(gy,gp,pos_label=1,zero_division=0),accuracy_score(gy,gp)))
        patient=np.asarray(patient)
        rows.append({"model":model,"seed":a.seed,"accuracy":accuracy_score(y,pred),"balanced_accuracy":balanced_accuracy_score(y,pred),"precision_macro":precision_score(y,pred,average="macro",zero_division=0),"recall_macro":recall_score(y,pred,average="macro",zero_division=0),"f1_macro":f1_score(y,pred,average="macro",zero_division=0),"f1_weighted":f1_score(y,pred,average="weighted",zero_division=0),"precision_nez":precision_score(y,pred,pos_label=1,zero_division=0),"recall_nez":recall_score(y,pred,pos_label=1,zero_division=0),"f1_nez":f1_score(y,pred,pos_label=1,zero_division=0),"precision_ez":precision_score(y,pred,pos_label=0,zero_division=0),"recall_ez":recall_score(y,pred,pos_label=0,zero_division=0),"f1_ez":f1_score(y,pred,pos_label=0,zero_division=0),"AUROC_NEZ":safe(lambda:roc_auc_score(y,score)),"AUPRC_NEZ":safe(lambda:average_precision_score(y,score)),"AUROC_EZ":safe(lambda:roc_auc_score(1-y,1-score)),"AUPRC_EZ":safe(lambda:average_precision_score(1-y,1-score)),"MCC":matthews_corrcoef(y,pred),"TN":tn,"FP":fp,"FN":fn,"TP":tp,"n_channels":len(x),"n_patients":x.subject_id.nunique(),"patient_macro_f1":patient[:,0].mean(),"patient_ez_f1":patient[:,1].mean(),"patient_nez_f1":patient[:,2].mean(),"patient_accuracy":patient[:,3].mean()})
    a.output.parent.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_csv(a.output,index=False); print(pd.DataFrame(rows).to_string(index=False)); print(a.output)
    return 0

if __name__ == "__main__": raise SystemExit(main())
