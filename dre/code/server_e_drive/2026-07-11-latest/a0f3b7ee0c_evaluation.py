from __future__ import annotations
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, f1_score, accuracy_score

def metric_row(frame):
    y, p = np.asarray(frame.outcome_true, int), np.asarray(frame.probability_success, float)
    pred = (p >= .5).astype(int); one = len(np.unique(y)) < 2
    return {"n_patients": len(y), "accuracy": accuracy_score(y,pred), "balanced_accuracy": balanced_accuracy_score(y,pred), "macro_f1": f1_score(y,pred,average="macro",zero_division=0), "auroc": float("nan") if one else roc_auc_score(y,p), "success_auprc": float("nan") if one else average_precision_score(y,p), "failure_auprc": float("nan") if one else average_precision_score(1-y,1-p), "success_recall": ((pred[y==1] == 1).mean() if (y==1).any() else float("nan")), "failure_recall": ((pred[y==0] == 0).mean() if (y==0).any() else float("nan")), "bce": float(-(y*np.log(np.clip(p,1e-7,1-1e-7))+(1-y)*np.log(np.clip(1-p,1e-7,1-1e-7))).mean()), "brier": float(((p-y)**2).mean()), "probability_mean":float(p.mean()), "probability_std":float(p.std())}
