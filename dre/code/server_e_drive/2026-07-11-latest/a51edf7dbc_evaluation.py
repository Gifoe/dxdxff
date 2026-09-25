from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score,balanced_accuracy_score,f1_score,roc_auc_score,
                             confusion_matrix,average_precision_score,precision_score,brier_score_loss)

def expected_calibration_error(y,p,bins=10):
    y=np.asarray(y); p=np.asarray(p); edges=np.linspace(0,1,bins+1); value=0.0
    for lo,hi in zip(edges[:-1],edges[1:]):
        keep=(p>=lo)&((p<hi) if hi<1 else (p<=hi))
        if keep.any(): value += keep.mean()*abs(y[keep].mean()-p[keep].mean())
    return float(value)


def metric_row(frame):
    y=frame.outcome_true.to_numpy(int); p=frame.probability_success.to_numpy(float); pred=(p>=.5).astype(int)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {"n_patients":len(y),"accuracy":accuracy_score(y,pred),"balanced_accuracy":balanced_accuracy_score(y,pred),"macro_f1":f1_score(y,pred,average="macro"),"auroc":roc_auc_score(y,p) if len(np.unique(y))==2 else np.nan,
            "success_auprc":average_precision_score(y,p) if (y==1).any() else np.nan,"failure_auprc":average_precision_score(1-y,1-p) if (y==0).any() else np.nan,
            "success_precision":precision_score(y,pred,pos_label=1,zero_division=0),"failure_precision":precision_score(y,pred,pos_label=0,zero_division=0),
            "failure_recall":tn/max(tn+fp,1),"success_recall":tp/max(tp+fn,1),"brier":brier_score_loss(y,p),"ece":expected_calibration_error(y,p),"threshold":.5}


def bootstrap_ci(frame,repeats=2000,seed=42):
    rng=np.random.default_rng(seed); rows=[]
    for _ in range(repeats):
        sample=frame.iloc[rng.integers(0,len(frame),len(frame))]
        rows.append(metric_row(sample))
    result=[]
    for key in ("accuracy","balanced_accuracy","macro_f1","auroc","success_auprc","failure_auprc","success_precision","success_recall","failure_precision","failure_recall","brier","ece"):
        v=np.asarray([x[key] for x in rows],float); result.append({"metric":key,"estimate":metric_row(frame)[key],"ci_lower":np.nanquantile(v,.025),"ci_upper":np.nanquantile(v,.975),"repeats":repeats})
    return pd.DataFrame(result)


__all__=["expected_calibration_error","metric_row","bootstrap_ci"]
