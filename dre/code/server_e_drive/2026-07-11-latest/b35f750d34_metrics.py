from __future__ import annotations
import numpy as np
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,roc_auc_score,average_precision_score
def _safe(fn,*a):
    try:return float(fn(*a))
    except ValueError:return np.nan
def patient_metrics(rows,threshold=None):
    per=[]
    for sid in sorted({r["subject_id"] for r in rows}):
      rr=[r for r in rows if r["subject_id"]==sid]; y=np.array([int(r["true_nez"]) for r in rr]); p=np.array([float(r["score_nez_probability"]) for r in rr]); pred=np.array([int(r["predicted_nez"]) for r in rr]) if threshold is None and "predicted_nez" in rr[0] else p>=threshold; ez=1-y; risk=1-p; order=np.argsort(-risk); k=int(ez.sum()); rank=np.where(ez[order]>0)[0]
      per.append({"f1":f1_score(y,pred,average="macro",zero_division=0),"nez_f1":f1_score(y,pred,pos_label=1,zero_division=0),"ez_f1":f1_score(ez,1-pred,pos_label=1,zero_division=0),"accuracy":accuracy_score(y,pred),"balanced_accuracy":balanced_accuracy_score(y,pred),"auroc_ez":_safe(roc_auc_score,ez,risk),"auprc_ez":_safe(average_precision_score,ez,risk),"ez_mrr":1/(rank[0]+1) if len(rank) else 0,"ez_recall_at_true_count":ez[order[:k]].sum()/max(k,1),"top1_is_ez_rate":float(ez[order[0]])})
    result={"patient_macro_"+k:float(np.nanmean([x[k] for x in per])) if np.isfinite([x[k] for x in per]).any() else np.nan for k in per[0]} if per else {}
    if per:
      for k in ("auroc_ez","auprc_ez"):result["patient_macro_"+k+"_n_valid"]=int(np.isfinite([x[k] for x in per]).sum())
    return result
def pooled_metrics(rows):
    y=np.array([int(r["true_nez"]) for r in rows]); pred=np.array([int(r["predicted_nez"]) for r in rows]); p=np.array([float(r["score_nez_probability"]) for r in rows]); ez=1-y
    return {"pooled_macro_f1":f1_score(y,pred,average="macro",zero_division=0),"pooled_nez_f1":f1_score(y,pred,pos_label=1,zero_division=0),"pooled_ez_f1":f1_score(ez,1-pred,pos_label=1,zero_division=0),"pooled_balanced_accuracy":balanced_accuracy_score(y,pred),"pooled_auroc_ez":_safe(roc_auc_score,ez,1-p),"pooled_auprc_ez":_safe(average_precision_score,ez,1-p)}
def select_validation_threshold(rows):
    best=None
    for t in np.arange(.05,.951,.01):
      m=patient_metrics(rows,float(t)); key=(m["patient_macro_f1"],m["patient_macro_nez_f1"],m["patient_macro_balanced_accuracy"],-abs(t-.5))
      if best is None or key>best[0]:best=(key,float(round(t,2)),m)
    return best[1],best[2]
def build_oof_summary(rows,fold_metrics,protocol,feature_audit,protocol_audit):
    oof=patient_metrics(rows,None)|pooled_metrics(rows); centers={};
    for center in sorted({r["center"] for r in rows}): centers[center]=patient_metrics([r for r in rows if r["center"]==center],None)
    keys=[k for k in fold_metrics[0] if k!="fold_idx"] if fold_metrics else []; mean={k:float(np.mean([f[k] for f in fold_metrics])) for k in keys}; std={k:float(np.std([f[k] for f in fold_metrics])) for k in keys}; vals=[m["patient_macro_f1"] for m in centers.values()]
    return {"protocol":protocol,"oof_metrics":oof,"fold_mean":mean,"fold_std":std,"center_metrics":centers,"worst_center_patient_macro_f1":min(vals),"best_center_patient_macro_f1":max(vals),"center_gap_patient_macro_f1":max(vals)-min(vals),"fusion_weights":{k:mean[k] for k in ("a_anchor","a_early","a_recurrence")},"feature_audit":feature_audit,"protocol_audit":protocol_audit}
__all__=["patient_metrics","pooled_metrics","select_validation_threshold","build_oof_summary"]
