from __future__ import annotations
import numpy as np,pandas as pd
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,roc_auc_score,average_precision_score,precision_score,recall_score
def metrics(frame):
    y=frame.outcome_true.to_numpy(int);p=frame.probability_success.to_numpy(float);pred=(p>=.5).astype(int);both=len(set(y))==2
    bins=np.minimum((p*10).astype(int),9);ece=sum((bins==b).mean()*abs(y[bins==b].mean()-p[bins==b].mean()) for b in set(bins))
    return {'n_patients':len(y),'accuracy':accuracy_score(y,pred),'balanced_accuracy':balanced_accuracy_score(y,pred),'macro_f1':f1_score(y,pred,average='macro',zero_division=0),'auroc':roc_auc_score(y,p) if both else np.nan,'success_auprc':average_precision_score(y,p) if both else np.nan,'failure_auprc':average_precision_score(1-y,1-p) if both else np.nan,'success_precision':precision_score(y,pred,pos_label=1,zero_division=0),'failure_precision':precision_score(y,pred,pos_label=0,zero_division=0),'success_recall':recall_score(y,pred,pos_label=1,zero_division=0),'failure_recall':recall_score(y,pred,pos_label=0,zero_division=0),'specificity':recall_score(y,pred,pos_label=0,zero_division=0),'brier':float(np.mean((p-y)**2)),'ece':float(ece),'probability_mean':float(p.mean()),'probability_std':float(p.std())}
def assert_oof(frame,expected=None):
    if frame.patient_key.duplicated().any():raise RuntimeError('duplicate OOF patient_key')
    if expected is not None and len(frame)!=expected:raise RuntimeError('OOF row count mismatch')
def bootstrap(frame,repeats=2000,seed=42):
    rng=np.random.default_rng(seed);rows=[];skipped=0
    for _ in range(repeats):
        sample=frame.iloc[rng.integers(0,len(frame),len(frame))]
        if sample.outcome_true.nunique()<2:skipped+=1;continue
        m=metrics(sample);rows.append({'auroc':m['auroc'],'balanced_accuracy':m['balanced_accuracy'],'macro_f1':m['macro_f1']})
    result=[]
    for key in ('auroc','balanced_accuracy','macro_f1'):
        values=np.array([x[key] for x in rows]);result.append({'metric':key,'estimate':metrics(frame)[key],'ci_low':np.quantile(values,.025),'ci_high':np.quantile(values,.975),'valid_repeats':len(values),'skipped_single_class':skipped})
    return pd.DataFrame(result)
