"""Atomic artifacts, exact original patient metrics, appearance-cluster statistics."""
import hashlib
import json
from pathlib import Path
import numpy as np

METRICS=['macro_f1','ez_f1','nez_f1','balanced_accuracy','ez_auprc','ez_auroc',
         'ez_mrr','top1_is_ez','sensitivity','specificity','accuracy']

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,default=str).encode()).hexdigest()

def write_json(path,obj):
    def clean(x):
        if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
        if isinstance(x,(list,tuple,np.ndarray)):return [clean(v) for v in x]
        if isinstance(x,np.integer):return int(x)
        if isinstance(x,np.bool_):return bool(x)
        if isinstance(x,(float,np.floating)):return float(x) if np.isfinite(x) else None
        return x
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.partial'); tmp.write_text(json.dumps(clean(obj),indent=2,sort_keys=True,allow_nan=False),encoding='utf-8'); tmp.replace(path)

def metric(y,score,tau,pred=None,order_score=None):
    y=np.asarray(y); score=np.asarray(score,dtype=float)
    assert len(y)>0 and set(y)<={0,1} and np.isfinite(score).all()
    pred=score>=tau if pred is None else np.asarray(pred,dtype=bool)
    tp=int(((y==0)&~pred).sum()); fp=int(((y==1)&~pred).sum()); tn=int(((y==1)&pred).sum()); fn=int(((y==0)&pred).sum())
    ef=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.
    nf=2*tn/(2*tn+fp+fn) if 2*tn+fp+fn else 0.
    se=tp/(tp+fn) if tp+fn else np.nan; sp=tn/(tn+fp) if tn+fp else np.nan
    rank=score if order_score is None else np.asarray(order_score,dtype=float)
    order=np.argsort(rank,kind='stable'); positives=np.flatnonzero(y[order]==0)
    ends=np.r_[np.flatnonzero(np.diff(rank[order])!=0),len(y)-1]
    ctp=np.cumsum(y[order]==0)[ends].astype(float); cfp=ends+1-ctp
    np_=int((y==0).sum()); nn=len(y)-np_
    ap=float(np.sum(np.diff(np.r_[0,ctp])*ctp/(ends+1))/np_) if np_ else 0.
    auc=float(np.sum(np.diff(np.r_[0,cfp])*(ctp+np.r_[0,ctp[:-1]])/2)/(np_*nn)) if np_ and nn else np.nan
    return {'macro_f1':(ef+nf)/2,'ez_f1':ef,'nez_f1':nf,'balanced_accuracy':float(np.nanmean([se,sp])),
        'ez_auprc':ap,'ez_auroc':auc,'ez_mrr':float(1/(positives[0]+1)) if len(positives) else 0.,
        'top1_is_ez':float(y[order[0]]==0),'sensitivity':se,'specificity':sp,'accuracy':(tp+tn)/len(y),
        'TP':tp,'FP':fp,'TN':tn,'FN':fn,'channels':len(y),
        'predicted_ez_fraction':float(np.mean(~pred)),'observed_ez_fraction':float(np.mean(y==0))}

def cluster_bootstrap(patient,values,draws=10000):
    patient=np.asarray(patient).astype(str); values=np.asarray(values,dtype=float)
    if values.ndim==1:values=values[:,None]
    ids=np.unique(patient); sums=[]; counts=[]
    for p in ids:
        a=values[patient==p]; sums.append(np.nansum(a,axis=0)); counts.append(np.isfinite(a).sum(0))
    sums=np.asarray(sums); counts=np.asarray(counts)
    ix=np.random.default_rng(42).integers(0,len(ids),size=(draws,len(ids)))
    num=sums[ix].sum(1); den=counts[ix].sum(1)
    return np.divide(num,den,out=np.full_like(num,np.nan),where=den>0)

