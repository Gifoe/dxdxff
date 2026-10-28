"""Patient endpoints and independent finite-patient bootstrap."""
import math,itertools,hashlib
from functools import lru_cache
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score

RANK=['auroc','auprc','mrr','hit_at_1','hit_at_3','hit_at_5']
METRICS=RANK+['macro_f1','f1','nez_f1','accuracy','ez_fraction_bias']
SOURCES=['hup','lzu','multicenter','pediatric']
BUDGETS=[('fraction',.05),('fraction',.10),('fraction',.20),('fixed',1),('fixed',3),('fixed',5)]

def order(score,channels):
    return np.array(sorted(range(len(score)),key=lambda i:(-float(score[i]),str(channels[i]))))

def metrics(y,score,pred,channels,independent=False):
    y=np.asarray(y,int);score=np.asarray(score,float);pred=np.asarray(pred,bool)
    assert len(y) and len(y)==len(score)==len(pred)==len(channels) and np.isfinite(score).all()
    n=len(y);pos=int(y.sum());neg=n-pos;o=order(score,channels);ranked=y[o]
    tp=int(np.sum((y==1)&pred));tn=int(np.sum((y==0)&~pred));fp=int(np.sum((y==0)&pred));fn=pos-tp
    f1=2*tp/max(1,2*tp+fp+fn);f0=2*tn/max(1,2*tn+fp+fn)
    if independent:
        auc=float(np.mean((score[y==1,None]>score[None,y==0])+.5*(score[y==1,None]==score[None,y==0]))) if pos and neg else np.nan
        if pos:
            ends=np.flatnonzero(np.r_[score[o][1:]!=score[o][:-1],True]);c=np.cumsum(ranked)[ends];ap=float(np.sum(np.diff(np.r_[0,c])/pos*c/(ends+1)))
        else:ap=np.nan
    else:
        auc=float(roc_auc_score(y,score)) if pos and neg else np.nan
        ap=float(average_precision_score(y,score)) if pos else np.nan
    out=dict(auroc=auc,auprc=ap,mrr=1/float(np.flatnonzero(ranked)[0]+1) if pos else np.nan,
             macro_f1=(f1+f0)/2,f1=f1,nez_f1=f0,accuracy=(tp+tn)/n,ez_fraction_bias=(fp-fn)/n)
    out.update({f'hit_at_{k}':float(ranked[:min(k,n)].any()) if pos else np.nan for k in [1,3,5]})
    return out

def budget_values(y,score,channels,kind,value):
    y=np.asarray(y,int);n=len(y);k=max(1,int(math.ceil(value*n))) if kind=='fraction' else min(int(value),n)
    hit=int(y[order(score,channels)[:k]].sum());npos=int(y.sum());precision=hit/k
    return dict(recall=hit/npos if npos else np.nan,precision=precision,hit=float(hit>0) if npos else np.nan,
                nez_count=k-hit,lift=precision/(npos/n) if npos else np.nan,review_channels=k)

def draw(source):
    rng=np.random.default_rng(42);out=[]
    for s in SOURCES:
        ix=np.flatnonzero(np.asarray(source)==s)
        if len(ix):out.append(ix[rng.integers(0,len(ix),(10000,len(ix)))])
    return np.concatenate(out,axis=1)

@lru_cache(maxsize=128)
def resamples(source):
    ix=draw(source);counts=np.array([np.bincount(r,minlength=len(source)) for r in ix])
    return ix,counts

def interval(values,source,family=None):
    a=np.asarray(values,float);good=np.isfinite(a);a=a[good];source=np.asarray(source)[good]
    if not len(a):return dict(mean=np.nan,low=np.nan,high=np.nan,patients=0)
    ix,counts=resamples(tuple(source));bs=counts@a/len(a)
    direct=np.concatenate([a[ix[k:k+250]].mean(1) for k in range(0,10000,250)])
    assert np.max(np.abs(bs-direct))<1e-12
    low,high=np.quantile(bs,[.025,.975]);out=dict(mean=float(a.mean()),low=float(low),high=float(high),patients=len(a))
    if family:
        lo,hi=np.quantile(bs,[.05/(2*family),1-.05/(2*family)])
        out.update(adjusted_low=float(lo),adjusted_high=float(hi),family_size=family)
    return out

def seizure_subsets(patient,n,k):
    combos=list(itertools.combinations(range(n),k)) if k<n else [tuple(range(n))]
    if len(combos)<=50:return combos
    seed=int.from_bytes(hashlib.sha256(f'42|{patient}|K={k}|seizure_subsets_v1'.encode()).digest()[:8],'little')
    take=np.random.default_rng(seed).choice(len(combos),50,replace=False)
    return [combos[int(i)] for i in sorted(take)]
