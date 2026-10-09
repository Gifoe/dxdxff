"""Deterministic label-budget probes and exhaustive threshold decisions."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.special import expit, logit
from sklearn.decomposition import PCA

METRICS=['macro_f1','ez_f1','nez_f1','ez_auprc','ez_auroc','ez_mrr','top1_is_ez','sensitivity','specificity','accuracy']

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def write_json(path,obj):
    def clean(x):
        if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
        if isinstance(x,(list,tuple,np.ndarray)):return [clean(v) for v in x]
        if isinstance(x,(np.integer,)):return int(x)
        if isinstance(x,(float,np.floating)):return float(x) if np.isfinite(x) else None
        return x
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.partial'); tmp.write_text(json.dumps(clean(obj),indent=2,sort_keys=True,allow_nan=False),encoding='utf-8'); tmp.replace(path)

def seed(*parts):
    return int.from_bytes(hashlib.sha256(json.dumps([42,*parts]).encode()).digest()[:8],'little')

def metric(y,score,tau,order_score=None,pred=None):
    y=np.asarray(y); score=np.asarray(score,dtype=float)
    assert len(y)>0 and set(y)<= {0,1} and np.isfinite(score).all()
    pred=score>=tau if pred is None else np.asarray(pred,dtype=bool)
    tp=int(((y==0)&~pred).sum()); fp=int(((y==1)&~pred).sum()); tn=int(((y==1)&pred).sum()); fn=int(((y==0)&pred).sum())
    divide=lambda a,b:a/b if b else np.nan
    ef=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.
    nf=2*tn/(2*tn+fp+fn) if 2*tn+fp+fn else 0.
    rank=score if order_score is None else np.asarray(order_score,dtype=float)
    order=np.argsort(rank,kind='stable'); positive=np.flatnonzero(y[order]==0)
    # Same non-interpolated AP and trapezoidal ROC as sklearn, including tied-score groups.
    # Avoid thousands of sklearn decorator/inspect calls in the native-crash-prone host.
    ordered_ez=(y[order]==0).astype(int); ends=np.r_[np.flatnonzero(np.diff(rank[order])!=0),len(y)-1]
    cumulative_tp=np.cumsum(ordered_ez)[ends].astype(float); cumulative_fp=ends+1-cumulative_tp
    npos=int((y==0).sum()); nneg=len(y)-npos
    ap=float(np.sum(np.diff(np.r_[0,cumulative_tp])*cumulative_tp/(ends+1))/npos) if npos else np.nan
    auc=float(np.sum(np.diff(np.r_[0,cumulative_fp])*(cumulative_tp+np.r_[0,cumulative_tp[:-1]])/2)/(npos*nneg)) if npos and nneg else np.nan
    return {'macro_f1':(ef+nf)/2,'ez_f1':ef,'nez_f1':nf,
        'ez_auprc':ap,'ez_auroc':auc,
        'ez_mrr':float(1/(positive[0]+1)) if len(positive) else np.nan,
        'top1_is_ez':float(y[order[0]]==0),'sensitivity':divide(tp,tp+fn),'specificity':divide(tn,tn+fp),
        'accuracy':divide(tp+tn,len(y)),'TP':tp,'FP':fp,'TN':tn,'FN':fn,
        'predicted_ez_fraction':float(np.mean(~pred)),'observed_ez_fraction':float(np.mean(y==0)), 'channels':len(y)}

def oracle(y,score,tau):
    v=np.unique(score); representatives=np.r_[v,np.nextafter(v[-1],np.inf)]
    base=metric(y,score,tau)
    def decisions(thresholds):
        pred=np.asarray(score)[:,None]>=np.asarray(thresholds)[None,:]; target=np.asarray(y)[:,None]
        tp=((target==0)&~pred).sum(0); fp=((target==1)&~pred).sum(0); tn=((target==1)&pred).sum(0); fn=((target==0)&pred).sum(0)
        ef=np.divide(2.*tp,2*tp+fp+fn,out=np.zeros(len(thresholds)),where=(2*tp+fp+fn)>0)
        nf=np.divide(2.*tn,2*tn+fp+fn,out=np.zeros(len(thresholds)),where=(2*tn+fp+fn)>0)
        return [{'macro_f1':float((e+n)/2),'ez_f1':float(e)} for e,n in zip(ef,nf)]
    values=decisions(representatives)
    maximum=max(m['macro_f1'] for m in values)
    indices=[i for i,m in enumerate(values) if abs(m['macro_f1']-maximum)<=1e-14]
    intervals=[(-np.inf if i==0 else v[i-1], np.inf if i==len(v) else v[i]) for i in indices]
    distance=lambda ab:max(ab[0]-tau,0,tau-ab[1])
    pick=min(indices,key=lambda i:(distance(intervals[indices.index(i)]),-values[i]['ez_f1'],i))
    grid=np.round(np.arange(0,1.0001,.005),3); gm=decisions(grid)
    gi=max(range(len(grid)),key=lambda i:(gm[i]['macro_f1'],gm[i]['ez_f1'],-abs(grid[i]-tau),-grid[i]))
    order=np.argsort(score,kind='stable'); pred=np.ones(len(y),dtype=bool); k=int((y==0).sum()); pred[order[:k]]=False
    topk=metric(y,score,tau,pred=pred)
    assert maximum>=base['macro_f1']-1e-14
    assert maximum>=gm[gi]['macro_f1']-1e-14
    selected=metric(y,score,representatives[pick]); grid_selected=metric(y,score,grid[gi])
    return {'baseline':base,'oracle':selected,'grid_oracle':grid_selected,'true_count_topk':topk,
        'headroom':maximum-base['macro_f1'],'grid_resolution_loss':maximum-gm[gi]['macro_f1'],
        'optimal_intervals':intervals,'optimal_decision_sets':len(indices),
        'disconnected':any(b!=a+1 for a,b in zip(indices,indices[1:])),
        'distance':min(map(distance,intervals)),
        'cardinality_change':selected['TP']+selected['FP']-base['TP']-base['FP'],
        'topk_cuts_tie':bool(0<k<len(y) and score[order[k-1]]==score[order[k]])}

def support_indices(score,channels,budget,policy,rngseed):
    score=np.asarray(score); channels=np.asarray(channels).astype(str)
    assert 0<budget<len(score) and len(set(channels))==len(channels)
    if policy=='uncertainty':order=np.lexsort((channels,score))
    elif policy=='random':
        # Canonical input order, no labels or centers are accepted by this API.
        order=np.random.default_rng(rngseed).permutation(np.argsort(channels,kind='stable'))
    else:raise ValueError(policy)
    support=np.sort(order[:budget]); query=np.setdiff1d(np.arange(len(score)),support)
    assert not set(support)&set(query)
    return support,query

def fit_probe(offset,z,y,lam,direction):
    """Mean BCE + lambda/2 ||theta||², zero-start damped Newton, float64."""
    offset=np.asarray(offset,dtype=float); z=np.asarray(z,dtype=float); y=np.asarray(y,dtype=float)
    assert lam>0 and offset.shape==y.shape and np.isfinite(offset).all() and np.isfinite(z).all()
    a=np.column_stack([np.ones(len(y)),z]) if direction else np.ones((len(y),1))
    theta=np.zeros(a.shape[1]); objective=lambda t:np.mean(np.logaddexp(0,offset+a@t)-y*(offset+a@t))+.5*lam*(t@t)
    for step in range(200):
        p=expit(offset+a@theta); g=a.T@(p-y)/len(y)+lam*theta
        if np.max(np.abs(g))<1e-9:return theta,{'converged':True,'iterations':step,'gradient_inf':float(np.max(np.abs(g))),'one_class':len(set(y))==1}
        h=(a.T*(p*(1-p)))@a/len(y)+lam*np.eye(a.shape[1]); delta=np.linalg.solve(h,g)
        rate=1.; old=objective(theta)
        while rate>2**-35:
            candidate=theta-rate*delta
            if objective(candidate)<=old-1e-4*rate*(g@delta)+1e-15:break
            rate*=.5
        theta=candidate
    return np.zeros_like(theta),{'converged':False,'iterations':200,'gradient_inf':float(np.max(np.abs(g))),'one_class':len(set(y))==1}

def episode(y,score,z,channels,tau,budget,policy,rngseed,lambdas):
    s,q=support_indices(np.abs(score-tau),channels,budget,policy,rngseed)
    offset=logit(np.clip(score,1e-12,1-1e-12)); out={'P0':metric(y[q],score[q],tau)}; checks=[]
    coeff={}
    for method,direction,lam in [('P1',False,lambdas[0]),('P2',True,lambdas[1]),('P2_permuted',True,lambdas[1])]:
        sy=y[s].copy()
        if method=='P2_permuted':sy=np.random.default_rng(seed(rngseed,'permute')).permutation(sy)
        theta,audit=fit_probe(offset[s],z[s],sy,lam,direction); checks.append(audit); coeff[method]=theta
        design=np.column_stack([np.ones(len(q)),z[q]]) if direction else np.ones((len(q),1))
        newoffset=offset[q]+design@theta
        # Ranking represented by logits, so constant shifts cannot introduce sigmoid saturation ties.
        out[method]=metric(y[q],expit(newoffset),tau,order_score=offset[q] if method=='P1' else newoffset)
    assert all(abs(out['P1'][m]-out['P0'][m])<1e-12 or (np.isnan(out['P1'][m]) and np.isnan(out['P0'][m])) for m in ['ez_auprc','ez_auroc','ez_mrr','top1_is_ez'])
    return out,{'support':s,'query':q,'one_class':len(set(y[s]))==1,'checks':checks,'coeff':coeff,
        'ranking_change_fraction':float(np.mean(np.argsort(offset[q],kind='stable')!=np.argsort(offset[q]+np.column_stack([np.ones(len(q)),z[q]])@coeff['P2'],kind='stable')))}

def fit_pca(x,fit):
    pca=PCA(n_components=4,svd_solver='full'); pca.fit(np.asarray(x)[fit]); return pca,pca.transform(x).astype(float)

def cluster_bootstrap(patient,values,draws=10000):
    """Equal appearance estimand; resample unique ID, retain all of its appearances."""
    patient=np.asarray(patient).astype(str); values=np.asarray(values,dtype=float)
    if values.ndim==1:values=values[:,None]
    ids=np.unique(patient); sums=[]; counts=[]
    for p in ids:
        a=values[patient==p]; sums.append(np.nansum(a,axis=0)); counts.append(np.isfinite(a).sum(0))
    sums=np.asarray(sums); counts=np.asarray(counts)
    draws_ix=np.random.default_rng(42).integers(0,len(ids),size=(draws,len(ids)))
    numerator=sums[draws_ix].sum(1); denominator=counts[draws_ix].sum(1)
    boot=np.divide(numerator,denominator,out=np.full_like(numerator,np.nan),where=denominator>0)
    return boot
