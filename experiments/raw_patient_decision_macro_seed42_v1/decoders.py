"""One-scalar score-only decisions; no Gaussian/posterior or identity features."""
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score
from common import probability


def score_z(logit,patient):
    x=np.asarray(logit,np.float64);patient=np.asarray(patient);out=np.empty_like(x)
    assert np.isfinite(x).all() and len(x)>0
    for pid in np.unique(patient):
        ix=patient==pid;q=x[ix];sd=q.std(ddof=0)
        out[ix]=(q-q.mean())/(sd if sd>1e-8 else 1.)
    return out


def fraction_decisions(logit,q):
    x=np.asarray(logit,np.float64);assert len(x)>0 and np.isfinite(x).all() and 0<=q<=1
    values,counts=np.unique(x,return_counts=True);cumulative=np.cumsum(counts);feasible=np.r_[0,cumulative]
    k=int(feasible[np.argmin(np.abs(feasible-float(q)*len(x)))])
    if k==0:return np.zeros(len(x),bool)
    return x<=values[int(np.searchsorted(cumulative,k))]


def predict(config,logit,patient):
    x=np.asarray(logit,np.float64);patient=np.asarray(patient);assert np.isfinite(x).all() and len(x)==len(patient)>0
    family=config['family']
    if family=='ORIGINAL':return probability(x)<config['value']
    if family=='MARGIN_GRID':
        if config['mode']=='none':return np.zeros(len(x),bool)
        if config['mode']=='all':return np.ones(len(x),bool)
        return x<config['value']
    if family=='PATIENT_Z':return score_z(x,patient)<config['value']
    assert family=='TOP_FRACTION';out=np.empty(len(x),bool)
    for pid in np.unique(patient):
        ix=patient==pid;out[ix]=fraction_decisions(x[ix],config['value'])
    return out


def candidates(family,logit):
    if family=='MARGIN_GRID':
        values=np.quantile(np.asarray(logit,float),np.linspace(.005,.995,199),method='linear')
        return [dict(family=family,mode='none',value=None)]+[dict(family=family,mode='threshold',value=float(v)) for v in values]+[dict(family=family,mode='all',value=None)]
    values=np.linspace(-3,3,201) if family=='PATIENT_Z' else np.linspace(0,1,101)
    return [dict(family=family,value=float(v)) for v in values]


def utility(y,pred):
    y=np.asarray(y,int);pred=np.asarray(pred,bool)
    if pred.ndim==1:pred=pred[:,None]
    tp=((y[:,None]==1)&pred).sum(0);tn=((y[:,None]==0)&~pred).sum(0)
    fp=((y[:,None]==0)&pred).sum(0);fn=((y[:,None]==1)&~pred).sum(0)
    a=np.divide(2.*tp,2*tp+fp+fn,out=np.zeros(pred.shape[1]),where=(2*tp+fp+fn)>0)
    b=np.divide(2.*tn,2*tn+fp+fn,out=np.zeros(pred.shape[1]),where=(2*tn+fp+fn)>0)
    if not y.sum():a[:]=np.nan
    if not (1-y).sum():b[:]=np.nan
    return np.nanmean(np.stack([a,b]),axis=0),a


def fit(family,logit,y_nez,patient,original):
    x=np.asarray(logit,np.float64);patient=np.asarray(patient);y=1-np.asarray(y_nez,int)
    configs=candidates(family,x);grid=np.stack([predict(c,x,patient) for c in configs],1)
    macro=[];ez=[];changed=[]
    base=predict(original,x,patient)
    for pid in np.unique(patient):
        ix=patient==pid;m,f=utility(y[ix],grid[ix]);macro.append(m);ez.append(f)
        changed.append(np.mean(grid[ix]!=base[ix,None],axis=0))
    macro=np.nanmean(macro,axis=0);ez=np.nanmean(ez,axis=0);changed=np.mean(changed,axis=0)
    keys=[]
    for i,c in enumerate(configs):
        v=c['value'];neutral=.5 if family=='TOP_FRACTION' else 0.
        distance=abs(v-neutral) if v is not None else float('inf')
        keys.append((float(macro[i]),float(ez[i]),-float(changed[i]),-distance,-i))
    j=max(range(len(keys)),key=keys.__getitem__)
    return configs[j],dict(calibration_macro_f1=float(macro[j]),calibration_ez_f1=float(ez[j]),
                           changed_fraction=float(changed[j]),candidate_count=len(configs))


def patient_metrics(y,logit,pred,channel):
    y=np.asarray(y,int);logit=np.asarray(logit,np.float64);pred=np.asarray(pred,bool);channel=np.asarray(channel,str)
    macro,ez=utility(y,pred);_,nez=utility(1-y,~pred)
    order=np.lexsort((channel,logit));positive=int(y.sum())
    first=int(np.flatnonzero(y[order])[0]) if positive else None
    return dict(macro_f1=float(macro[0]),f1=float(ez[0]),nez_f1=float(nez[0]),accuracy=float(np.mean(y==pred)),
                auroc=float(roc_auc_score(y,-logit)) if positive and (1-y).sum() else np.nan,
                auprc=float(average_precision_score(y,-logit)) if positive else np.nan,
                mrr=1/(first+1) if positive else np.nan,hit_at_1=float(first==0) if positive else np.nan,
                ez_fraction_bias=float(pred.mean()-y.mean()))


def rows(d,ix,logit,pred):
    result=[]
    for pid in sorted(set(d['patient'][ix])):
        loc=d['patient'][ix]==pid
        result.append(dict(patient=str(pid),center=str(d['center'][ix][loc][0]),
                      **patient_metrics(1-d['y'][ix][loc],logit[loc],pred[loc],d['channel'][ix][loc])))
    return result


def oracle_macro(y,logit):
    x=np.asarray(logit,np.float64);values=np.unique(x)
    grid=np.column_stack([np.zeros(len(x),bool)]+[x<=v for v in values])
    macro,_=utility(y,grid)
    return float(np.nanmax(macro))
