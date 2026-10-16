"""EZ-positive patient metrics and explicitly declared threshold objectives."""
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score


def patient_rows(y,p,pid,channel,center,threshold):
    out=[]
    for who in np.unique(pid):
        ix=pid==who;a=np.asarray(y[ix],int);b=p[ix]>=threshold;score=p[ix]
        tp=int(((a==1)&b).sum());tn=int(((a==0)&~b).sum())
        fp=int(((a==0)&b).sum());fn=int(((a==1)&~b).sum())
        f1=2*tp/(2*tp+fp+fn) if a.sum() else np.nan
        f0=2*tn/(2*tn+fp+fn) if (a==0).sum() else np.nan
        order=np.lexsort((channel[ix].astype(str),-score));where=np.flatnonzero(a[order])
        out.append(dict(patient=str(who),center=str(center[ix][0]),auroc=float(roc_auc_score(a,score)) if len(set(a))==2 else np.nan,
            auprc=float(average_precision_score(a,score)) if a.sum() else np.nan,
            f1=f1,nez_f1=f0,macro_f1=float(np.nanmean([f0,f1])),accuracy=(tp+tn)/len(a),
            mrr=1/(int(where[0])+1) if len(where) else np.nan,hit_at_1=float(a[order[0]]) if a.sum() else np.nan,
            ez_fraction_bias=float(b.mean()-a.mean()),TP=tp,TN=tn,FP=fp,FN=fn,n=len(a)))
    return out


def threshold(y,p,pid,kind='ez'):
    grid=np.round(np.arange(0,1.0001,.005),3);pred=p[:,None]>=grid[None];ez=[];macro=[]
    for who in np.unique(pid):
        ix=pid==who;truth=y[ix,None];q=pred[ix]
        tp=((truth==1)&q).sum(0);tn=((truth==0)&~q).sum(0)
        fp=((truth==0)&q).sum(0);fn=((truth==1)&~q).sum(0)
        f1=np.divide(2.*tp,2*tp+fp+fn,out=np.zeros(len(grid)),where=(2*tp+fp+fn)>0)
        f0=np.divide(2.*tn,2*tn+fp+fn,out=np.zeros(len(grid)),where=(2*tn+fp+fn)>0)
        if not y[ix].sum():f1[:]=np.nan
        if y[ix].sum()==ix.sum():f0[:]=np.nan
        ez.append(f1);macro.append(np.nanmean(np.stack([f0,f1]),0))
    e=np.nanmean(ez,0);m=np.nanmean(macro,0)
    j=max(range(len(grid)),key=lambda i:((e[i],m[i]) if kind=='ez' else (m[i],e[i]))+(-abs(grid[i]-.5),-grid[i]))
    return dict(threshold=float(grid[j]),f1=float(e[j]),macro_f1=float(m[j]))
