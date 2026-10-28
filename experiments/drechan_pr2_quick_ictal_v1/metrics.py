"""Patient-unit endpoints and unchanged historical threshold policies."""
import numpy as np
from scipy.special import expit
from sklearn.metrics import roc_auc_score,average_precision_score
KEYS=['auroc','ap','macro_f1','ez_f1','nez_f1','accuracy','mrr','top1','ez_fraction_bias']

def one(y,rank,probability,threshold,channel,kind='EZ_GE'):
    y=np.asarray(y,int);rank=np.asarray(rank,float);p=np.asarray(probability,float)
    pred=p>=threshold if kind=='EZ_GE' else (1-p)<threshold
    tp=int((pred&(y==1)).sum());tn=int((~pred&(y==0)).sum())
    fp=int((pred&(y==0)).sum());fn=int((~pred&(y==1)).sum())
    f1=2*tp/(2*tp+fp+fn) if y.sum() else np.nan
    f0=2*tn/(2*tn+fp+fn) if (1-y).sum() else np.nan
    order=np.lexsort((np.asarray(channel,str),-rank));hits=np.flatnonzero(y[order])
    return dict(auroc=float(roc_auc_score(y,rank)) if len(np.unique(y))==2 else np.nan,
                ap=float(average_precision_score(y,rank)) if y.sum() else np.nan,
                macro_f1=float(np.nanmean([f1,f0])),ez_f1=f1,nez_f1=f0,accuracy=(tp+tn)/len(y),
                mrr=1/(1+int(hits[0])) if len(hits) else np.nan,top1=float(y[order[0]]),
                ez_fraction_bias=float(pred.mean()-y.mean()))

def threshold(y,p,pid,kind='EZ_GE'):
    grid=np.round(np.arange(0,1.0001,.005),3);means=[];positive=[]
    pred=p[:,None]>=grid[None,:] if kind=='EZ_GE' else (1-p[:,None])<grid[None,:]
    for patient in np.unique(pid):
        ix=pid==patient;t=y[ix,None];v=pred[ix]
        tp=((t==1)&v).sum(0);tn=((t==0)&~v).sum(0);fp=((t==0)&v).sum(0);fn=((t==1)&~v).sum(0)
        f1=np.divide(2.*tp,2*tp+fp+fn,out=np.zeros(len(grid)),where=(2*tp+fp+fn)>0)
        f0=np.divide(2.*tn,2*tn+fp+fn,out=np.zeros(len(grid)),where=(2*tn+fp+fn)>0)
        if not y[ix].sum():f1[:]=np.nan
        if y[ix].sum()==ix.sum():f0[:]=np.nan
        means.append(np.nanmean([f1,f0],axis=0));positive.append(f1)
    means=np.nanmean(means,axis=0);positive=np.nanmean(positive,axis=0)
    j=max(range(len(grid)),key=lambda i:(means[i],positive[i],-abs(grid[i]-.5),-grid[i]))
    return dict(threshold=float(grid[j]),macro_f1=float(means[j]),ez_f1=float(positive[j]))

def patients(data,components,threshold_value,kind='EZ_GE'):
    p=components.get('probability_exact',expit(components['final']));rows=[]
    for patient in sorted(set(data['patient'])):
        ix=data['patient']==patient
        rows.append(dict(patient=str(patient),source=str(data['source'][ix][0]),
                    **one(data['y'][ix],components['rank_score'][ix],p[ix],threshold_value,data['channel'][ix],kind)))
    return rows

def validation(data,components,kind='EZ_GE'):
    p=expit(components['final']);sel=threshold(data['y'],p,data['patient'],kind)
    rows=patients(data,components,sel['threshold'],kind)
    sel.update(auroc=float(np.mean([r['auroc'] for r in rows])),ap=float(np.mean([r['ap'] for r in rows])))
    return sel
