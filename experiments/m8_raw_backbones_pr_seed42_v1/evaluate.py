"""EZ-positive metrics; patient-local ranking; no balanced-accuracy calculation."""
import warnings
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score

CLASSIFICATION=['accuracy','precision','recall','specificity','f1','macro_f1','nez_f1','weighted_f1','macro_precision','macro_recall','auroc','auprc','mcc','kappa','npv','fpr','fnr','brier','log_loss','ece','predicted_ez_fraction','true_ez_fraction','ez_fraction_bias']
RANKING=['mrr','hit_at_1','hit_at_3','hit_at_5','precision_at_3','recall_at_3','ndcg_at_5']
COUNTS=['TP','TN','FP','FN','n','positives','negatives']

def div(a,b):return float(a/b) if b else np.nan
def nanmean(a):
    a=np.asarray(a,dtype=float);return float(a[np.isfinite(a)].mean()) if np.isfinite(a).any() else np.nan

def classification(y,p,pred):
    y=np.asarray(y,dtype=int);p=np.asarray(p,dtype=np.float64);pred=np.asarray(pred,dtype=bool)
    assert y.shape==p.shape==pred.shape and len(y)>0 and set(y)<=set([0,1])
    assert np.isfinite(p).all() and (p>=0).all() and (p<=1).all()
    tp=int(((y==1)&pred).sum());fp=int(((y==0)&pred).sum());tn=int(((y==0)&~pred).sum());fn=int(((y==1)&~pred).sum());n=len(y)
    pos=tp+fn;neg=tn+fp;precision=div(tp,tp+fp);npv=div(tn,tn+fn)
    recall=div(tp,pos);specificity=div(tn,neg)
    # No true-positive reference => positive-class F1 unavailable, not a silent zero.
    f1=div(2*tp,2*tp+fp+fn) if pos else np.nan
    f0=div(2*tn,2*tn+fp+fn) if neg else np.nan
    den=float((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn));expected=((tp+fn)*(tp+fp)+(tn+fp)*(tn+fn))/n**2
    ece=0.;bins=np.minimum((p*10).astype(int),9)
    for k in range(10):
        ix=bins==k
        if ix.any():ece+=ix.mean()*abs(float(y[ix].mean()-p[ix].mean()))
    q=np.clip(p,1e-15,1-1e-15)
    return dict(accuracy=(tp+tn)/n,precision=precision,recall=recall,specificity=specificity,f1=f1,macro_f1=nanmean([f0,f1]),nez_f1=f0,weighted_f1=((f1*pos if pos else 0)+(f0*neg if neg else 0))/n,macro_precision=nanmean([precision,npv]),macro_recall=nanmean([recall,specificity]),auroc=float(roc_auc_score(y,p)) if pos and neg else np.nan,auprc=float(average_precision_score(y,p)) if pos else np.nan,mcc=(tp*tn-fp*fn)/np.sqrt(den) if den else np.nan,kappa=div((tp+tn)/n-expected,1-expected),npv=npv,fpr=div(fp,neg),fnr=div(fn,pos),brier=float(np.mean((p-y)**2)),log_loss=float(-np.mean(y*np.log(q)+(1-y)*np.log1p(-q))),ece=float(ece),TP=tp,TN=tn,FP=fp,FN=fn,n=n,positives=pos,negatives=neg,predicted_ez_fraction=float(pred.mean()),true_ez_fraction=float(pos/n),ez_fraction_bias=float((fp-fn)/n))

def ranking(y,p,channel):
    y=np.asarray(y,dtype=int);p=np.asarray(p,dtype=float);channel=np.asarray(channel,dtype=str)
    # Canonical name is the tie key; scores never rounded before ordering.
    order=np.lexsort((channel,-p));yy=y[order];positive=int(y.sum())
    if not positive:return dict.fromkeys(RANKING,np.nan)
    first=int(np.flatnonzero(yy)[0]);k3=min(3,len(y));k5=min(5,len(y))
    gains=1/np.log2(np.arange(k5)+2);dcg=float((yy[:k5]*gains).sum());ideal=float(gains[:min(positive,k5)].sum())
    return dict(mrr=1/(first+1),hit_at_1=float(yy[:1].any()),hit_at_3=float(yy[:k3].any()),hit_at_5=float(yy[:k5].any()),precision_at_3=float(yy[:k3].sum()/k3),recall_at_3=float(yy[:k3].sum()/positive),ndcg_at_5=dcg/ideal)

def patient_rows(frame):
    out=[]
    for (model,version,fold,patient),q in frame.groupby(['model','version','fold','patient'],sort=True):
        q=q.sort_values('channel',kind='stable')
        out.append(dict(model=model,version=version,fold=fold,patient=patient,center=q.center.iloc[0],**classification(q.y,q.probability,q.prediction),**ranking(q.y,q.probability,q.channel)))
    return out

def threshold(y,p,patient):
    """Common validation-only grid/ties; no forbidden auxiliary metric in ties."""
    y=np.asarray(y,dtype=int);p=np.asarray(p,dtype=float);patient=np.asarray(patient)
    grid=np.round(np.arange(0,1.0001,.005),3);pred=p[:,None]>=grid[None,:];macro=[];positive=[]
    for pid in np.unique(patient):
        ix=patient==pid;truth=y[ix,None];b=pred[ix]
        tp=((truth==1)&b).sum(0);tn=((truth==0)&~b).sum(0);fp=((truth==0)&b).sum(0);fn=((truth==1)&~b).sum(0)
        f1=np.divide(2.*tp,2*tp+fp+fn,out=np.zeros(len(grid)),where=(2*tp+fp+fn)>0)
        f0=np.divide(2.*tn,2*tn+fp+fn,out=np.zeros(len(grid)),where=(2*tn+fp+fn)>0)
        if not y[ix].sum():f1[:]=np.nan
        if y[ix].sum()==ix.sum():f0[:]=np.nan
        macro.append(np.nanmean(np.stack([f0,f1]),axis=0));positive.append(f1)
    means=np.nanmean(macro,axis=0);pos=np.nanmean(positive,axis=0)
    keys=[(float(means[i]),float(pos[i]) if np.isfinite(pos[i]) else -np.inf,-abs(float(t)-.5),-float(t),i) for i,t in enumerate(grid)]
    j=max(keys)[-1]
    return dict(threshold=float(grid[j]),validation_macro_f1=float(means[j]),validation_positive_f1=float(pos[j]))
