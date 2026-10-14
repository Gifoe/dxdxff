"""Exact validation confusion plateaus; no TEST score/label input."""
import numpy as np

def select_exact(y,score,patient):
    score=np.asarray(score,dtype=float);values=np.unique(score);grid=np.unique(np.r_[0.,.5,1.,values[:-1]+(values[1:]-values[:-1])/2]);pred=score[:,None]>=grid[None,:];macros=[];ez=[];counts=[]
    for pid in np.unique(patient):
        k=patient==pid;a=np.asarray(y)[k,None];b=pred[k];tp=((a==1)&b).sum(0);tn=((a==0)&~b).sum(0);fp=((a==0)&b).sum(0);fn=((a==1)&~b).sum(0);f1=np.divide(2.*tp,2*tp+fp+fn,out=np.zeros(len(grid)),where=2*tp+fp+fn>0);f0=np.divide(2.*tn,2*tn+fp+fn,out=np.zeros(len(grid)),where=2*tn+fp+fn>0);macros.append((f0+f1)/2);ez.append(f0);counts.append((tp,tn,fp,fn))
    tp,tn,fp,fn=np.asarray(counts).sum(0);recalls=[]
    if (y==1).any():recalls.append(tp/(tp+fn))
    if (y==0).any():recalls.append(tn/(tn+fp))
    ba=np.mean(recalls,0);m=np.mean(macros,0);e=np.mean(ez,0);best=max((float(m[j]),float(e[j]),float(ba[j]),-abs(float(t)-.5),-float(t),float(t)) for j,t in enumerate(grid))
    return dict(threshold=best[-1],macro_f1=best[0],ez_f1=best[1],pooled_ba=best[2],candidate_plateaus=len(grid))

def source_map(d,score,default):
    va=d['val'];known=set(d['center'][d['train']].astype(str));mapping={};counts={}
    for source in sorted(set(d['center'][va].astype(str))):
        if source not in known:continue
        k=np.flatnonzero(d['center'][va].astype(str)==source);sel=select_exact(d['y'][va][k],score[k],d['patient'][va][k]);mapping[source]=sel['threshold'];counts[source]=len(set(d['patient'][va][k]))
    return mapping,counts
