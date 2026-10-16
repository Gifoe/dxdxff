"""Exact cached block solution of weighted ridge on Raw plus selected groups."""
import numpy as np
from scipy.special import expit


def prepare(x,train):
    a=np.asarray(x,dtype=float);tr=a[train];finite=np.isfinite(tr);count=finite.sum(0)
    fill=np.divide(np.nansum(tr,axis=0),count,out=np.zeros(a.shape[1]),where=count>0)
    filled=np.where(np.isfinite(a),a,fill);mean=filled[train].mean(0);scale=filled[train].std(0)
    scale=np.where(scale>0,scale,1.)
    return (filled-mean)/scale,dict(imputer=fill,mean=mean,scale=scale)


class ConditionalRidge:
    def __init__(self,x,target,weight,train,raw_columns=192,alpha=1.):
        self.x=np.asarray(x,dtype=float);self.nraw=raw_columns;self.alpha=float(alpha)
        a=self.x[train];y=np.asarray(target,float)[train];w=np.asarray(weight,float)
        assert len(w)==len(train) and np.isfinite(a).all() and np.isfinite(y).all() and (w>=0).all() and w.sum()>0
        w=w*len(w)/w.sum();self.weights=w;self.mean=np.average(a,axis=0,weights=w);self.target_mean=np.average(y,weights=w)
        c=a-self.mean;self.gram=c.T@(w[:,None]*c);self.cross=c.T@(w*(y-self.target_mean))
        A=self.gram[:raw_columns,:raw_columns]+alpha*np.eye(raw_columns)
        self.raw_inverse_cross=np.linalg.solve(A,np.column_stack([self.cross[:raw_columns],self.gram[:raw_columns,raw_columns:]]))
        self.base=self.raw_inverse_cross[:,0];self.raw_other=self.raw_inverse_cross[:,1:]

    def fit(self,groups):
        columns=np.array([2*g+j for g in sorted(groups) for j in [0,1]],dtype=int)
        full=self.nraw+columns;raw=self.base.copy()
        if len(full):
            B=self.gram[:self.nraw,full];S=self.gram[np.ix_(full,full)]+self.alpha*np.eye(len(full))-B.T@self.raw_other[:,columns]
            rhs=self.cross[full]-B.T@self.base;phys=np.linalg.solve(S,rhs);raw-=self.raw_other[:,columns]@phys
        else:phys=np.empty(0)
        used=np.r_[np.arange(self.nraw),full];coef=np.r_[raw,phys];intercept=self.target_mean-self.mean[used]@coef
        score=self.x[:,used]@coef+intercept
        assert np.isfinite(score).all()
        return dict(score=score,probability=expit(score),coef=coef,intercept=float(intercept),columns=used)
