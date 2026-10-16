"""Dense canonical ridge and deterministic finite subset selection."""
import itertools
import warnings
import numpy as np
from scipy.special import expit
from sklearn.linear_model import Ridge


def prepare(x,train):
    finite=np.isfinite(x[train]);count=finite.sum(0)
    mean=np.divide(np.nansum(x[train],axis=0),count,out=np.zeros(x.shape[1]),where=count>0)
    filled=np.where(np.isfinite(x),x,mean);center=filled[train].mean(0);sd=filled[train].std(0);sd[sd==0]=1.
    return (filled-center)/sd,dict(imputer=mean,mean=center,scale=sd)


def subsets(eligible):return [tuple(c) for k in range(len(eligible)+1) for c in itertools.combinations(eligible,k)]


def fit(x,base,groups,target,weights,train):
    cols=np.r_[np.arange(base),base+np.array([2*g+j for g in groups for j in [0,1]],int)]
    w=np.asarray(weights,float);w=w*len(w)/w.sum();t=target[train]
    assert np.isfinite(t).all() and np.isfinite(x).all()
    if len(cols):
        model=Ridge(alpha=1.,fit_intercept=True,solver='cholesky')
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always');model.fit(x[train][:,cols],t,sample_weight=w)
        assert not caught,[type(r.message).__name__ for r in caught]
        coef=model.coef_;intercept=float(model.intercept_)
    else:coef=np.empty(0);intercept=float(np.average(t,weights=w))
    score=x[:,cols]@coef+intercept
    return dict(coef=coef,intercept=intercept,columns=cols,probability=expit(score),score=score)


def choose(rows):
    best=max(r['choice_score'] for r in rows)
    k=min(r['k'] for r in rows if r['choice_score']>=best-.005)
    own=[r for r in rows if r['k']==k]
    return min(own,key=lambda r:(-r['choice_score'],tuple(r['groups'])))


def alternative(rows,selected):
    own=[r for r in rows if r['k']==selected['k'] and r['groups']!=selected['groups']]
    return min(own,key=lambda r:(-r['choice_score'],tuple(r['groups']))) if own else selected
