"""Fixed normalization and score transforms; labels are not accepted by this API."""
import copy
import random
from contextlib import contextmanager
import numpy as np
from scipy.special import expit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

CONDITIONS=['NO_PR','GLOBAL','CENTER','SCALE','FULL_PR','SCORE_CENTER','SCORE_Z']
KEEP12=np.array([0,1,2,3,4,5,6,7,10,11,12,13])

def moments(x):
    a=np.asarray(x,np.float64);assert a.ndim==2 and len(a) and np.isfinite(a).all()
    return a.mean(0),a.std(0,ddof=0)

def feature_view(raw,pid,condition,mu,unit,physical_zero=True):
    x=np.asarray(raw,np.float64);pid=np.asarray(pid).astype(str)
    assert condition in ['NO_PR','GLOBAL','CENTER','SCALE','FULL_PR'] and np.isfinite(x).all()
    if condition in ['NO_PR','GLOBAL']:return x.copy()
    out=np.empty_like(x)
    for p in np.unique(pid):
        ix=pid==p;a=x[ix];m,s=moments(a)
        loc=m if condition in ['CENTER','FULL_PR'] else mu
        if condition=='CENTER':out[ix]=(a-loc)/unit
        elif physical_zero:
            out[ix]=np.divide(a-loc,s,out=np.zeros_like(a),where=s>1e-8*unit)
        else:out[ix]=(a-loc)/np.where(s>1e-8,s,1.)
    return out

def fit_feature(raw,pid,fit,condition,physical_zero=True):
    x=np.asarray(raw,np.float64);fit=np.asarray(fit,int)
    initial=SimpleImputer().fit(x[fit]);units=StandardScaler().fit(initial.transform(x[fit]))
    value=feature_view(initial.transform(x),pid,condition,units.mean_,units.scale_,physical_zero)
    imp=SimpleImputer().fit(value[fit]);sc=StandardScaler().fit(imp.transform(value[fit]))
    pre=dict(condition=condition,initial=initial,mu=units.mean_.copy(),unit=units.scale_.copy(),imputer=imp,scaler=sc,physical_zero=physical_zero)
    return sc.transform(imp.transform(value)),pre

def query_feature(raw,pid,pre):
    value=feature_view(pre['initial'].transform(np.asarray(raw,np.float64)),pid,pre['condition'],pre['mu'],pre['unit'],pre['physical_zero'])
    return pre['scaler'].transform(pre['imputer'].transform(value))

def score_transform(logit,pid,condition):
    x=np.asarray(logit,np.float64);pid=np.asarray(pid).astype(str)
    assert x.ndim==1 and x.shape==pid.shape and np.isfinite(x).all()
    assert condition in ['SCORE_CENTER','SCORE_Z']
    out=np.empty_like(x)
    for p in np.unique(pid):
        ix=pid==p;v=x[ix];z=v-v.mean()
        out[ix]=z if condition=='SCORE_CENTER' else z/(v.std(ddof=0)+1e-8)
    assert np.isfinite(out).all()
    return out

class StreamingMoments:
    def __init__(self):self.n=0;self.mean=None;self.m2=None
    def add(self,x):
        a=np.asarray(x,np.float64);assert a.ndim==2 and len(a) and np.isfinite(a).all()
        n=len(a);mu=a.mean(0);m2=((a-mu)**2).sum(0)
        if not self.n:self.n=n;self.mean=mu;self.m2=m2;return
        delta=mu-self.mean;total=self.n+n
        self.m2+=m2+delta**2*(self.n*n/total);self.mean+=delta*(n/total);self.n=total
    def result(self):
        assert self.n>0
        return dict(n=self.n,mean=self.mean.copy(),sd=np.sqrt(np.maximum(self.m2/self.n,0)))

@contextmanager
def preserved_eval(model):
    import torch
    states=(random.getstate(),copy.deepcopy(np.random.get_state()),torch.get_rng_state().clone(),[s.clone() for s in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [])
    modes={k:m.training for k,m in model.named_modules()};buffers={k:v.detach().clone() for k,v in model.named_buffers()}
    model.eval()
    try:
        with torch.no_grad():yield
    finally:
        for k,m in model.named_modules():m.training=modes[k]
        with torch.no_grad():
            for k,v in model.named_buffers():v.copy_(buffers[k])
        random.setstate(states[0]);np.random.set_state(states[1]);torch.set_rng_state(states[2])
        if states[3]:torch.cuda.set_rng_state_all(states[3])

def normalize_tensor(x,condition,stats):
    import torch
    v=x.double();mu=torch.as_tensor(stats['mean'],device=x.device);sd=torch.as_tensor(stats['sd'],device=x.device)
    loc=v.mean(0) if condition in ['CENTER','FULL_PR'] else mu
    scale=v.std(0,unbiased=False) if condition in ['SCALE','FULL_PR'] else sd
    return ((v-loc)/torch.where(scale>1e-8,scale,torch.ones_like(scale))).to(x.dtype)

