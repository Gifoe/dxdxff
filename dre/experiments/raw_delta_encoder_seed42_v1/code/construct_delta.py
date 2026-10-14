"""Original all-pre float32 baseline, mean windows, then available-seizure stats."""
import numpy as np
import warnings

def window_delta(x,t,valid):
 x=np.asarray(x,dtype=np.float32);t=np.asarray(t,dtype=np.float64);valid=np.asarray(valid,dtype=bool)
 assert x.ndim==3 and valid.shape==x.shape[:2] and t.shape==(len(x),)
 out=np.full_like(x,np.nan);baseline=np.full(x.shape[1:],np.nan,np.float32);counts=np.zeros(x.shape[1],int)
 for c in range(x.shape[1]):
  pre=(t<0)&valid[:,c]&np.isfinite(x[:,c]).all(-1);counts[c]=pre.sum()
  if counts[c]:
   baseline[c]=x[pre,c].mean(0);take=valid[:,c]&np.isfinite(x[:,c]).all(-1);out[take,c]=x[take,c]-baseline[c]
 return out,baseline,counts

def seizure_mean(x):
 with warnings.catch_warnings():warnings.simplefilter('ignore',RuntimeWarning);return np.nanmean(np.asarray(x,dtype=np.float32),axis=0)

def aggregate(items):
 a=np.asarray(items,dtype=np.float32)
 mu=np.empty(a.shape[1:],np.float32);sd=mu.copy()
 with warnings.catch_warnings():
  warnings.simplefilter('ignore',RuntimeWarning)
  for c in range(a.shape[1]):
   for f in range(a.shape[2]):
    v=a[:,c,f];v=v[np.isfinite(v)];mu[c,f]=np.nanmean(v) if len(v) else np.nan;sd[c,f]=np.nanstd(v,ddof=0) if len(v) else np.nan
 return np.stack([mu,sd],-1).reshape(mu.shape[0],-1),np.isfinite(a).sum(0)

def historical_record_mean(module,features,names,t,valid):
 valid=np.asarray(valid,dtype=bool);assert valid.any() and (np.asarray(t)[valid]<0).any(),'Historical all-window baseline fallback forbidden'
 values,derived_names=module._p2_window_features(dict(features=np.asarray(features,dtype=np.float32)[valid],window_centers=np.asarray(t)[valid],feature_names=names));out=[]
 for c in range(values.shape[1]):
  local=values[:,c,:];good=local[np.isfinite(local).all(axis=1)];assert len(good)>0;out.append(np.nanmean(good,axis=0))
 return np.stack(out),derived_names
