"""Label-free same-seizure query-excluded median; no normal-tissue assumption."""
import warnings
import numpy as np

def construct(x,t,mask):
 # [time,source-channel,7]. Missing/invalid fields are NEVER observed zeros.
 assert x.shape[:2]==mask.shape and x.shape[2]==7 and len(t)==len(x)
 raw=np.where(mask[...,None]&np.isfinite(x),x,np.nan)
 with warnings.catch_warnings():
  warnings.simplefilter('ignore',RuntimeWarning);b=np.nanmedian(raw[(t>=-10)&(t<0)],axis=0)
 counts=np.isfinite(raw[(t>=-10)&(t<0)]).sum(0);b=np.where(counts>0,b,np.nan)
 d=raw-b[None];m=np.full_like(d,np.nan);other=np.zeros(d.shape,dtype=np.int32)
 for ti in range(len(t)):
  for k in range(7):
   ix=np.flatnonzero(np.isfinite(d[ti,:,k]));n=len(ix)
   other[ti,ix,k]=n-1
   if n<5:continue
   order=np.argsort(d[ti,ix,k],kind='stable');v=d[ti,ix[order],k];rank=np.empty(n,int);rank[order]=np.arange(n)
   # Median after removing each query's sorted position, including ties.
   lo=(n-2)//2;hi=(n-1)//2
   m[ti,ix,k]=(v[lo+(rank<=lo)]+v[hi+(rank<=hi)])/2
 u=d-m
 assert np.array_equal(np.isfinite(u),np.isfinite(d)&np.isfinite(m))
 return dict(b=b,baseline_counts=counts,d=d,m=m,u=u,other_counts=other)
