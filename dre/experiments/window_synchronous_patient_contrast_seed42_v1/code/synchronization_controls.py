"""Predeclared circular stage-local shifts; same query, multiset and masks."""
import hashlib
import numpy as np
from summarize_temporal_features import STAGES

def correlation(a,b,floor=1e-12):
 a=a-a.mean();b=b-b.mean();sa=np.sqrt(np.mean(a*a));sb=np.sqrt(np.mean(b*b))
 return float(np.dot(a,b)/len(a)/sa/sb) if len(a)>=3 and sa>floor and sb>floor else np.nan

def shift(d,m,t,names,identity):
 out=m.copy();rows=[]
 for c,name in enumerate(names):
  for stage,lo,hi in STAGES:
   h=int.from_bytes(hashlib.sha256('|'.join(map(str,[42,*identity,name,stage])).encode()).digest()[:8],'little')
   for k in range(7):
    ix=np.flatnonzero((t>=lo)&(t<hi)&np.isfinite(m[:,c,k])&np.isfinite(d[:,c,k]));n=len(ix);original=m[ix,c,k];delta=d[ix,c,k]
    offset=1+h%(n-1) if n>=2 else 0;changed=np.roll(original,offset);out[ix,c,k]=changed
    assert np.array_equal(np.sort(original),np.sort(changed))
    sd=float(original.std()) if n else np.nan;mad=float(np.abs(original-changed).mean()) if n else np.nan
    rho=correlation(delta,original) if n else np.nan;wrong=correlation(delta,changed) if n else np.nan
    cyc=correlation(original,np.roll(original,1)) if n else np.nan;wc=correlation(changed,np.roll(changed,1)) if n else np.nan
    ordinary=correlation(original[:-1],original[1:]) if n>1 else np.nan;so=correlation(changed[:-1],changed[1:]) if n>1 else np.nan
    rows.append(dict(channel_index=c,stage=stage,feature_index=k,valid_positions=n,eligible=n>=2,shifted=offset>0,offset=offset,context_std=sd,raw_mean_abs_change=mad,relative_change=mad/max(sd,1e-12) if n else np.nan,material_changed=bool(n>=2 and mad/max(sd,1e-12)>1e-6),query_context_abs_correlation=abs(rho),shifted_query_context_abs_correlation=abs(wrong),correlation_reduction=abs(rho)-abs(wrong),cyclic_autocorrelation_drift=abs(cyc-wc),ordinary_lag1_correlation=ordinary,shifted_ordinary_lag1_correlation=so,ordinary_lag1_change=abs(ordinary-so),context_mean_drift=float(abs(original.mean()-changed.mean())) if n else np.nan,context_std_drift=float(abs(original.std()-changed.std())) if n else np.nan,multiset_identity=True))
 assert np.array_equal(np.isfinite(m),np.isfinite(out));u=d-m;us=d-out;assert np.array_equal(np.isfinite(u),np.isfinite(us))
 return out,us,rows
