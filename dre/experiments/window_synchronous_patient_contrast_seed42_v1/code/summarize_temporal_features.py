"""Fixed actual-time mean/population-SD/OLS, then masked equal-seizure mean/SD."""
import numpy as np

STAGES=[('early',0,8),('later',8,20)]

def summarize(v,t):
 # v[time,channel,7]. Output[channel,2*7*3], stage/feature/stat order.
 out=np.full((v.shape[1],2,7,3),np.nan);count=np.zeros((v.shape[1],2,7),int)
 for si,(_,lo,hi) in enumerate(STAGES):
  for c in range(v.shape[1]):
   for k in range(7):
    valid=(t>=lo)&(t<hi)&np.isfinite(v[:,c,k]);a=v[valid,c,k];tt=t[valid];n=len(a);count[c,si,k]=n
    if n==0:continue
    mean=a.mean();sd=a.std(ddof=0);dt=tt-tt.mean();den=dt@dt;slope=float(dt@(a-mean)/den) if n>1 and den>0 else 0.
    out[c,si,k]=[mean,sd,slope]
 return out.reshape(v.shape[1],42),count

def across_seizures(values):
 v=np.stack(values);valid=np.isfinite(v);n=valid.sum(0)
 mean=np.divide(np.nansum(v,0),n,out=np.full(v.shape[1:],np.nan),where=n>0)
 delta=np.where(valid,v-mean,0);var=np.divide((delta*delta).sum(0),n,out=np.full_like(mean,np.nan),where=n>0)
 return np.concatenate([mean,np.sqrt(var)],axis=-1),n

def schema(features):
 base=[f'baseline_{stat}__{f}' for stat in ['seizure_mean','seizure_std'] for f in features]
 temporal=[f'{group}__{agg}__{stage}__{f}__{stat}' for group in ['self','context'] for agg in ['seizure_mean','seizure_std'] for stage,_,_ in STAGES for f in features for stat in ['mean','std','slope']]
 assert len(base+temporal)==182
 return base+temporal

def stage_indices(stage):
 si=0 if stage=='early' else 1
 return np.r_[np.arange(98+si*21,98+(si+1)*21),np.arange(98+42+si*21,98+42+(si+1)*21)]
