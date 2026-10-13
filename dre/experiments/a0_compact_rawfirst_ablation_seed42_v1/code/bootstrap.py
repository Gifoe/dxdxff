"""Fixed-checkpoint paired ID-cluster draws; independent multiplicity verification."""
import numpy as np


def compare(c,left,right,metrics):
 ll=left.set_index(['fold','patient']).sort_index();rr=right.set_index(['fold','patient']).sort_index();assert ll.index.equals(rr.index)
 ids=ll.index.get_level_values('patient').to_numpy().astype(str);delta=ll[metrics].to_numpy()-rr[metrics].to_numpy();boot=c.cluster_bootstrap(ids,delta,10000)
 return [dict(metric=m,delta=float(np.nanmean(delta[:,i])),ci_low=float(np.nanpercentile(boot[:,i],2.5)),ci_high=float(np.nanpercentile(boot[:,i],97.5)),
  draws=10000,seed=42,unique_clusters=len(np.unique(ids)),appearances=len(ids),threshold_reoptimized=False) for i,m in enumerate(metrics)]


def independent(left,right,metrics):
 ll=left.set_index(['fold','patient']).sort_index();rr=right.set_index(['fold','patient']).sort_index();assert ll.index.equals(rr.index)
 pid=ll.index.get_level_values('patient').to_numpy().astype(str);ids=np.unique(pid);d=ll[metrics].to_numpy()-rr[metrics].to_numpy()
 ix=np.random.default_rng(42).integers(0,len(ids),size=(10000,len(ids)));counts=np.stack([(ix==j).sum(1) for j in range(len(ids))],axis=1)
 sums=np.stack([np.nansum(d[pid==p],axis=0) for p in ids]);n=np.stack([np.isfinite(d[pid==p]).sum(0) for p in ids]);num=counts@sums;den=counts@n
 boot=np.divide(num,den,out=np.full_like(num,np.nan),where=den>0)
 return {m:(np.nanmean(d[:,j]),*np.nanpercentile(boot[:,j],[2.5,97.5])) for j,m in enumerate(metrics)}
