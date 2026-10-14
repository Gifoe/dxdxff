"""Paired unique-ID clusters preserving repeated validation appearances."""
import numpy as np
def compare(c,left,right,metrics):
 l=left.set_index(['fold','patient']).sort_index();r=right.set_index(['fold','patient']).sort_index();assert l.index.equals(r.index)
 pid=l.index.get_level_values('patient').to_numpy().astype(str);d=l[metrics].to_numpy()-r[metrics].to_numpy();b=c.cluster_bootstrap(pid,d,10000)
 ids=np.unique(pid);ix=np.random.default_rng(42).integers(0,len(ids),size=(10000,len(ids)));counts=np.stack([(ix==j).sum(1) for j in range(len(ids))],axis=1)
 sums=np.stack([np.nansum(d[pid==p],axis=0) for p in ids]);n=np.stack([np.isfinite(d[pid==p]).sum(0) for p in ids]);num=counts@sums;den=counts@n
 independent=np.divide(num,den,out=np.full_like(num,np.nan),where=den>0);assert np.allclose(b,independent,atol=1e-12,rtol=0,equal_nan=True)
 return [dict(metric=m,delta=float(np.nanmean(d[:,j])),ci_low=float(np.nanpercentile(b[:,j],2.5)),ci_high=float(np.nanpercentile(b[:,j],97.5)),draws=10000,seed=42,unique_clusters=len(ids),appearances=len(pid),threshold_reoptimized=False) for j,m in enumerate(metrics)]
