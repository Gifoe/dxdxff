"""Unique-ID paired bootstrap; same seed42 draws for every fixed comparison."""
import numpy as np


def compare(c,left,right,metrics,draws=10000):
 ll=left.set_index(['fold','patient']).sort_index();rr=right.set_index(['fold','patient']).sort_index()
 assert ll.index.equals(rr.index)
 patient=np.asarray(ll.index.get_level_values('patient')).astype(str)
 assert len(patient)==65 and len(np.unique(patient))==47
 delta=ll[metrics].to_numpy()-rr[metrics].to_numpy();boot=c.cluster_bootstrap(patient,delta,draws)
 return [dict(metric=m,delta=float(np.nanmean(delta[:,j])),ci_low=float(np.nanpercentile(boot[:,j],2.5)),
   ci_high=float(np.nanpercentile(boot[:,j],97.5)),draws=draws,seed=42,unique_clusters=47,
   retained_appearances=65,thresholds_reoptimized=False) for j,m in enumerate(metrics)]


def independently_verify(left,right,metrics):
 ll=left.set_index(['fold','patient']).sort_index();rr=right.set_index(['fold','patient']).sort_index();assert ll.index.equals(rr.index)
 patient=np.asarray(ll.index.get_level_values('patient')).astype(str);ids=np.unique(patient)
 delta=ll[metrics].to_numpy()-rr[metrics].to_numpy();draws=np.random.default_rng(42).integers(0,len(ids),size=(10000,len(ids)))
 counts=np.stack([(draws==i).sum(1) for i in range(len(ids))],axis=1)
 sums=np.stack([np.nansum(delta[patient==p],axis=0) for p in ids]);n=np.stack([np.isfinite(delta[patient==p]).sum(0) for p in ids])
 den=counts@n;num=counts@sums;boot=np.divide(num,den,out=np.full_like(num,np.nan),where=den>0)
 return {m:(np.nanmean(delta[:,j]),*np.nanpercentile(boot[:,j],[2.5,97.5])) for j,m in enumerate(metrics)}
