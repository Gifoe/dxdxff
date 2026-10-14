"""Common seed42 unique-ID multiplicities retaining every repeated appearance."""
import numpy as np

def paired(df,left,right,metrics):
 a=df[df.arm==left].set_index(['fold','inner','patient']).sort_index();b=df[df.arm==right].set_index(['fold','inner','patient']).sort_index();assert a.index.equals(b.index)
 p=a.index.get_level_values('patient').to_numpy().astype(str);ids=np.unique(p);delta=a[metrics].to_numpy()-b[metrics].to_numpy();ix=np.random.default_rng(42).integers(0,len(ids),(10000,len(ids)))
 sums=np.stack([np.nansum(delta[p==q],0) for q in ids]);counts=np.stack([np.isfinite(delta[p==q]).sum(0) for q in ids]);num=sums[ix].sum(1);den=counts[ix].sum(1);boot=np.divide(num,den,out=np.full_like(num,np.nan),where=den>0)
 # Separate direct multiplicity implementation, not a second stochastic run.
 mult=np.stack([(ix==j).sum(1) for j in range(len(ids))],1);n2=mult@sums;d2=mult@counts;two=np.divide(n2,d2,out=np.full_like(n2,np.nan),where=d2>0);assert np.allclose(boot,two,atol=1e-12,rtol=0,equal_nan=True)
 return [dict(comparison=left+' minus '+right,metric=m,delta=float(np.nanmean(delta[:,j])),ci_low=float(np.nanpercentile(boot[:,j],2.5)),ci_high=float(np.nanpercentile(boot[:,j],97.5)),draws=10000,seed=42,unique_patient_clusters=len(ids),appearances=len(p),improved=int((delta[:,j]>0).sum()),degraded=int((delta[:,j]<0).sum()),ties=int((delta[:,j]==0).sum()),estimable_appearances=int(np.isfinite(delta[:,j]).sum()),threshold_reoptimized=False) for j,m in enumerate(metrics)]
