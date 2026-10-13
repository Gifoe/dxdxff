"""Label-free patient-cross-fitted Ridge; one total weight per training patient."""
import numpy as np
from sklearn.linear_model import Ridge

FEATURES=['log_bp_beta','log_bp_low_gamma','log_bp_high_gamma','rms','variance','line_length_per_sec','spectral_entropy']

def mapping(columns):
 triples=[]
 for feature in FEATURES:
  for stat in ['mean','std']:
   record={'group':len(triples),'feature':feature,'aggregation':stat}
   for view in ['abs','delta','ratio']:
    found=[q for q in columns if q['block']=='b0' and q['view']==view and q['physiological_feature']==feature and q['aggregation']==stat]
    assert len(found)==1 and not found[0]['constant_zero_in_frozen_export'];record[view]=found[0]['index'];record[view+'_name']=found[0]['name']
   triples.append(record)
 retain=[q for q in columns if q['view']!='zdelta'];zero=[q['index'] for q in retain if q['constant_zero_in_frozen_export']]
 duplicates=[(q['index'],q['duplicate_of']) for q in retain if 'duplicate_of' in q]
 assert len(triples)==14 and len(retain)==62 and len(zero)==12 and len(duplicates)==8
 assert len({r[v] for r in triples for v in ['abs','delta','ratio']})==42
 assert all(columns[i]['view']==columns[j]['view']=='delta' for i,j in duplicates)
 return dict(triples=triples,zero=zero,duplicates=duplicates,masked=[q['index'] for q in columns if q['view']=='zdelta'])

def groups(patient,seed):
 ids=np.array(sorted(set(patient)));order=np.random.default_rng(seed).permutation(ids);parts=np.array_split(order,4)
 assert all(len(p)>0 for p in parts)
 assignment={p:k for k,part in enumerate(parts) for p in part}
 return np.array([assignment[p] for p in patient]),parts

def weights(patient):
 return np.array([1./np.sum(patient==p) for p in patient],dtype=float)

def mean(x,w):return np.sum(np.asarray(x)*w.reshape((-1,)+(1,)*(np.asarray(x).ndim-1)),axis=0)/w.sum()
def sd(x,w):return np.sqrt(np.maximum(mean((x-mean(x,w))**2,w),0))

def weighted_quantiles(x,w):
 order=np.argsort(x,kind='stable');cum=np.cumsum(w[order])/w.sum()
 return np.interp([0,.01,.05,.5,.95,.99,1],cum,x[order],left=x[order[0]],right=x[order[-1]])

def fit_models(x,patient,mp):
 w=weights(patient);fits=[]
 for q in mp['triples']:
  a=x[:,q['abs']].astype(float)[:,None];d=x[:,q['delta']].astype(float);r=x[:,q['ratio']].astype(float)
  f=Ridge(alpha=1.,fit_intercept=True).fit(a,d,sample_weight=w)
  g=Ridge(alpha=1.,fit_intercept=True).fit(np.column_stack([a[:,0],d]),r,sample_weight=w)
  fits.append(dict(Dcoef=float(f.coef_[0]),Dintercept=float(f.intercept_),Rcoef=g.coef_.copy(),Rintercept=float(g.intercept_)))
 return fits

def residuals(x,fits,mp):
 e=np.zeros((len(x),14,2),dtype=float)
 for j,q in enumerate(mp['triples']):
  a=x[:,q['abs']].astype(float);d=x[:,q['delta']].astype(float);r=x[:,q['ratio']].astype(float);f=fits[j]
  e[:,j,0]=d-(a*f['Dcoef']+f['Dintercept']);e[:,j,1]=r-(a*f['Rcoef'][0]+d*f['Rcoef'][1]+f['Rintercept'])
 assert np.isfinite(e).all();return e

def fit_cross(x,patient,mp,seed):
 assignment,parts=groups(patient,seed);e=np.empty((len(x),14,2));inner=[]
 for k in range(4):
  tr=assignment!=k;va=assignment==k;assert set(patient[tr]).isdisjoint(set(patient[va]))
  fits=fit_models(x[tr],patient[tr],mp);e[va]=residuals(x[va],fits,mp)
  inner.append(dict(fits=fits,fit_ids=sorted(set(patient[tr])),heldout_ids=sorted(set(patient[va]))))
 w=weights(patient);rawsd=sd(e,w);normal=np.maximum(rawsd,1e-6)
 full=fit_models(x,patient,mp)
 stats=dict(full=full,scale=normal,raw_std=rawsd,inner=inner,assignment=assignment,fit_ids=sorted(set(patient)),seed=seed)
 assert np.isfinite(e/normal).all();return stats,e

def apply(x,stats,mp):return residuals(x,stats['full'],mp)/stats['scale']

def wrong_permutation(mp,seed):
 rng=np.random.default_rng(seed);perm=np.arange(14)
 for stat in ['mean','std']:
  ix=np.array([q['group'] for q in mp['triples'] if q['aggregation']==stat]);assert len(ix)==7
  p=rng.permutation(7)
  while np.any(p==np.arange(7)):p=rng.permutation(7)
  perm[ix]=ix[p]
 assert np.all(perm!=np.arange(14)) and sorted(perm.tolist())==list(range(14))
 return perm

def coordinates(e,perm=None):
 d=np.tanh(e[:,:,0]);r=np.tanh(e[:,:,1]);return d*r[:,np.arange(14) if perm is None else perm]

def diagnostics(x,patient,center,e,stats,mp,fold,phase):
 w=weights(patient);out=[]
 for j,q in enumerate(mp['triples']):
  vals=e[:,j,:];s=sd(vals,w);mu=mean(vals,w);cov=mean((vals[:,0]-mu[0])*(vals[:,1]-mu[1]),w)
  corr=float(cov/(s[0]*s[1])) if np.all(s>0) else np.nan
  for v,label in [(0,'DELTA'),(1,'RATIO')]:
   quant=weighted_quantiles(vals[:,v],w);pm=np.array([vals[patient==p,v].mean() for p in sorted(set(patient))]);ps=np.array([vals[patient==p,v].std() for p in sorted(set(patient))])
   target=x[:,q['delta' if v==0 else 'ratio']].astype(float);den=mean((target-mean(target,w))**2,w)
   out.append(dict(fold=fold,phase=phase,group=j,feature=q['feature'],aggregation=q['aggregation'],residual=label,patients=len(set(patient)),channels=len(x),
    R2=float(1-mean(vals[:,v]**2,w)/den) if den>0 else np.nan,mean=float(mu[v]),std=float(s[v]),normalized_mean=float(mu[v]/stats['scale'][j,v]),normalized_std=float(s[v]/stats['scale'][j,v]),OOF_FIT_scale=float(stats['scale'][j,v]),degenerate_OOF_std=bool(stats['raw_std'][j,v]<1e-6),
    near_zero_fraction=float(mean((np.abs(vals[:,v])<=1e-6).astype(float),w)),residual_D_R_correlation=corr,
    between_patient_mean_std=float(pm.std()),median_patient_residual_std=float(np.median(ps)),min_patient_residual_std=float(ps.min()),max_patient_residual_std=float(ps.max()),
    **dict(zip(['q00','q01','q05','q50','q95','q99','q100'],quant.tolist()))))
 return out
