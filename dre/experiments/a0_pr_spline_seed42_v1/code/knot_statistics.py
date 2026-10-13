"""Deterministic FIT-patient-equal empirical quantiles and authoritative mapping."""
import numpy as np

Q=np.array([.02,.10,.25,.50,.75,.90,.98],dtype=np.float64)


def mapping(manifest,names):
 cols=manifest['columns'];assert len(cols)==88 and [c['name'] for c in cols]==list(names)
 zero=[c['index'] for c in cols if c['constant_zero_in_frozen_export']]
 reps=[c['index'] for c in cols if not c['constant_zero_in_frozen_export'] and 'duplicate_of' not in c]
 duplicate={c['index']:c['duplicate_of'] for c in cols if 'duplicate_of' in c}
 assert len(zero)==16 and len(reps)==56 and len(duplicate)==16
 cm=np.full(88,-1,np.int64)
 for i,j in enumerate(reps):cm[j]=i
 for j,k in duplicate.items():assert k in reps;cm[j]=cm[k]
 assert (cm[zero]==-1).all() and (cm>=0).sum()==72
 return {'representatives':np.array(reps),'zeros':np.array(zero),'duplicates':duplicate,'column_to_curve':cm}


def weighted_quantiles(values,weights,q=Q):
 x=np.asarray(values,np.float64);w=np.asarray(weights,np.float64)
 assert x.ndim==w.ndim==1 and x.shape==w.shape and np.isfinite(x).all() and np.isfinite(w).all() and (w>0).all()
 order=np.argsort(x,kind='stable');v=x[order];ww=w[order]
 starts=np.r_[0,np.flatnonzero(v[1:]!=v[:-1])+1];unique=v[starts];mass=np.add.reduceat(ww,starts)
 cdf=np.cumsum(mass);cdf/=cdf[-1];indices=np.searchsorted(cdf,q,side='left')
 return unique[np.minimum(indices,len(unique)-1)]


def fit(x,patient,train,feature_map):
 tr=np.asarray(train);ids=np.unique(patient[tr]);weights=np.zeros(len(x),np.float64)
 for p in ids:
  ix=tr[patient[tr]==p];weights[ix]=1/(len(ids)*len(ix))
 assert np.isclose(weights.sum(),1,atol=1e-12) and np.isfinite(x[tr]).all()
 for p in ids:assert abs(weights[patient==p].sum()-1/len(ids))<1e-12
 knots=np.zeros((56,7),np.float64);slopes_mask=np.zeros((56,8),bool);boundaries=np.zeros((56,7),bool);raw=[];counts=[]
 for i,j in enumerate(feature_map['representatives']):
  qq=weighted_quantiles(x[tr,j],weights[tr]);distinct=np.unique(qq);n=len(distinct);raw.append(qq);counts.append(n)
  if n<3:continue
  knots[i,:n]=distinct;boundaries[i,:n]=True;slopes_mask[i,:n+1]=True
 return dict(knots=knots,slope_mask=slopes_mask,boundary_mask=boundaries,
   column_to_curve=feature_map['column_to_curve'],representatives=feature_map['representatives'],
   raw_quantiles=np.array(raw),distinct_count=np.array(counts),effective_parameters=int(slopes_mask.sum()),
   FIT_patients=len(ids),FIT_channels=len(tr)),weights
