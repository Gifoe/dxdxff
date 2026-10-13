"""Immutable manifest-by-name statistic set; not feature selection or deduplication."""
import numpy as np


def mapping(manifest,names):
 cols=manifest['columns'];assert len(cols)==88 and [c['name'] for c in cols]==list(names)
 assert [c['index'] for c in cols]==list(range(88)) and len(set(names))==88
 zeros=[c['index'] for c in cols if c['constant_zero_in_frozen_export']]
 duplicate={c['index']:c['duplicate_of'] for c in cols if 'duplicate_of' in c}
 unique=[c['index'] for c in cols if not c['constant_zero_in_frozen_export'] and 'duplicate_of' not in c]
 assert zeros==[0,1,2,3,18,19,20,21,36,37,38,39,54,55,56,57]
 assert sorted(duplicate)==list(range(72,88)) and len(unique)==len(set(unique))==56
 assert not set(unique)&(set(zeros)|set(duplicate)) and set(unique)|set(zeros)|set(duplicate)==set(range(88))
 for j,k in duplicate.items():
  assert k in unique
  a,b=cols[j],cols[k]
  assert all(a[key]==b[key] for key in ['view','physiological_feature','aggregation'])
 return dict(stat_indices=unique,zeros=zeros,duplicates=duplicate)


def validate_values(d,fmap):
 x=np.asarray(d['x']);assert x.ndim==2 and x.shape[1]==88 and np.isfinite(x).all()
 assert (x[:,fmap['zeros']]==0).all()
 for j,k in fmap['duplicates'].items():
  assert np.array_equal(d['raw_x'][:,j],d['raw_x'][:,k],equal_nan=True) and np.array_equal(x[:,j],x[:,k])
 assert (np.ptp(x[d['train']][:,fmap['stat_indices']].astype(np.float64),axis=0)>0).all()
 return True
