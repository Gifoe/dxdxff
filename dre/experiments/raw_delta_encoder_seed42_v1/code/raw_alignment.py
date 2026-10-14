"""Explicit original Raw support and independent compound/channel slice checks."""
import numpy as np

def support(t,fs,total,valid_start,valid_samples,widths):
 t=np.asarray(t,dtype=float);w=np.asarray(widths,dtype=float)
 assert fs==250 and total==15000 and np.all(w==2) and len(t)==len(w)
 assert np.isfinite(t).all() and len(np.unique(t))==len(t)
 assert 0<=valid_start<total and 0<valid_samples<=total-valid_start
 starts=np.rint((30+t)*fs).astype(int)-250;assert len(np.unique(starts))==len(starts)
 valid=(starts>=valid_start)&(starts+500<=valid_start+valid_samples)&(starts>=0)&(starts+500<=total)
 return starts,valid

def exact_slices(wave,names,raw_names,starts,valid,aligned):
 assert len(set(names))==len(names) and len(set(raw_names))==len(raw_names)
 assert np.asarray(wave).dtype==np.float32,'No hidden dtype-conversion sample parity claim'
 lookup={n:i for i,n in enumerate(raw_names)};assert set(names)<=set(raw_names)
 for j,n in enumerate(names):
  expected=np.stack([wave[lookup[n],p:p+500] for p in starts[valid]])
  assert np.array_equal(aligned[valid,j],expected)
 return int(valid.sum()*len(names))
