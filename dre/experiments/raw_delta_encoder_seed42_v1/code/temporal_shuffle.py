"""Fixed SHA-derived stage-local permutations, labels never enter identities."""
import hashlib
import json
import numpy as np
def permutation(patient,compound_key,channel,times):
 t=np.asarray(times);out=np.arange(len(t))
 for stage,(low,high) in enumerate([(0,8),(8,20)]):
  ix=np.flatnonzero((t>=low)&(t<high))
  if len(ix)<2:continue
  key=json.dumps([42,str(patient),list(compound_key),str(channel),stage],separators=(',',':'))
  seed=int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],'little');p=np.random.default_rng(seed).permutation(ix)
  if np.array_equal(p,ix):p=np.roll(p,1)
  out[ix]=p
  assert not np.array_equal(out[ix],ix) and set(out[ix])==set(ix)
 assert sorted(out.tolist())==list(range(len(t)))
 return out
