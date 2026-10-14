from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parents[1]/'code'))
import numpy as np
import pandas as pd
import torch
from compact_delta_classifier import CompactDelta,inflate
from bootstrap import compare

def synthetic_records():return [dict(index=i,copies=[i,i+14] if i<8 else [i],multiplicity=2 if i<8 else 1) for i in range(14)]
def network(dtype):return torch.nn.Sequential(torch.nn.LayerNorm(88,dtype=dtype),torch.nn.Linear(88,96,dtype=dtype),torch.nn.GELU(),torch.nn.Dropout(.15),torch.nn.Linear(96,1,dtype=dtype)).eval()
def test_exact_original_LN88_algebra_and_parameter_budget():
 torch.manual_seed(42);n=network(torch.float64);r=synthetic_records();m=CompactDelta(n,r).eval();x=torch.randn(100,14,dtype=torch.float64)
 assert sum(p.numel() for p in m.parameters())==1537
 assert (n(inflate(x,r)).squeeze(-1)-m(x)).abs().max()<1e-12
 assert not torch.allclose(m.normalized(x),torch.nn.functional.layer_norm(x,(14,)))
def test_same_seed_compact_initializations():
 torch.manual_seed(42);a=CompactDelta(network(torch.float32),synthetic_records());torch.manual_seed(42);b=CompactDelta(network(torch.float32),synthetic_records())
 assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
def test_cluster_repeated_appearances_preserved():
 class C:
  @staticmethod
  def cluster_bootstrap(pid,d,draws):
   ids=np.unique(pid);ix=np.random.default_rng(42).integers(0,len(ids),size=(draws,len(ids)));s=np.stack([d[pid==p].sum(0) for p in ids]);n=np.stack([np.isfinite(d[pid==p]).sum(0) for p in ids]);return s[ix].sum(1)/n[ix].sum(1)
 l=pd.DataFrame(dict(fold=[1,2,1],patient=['A','A','B'],metric=[1.,2.,3.]));r=l.copy();r['metric']=0
 q=compare(C,l,r,['metric'])[0];assert q['appearances']==3 and q['unique_clusters']==2 and q['delta']==2
