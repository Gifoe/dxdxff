from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parents[1]/'code'))
import numpy as np
import torch
from compact_delta_classifier import CompactDelta
from raw_delta_encoder import RawDeltaEncoder
from temporal_shuffle import permutation

def model():
 r=[dict(index=i,copies=[i,i+14] if i<8 else [i],multiplicity=2 if i<8 else 1) for i in range(14)]
 n=torch.nn.Sequential(torch.nn.LayerNorm(88),torch.nn.Linear(88,96),torch.nn.GELU(),torch.nn.Dropout(.15),torch.nn.Linear(96,1))
 return RawDeltaEncoder(CompactDelta(n,r))
def inputs():
 times=torch.tensor([[-2.,-1.,0.,1.,2.,8.,9.,10.],[-2.,-1.,0.,1.,2.,8.,9.,10.]])
 raw=torch.randn(2,4,8,500);mask=torch.ones(2,4,8,dtype=torch.bool);perms=[torch.tensor(np.stack([permutation('synthetic',['run',str(s)],str(c),[0.,1.,2.,8.,9.,10.]) for c in range(4)])) for s in range(2)]
 return torch.randn(4,14),raw,mask,times,perms
def test_shuffle_contents_stages_and_determinism():
 t=np.array([-2,-1,0,1,2,8,9,10]);a=permutation('A',['r','s'],'c',t);assert np.array_equal(a,permutation('A',['r','s'],'c',t));assert np.array_equal(a[:2],[0,1])
 for ix in [np.arange(2,5),np.arange(5,8)]:assert set(a[ix])==set(ix) and not np.array_equal(a[ix],ix)
def test_initial_identity_frozen_base_auxiliary_gradient_and_budget():
 torch.set_num_threads(1);torch.manual_seed(42);m=model();x=inputs();m.train();assert not m.base.training and not any(p.requires_grad for p in m.base.parameters());ll,rr,base,res=m(*x)
 assert torch.equal(ll,base) and torch.equal(res,torch.zeros_like(res))
 y=torch.tensor([0.,1.,0.,1.]);loss=torch.nn.functional.binary_cross_entropy_with_logits(ll,y)+.25*torch.nn.functional.binary_cross_entropy_with_logits(rr,y);loss.backward()
 assert m.cnn[0].weight.grad.abs().sum()>0 and m.gru.weight_ih_l0.grad.abs().sum()>0 and all(p.grad is None for p in m.base.parameters())
 assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
 assert sum(p.numel() for p in m.parameters())==13643 and sum(p.numel() for p in m.parameters() if p.requires_grad)==12106
 with torch.no_grad():m.gamma.fill_(.2)
 ll,rr,base,res=m(*x,gamma_zero=True);assert torch.equal(ll,base)
def test_D2_D3_exact_initial_parameters_and_padding_excluded():
 torch.manual_seed(42);a=model();torch.manual_seed(42);b=model();assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
 x,raw,mask,times,p=inputs();raw=torch.cat([raw,torch.full((2,4,1,500),float('nan'))],2);mask=torch.cat([mask,torch.zeros(2,4,1,dtype=torch.bool)],2);times=torch.cat([times,torch.full((2,1),float('nan'))],1)
 assert torch.isfinite(a(x,raw,mask,times,p)[0]).all()
