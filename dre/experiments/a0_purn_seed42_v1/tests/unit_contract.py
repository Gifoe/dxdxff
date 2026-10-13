"""Independent NumPy formulas and label-free normalization distinctions."""
import json
import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).parents[1]/'code'))
from model import UniqueFeatureNorm
from feature_mapping import mapping


def reference(x,indices,centered,weight,bias,eps=1e-5):
 a=np.asarray(x,np.float64);part=np.take(a,indices,axis=-1)
 mean=np.mean(part,axis=-1,keepdims=True) if centered else np.zeros_like(part[...,0:1])
 second=np.mean(np.square(part-mean),axis=-1,keepdims=True)
 return ((a-mean)/np.sqrt(second+eps))*np.asarray(weight,np.float64)+np.asarray(bias,np.float64)


def run_unit(manifest):
 fmap=mapping(manifest,[c['name'] for c in manifest['columns']]);u=fmap['stat_indices'];checks=[]
 rng=np.random.default_rng(42);random=rng.normal(size=(24,88)).astype(np.float32)
 fixtures={'random':random,'zeros':np.zeros((3,88),np.float32),'identical':np.full((3,88),2,np.float32),
  'small':random*1e-10,'large':random*1e30,'observed_extremes':np.linspace(-40,68,264,dtype=np.float32).reshape(3,88)}
 duplicate=random.copy();duplicate[:,fmap['zeros']]=0
 for j,k in fmap['duplicates'].items():duplicate[:,j]=duplicate[:,k]
 fixtures['contract_duplicates']=duplicate
 for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
  for centered,indices in [(False,list(range(88))),(True,u),(False,u)]:
   for dtype in [torch.float32,torch.float64]:
    norm=UniqueFeatureNorm(stat_indices=indices,centered=centered).to(device=device,dtype=dtype)
    with torch.no_grad():norm.weight.copy_(torch.linspace(.5,1.5,88,device=device,dtype=dtype));norm.bias.copy_(torch.linspace(-.2,.2,88,device=device,dtype=dtype))
    for name,value in fixtures.items():
     before=value.copy();x=torch.tensor(value,device=device,dtype=dtype,requires_grad=True);out=norm(x)
     ref=reference(value,indices,centered,norm.weight.detach().cpu().numpy(),norm.bias.detach().cpu().numpy())
     atol,rtol=(3e-6,3e-6) if dtype==torch.float32 else (1e-10,1e-10)
     assert out.shape==x.shape and np.allclose(out.detach().cpu().numpy(),ref,atol=atol,rtol=rtol)
     assert torch.equal(out.detach(),norm(x).detach()) and torch.isfinite(out).all() and np.array_equal(value,before)
     out.square().mean().backward();assert torch.isfinite(x.grad).all()
     assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in norm.parameters())
     norm.zero_grad();checks.append(f'{device}/{dtype}/{centered}/{len(indices)}/{name}/reference_backward')
  # No per-column reordering, despite exclusion from statistics.
  x=torch.tensor(duplicate,device=device);n=UniqueFeatureNorm(stat_indices=u).to(device)
  yy=n(x);assert (yy[:,fmap['zeros']]==0).all()
  for j,k in fmap['duplicates'].items():assert torch.equal(yy[:,j],yy[:,k])
  c=UniqueFeatureNorm(stat_indices=u,centered=True).to(device);assert c(x)[:,fmap['zeros']].abs().sum()>0
  with torch.no_grad():n.bias.fill_(.3)
  assert torch.allclose(n(x)[:,fmap['zeros']],torch.full_like(x[:,fmap['zeros']],.3))
  checks.append(device+'/duplicate_zero_affine_semantics')
  constructed=torch.ones(2,88,device=device);constructed[:,fmap['zeros']]=0
  for j,k in fmap['duplicates'].items():constructed[:,j]=3;constructed[:,k]=3
  layer=torch.nn.LayerNorm(88).to(device);r88=UniqueFeatureNorm().to(device)
  l56=UniqueFeatureNorm(stat_indices=u,centered=True).to(device);r56=UniqueFeatureNorm(stat_indices=u).to(device)
  outs=[z(constructed) for z in [layer,r88,l56,r56]]
  for a,b in [(0,1),(0,2),(1,3),(2,3)]:assert not torch.allclose(outs[a],outs[b])
  checks.append(device+'/four_factor_distinctions')
  for bad in [torch.ones(2,87,device=device),torch.ones(2,89,device=device)]:
   try:n(bad)
   except AssertionError:pass
   else:raise AssertionError('Invalid width accepted')
  checks.append(device+'/width_validation')
 return checks


if __name__=='__main__':
 path=Path(__file__).parents[2]/'d2_ictal_ablation_seed42_v1/FEATURE_GROUP_MANIFEST.json'
 print('PURN_UNIT_CONTRACT_PASS',len(run_unit(json.loads(path.read_text()))))
