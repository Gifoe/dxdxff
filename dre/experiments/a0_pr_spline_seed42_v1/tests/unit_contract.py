"""Label-free quantile, degeneracy and analytic curve contracts."""
import json
import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).parents[1]/'code'))
import knot_statistics as ks
from model import Curves


def run_unit(manifest):
 fmap=ks.mapping(manifest,[c['name'] for c in manifest['columns']]);rng=np.random.default_rng(42)
 x=rng.normal(size=(64,88)).astype(np.float32);x[:,fmap['zeros']]=0
 for j,k in fmap['duplicates'].items():x[:,j]=x[:,k]
 pid=np.repeat(['f1','f2','v1','v2'],16);tr=np.arange(32)
 stats,w=ks.fit(x,pid,tr,fmap);changed=x.copy();changed[32:]=changed[32:]*1e5+1e6
 st2,_=ks.fit(changed,pid,tr,fmap)
 for key in ['knots','raw_quantiles','slope_mask','boundary_mask']:assert np.array_equal(stats[key],st2[key])
 xx=np.r_[x[:16],np.repeat(x[16:32],3,axis=0)];pp=np.r_[np.repeat('f1',16),np.repeat('f2',48)]
 st3,_=ks.fit(xx,pp,np.arange(64),fmap);assert np.array_equal(stats['raw_quantiles'],st3['raw_quantiles'])
 assert np.allclose([w[pid==p].sum() for p in ['f1','f2']],[.5,.5],atol=1e-12)
 for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
  m=Curves(stats).to(device);tx=torch.as_tensor(x,device=device)
  assert torch.equal(m(tx),tx)
  assert sum(p.numel() for p in m.parameters())==448
  with torch.no_grad():m.theta.copy_(torch.linspace(-.8,.8,448,device=device).reshape(56,8))
  assert torch.equal(m(torch.zeros_like(tx)),torch.zeros_like(tx))
  assert (m.slopes()>=.25).all() and (m.slopes()<=1.75).all()
  grid=torch.linspace(-100,100,2001,device=device,dtype=torch.float64)[:,None].expand(-1,88)
  transformed=m(grid);assert (torch.diff(transformed,dim=0)>0).all()
  for j,k in fmap['duplicates'].items():assert torch.equal(m(tx)[:,j],m(tx)[:,k])
  assert torch.equal(m(tx)[:,fmap['zeros']],tx[:,fmap['zeros']])
  # Analytic interval derivatives and continuity around every active boundary.
  for curve,j in enumerate(fmap['representatives']):
   knots=stats['knots'][curve,stats['boundary_mask'][curve]]
   points=np.r_[knots[0]-10,(knots[:-1]+knots[1:])/2,knots[-1]+10]
   z=torch.zeros(len(points),88,device=device,dtype=torch.float64);z[:,j]=torch.as_tensor(points,device=device);z.requires_grad_()
   grad=torch.autograd.grad(m(z)[:,j].sum(),z)[0][:,j]
   assert torch.allclose(grad,m.slopes()[curve,:len(points)],atol=1e-12,rtol=0)
   z=torch.zeros(2*len(knots),88,device=device,dtype=torch.float64);z[0::2,j]=torch.tensor(knots-1e-8,device=device);z[1::2,j]=torch.tensor(knots+1e-8,device=device)
   v=m(z)[:,j];assert ((v[1::2]-v[0::2]).abs()<=3.5e-8+1e-12).all()
  # Cover both tails as well as fitted intervals. A tiny synthetic FIT set
  # may have no observation beyond a boundary quantile equal to its minimum.
  m.theta.grad=None;weights=torch.tensor(rng.normal(size=grid.shape),device=device)
  (m(grid)*weights).sum().backward()
  assert torch.isfinite(m.theta.grad).all() and (m.theta.grad[torch.tensor(stats['slope_mask'],device=device)].abs()>0).all()
 # Degenerate/repeated knots retain fixed storage but inactive identity.
 deg=x.copy();deg[:,fmap['representatives'][0]]=0;deg[:,fmap['representatives'][1]]=np.arange(64)%2
 ds,_=ks.fit(deg,pid,tr,fmap);dm=Curves(ds);dm.theta.data.fill_(.5)
 assert not ds['slope_mask'][:2].any()
 dt=torch.tensor(x,requires_grad=True);out=dm(dt)
 assert torch.equal(out[:,fmap['representatives'][:2]],dt[:,fmap['representatives'][:2]])
 out.sum().backward();assert (dm.theta.grad[~dm.slope_mask]==0).all()
 return ['FIT_only','patient_equal_quantile','identity','zero_anchor','bounded_monotonic_derivatives','continuity_tails','duplicate_sharing','constant_unchanged','active_gradients','degenerate_mask']


if __name__=='__main__':
 path=Path(__file__).parents[2]/'d2_ictal_ablation_seed42_v1/FEATURE_GROUP_MANIFEST.json'
 print('PR_SPLINE_UNIT_CONTRACT_PASS',len(run_unit(json.loads(path.read_text()))))
