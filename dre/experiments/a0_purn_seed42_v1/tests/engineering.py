"""Actual legal-bank references, parameter/backbone parity, exact synthetic resume."""
import json
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from unit_contract import run_unit,reference
from runtime import args,setup


def main():
 a=args();lock,ref,c,core,original,trainer,models,fm,banks,binding=setup(a)
 pre=json.loads((a.runtime/'public/PRETRAIN_AUDIT.json').read_text());assert pre['binding']==binding
 manifest=json.loads((a.previous/'public/FEATURE_GROUP_MANIFEST.json').read_text());checks=run_unit(manifest);params=[]
 def check(name,ok):assert bool(ok),name;checks.append(name)
 for fold in range(1,6):
  d=torch.load(a.runtime/f'fold{fold}/BANK_PRIVATE.pt',map_location='cpu',weights_only=False);seed=42+1009*fold
  states={}
  for arm in lock['arms']:
   core.seed_all(seed);m=models.constructor(original.SourceA0,arm,d['mapping'])(seed).eval();states[arm]=m
   check(f'{fold}/{arm}/parameters',sum(p.numel() for p in m.parameters())==8817 and sum(p.numel() for p in m.network[0].parameters())==176)
   check(f'{fold}/{arm}/native_B0',isinstance(states['B0'].network[0],torch.nn.LayerNorm))
   for key,v in m.network[1:].state_dict().items():check(f'{fold}/{arm}/shared/{key}',torch.equal(v,states['B0'].network[1:].state_dict()[key]))
   check(f'{fold}/{arm}/initial_affine',torch.equal(m.network[0].weight,torch.ones(88)) and not m.network[0].bias.count_nonzero())
   params.append(dict(fold=fold,arm=arm,parameters=8817,affine_parameters=176,added_parameters=0,
    linear_initial_hash=core.state_hash(m.network[1:].state_dict()),centered=arm in ['B0','B2'],statistic_coordinates=88 if arm in ['B0','B1'] else 56))
  for device in ['cpu','cuda']:
   for arm,m in states.items():m.to(device)
   for role,ix in [('FIT',d['train']),('VAL',d['val'])]:
    xx=d['x'][ix];copy=xx.copy()
    for arm,m in states.items():
     x=torch.tensor(xx,device=device,requires_grad=True);z=m.network[0](x)
     norm=m.network[0];indices=list(range(88)) if arm in ['B0','B1'] else d['mapping']['stat_indices']
     pred=reference(xx,indices,arm in ['B0','B2'],norm.weight.detach().cpu().numpy(),norm.bias.detach().cpu().numpy())
     check(f'{fold}/{device}/{arm}/{role}/numpy',np.allclose(z.detach().cpu().numpy(),pred,atol=3e-6,rtol=3e-6))
     ll=m(x);ll.square().mean().backward()
     check(f'{fold}/{device}/{arm}/{role}/finite_gradient',torch.isfinite(ll).all() and torch.isfinite(x.grad).all() and all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))
     check(f'{fold}/{device}/{arm}/{role}/input_unchanged',np.array_equal(xx,copy))
     m.zero_grad()
 fmap=d['mapping'];rng=np.random.default_rng(42);x=rng.normal(size=(32,88)).astype(np.float32);x[:,fmap['zeros']]=0
 for j,k in fmap['duplicates'].items():x[:,j]=x[:,k]
 pid=np.repeat(['f1','f2','v1','v2'],8);tr=np.arange(16);va=np.arange(16,32);y=np.tile([0,1],16);g=np.arange(32)%4
 directory=Path(tempfile.mkdtemp(prefix='synthetic_purn_resume_',dir=a.runtime))
 for arm in ['B1','B2','B3']:
  old=trainer.SourceA0;trainer.SourceA0=models.constructor(original.SourceA0,arm,fmap)
  try:
   trainer.train(x,y,pid,g,tr,va,directory/arm/'full',1051,'synthetic_only',penalty=0.,max_epochs=6)
   trainer.train(x,y,pid,g,tr,va,directory/arm/'resume',1051,'synthetic_only',penalty=0.,max_epochs=6,stop_after=2)
   trainer.train(x,y,pid,g,tr,va,directory/arm/'resume',1051,'synthetic_only',penalty=0.,max_epochs=6)
  finally:trainer.SourceA0=old
  p=torch.load(directory/arm/'full/LAST_PRIVATE.pt',map_location='cpu',weights_only=False)
  q=torch.load(directory/arm/'resume/LAST_PRIVATE.pt',map_location='cpu',weights_only=False)
  check(arm+'/resume_model_history',all(torch.equal(v,q['model'][k]) for k,v in p['model'].items()) and p['history']==q['history'] and p['best_key']==q['best_key'])
  check(arm+'/resume_Adam_RNG',all(torch.equal(v,q['optimizer']['state'][k][n]) for k,s in p['optimizer']['state'].items() for n,v in s.items()) and torch.equal(p['rng']['torch'],q['rng']['torch']) and all(torch.equal(v,w) for v,w in zip(p['rng']['cuda'],q['rng']['cuda'])))
 pd.DataFrame(params).to_csv(a.runtime/'public/PARAMETER_AUDIT.csv',index=False)
 assert setup(a)[-1]==binding
 c.write_json(a.runtime/'public/ENGINEERING_TESTS.json',dict(status='PASS',binding=binding,checks=checks,checks_passed=len(checks),CPU_GPU_numpy_reference=True,parameter_and_initial_linear_parity=True,exact_resume=True,new_outer_accessed=False))
 print('PURN_ALL_ENGINEERING_TESTS_PASS',len(checks),flush=True)


if __name__=='__main__':main()
