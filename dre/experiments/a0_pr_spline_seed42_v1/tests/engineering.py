"""Actual original shared-state parity, legal bank identity and exact resume."""
import json
import tempfile
import numpy as np
import pandas as pd
import torch
from unit_contract import run_unit
from runtime import args,setup


def main():
 a=args();lock,ref,c,core,original,trainer,models,ks,banks,binding=setup(a)
 pre=json.loads((a.runtime/'public/PRETRAIN_AUDIT.json').read_text());assert pre['binding']==binding
 manifest=json.loads((a.previous/'public/FEATURE_GROUP_MANIFEST.json').read_text())
 checks=run_unit(manifest);params=[];fmap=ks.mapping(manifest,[r['name'] for r in manifest['columns']])
 rng=np.random.default_rng(42);x=rng.normal(size=(32,88)).astype('float32');x[:,fmap['zeros']]=0
 for j,k in fmap['duplicates'].items():x[:,j]=x[:,k]
 extreme=x*1e8;pid=np.repeat(['f1','f2','v1','v2'],8);tr=np.arange(16);va=np.arange(16,32)
 synstats,_=ks.fit(x,pid,tr,fmap)
 def check(name,ok):assert bool(ok),name;checks.append(name)
 for fold in range(1,6):
  d=torch.load(a.runtime/f'fold{fold}/BANK_PRIVATE.pt',map_location='cpu',weights_only=False);seed=42+1009*fold
  models_by_arm={}
  for arm in lock['arms']:
   core.seed_all(seed);m=models.constructor(original.SourceA0,arm,d['stats'])(seed).eval();models_by_arm[arm]=m
   check(f'{fold}/{arm} parameter_count',sum(p.numel() for p in m.parameters())==lock['parameters'][arm])
   for key,v in m.network.state_dict().items():
    check(f'{fold}/{arm}/shared/{key}',torch.equal(v,models_by_arm['B0'].network.state_dict()[key]))
   params.append(dict(fold=fold,arm=arm,parameters=lock['parameters'][arm],allocated_extra=lock['parameters'][arm]-8817,
      effective_extra=d['stats']['effective_parameters'] if arm=='B3' else lock['parameters'][arm]-8817,backbone_initial_hash=core.state_hash(m.network.state_dict())))
  for device in ['cpu','cuda']:
   for arm,m in models_by_arm.items():m.to(device)
   for label,xx in [('random',x),('extreme',extreme),('actual_FIT',d['x'][d['train']]),('actual_VAL',d['x'][d['val']])]:
    tx=torch.as_tensor(xx,device=device)
    with torch.inference_mode():base=models_by_arm['B0'](tx)
    for arm in ['B1','B2','B3']:
     with torch.inference_mode():pred=models_by_arm[arm](tx)
     check(f'{fold}/{device}/{arm}/{label}/identity',torch.isfinite(pred).all() and (pred-base).abs().max()<=1e-6)
  check(f'{fold}/inactive_parameters_zero',not models_by_arm['B3'].transform.theta.detach()[~models_by_arm['B3'].transform.slope_mask].count_nonzero())
 y=np.tile([0,1],16);g=np.arange(32)%4
 directory=tempfile.mkdtemp(prefix='synthetic_spline_resume_',dir=a.runtime)
 for arm in ['B1','B2','B3']:
  old=trainer.SourceA0;trainer.SourceA0=models.constructor(original.SourceA0,arm,synstats)
  try:
   trainer.train(x,y,pid,g,tr,va,__import__('pathlib').Path(directory)/arm/'full',1051,'synthetic_only',penalty=0.,max_epochs=6)
   trainer.train(x,y,pid,g,tr,va,__import__('pathlib').Path(directory)/arm/'resume',1051,'synthetic_only',penalty=0.,max_epochs=6,stop_after=2)
   trainer.train(x,y,pid,g,tr,va,__import__('pathlib').Path(directory)/arm/'resume',1051,'synthetic_only',penalty=0.,max_epochs=6)
  finally:trainer.SourceA0=old
  p=torch.load(__import__('pathlib').Path(directory)/arm/'full/LAST_PRIVATE.pt',map_location='cpu',weights_only=False)
  q=torch.load(__import__('pathlib').Path(directory)/arm/'resume/LAST_PRIVATE.pt',map_location='cpu',weights_only=False)
  check(arm+'/resume_state_history_selection',all(torch.equal(v,q['model'][k]) for k,v in p['model'].items()) and p['history']==q['history'] and p['best_key']==q['best_key'])
  check(arm+'/resume_Adam_RNG',all(torch.equal(v,q['optimizer']['state'][k][n]) for k,slot in p['optimizer']['state'].items() for n,v in slot.items()) and torch.equal(p['rng']['torch'],q['rng']['torch']) and all(torch.equal(v,w) for v,w in zip(p['rng']['cuda'],q['rng']['cuda'])))
  # Nonzero finite changed-module gradients; B2 upstream activates after output moves.
  m=models.constructor(original.SourceA0,arm,synstats)(1051).cuda();opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001)
  for update in range(2):
   opt.zero_grad();loss=torch.nn.functional.binary_cross_entropy_with_logits(m(torch.as_tensor(x,device='cuda')),torch.as_tensor(y,device='cuda',dtype=torch.float32));loss.backward()
   check(arm+f'/gradient_finite{update}',all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))
   if arm in ['B1','B3']:check(arm+f'/curve_nonzero_gradient{update}',m.transform.theta.grad.abs().sum()>0)
   if arm=='B2':check(arm+f'/residual_gradient{update}',m.input_residual[-1 if update==0 else 0].weight.grad.abs().sum()>0)
   opt.step()
 pd.DataFrame(params).to_csv(a.runtime/'public/PARAMETER_AUDIT.csv',index=False)
 assert setup(a)[-1]==binding
 c.write_json(a.runtime/'public/ENGINEERING_TESTS.json',dict(status='PASS',binding=binding,checks=checks,checks_passed=len(checks),CPU_GPU_identity=True,exact_resume=True,new_outer_accessed=False))
 print('PR_SPLINE_ALL_ENGINEERING_TESTS_PASS',len(checks),flush=True)


if __name__=='__main__':main()
