"""Synthetic forward/gradient/AdamW interchange check; no clinical data."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np  # Load the conda MKL runtime before PyTorch; no duplicate-OMP bypass.
import torch

def main():
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['reference','check']);a=p.parse_args()
 source=Path(r'E:\DRE-nips\new-pipeline\7-11\a0_source_posterior_decoder_seed42_v1\code\model.py');spec=importlib.util.spec_from_file_location('parity_source',source);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 path=Path(r'D:\COVE_BACKBONE_RUNTIME_REFERENCE_PRIVATE.pt');reference=None if a.mode=='reference' else torch.load(path,map_location='cpu',weights_only=False)
 results={};record={}
 for device in ['cpu','cuda']:
  torch.manual_seed(1051);model=module.SourceA0(1051).network.to(device).eval();initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
  if reference is not None:model.load_state_dict(reference[device]['initial'],strict=True)
  x=(torch.arange(32*88,dtype=torch.float32).reshape(32,88)/1024-1.).to(device);y=(torch.arange(32)%2).to(device,dtype=torch.float32)
  opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);ll=model(x).squeeze(-1);loss=torch.nn.functional.binary_cross_entropy_with_logits(ll,y,pos_weight=torch.tensor(.5,device=device));loss.backward()
  grad=torch.cat([v.grad.flatten() for v in model.parameters()]).detach().cpu();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
  state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
  if reference is None:record[device]=dict(initial=initial,logits=ll.detach().cpu(),grad=grad,updated=state)
  else:
   q=reference[device];fd=float((ll.detach().cpu()-q['logits']).abs().max());gd=float((grad-q['grad']).abs().max());ud=max(float((state[k]-q['updated'][k]).abs().max()) for k in state)
   assert fd<=1e-6 and gd<=1e-6 and ud<=1e-6;results[device]=dict(logit_drift=fd,gradient_drift=gd,AdamW_update_drift=ud)
 if reference is None:
  assert not path.exists();torch.save(record,path);print('SYNTHETIC_REFERENCE_SAVED',sys.version.split()[0],flush=True)
 else:
  result=dict(status='PASS',synthetic_only=True,python=sys.version.split()[0],torch=torch.__version__,cudnn=torch.backends.cudnn.version(),CPU_CUDA=results)
  path=Path(r'D:\cove_backbone_runtime_parity.json');assert not path.exists();path.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)

if __name__=='__main__':main()
