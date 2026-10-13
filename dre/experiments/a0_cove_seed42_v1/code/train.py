"""Original trainer, indexed OOF bank adapter; all fifteen scratch-trained selections."""
import json
import time
import numpy as np
import pandas as pd
import torch
from runtime import args,setup,checkpoint
from cove_model import constructor

def main():
 a=args();o=setup(a);c,core=o.c,o.core;pub=o.pub
 fr=json.loads((a.runtime/'EXPERIMENT_FREEZE.json').read_text());assert fr['binding']==o.bind and fr['status']=='PASS'
 assert c.sha(a.runtime/'NUISANCE_FREEZE.json')==fr['nuisance_sha256'] and c.sha(pub/'ENGINEERING_TESTS.json')==fr['engineering_sha256']
 assert json.loads((pub/'STAGE_A_GATE.json').read_text())['status']=='PASS'
 assert not (a.runtime/'DEVELOPMENT_FREEZE.json').exists();selections=[];history=[];resources=[]
 for fold in range(1,6):
  bp=a.runtime/f'fold{fold}/BANK_PRIVATE.pt';assert c.sha(bp)==fr['files'][f'bank{fold}'];b=torch.load(bp,map_location='cpu',weights_only=False);seed=42+1009*fold
  for arm in ['B1','B2','B3']:
   fac=constructor(o.original.SourceA0,arm,b['mapping'],b['stats'],b['permutation'],b);core.seed_all(seed);initial=core.state_hash(fac(seed).state_dict());cp=checkpoint(a,arm,fold)
   old=o.trainer.SourceA0;o.trainer.SourceA0=fac;t=time.perf_counter();torch.cuda.reset_peak_memory_stats()
   try:o.trainer.train(np.arange(len(b['x']),dtype=np.float32)[:,None],b['y'],b['patient'],b['g'],b['train'],b['val'],cp.parent,seed,c.digest(dict(binding=o.bind,bank=c.sha(bp),arm=arm,nuisance=fr['nuisance_sha256'])),device='cuda',penalty=0.)
   finally:o.trainer.SourceA0=old
   ck=torch.load(cp,map_location='cpu',weights_only=False);last=torch.load(cp.parent/'LAST_PRIVATE.pt',map_location='cpu',weights_only=False);done=json.loads((cp.parent/'COMPLETE_PRIVATE.json').read_text())
   assert c.sha(cp)==done['best_sha256'] and c.sha(cp.parent/'LAST_PRIVATE.pt')==done['last_sha256'] and ck['binding']['initial_hash']==initial
   selections.append(dict(arm=arm,fold=fold,epoch=ck['epoch'],threshold=ck['threshold'],bank_sha256=c.sha(bp),checkpoint_sha256=c.sha(cp),parameters=8845))
   history.extend(dict(arm=arm,fold=fold,**r) for r in last['history']);resources.append(dict(arm=arm,fold=fold,epochs=last['epoch'],wall_seconds=time.perf_counter()-t,peak_GPU_bytes=torch.cuda.max_memory_allocated()))
   c.write_json(pub/'TRAIN_PROGRESS.json',dict(status='RUNNING',cells_complete=len(selections),total=15,last_arm=arm,last_fold=fold));print('SELECTED',arm,fold,ck['epoch'],ck['threshold'],flush=True)
 assert setup(a).bind==o.bind
 c.write_json(a.runtime/'DEVELOPMENT_FREEZE.json',dict(status='PASS',binding=o.bind,selections=selections,nuisance_sha256=fr['nuisance_sha256'],all_selections_before_comparative_metrics=True,new_outer_accessed=False))
 pd.DataFrame(selections).to_csv(pub/'MODEL_SELECTION.csv',index=False);pd.DataFrame(history).to_csv(pub/'TRAINING_HISTORY.csv',index=False);pd.DataFrame(resources).to_csv(pub/'RESOURCE_AUDIT.csv',index=False)
 c.write_json(pub/'TRAIN_PROGRESS.json',dict(status='ALL_TRAIN_VALIDATION_COMPLETE',cells_complete=15,total=15))

if __name__=='__main__':main()
