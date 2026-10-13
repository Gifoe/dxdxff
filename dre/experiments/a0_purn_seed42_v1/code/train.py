"""Exactly15 new cells through the unchanged original trainer, no outer path."""
import json
import time
import numpy as np
import pandas as pd
import torch
from runtime import args,setup,checkpoint


def main():
 a=args();lock,ref,c,core,original,trainer,models,ks,banks,binding=setup(a)
 pre=json.loads((a.runtime/'public/PRETRAIN_AUDIT.json').read_text());eng=json.loads((a.runtime/'public/ENGINEERING_TESTS.json').read_text())
 assert pre['binding']==eng['binding']==binding and pre['status']==eng['status']=='PASS'
 rows=json.loads((a.runtime/'B0_SELECTIONS_PRIVATE.json').read_text());histories=[];durations=[];complete=0
 for fold in range(1,6):
  bp=a.runtime/f'fold{fold}/BANK_PRIVATE.pt';assert c.sha(bp)==pre['folds'][fold-1]['bank_sha256']
  d=torch.load(bp,map_location='cpu',weights_only=False)
  for arm in ['B1','B2','B3']:
   factory=models.constructor(original.SourceA0,arm,d['mapping']);old=trainer.SourceA0;trainer.SourceA0=factory
   started=time.perf_counter()
   try:
    trainer.train(d['x'],d['y'],d['patient'],d['g'],d['train'],d['val'],a.runtime/f'fold{fold}/{arm}',42+1009*fold,
      c.digest(dict(binding=binding,bank=c.sha(bp),arm=arm)),penalty=0.,device='cuda')
   finally:trainer.SourceA0=old
   elapsed=time.perf_counter()-started;cp=checkpoint(a,arm,fold);ck=torch.load(cp,map_location='cpu',weights_only=False)
   done=json.loads((cp.parent/'COMPLETE_PRIVATE.json').read_text());assert done['best_sha256']==c.sha(cp)
   last=torch.load(cp.parent/'LAST_PRIVATE.pt',map_location='cpu',weights_only=False)
   m=factory().cuda().eval();m.load_state_dict(ck['model'],strict=True)
   pos=torch.tensor(max(int((1-d['y'][d['train']]).sum()),1)/max(int(d['y'][d['train']].sum()),1),device='cuda')
   losses={}
   with torch.inference_mode():
    for role,ix in [('FIT',d['train']),('validation',d['val'])]:
     patient_losses=[]
     for p in sorted(set(d['patient'][ix])):
      ii=ix[d['patient'][ix]==p];x=torch.as_tensor(d['x'][ii],device='cuda');y=torch.as_tensor(d['y'][ii],device='cuda',dtype=torch.float32)
      patient_losses.append(float(torch.nn.functional.binary_cross_entropy_with_logits(m(x),y,pos_weight=pos)))
     losses[role+'_loss_eval_selected']=float(np.mean(patient_losses))
   rows.append(dict(arm=arm,fold=fold,epoch=ck['epoch'],threshold=ck['threshold'],checkpoint_sha256=c.sha(cp),bank_sha256=c.sha(bp),reused=False))
   histories.extend(dict(arm=arm,fold=fold,**h) for h in last['history'])
   durations.append(dict(arm=arm,fold=fold,wall_seconds_this_invocation=elapsed,completed_epochs=last['epoch'],**losses))
   complete+=1;c.write_json(a.runtime/'public/TRAIN_PROGRESS.json',dict(status='RUNNING',cells_complete=complete,total=15,last_arm=arm,last_fold=fold))
   print('PURN_CELL_SELECTED',arm,fold,ck['epoch'],flush=True)
 assert setup(a)[-1]==binding
 freeze=a.runtime/'DEVELOPMENT_FREEZE.json';assert not freeze.exists()
 c.write_json(freeze,dict(status='PASS',binding=binding,selections=rows,new_outer_authorized=False))
 pd.DataFrame(rows).to_csv(a.runtime/'public/MODEL_SELECTION.csv',index=False)
 pd.DataFrame(histories).to_csv(a.runtime/'public/TRAINING_HISTORY.csv',index=False)
 pd.DataFrame(durations).to_csv(a.runtime/'public/TRAINING_EFFICIENCY.csv',index=False)
 c.write_json(a.runtime/'public/TRAIN_PROGRESS.json',dict(status='ALL15_NEW_CELLS_COMPLETE',cells_complete=15,total=15,new_outer_accessed=False))
 print('ALL_PURN_DEVELOPMENT_SELECTIONS_FROZEN',flush=True)


if __name__=='__main__':main()

