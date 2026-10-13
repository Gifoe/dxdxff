"""All compact deletions/restorations: exact name maps, no silent ZDELTA return."""
import copy
import json
import numpy as np
import pandas as pd
import torch
from audit_sources import arguments,context
from dependencies import load


def factory(original,helper,arm):
 parent=helper.source_free_factory(original.SourceA0)
 if arm!='F5':return parent
 class WithoutLN(parent):
  def __init__(self,seed=42):
   super().__init__(seed);self.network[0]=torch.nn.Identity()
   assert sum(p.numel() for p in self.parameters())==8641
 return WithoutLN


def main():
 a=arguments();lock,ref,c,core,original,trainer,helper,banks,manifest,bind=context(a)
 fr=json.loads((a.runtime/'EXPERIMENT_FREEZE.json').read_text());assert fr['binding']==bind
 absolute=load('compact_absolute_transform',a.source/'d2_no_patient_z_outer_seed42_v1/code/run_ablation.py');rows=[];audits=[]
 for arm,spec in lock['matrix'].items():
  if arm=='R0':continue
  mask=np.asarray([q['view'] in spec['views'] for q in manifest['columns']]);assert mask.sum()==spec['retained']
  if arm not in ['F6']:assert not any(k and q['view']=='zdelta' for q,k in zip(manifest['columns'],mask))
  for q in manifest['columns']:
   if 'duplicate_of' in q:assert mask[q['index']]==mask[q['duplicate_of']]
  rows.extend(dict(arm=arm,**q,retained=bool(k)) for q,k in zip(manifest['columns'],mask))
  if arm not in ['F1','F2','F3','F4','F5']:continue
  for fold,d in enumerate(banks,1):
   if arm=='F1':
    x,pre=absolute.prepare_absolute(d['raw_x'],d['train']);pre['fit_patient_ids']=d['pre']['fit_patient_ids']
    assert np.array_equal(absolute.frozen_absolute(d['raw_x'],pre),x)
    changed=d['raw_x'].copy();changed[d['val']]=changed[d['val']]*7+11;_,p2=absolute.prepare_absolute(changed,d['train'])
    for k in p2:assert np.array_equal(p2[k],pre[k])
   else:
    x,pre=core.prepare(d['raw_x'],d['patient'],d['train']);assert np.array_equal(x,d['x'])
    for k in ['mean','scale','var','imputer_statistics']:assert np.array_equal(pre[k],d['pre'][k])
   xx=helper.masked(x,mask);assert np.array_equal(xx[:,mask],x[:,mask]) and np.all(xx[:,~mask]==0)
   seed=42+1009*fold;core.seed_all(seed);m=factory(original,helper,arm)(seed)
   core.seed_all(seed);base=helper.source_free_factory(original.SourceA0)(seed)
   for name,v in m.state_dict().items():assert torch.equal(v,base.state_dict()[name])
   assert sum(p.numel() for p in m.parameters())==spec['parameters']
   dd=copy.deepcopy(d);dd.update(x=xx,pre=pre,mask=mask)
   path=a.runtime/f'features/{arm}/fold{fold}/BANK_PRIVATE.pt';assert not path.exists();core.torch_write(path,dd)
   audits.append(dict(arm=arm,fold=fold,bank_sha256=c.sha(path),retained_columns=int(mask.sum()),parameters=spec['parameters'],retained_values_bitwise=True,initial_weights_exact=True,FIT_only_preprocessor=True))
 pd.DataFrame(rows).to_csv(a.runtime/'public/FEATURE_MASK_AUDIT.csv',index=False)
 c.write_json(a.runtime/'public/FEATURE_PRETRAIN_AUDIT.json',dict(status='PASS',binding=bind,folds=audits,feature_training_started=False,raw_first_required=True))
 print('ALL_FIVE_COMPACT_DELETION_MASKS_AND_INITIALIZATION_PASS',flush=True)


if __name__=='__main__':main()
