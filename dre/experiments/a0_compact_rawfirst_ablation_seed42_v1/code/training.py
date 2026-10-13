"""Use original trainer unchanged; all selections before outer, stage ordering gate."""
import json
import time
import numpy as np
import pandas as pd
import torch
from audit_sources import context
from raw_alignment import RawBankStore
from raw_only_model import constructor as raw_constructor
from verify_feature_masks import factory


def path(a,arm,fold):return a.runtime/(f'raw_training/fold{fold}' if arm=='R0' else f'features/{arm}/fold{fold}/TRAIN')


def train_stage(a,raw):
 o=context(a);lock,ref,c,core,original,trainer,helper,banks,manifest,bind=o;pub=a.runtime/'public'
 fr=json.loads((a.runtime/'EXPERIMENT_FREEZE.json').read_text());assert fr['binding']==bind
 eng=json.loads((pub/'ENGINEERING_TESTS.json').read_text());assert eng['status']=='PASS' and eng['binding']==bind
 stage='RAW' if raw else 'FEATURE';freeze=a.runtime/f'{stage}_DEVELOPMENT_FREEZE.json';assert not freeze.exists()
 if not raw:
  assert json.loads((pub/'RAW_INDEPENDENT_VERIFICATION.json').read_text())['status']=='PASS'
  publication=json.loads((a.runtime/'RAW_PUBLICATION_PRIVATE.json').read_text());assert publication['verified_raw_outer_status_sha256']==c.sha(pub/'RAW_OUTER_STATUS.json') and publication['verified_raw_independent_sha256']==c.sha(pub/'RAW_INDEPENDENT_VERIFICATION.json')
  assert publication['github_commit'] and publication['raw_report_sha256']
 maskaudit=json.loads((pub/'FEATURE_PRETRAIN_AUDIT.json').read_text());assert maskaudit['binding']==bind and maskaudit['status']=='PASS'
 store=RawBankStore(a.runtime,c) if raw else None;selections=[];history=[];resources=[]
 for fold in range(1,6):
  for arm in (['R0'] if raw else ['F1','F2','F3','F4','F5']):
   bp=a.original/f'fold{fold}/BANK_PRIVATE.pt' if raw else a.runtime/f'features/{arm}/fold{fold}/BANK_PRIVATE.pt'
   if raw:assert c.sha(bp)==ref['bank_sha256'][fold-1]
   else:assert c.sha(bp)==next(r['bank_sha256'] for r in maskaudit['folds'] if r['arm']==arm and r['fold']==fold)
   d=torch.load(bp,map_location='cpu',weights_only=False);seed=42+1009*fold
   fac=raw_constructor(original,a.source,store,{k:d[k] for k in ['patient','channel']},lock['raw']['encoder_chunk']) if raw else factory(original,helper,arm)
   x=np.arange(len(d['patient']),dtype=np.float32)[:,None] if raw else d['x']
   core.seed_all(seed);m=fac(seed);initial=core.state_hash(m.state_dict());core.seed_all(seed);baseline=helper.source_free_factory(original.SourceA0)(seed)
   for k,v in m.network.state_dict().items():assert torch.equal(v,baseline.network.state_dict()[k])
   del m,baseline
   old=trainer.SourceA0;trainer.SourceA0=fac;torch.cuda.reset_peak_memory_stats();t=time.perf_counter()
   try:trainer.train(x,d['y'],d['patient'],d['g'],d['train'],d['val'],path(a,arm,fold),seed,c.digest(dict(binding=bind,bank=c.sha(bp),raw_registry=c.sha(a.runtime/'RAW_REGISTRY_PRIVATE.json') if raw else None,arm=arm)),device='cuda',penalty=0.)
   finally:trainer.SourceA0=old
   cp=path(a,arm,fold)/'BEST_PRIVATE.pt';last=torch.load(cp.parent/'LAST_PRIVATE.pt',map_location='cpu',weights_only=False);ck=torch.load(cp,map_location='cpu',weights_only=False)
   done=json.loads((cp.parent/'COMPLETE_PRIVATE.json').read_text());assert c.sha(cp)==done['best_sha256'] and c.sha(cp.parent/'LAST_PRIVATE.pt')==done['last_sha256'] and ck['binding']['initial_hash']==initial
   selections.append(dict(arm=arm,fold=fold,epoch=ck['epoch'],threshold=ck['threshold'],checkpoint_sha256=c.sha(cp),bank_sha256=c.sha(bp),parameters=18153 if raw else lock['matrix'][arm]['parameters']))
   history.extend(dict(arm=arm,fold=fold,**r) for r in last['history']);resources.append(dict(arm=arm,fold=fold,wall_seconds_this_invocation=time.perf_counter()-t,epochs=last['epoch'],gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(),parameters=selections[-1]['parameters']))
   c.write_json(pub/(stage+'_TRAIN_PROGRESS.json'),dict(status='RUNNING',cells_complete=len(selections),total=5 if raw else 25,last_arm=arm,last_fold=fold))
   print('SELECTED',arm,fold,ck['epoch'],ck['threshold'],flush=True)
 assert context(a)[-1]==bind
 c.write_json(freeze,dict(status='PASS',binding=bind,selections=selections,raw_registry_sha256=c.sha(a.runtime/'RAW_REGISTRY_PRIVATE.json') if raw else None,all_selections_before_new_outer=True))
 pd.DataFrame(selections).to_csv(pub/(stage+'_MODEL_SELECTION.csv'),index=False);pd.DataFrame(history).to_csv(pub/(stage+'_TRAINING_HISTORY.csv'),index=False)
 pd.DataFrame(resources).to_csv(pub/(stage+'_RESOURCE_AUDIT.csv'),index=False)
 c.write_json(pub/(stage+'_TRAIN_PROGRESS.json'),dict(status='ALL_TRAIN_VALIDATION_SELECTIONS_COMPLETE',cells_complete=len(selections),total=5 if raw else 25))
