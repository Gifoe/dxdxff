"""Hash admission. Legal FIT/VAL banks only; never opens full feature x/y or outer scores."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from dependencies import dependencies

def args():
 p=argparse.ArgumentParser();p.add_argument('--runtime',type=Path,default=Path(r'D:\a0_cove_seed42_v1_runtime'));p.add_argument('--source',type=Path,default=Path(r'E:\DRE-nips\new-pipeline\7-11'));opt=p.parse_args();root=Path(__file__).parents[1]
 return SimpleNamespace(**vars(opt),root=root,protocol=root/'PROTOCOL_LOCK.json',original=Path(r'C:\a0_source_posterior_decoder_seed42_runtime'),previous=Path(r'C:\d2_ictal_ablation_seed42_v1_runtime'),
  compact=Path(r'C:\d2_no_source_no_zdelta_outer_seed42_v1_runtime'),frozenval=Path(r'D:\a0_compact_delta_ratio_only_seed42_v1_runtime'),features=Path(r'C:\a0_patient_local_label_reliability_seed42_runtime\gate\FEATURES_PRIVATE.npz'))

def setup(a):
 ref,c,core,original,trainer,helper,banks,base=dependencies(a);lock=json.loads(a.protocol.read_text());assert lock['no_new_outer'] and lock['arms']==['B0','B1','B2','B3']
 paths={'protocol':a.protocol,'manifest':a.previous/'public/FEATURE_GROUP_MANIFEST.json','evidence_views':a.source/'neuroez_c/evidence_views.py','aggregation':a.source/'task1_baselines/feature_aggregation.py',
  'F0_selection_registry':a.compact/'FROZEN_BEFORE_OUTER.json','frozen_validation_registry':a.frozenval/'EXPERIMENT_FREEZE.json','frozen_validation_verification':a.frozenval/'public/INDEPENDENT_VERIFICATION.json'}
 for tag,p in paths.items():
  if tag!='protocol':assert c.sha(p)==lock[tag+'_sha256']
 assert json.loads(paths['frozen_validation_verification'].read_text())['status']=='PASS'
 for folder in ['code','tests']:
  for p in sorted((a.root/folder).glob('*.py')):paths[folder+'/'+p.name]=p
 bind=dict(original=base,new={k:c.sha(p) for k,p in paths.items()});manifest=json.loads(paths['manifest'].read_text())
 return SimpleNamespace(lock=lock,ref=ref,c=c,core=core,original=original,trainer=trainer,helper=helper,banks=banks,manifest=manifest,bind=bind,pub=a.runtime/'public')

def baseline(a,fold):
 root=a.compact/f'fold{fold}';r=json.loads((a.compact/'FROZEN_BEFORE_OUTER.json').read_text())['selections'][fold-1]
 pp=a.frozenval/f'scores/F0_fold{fold}_VAL_PRIVATE.pt';h=json.loads((a.frozenval/'EXPERIMENT_FREEZE.json').read_text())['historical_files']
 return root/'BANK_PRIVATE.pt',root/'TRAIN/BEST_PRIVATE.pt',pp,dict(r,threshold=r['nez_threshold']),h

def checkpoint(a,arm,fold):return a.runtime/f'fold{fold}/{arm}/BEST_PRIVATE.pt'

def patient_rows(c,d,ix,scores,tau,arm,fold):
 rows=[]
 for p in sorted(set(d['patient'][ix])):
  k=np.flatnonzero(d['patient'][ix]==p);k=k[np.argsort(d['channel'][ix][k].astype(str),kind='stable')];ii=ix[k]
  n=len(k);stratum='<=64' if n<=64 else ('65-128' if n<=128 else '>128')
  rows.append(dict(arm=arm,fold=fold,patient=p,center=str(d['center'][ii[0]]),channel_stratum=stratum,**c.metric(d['y'][ii],scores[k],tau)))
 return rows
