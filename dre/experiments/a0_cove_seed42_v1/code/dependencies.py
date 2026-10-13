"""Exact original source-free A0 admission; legal FIT/VAL only, no new model."""
import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch


def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
 sys.modules[name]=m;spec.loader.exec_module(m);return m


def args():
 p=argparse.ArgumentParser();p.add_argument('--runtime',type=Path,default=Path(r'D:\a0_cove_seed42_v1_runtime'))
 p.add_argument('--source',type=Path,default=Path(r'E:\DRE-nips\new-pipeline\7-11'))
 p.add_argument('--original',type=Path,default=Path(r'C:\a0_source_posterior_decoder_seed42_runtime'))
 p.add_argument('--previous',type=Path,default=Path(r'C:\d2_ictal_ablation_seed42_v1_runtime'))
 p.add_argument('--features',type=Path,default=Path(r'C:\a0_patient_local_label_reliability_seed42_runtime\gate\FEATURES_PRIVATE.npz'))
 p.add_argument('--protocol',type=Path,default=Path(__file__).parents[1]/'PROTOCOL_LOCK.json')
 return p.parse_args()


def dependencies(a):
 oc=a.source/'a0_source_posterior_decoder_seed42_v1/code';c=load('common',oc/'common.py')
 amendment=a.source/'d2_ictal_outer_seed42_v1/PROTOCOL_AMENDMENT.json'
 assert c.sha(amendment)=='452a4f6957d18125a9ea1b32d827c5f50fbc81c22d9d41100173bddcd2e22503'
 ref=json.loads(amendment.read_text());cp=a.source/'pr_uncertainty_aware_supervision_seed42_v1/code/uas_core.py'
 assert c.sha(cp)==ref['core_sha256'];core=load('uas_core',cp)
 original=load('model',oc/'model.py');trainer=load('nozdelta_original_trainer',oc/'train.py')
 orig_hashes={p.name:c.sha(p) for p in oc.glob('*.py')}
 assert orig_hashes==json.loads((a.original/'public/DECODER_UNIT_TEST_AUDIT.json').read_text())['code_sha256']
 ap=a.source/'d2_ictal_ablation_seed42_v1/code/ablation_core.py'
 assert c.sha(ap)=='c567196692be1149795d93586fe25e2b474f071511bd2fbd0f561db82e00b7d5'
 helper=load('original_ablation_core',ap)
 split=Path(r'D:\nips-temp\task1_aaai_completion_training\audit\fixed_partition_manifest.csv')
 assert c.sha(a.features)==ref['feature_sha256']=='1bdcc479a2bd148bdc94614c8253bea67e4008bd98bc6dad9e8030df827db9bd'
 assert c.sha(split)==ref['split_sha256']=='fd897fa7eed2c521fd5b14c1ae95d91b5d43b2ae85822317dcda08a878f58278'
 of=a.previous/'FROZEN_BEFORE_OUTER.json';assert c.sha(of)=='c001a58553ec16c303e96256384f6089c4a56bcccbe1ae1aa418f93fb3e2e16b'
 selections=json.loads(of.read_text())['selections']
 with np.load(a.features,allow_pickle=True) as f:
  meta={k:f[k] for k in ['patient','channel','center','features','split_patient','split_fold','split_role']}
 assert len(meta['patient'])==7635 and len(set(meta['patient']))==80
 assert len(set(zip(meta['patient'],meta['channel'])))==7635
 assert hashlib.sha256(json.dumps(meta['features'].tolist(),separators=(',',':')).encode()).hexdigest()==ref['feature_order_sha256']=='b29b29129d4d4a275d9cd077aa47e46cf428fe0f25913dea75a51ff0c7ea6ca9'
 banks=[];paths={'feature_archive':a.features,'split':split,'source_free_factory':ap,'core':cp,'source_selection_metadata':of}
 for fold in range(1,6):
  bp=a.original/f'fold{fold}/BANK_PRIVATE.pt';assert c.sha(bp)==ref['bank_sha256'][fold-1]
  d=torch.load(bp,map_location='cpu',weights_only=False)
  s=meta['split_fold']==fold;ids={role:set(meta['split_patient'][s&(meta['split_role']==role)]) for role in ['fit','validation','test']}
  assert all(ids[x].isdisjoint(ids[y]) for x,y in [('fit','validation'),('fit','test'),('validation','test')])
  assert set(d['patient'][d['train']])==ids['fit'] and set(d['patient'][d['val']])==ids['validation']
  assert not set(d['patient'])&ids['test'] and set(d['pre']['fit_patient_ids'])==ids['fit']
  assert np.array_equal(d['features'],meta['features'])
  checkpoint=a.previous/f'no_source/fold{fold}/TRAIN/BEST_PRIVATE.pt'
  rec=next(r for r in selections if r['arm']=='w/o Source' and r['fold']==fold)
  assert c.sha(checkpoint)==rec['checkpoint_sha256']
  bank=a.previous/f'no_source/fold{fold}/BANK_PRIVATE.pt';assert c.sha(bank)==rec['bank_sha256']
  paths[f'bank{fold}']=bp;paths[f'baseline_checkpoint{fold}']=checkpoint;paths[f'baseline_bank{fold}']=bank
  banks.append(d)
 torch.set_num_threads(2);a.runtime.mkdir(parents=True,exist_ok=True);(a.runtime/'public').mkdir(exist_ok=True)
 return ref,c,core,original,trainer,helper,banks,{'original_code':orig_hashes,'dependencies':{k:c.sha(p) for k,p in paths.items()}}
