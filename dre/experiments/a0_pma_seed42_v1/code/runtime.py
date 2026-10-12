"""Pinned original dependencies, legal FIT/VAL banks, immutable current B0."""
import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np
import torch


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m


def args():
    p=argparse.ArgumentParser()
    p.add_argument('--runtime',type=Path,default=Path(r'C:\a0_pma_seed42_v1_runtime'))
    p.add_argument('--source',type=Path,default=Path(r'E:\DRE-nips\new-pipeline\7-11'))
    p.add_argument('--previous',type=Path,default=Path(r'C:\d2_ictal_ablation_seed42_v1_runtime'))
    p.add_argument('--original',type=Path,default=Path(r'C:\a0_source_posterior_decoder_seed42_runtime'))
    p.add_argument('--features',type=Path,default=Path(r'C:\a0_patient_local_label_reliability_seed42_runtime\gate\FEATURES_PRIVATE.npz'))
    p.add_argument('--protocol',type=Path,default=Path(__file__).parents[1]/'PROTOCOL_LOCK.json')
    return p.parse_args()


def setup(a):
    oc=a.source/'a0_source_posterior_decoder_seed42_v1/code'
    common=load('common',oc/'common.py')
    lock=json.loads(a.protocol.read_text());assert lock['arms']==['B0','B1','B2','B3']
    amendment=a.source/'d2_ictal_outer_seed42_v1/PROTOCOL_AMENDMENT.json'
    assert common.sha(amendment)=='452a4f6957d18125a9ea1b32d827c5f50fbc81c22d9d41100173bddcd2e22503'
    ref=json.loads(amendment.read_text());corepath=a.source/'pr_uncertainty_aware_supervision_seed42_v1/code/uas_core.py'
    assert common.sha(corepath)==ref['core_sha256']
    core=load('uas_core',corepath);original=load('model',oc/'model.py')
    trainer=load('pma_original_trainer',oc/'train.py')
    hashes={p.name:common.sha(p) for p in oc.glob('*.py')}
    assert hashes==json.loads((a.original/'public/DECODER_UNIT_TEST_AUDIT.json').read_text())['code_sha256']
    assert common.sha(a.features)==lock['feature_sha256']==ref['feature_sha256']
    split=Path(r'D:\nips-temp\task1_aaai_completion_training\audit\fixed_partition_manifest.csv')
    assert common.sha(split)==ref['split_sha256']
    oldfreeze=a.previous/'FROZEN_BEFORE_OUTER.json';assert common.sha(oldfreeze)==lock['previous_freeze_sha256']
    selections=json.loads(oldfreeze.read_text())['selections'];paths={'protocol':a.protocol,'features':a.features,'split':split,'core':corepath,'oldfreeze':oldfreeze}
    banks=[]
    # Only unlabeled identity/order/split metadata are materialized from the full NPZ.
    with np.load(a.features,allow_pickle=True) as f:
        meta={k:f[k] for k in ['features','patient','channel','center','split_fold','split_role','split_patient']}
    assert len(meta['patient'])==7635 and len(set(meta['patient']))==80
    assert len(set(zip(meta['patient'],meta['channel'])))==7635
    assert hashlib.sha256(json.dumps(meta['features'].tolist(),separators=(',',':')).encode()).hexdigest()==ref['feature_order_sha256']
    for fold in range(1,6):
        bp=a.original/f'fold{fold}/BANK_PRIVATE.pt';assert common.sha(bp)==ref['bank_sha256'][fold-1]
        d=torch.load(bp,map_location='cpu',weights_only=False)
        s=meta['split_fold']==fold;ids={r:set(meta['split_patient'][s&(meta['split_role']==r)]) for r in ['fit','validation','test']}
        assert all(ids[x].isdisjoint(ids[y]) for x,y in [('fit','validation'),('fit','test'),('validation','test')])
        assert set(d['patient'][d['train']])==ids['fit'] and set(d['patient'][d['val']])==ids['validation']
        assert set(d['pre']['fit_patient_ids'])==ids['fit'] and not set(d['patient'])&ids['test']
        assert np.array_equal(d['features'],meta['features'])
        cp=a.previous/f'no_source/fold{fold}/TRAIN/BEST_PRIVATE.pt'
        rec=next(r for r in selections if r['arm']=='w/o Source' and r['fold']==fold)
        assert common.sha(cp)==rec['checkpoint_sha256']
        previous_bank=a.previous/f'no_source/fold{fold}/BANK_PRIVATE.pt';assert common.sha(previous_bank)==rec['bank_sha256']
        paths[f'original_bank{fold}']=bp;paths[f'B0_ck{fold}']=cp;paths[f'B0_bank{fold}']=previous_bank
        banks.append(d)
    here=Path(__file__).parent
    for p in sorted(here.glob('*.py')):paths['new_'+p.name]=p
    for p in sorted(here.parent.joinpath('tests').glob('*.py')):paths['test_'+p.name]=p
    binding={'original_code':hashes,'dependencies':{k:common.sha(p) for k,p in paths.items()}}
    a.runtime.mkdir(parents=True,exist_ok=True);(a.runtime/'public').mkdir(exist_ok=True)
    models=load('pma_models',here/'model.py')
    torch.set_num_threads(2)
    return lock,ref,common,core,original,trainer,models,banks,binding


def checkpoint(a,arm,fold):
    return a.previous/f'no_source/fold{fold}/TRAIN/BEST_PRIVATE.pt' if arm=='B0' else a.runtime/f'fold{fold}/{arm}/BEST_PRIVATE.pt'


def patient_rows(common,d,ix,scores,tau,arm,fold):
    rows=[]
    for pat in sorted(set(d['patient'][ix])):
        k=np.flatnonzero(d['patient'][ix]==pat)
        k=k[np.argsort(d['channel'][ix][k].astype(str),kind='stable')];si=ix[k]
        rows.append({'arm':arm,'fold':fold,'patient':pat,'center':str(d['center'][si[0]]),**common.metric(d['y'][si],scores[k],tau)})
    return rows
