"""Pinned historical readers; private paths never emitted in public results."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).parents[1]
SOURCE = Path(r'E:\DRE-nips\new-pipeline\7-11')
BASE = SOURCE/'raw_delta_encoder_seed42_v1/code'
M8_CODE = Path(r'D:\d1_small_module_exploration_seed42_v1')
sys.path.insert(0, str(BASE))
from io_utils import sha, read, save, write, load
from dependencies import context
from compact_delta_classifier import CompactDelta
from raw_store import Store
from raw_physiological_features import Extractor, FEATURES
from construct_delta import window_delta, aggregate, historical_record_mean
EV = load('progressive_original_metrics', BASE/'evaluate.py')
BOOT = load('progressive_original_bootstrap', BASE/'bootstrap.py')
METRICS = EV.METRICS

def args():
    p=argparse.ArgumentParser()
    p.add_argument('--runtime',type=Path,required=True)
    p.add_argument('--baseline',type=Path,default=Path(r'D:\raw_delta_encoder_seed42_v1_exact_resume_runtime'))
    p.add_argument('--m8',type=Path,default=Path(r'D:\d1_small_module_exploration_seed42_v1_phase3_runtime'))
    p.add_argument('--old-outer',type=Path,default=Path(r'D:\d1_small_module_exploration_seed42_v1_outer_runtime'))
    p.add_argument('--source',type=Path,default=SOURCE)
    return p.parse_args()

def jread(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def roles(meta,fold):
    ix=meta['split_fold']==fold
    r={k:set(meta['split_patient'][ix&(meta['split_role']==k)]) for k in ['fit','validation','test']}
    assert all(r[a].isdisjoint(r[b]) for a,b in [('fit','validation'),('fit','test'),('validation','test')])
    return r

def setup(a):
    lock=jread(ROOT/'PROTOCOL_LOCK.json')
    for rel,h in lock['reference_file_hashes'].items():
        anchor,tail=rel.split('/',1)
        p=(BASE.parent if anchor=='D1' else M8_CODE)/tail
        assert sha(p)==h, rel
    for p,h in lock['new_code_hashes'].items():assert sha(ROOT/'code'/p)==h, p
    c,core,fac,trainer,records,rec,banks,hist,deps=context(a.source,a.baseline)
    meta=rec['meta'];assert len(meta['patient'])==7635 and len(set(meta['patient']))==80
    assert len(set(zip(meta['patient'],meta['channel'])))==7635
    assert sum(len(q['records']) for q in rec['ledgers'])==256
    for fold,d in enumerate(banks,1):
        r=roles(meta,fold);ix=np.flatnonzero(np.isin(meta['patient'],list(r['test'])))
        assert len(r['test'])==lock['patients_by_fold'][fold-1] and len(ix)==lock['channels_by_fold'][fold-1]
        assert set(d['patient'][d['train']])==r['fit'] and set(d['patient'][d['val']])==r['validation']
        assert set(d['patient']).isdisjoint(r['test'])
    a.runtime.mkdir(parents=True,exist_ok=True);(a.runtime/'results').mkdir(exist_ok=True)
    binding=dict(protocol=sha(ROOT/'PROTOCOL_LOCK.json'),code=lock['new_code_hashes'],references=lock['reference_file_hashes'],dependencies=deps,reconstruction=sha(a.baseline/'RECONSTRUCTION_PRIVATE.pkl'),raw_registry=sha(a.baseline/'RAW_DYNAMIC_REGISTRY_PRIVATE.pkl'))
    return lock,c,core,fac,trainer,records,rec,banks,binding

def summary(df,pub,scope):
    df.groupby('arm')[METRICS].mean().reset_index().to_csv(pub/f'{scope}_SUMMARY.csv',index=False)
    for key in ['fold','center']:
        pd.DataFrame([dict(arm=arm,**{key:v},appearances=len(g),unique_patients=g.patient.nunique(),channels=int(g.channels.sum()),**g[METRICS].mean().to_dict()) for (arm,v),g in df.groupby(['arm',key])]).to_csv(pub/f'{scope}_BY_{key.upper()}.csv',index=False)

def references(a,core,banks):
    """No neural replay or fitting of A2/A3: verify immutable original selections."""
    fr=[jread(a.baseline/'STAGE_B_MODEL_FREEZE_PRIVATE.json'),jread(a.m8/'MODEL_FREEZE_PRIVATE.json')]
    rows=[]
    for arm,old,root,freeze in [('A2','D1',a.baseline,fr[0]),('A3','M8_MACRO_AWARE',a.m8,fr[1])]:
        for fold in range(1,6):
            r=next(v for v in freeze['selections'] if v['arm']==old and v['fold']==fold)
            cp=root/f'{old}/fold{fold}/TRAIN/BEST_PRIVATE.pt';assert sha(cp)==r['checkpoint_sha256']
            ck=torch.load(cp,map_location='cpu',weights_only=False)
            assert ck['epoch']==r['epoch'] and ck['threshold']==r['threshold']
            done=jread(cp.parent/'COMPLETE_PRIVATE.json')
            assert done['best_sha256']==sha(cp) and done['last_sha256']==sha(cp.parent/'LAST_PRIVATE.pt')
            bp=a.baseline/f'D1/fold{fold}/BANK_PRIVATE.pt'
            if arm=='A2':assert sha(bp)==r['bank_sha256']
            else:assert r['D1_checkpoint_sha256']==sha(a.baseline/f'D1/fold{fold}/TRAIN/BEST_PRIVATE.pt')
            bd=torch.load(bp,map_location='cpu',weights_only=False)
            assert set(bd['pre']['fit_patient_ids'])==banks[fold-1]['roles']['fit']
            import hashlib
            ph=hashlib.sha256()
            for k in ['imputer_statistics','mean','scale','var']:ph.update(k.encode());ph.update(np.asarray(bd['pre'][k]).tobytes())
            rows.append(dict(arm=arm,old_arm=old,fold=fold,checkpoint=str(cp),checkpoint_sha256=sha(cp),bank_sha256=sha(bp),epoch=r['epoch'],threshold=r['threshold'],preprocessor_sha256=ph.hexdigest(),reused_frozen=True))
    return rows

def public_selection(r):return {k:v for k,v in r.items() if k not in ['checkpoint','old_arm']}
