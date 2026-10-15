"""Private historical artifacts and source pins; no waveform extraction."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
import torch

ROOT=Path(__file__).parent
SOURCE=Path(r'E:\DRE-nips\new-pipeline\7-11')
BASE=SOURCE/'raw_delta_encoder_seed42_v1/code'
sys.path.insert(0,str(BASE))
from io_utils import sha,read,save,write,load
from dependencies import context
EV=load('feature_pr_metrics',ROOT/'evaluate.py')
METRICS=EV.CLASSIFICATION+EV.RANKING+['predicted_ez_fraction']

def jread(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def digest(q):return hashlib.sha256(json.dumps(q,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def arguments():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['gate','train','outer','verify'])
    p.add_argument('--runtime',type=Path,required=True)
    p.add_argument('--baseline',type=Path,default=Path(r'D:\raw_delta_encoder_seed42_v1_exact_resume_runtime'))
    p.add_argument('--features',type=Path,default=Path(r'D:\d1_m8_progressive_ablation_seed42_v1_runtime'))
    p.add_argument('--references',type=Path,default=Path(r'D:\d1_small_module_exploration_seed42_v1_outer_runtime'))
    return p.parse_args()

def roles(meta,fold):
    ix=meta['split_fold']==fold;r={v:set(meta['split_patient'][ix&(meta['split_role']==v)]) for v in ['fit','validation','test']}
    assert all(r[l].isdisjoint(r[h]) for l,h in [('fit','validation'),('fit','test'),('validation','test')]);return r

def setup(a):
    lock=jread(ROOT/'configs/PROTOCOL_LOCK.json')
    for f,h in lock['code_sha256'].items():assert sha(ROOT/f)==h,f
    for f,h in lock['original_source_sha256'].items():assert sha(SOURCE/f)==h,f
    assert sha(a.features/'FEATURES_PRIVATE.pkl')==lock['feature_cache_sha256']
    assert sha(a.baseline/'RECONSTRUCTION_PRIVATE.pkl')==lock['reconstruction_sha256']
    assert sha(a.features/'results/FEATURE_PARITY.json')==lock['feature_gate_sha256']
    gate=jread(a.features/'results/FEATURE_PARITY.json');assert gate['status']=='PASS' and gate['DELTA14_exact_original_R14'] and gate['max_absolute_delta_error']==0
    c,core,fac,trainer,records,rec,banks,hist,deps=context(SOURCE,a.baseline)
    cache=read(a.features/'FEATURES_PRIVATE.pkl');x=cache['DELTA14'];meta=rec['meta']
    assert x.shape==(7635,14) and np.array_equal(x,rec['R14']) and np.isfinite(x).all()
    assert len(set(zip(meta['patient'],meta['channel'])))==7635 and len(set(meta['patient']))==80
    assert sum(len(q['records']) for q in rec['ledgers'])==256
    assert sha(Path(r'D:\nips-temp\task1_aaai_completion_training\audit\fixed_partition_manifest.csv'))==lock['split_sha256']
    visited=[]
    for fold,d in enumerate(banks,1):
        r=roles(meta,fold);ix=np.flatnonzero(np.isin(meta['patient'],list(r['test'])));visited.extend(ix.tolist())
        assert (len(r['test']),len(ix))==(lock['patients_by_fold'][fold-1],lock['channels_by_fold'][fold-1])
        assert set(d['patient'][d['train']])==r['fit'] and set(d['patient'][d['val']])==r['validation'] and set(d['patient']).isdisjoint(r['test'])
    assert sorted(visited)==list(range(7635))
    import scipy,xgboost
    versions=dict(numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,xgboost=xgboost.__version__,torch=torch.__version__)
    assert versions['sklearn']=='1.7.2' and versions['xgboost']=='3.0.5'
    binding=dict(protocol_sha256=sha(ROOT/'configs/PROTOCOL_LOCK.json'),code=lock['code_sha256'],feature_sha256=lock['feature_cache_sha256'],feature_gate_sha256=lock['feature_gate_sha256'],reconstruction_sha256=lock['reconstruction_sha256'],dependencies=deps,versions=versions)
    a.runtime.mkdir(parents=True,exist_ok=True);(a.runtime/'public').mkdir(exist_ok=True)
    return lock,core,rec,banks,x,binding

def transform_view(core,x,patient,pr):
    z=core.patient_z(x,patient) if pr else np.asarray(x,dtype=np.float64).copy()
    assert z.shape==x.shape and np.isfinite(z).all()
    if pr:
        independent=np.empty_like(z)
        for pid in np.unique(patient):
            ix=patient==pid;a=np.asarray(x[ix],dtype=np.float64);s=np.std(a,axis=0,ddof=0)
            independent[ix]=(a-np.mean(a,axis=0))/np.where(s>1e-8,s,1.)
        assert np.array_equal(z,independent)
    return z
