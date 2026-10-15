"""Pinned cohort/Raw readers; no feature tensor is passed to any new model."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).parent
SOURCE=Path(r'E:\DRE-nips\new-pipeline\7-11')
BASE=SOURCE/'raw_delta_encoder_seed42_v1/code'
BASELINE=Path(r'D:\raw_delta_encoder_seed42_v1_exact_resume_runtime')
RUNTIME=Path(r'D:\m8_raw_backbones_pr_seed42_v1_runtime')
sys.path.insert(0,str(BASE))
from io_utils import sha,read,save,write,load
from dependencies import context
from raw_store import Store
EV=load('raw_backbone_no_BA_metrics',ROOT/'evaluate.py')
from models.backbones import RawBackbone

def jr(p):return json.loads(Path(p).read_text())
def digest(o):return hashlib.sha256(json.dumps(o,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def statehash(state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def atomic_torch(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.partial');torch.save(obj,tmp);tmp.replace(p)

def roles(meta,fold):
    ix=meta['split_fold']==fold
    q={r:set(meta['split_patient'][ix&(meta['split_role']==r)]) for r in ['fit','validation','test']}
    assert all(q[a].isdisjoint(q[b]) for a,b in [('fit','validation'),('fit','test'),('validation','test')]);return q

def setup(runtime):
    lock=jr(ROOT/'configs/PROTOCOL_LOCK.json')
    source=jr(ROOT/'configs/SOURCE_AUDIT.json')
    for rel,h in source['vendor_byte_sha256'].items():assert sha(ROOT/rel)==h,rel
    assert sha(BASELINE/'RECONSTRUCTION_PRIVATE.pkl')=='73095eefbb3035200894560de3c3cfdb6259a4f28c343ecd3d7195be5b94f609'
    c,core,fac,trainer,records,rec,banks,hist,deps=context(SOURCE,BASELINE)
    meta=rec['meta'];assert len(meta['patient'])==7635 and len(set(meta['patient']))==80
    assert len(set(zip(meta['patient'],meta['channel'])))==7635
    assert sum(len(q['records']) for q in rec['ledgers'])==256
    for fold,d in enumerate(banks,1):
        r=roles(meta,fold);assert set(d['patient'][d['train']])==r['fit'] and set(d['patient'][d['val']])==r['validation']
        assert set(d['patient']).isdisjoint(r['test'])
        ix=np.flatnonzero(np.isin(meta['patient'],list(r['test'])))
        assert len(r['test'])==lock['patients_by_fold'][fold-1] and len(ix)==lock['channels_by_fold'][fold-1]
    raw=jr(BASELINE/'results/RAW_DYNAMIC_ALIGNMENT.json')
    assert raw['status']=='PASS' and raw['valid_channel_windows']==1471965 and raw['every_existing_bank_sample_reverified']
    assert raw['registry_sha256']==sha(BASELINE/'RAW_DYNAMIC_REGISTRY_PRIVATE.pkl')
    code={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(ROOT.rglob('*.py')) if '__pycache__' not in p.parts}
    binding=dict(protocol=sha(ROOT/'configs/PROTOCOL_LOCK.json'),source_audit=sha(ROOT/'configs/SOURCE_AUDIT.json'),code=code,dependencies=deps,reconstruction=sha(BASELINE/'RECONSTRUCTION_PRIVATE.pkl'),raw_registry=sha(BASELINE/'RAW_DYNAMIC_REGISTRY_PRIVATE.pkl'),raw_alignment=sha(BASELINE/'results/RAW_DYNAMIC_ALIGNMENT.json'),torch=torch.__version__,numpy=np.__version__)
    runtime=Path(runtime);runtime.mkdir(parents=True,exist_ok=True);(runtime/'public').mkdir(exist_ok=True)
    bp=runtime/'INPUT_FREEZE_PRIVATE.json'
    if bp.exists():assert jr(bp)==binding,'Training/inference source or dependency changed; do not silently resume'
    else:write(bp,binding)
    torch.set_num_threads(2);torch.set_float32_matmul_precision('highest');torch.backends.cudnn.allow_tf32=False
    return lock,core,rec,banks,binding

def build(name,pr,fold,lock,core):
    core.seed_all(42+1009*fold)
    return RawBackbone(name,pr,lock).cuda()

def raw_patient(store,pid,channels):
    raw,mask,times,perms=store.get(str(pid),list(map(str,channels)),'cuda')
    assert raw.shape[-1]==500 and torch.isfinite(raw).all() and (mask.sum((0,2))>0).all()
    return raw,mask

def arguments():
    p=argparse.ArgumentParser();p.add_argument('--runtime',type=Path,default=RUNTIME);p.add_argument('--model',choices=['TimeConvCNN','EEGNet','SEEGNet','SEEGformer'],required=True);return p.parse_args()
