"""Pinned qualified inputs and actual own-FIT labels; never read Outer results."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

HERE=Path(__file__).resolve().parent
ROOT=Path(r'D:\m8_nonlocal6_conditional_selection_seed42_v1_runtime');PUB=ROOT/'public'
OLD=Path(r'D:\m8_physiology_discovery_selection_seed42_v2')
OLDROOT=Path(r'D:\m8_physiology_discovery_selection_seed42_v2_runtime')
QUAL=Path(r'D:\m8_nonlocal6_measurement_qualification_seed42_v1_runtime\constant_safe')
QUALCODE=Path(r'D:\m8_nonlocal6_measurement_qualification_seed42_v1')
RAWROOT=Path(r'D:\m8_rawconditional_feature_selection_seed42_v1_runtime')
BASE=Path(r'D:\raw_delta_encoder_seed42_v1_exact_resume_runtime')
SRC=Path(r'E:\DRE-nips\new-pipeline\7-11')
sys.path.insert(0,str(SRC/'raw_delta_encoder_seed42_v1/code'))
from io_utils import read,save,write,sha,load
sys.path.insert(0,str(QUALCODE))
from constant_safe import patient_z
from measurement import GROUPS,nonconstant_rate

CONTEXTS=['PHYS','PHYS_OLD7','RAW','RAW_OLD7']
METRICS=['auroc','auprc','macro_f1','f1','nez_f1','accuracy','mrr','hit_at_1','ez_fraction_bias']
PRIMARY=['auroc','auprc','macro_f1','f1','mrr','hit_at_1']
UTILITY=['auroc','auprc','macro_f1','f1']
SOURCES=['hup','lzu','multicenter','pediatric']


def jr(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def freeze(p,v):
    if p.exists():assert jr(p)==v,p.name
    else:write(p,v)


def primitives():
    path=OLD/'probes.py';assert sha(path)=='9d29784cce73f67329164ff25343f9dae8838a849cfad9632ee2cef77e0bbf0f'
    return load('nonlocal6_original_metric_primitives',path)


def source_binding():
    return {n:sha(HERE/n) for n in ['PROTOCOL_LOCK.json','study_io.py','prepare_inputs.py','select_and_seal.py','ridge.py','test_ridge.py']}


def fit_identity_bank(fold,rec):
    path=Path(r'C:\a0_source_posterior_decoder_seed42_runtime')/f'fold{fold}/BANK_PRIVATE.pt'
    amend=SRC/'d2_ictal_outer_seed42_v1/PROTOCOL_AMENDMENT.json'
    assert sha(amend)=='452a4f6957d18125a9ea1b32d827c5f50fbc81c22d9d41100173bddcd2e22503'
    assert sha(path)==jr(amend)['bank_sha256'][fold-1]
    b=torch.load(path,map_location='cpu',weights_only=False);tr=b['train'];meta=rec['meta']
    lookup={(str(p),str(c)):i for i,(p,c) in enumerate(zip(meta['patient'],meta['channel']))}
    patients=np.asarray(b['patient'])[tr].astype(str);channels=np.asarray(b['channel'])[tr].astype(str)
    ix=np.array([lookup[p,c] for p,c in zip(patients,channels)])
    fit=set(meta['split_patient'][(meta['split_fold']==fold)&(meta['split_role']=='fit')])
    assert set(patients)==fit
    forbidden=set(meta['split_patient'][(meta['split_fold']==fold)&(meta['split_role']!='fit')])
    assert fit.isdisjoint(forbidden)
    return dict(archive_indices=ix,patient=patients,channel=channels,center=np.asarray(meta['center'])[ix],
        own_fit_ez=(np.asarray(b['y'])[tr]==0).astype(int),bank_sha256=sha(path))


def context_input(view,context):
    parts=[]
    if context.startswith('RAW'):parts.append(view['raw'])
    if context.endswith('OLD7'):parts.append(view['old14'])
    base=np.concatenate(parts,1) if parts else np.empty((len(view['patient']),0))
    return np.c_[base,view['new12']],base.shape[1]


def source_rates(measurements,archive_indices,train):
    wanted=set(archive_indices[train]);unit=0;finite=np.zeros(6,int);eligible=np.zeros(6,int)
    for r in measurements['records']:
        columns=np.array([int(i) in wanted for i in r['archive_indices']]);present=r['present'][:,columns]
        a=r['scalar'][:,columns];unit+=int(present.sum());finite+=np.isfinite(a[present]).sum(0)
        for j in [0,1,4,5]:eligible[j]+=int(present.sum())
        eligible[2:4]+=r['eligible'][:,columns][present].sum(0)
    assert unit>0
    return finite/unit,eligible/unit
