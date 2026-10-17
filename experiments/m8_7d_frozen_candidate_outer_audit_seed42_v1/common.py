"""Separate read-only adapter for ten historical frozen single-head candidates."""
import csv
import gc
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

HERE=Path(__file__).resolve().parent
ROOT=Path(r'D:\m8_7d_frozen_candidate_outer_audit_seed42_v1_runtime');PUBLIC=ROOT/'public'
BASE_CODE=Path(r'D:\m8_7d_balanced_single_head_seed42_v1')
BASE_ROOT=Path(r'D:\m8_7d_balanced_single_head_seed42_v1_runtime')
BASE_SEAL=BASE_ROOT/'final_stage_v1/OUTER_SCORE_SEAL_PRIVATE.json'
BASE_FREEZE_SHA='c4aaecb2be9dbaa6f0b0da0351a7fa0b3eec13057a763c5b863ed5b613192e10'
BASE_SEAL_SHA='9f5beef8b232cff825cf1a3b560d9a54f10bdfd4ee9bdf27f3275b3bb7774e6c'
STUDIES=[
 ('raw_dispersion','2442250873dc3e10fc173edafc1c0ce848917924212607a084475f218093c042','336086e5fa8f063d12e7a1baf77010cad4b19213810a4947c5b8ec942f2dc7f3'),
 ('raw_film','83ffe0516ce3f7ea8b13448b498948f5b04b3a630ad538a0037134b1243721f7','778e4544c01bfccc55fb0a44a35a15ae235daa22f93947a6842401674657915c'),
 ('context_affine','f32af4b55d91386abdec21d5c478601a45e281299bbbdb09d64e54da77d066f1','b79f87abff78ac22638bb1a2766e18d5b1d609b6175be8dabd63b5ceffcd1740'),
 ('channel_relation','21c4a00f797fba9a881d8e24175f0babc86332c6a439a797caf1ca0c11400fc9','fcfc6537d69b3b70bf984fcf80b6bea5f5507ecd980f3140665285542c5e9b5f'),
 ('patient_macro_training','9aeb0770c177993048dff7585ca23cb22acd06147766b4797f61eb607108d361','24eb24e9dcbdb38665a69b122b979cf7fc2e2e601d0b56c35b29ad3c40810658')]
ARMS=json.loads((HERE/'PROTOCOL_LOCK.json').read_text())['arms'];NEW=ARMS[1:]
METRICS=['macro_f1','auroc','auprc','f1','nez_f1','accuracy','mrr','hit_at_1','ez_fraction_bias']
PAIRS=[(a,'C2_REFERENCE') for a in NEW]+[(ARMS[j+1],ARMS[j]) for j in [1,3,5,7,9]]
FILES=['PROTOCOL_LOCK.json','common.py','admit.py','evaluate_outer.py','report_outer.py','verify_outer.py','test_contract.py','verify_public.py','run_outer.ps1']

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m

base=module('frozen_audit_base',BASE_CODE/'common.py')
io,legacy,EV=base.io,base.legacy,base.EV
sha,read,save,write,freeze,jr,digest=io.sha,io.read,io.save,io.write,io.freeze,io.jr,io.digest
state_hash=io.state_hash;probability=base.probability
MODULES={}
for name,_,_ in STUDIES:
    MODULES[name]=module('frozen_audit_'+name,Path('D:/m8_7d_'+name+'_seed42_v1/common.py'))
sys.path.insert(0,str(HERE));sys.modules['common']=sys.modules[__name__]

def setup():
    ROOT.mkdir(exist_ok=True);PUBLIC.mkdir(exist_ok=True);own=sys.modules[__name__]
    try:b,core,fac,records,rec,banks,_=base.setup()
    finally:sys.modules['common']=own;sys.path.insert(0,str(HERE))
    assert sha(BASE_ROOT/'MODEL_FREEZE_PRIVATE.json')==BASE_FREEZE_SHA
    assert sha(BASE_SEAL)==BASE_SEAL_SHA
    assert jr(BASE_ROOT/'public/INDEPENDENT_INTERNAL_VERIFICATION.json')['status']=='PASS'
    assert len(ARMS)==11 and len(set(ARMS))==11 and len(PAIRS)==15
    return b,core,fac,records,rec,banks,jr(BASE_ROOT/'MODEL_FREEZE_PRIVATE.json')

def sources():return {n:sha(HERE/n) for n in FILES}
def bank(fold,banks):return base.bank(fold,banks)
def scale(fold):return base.scale(fold)
def store():
    q=legacy.OLD.P3.Inputs(legacy.OLD.BASELINE,Path(r'D:\dgm_net_seed42_v1_runtime'),ROOT)
    q.arm='M1_SCALED_DGM';return q

def catalogue(old):
    result=[];studies=[]
    for name,fsha,csha in STUDIES:
        mod=MODULES[name];root=mod.ROOT
        assert sha(root/'MODEL_FREEZE_PRIVATE.json')==fsha and sha(root/'CANDIDATE_PRIVATE.json')==csha
        fr=jr(root/'MODEL_FREEZE_PRIVATE.json');inp=jr(root/'INPUT_FREEZE_PRIVATE.json')
        assert fr['input']==inp and len(fr['cells'])==15
        for n,h in inp['source'].items():assert sha(mod.HERE/n)==h
        assert jr(root/'public/INDEPENDENT_INTERNAL_VERIFICATION.json')['status']=='PASS'
        # Old selector is intentionally neither called nor used as an eligibility gate.
        assert jr(root/'CANDIDATE_PRIVATE.json')['model_freeze_sha256']==fsha
        studies.append(dict(study=name,model_freeze_sha256=fsha,candidate_sha256=csha,
            input_sha256=sha(root/'INPUT_FREEZE_PRIVATE.json'),verification_sha256=sha(root/'public/INDEPENDENT_INTERNAL_VERIFICATION.json'),source=inp['source']))
        for arm in mod.NEW:
            assert arm in NEW
            for fold in range(1,6):
                r=next(r for r in fr['cells'] if r['arm']==arm and r['fold']==fold)
                assert sha(r['checkpoint'])==r['checkpoint_sha256'] and sha(r['validation'])==r['validation_sha256']
                q=torch.load(r['checkpoint'],map_location='cpu',weights_only=False);v=read(r['validation'])
                assert q['binding']==v['binding'] and q['binding']['input']==inp and q['binding']['arm']==arm
                assert q['epoch']>=1 and 0<=q['threshold']<=1
                if name=='context_affine':assert sha(r['body_checkpoint'])==r['body_checkpoint_sha256']
                result.append(dict(r,study=name,threshold=float(q['threshold']),epoch=int(q['epoch'])))
    for fold in range(1,6):
        r=next(r for r in old['cells'] if r['arm']=='C2_INTERACTION' and r['fold']==fold)
        assert sha(r['checkpoint'])==r['checkpoint_sha256'] and sha(r['validation'])==r['validation_sha256']
        q=torch.load(r['checkpoint'],map_location='cpu',weights_only=False)
        result.append(dict(r,arm='C2_REFERENCE',study='reference',threshold=float(q['threshold']),epoch=int(q['epoch'])))
    result.sort(key=lambda r:(ARMS.index(r['arm']),r['fold']))
    assert len(result)==55 and {(r['arm'],r['fold']) for r in result}=={(a,f) for a in ARMS for f in range(1,6)}
    return result,studies

def make(r,core,fac,records,old):
    state=torch.load(r['checkpoint'],map_location='cpu',weights_only=False);val=read(r['validation'])
    if r['study']=='reference':m=base.make(core,fac,records,r['fold'],'C2_INTERACTION')
    elif r['study']=='context_affine':
        mod=MODULES['context_affine'];body,br,_,_=mod.body(old,r['fold'],core,fac,records)
        assert br['checkpoint_sha256']==r['body_checkpoint_sha256']
        head=mod.make(r['arm'],body.head[2],r['fold']);assert state_hash(head.state_dict())==val['binding']['initial']
        head.load_state_dict(state['model'])
        assert state_hash(head.readout.state_dict())==state_hash(body.head[2].state_dict())==val['binding']['readout']
        m=mod.MODEL.Fused(body,head).cuda();del body,head
    else:m=MODULES[r['study']].make(core,fac,records,r['fold'],r['arm'])
    if r['study']!='context_affine':
        assert state_hash(m.state_dict())==val['binding']['initial'];m.load_state_dict(state['model'])
    m.eval()
    assert sum(isinstance(z,torch.nn.Linear) and z.out_features==1 for z in m.modules())==1
    assert state['threshold']==r['threshold'] and state['epoch']==r['epoch']
    return m,val

def preprocessing(banks):
    rows=[]
    for fold in range(1,6):
        d=bank(fold,banks);ss=scale(fold)
        assert set(d['pre']['fit_patient_ids'])==d['roles']['fit'] and d['x'].shape[1]==14
        assert d['roles']['fit'].isdisjoint(d['roles']['test']) and d['roles']['validation'].isdisjoint(d['roles']['test'])
        rows.append(dict(fold=fold,preprocessor={k:legacy.array_sha(np.asarray(v)) for k,v in d['pre'].items() if k!='fit_patient_ids'},scale=legacy.array_sha(ss.cpu().numpy()),
            roles={k:digest(sorted(v)) for k,v in d['roles'].items()}))
    return rows

def cache_binding(fh,r,ii,x,ss):
    return dict(freeze=fh,checkpoint=r['checkpoint_sha256'],indices=legacy.array_sha(ii),input=legacy.array_sha(x),scale=legacy.array_sha(ss.cpu().numpy()))

def read_frozen():
    fr=jr(ROOT/'FROZEN_BEFORE_OUTER_PRIVATE.json')
    assert fr['sources']==sources() and fr['status']=='ALL55_CELLS_FROZEN_BEFORE_NEW_OUTER'
    assert sha(BASE_SEAL)==fr['old_reference_seal']==BASE_SEAL_SHA
    assert sha(ROOT/'ADMISSION_PRIVATE.json')==fr['admission_sha256']
    for s in fr['studies']:
        mod=MODULES[s['study']]
        for n,h in s['source'].items():assert sha(mod.HERE/n)==h
        for n,k in [('MODEL_FREEZE_PRIVATE.json','model_freeze_sha256'),('CANDIDATE_PRIVATE.json','candidate_sha256'),('INPUT_FREEZE_PRIVATE.json','input_sha256'),('public/INDEPENDENT_INTERNAL_VERIFICATION.json','verification_sha256')]:assert sha(mod.ROOT/n)==s[k]
    for p,h in fr['loaded_source_closure'].items():assert sha(p)==h
    for r in fr['models']:
        assert sha(r['checkpoint'])==r['checkpoint_sha256'] and sha(r['validation'])==r['validation_sha256']
        if 'body_checkpoint' in r:assert sha(r['body_checkpoint'])==r['body_checkpoint_sha256']
    return fr

def loaded_closure():
    paths={str(Path(m.__file__).resolve()) for m in list(sys.modules.values()) if getattr(m,'__file__',None) and str(m.__file__).lower().startswith('d:\\') and str(m.__file__).endswith('.py')}
    return {p:sha(p) for p in sorted(paths)}

def csv_out(name,rows):
    assert rows and all(set(r)==set(rows[0]) for r in rows)
    PUBLIC.mkdir(exist_ok=True)
    path=PUBLIC/(name+'.csv');tmp=path.with_suffix('.csv.tmp')
    with tmp.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    tmp.replace(path)

def vector(rows):return np.array([[r[m] for m in METRICS] for r in rows],dtype=np.float64)

def intervals(values,draw,levels=(.025,.975)):
    assert values.shape==(80,9) and draw.shape==(10000,80) and np.isfinite(values).all()
    # Chunked ordinary NumPy mean: no pandas Cython grouping on the bootstrap path.
    samples=np.concatenate([values[draw[k:k+250]].mean(axis=1) for k in range(0,10000,250)])
    return values.mean(0),np.quantile(samples,levels,axis=0)
