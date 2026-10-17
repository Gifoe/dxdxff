"""Exact original7D/Raw inputs, sealed comparison, no new Outer reader."""
import importlib.util
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

HERE=Path(__file__).resolve().parent
ROOT=Path(r'D:\m8_7d_hidden_anchor_seed42_v1_runtime');PUB=ROOT/'public'
BASE_CODE=Path(r'D:\m8_7d_balanced_single_head_seed42_v1');BASE_ROOT=Path(r'D:\m8_7d_balanced_single_head_seed42_v1_runtime')


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m


previous=module('hidden_anchor_exact_inputs',BASE_CODE/'common.py')
MODEL=module('hidden_anchor_model',HERE/'model.py');sys.path.insert(0,str(HERE))
io,legacy=previous.io,previous.legacy
sha,read,save,write,freeze,jr,digest=io.sha,io.read,io.save,io.write,io.freeze,io.jr,io.digest
atomic_torch,state_hash,seed_all=io.atomic_torch,io.state_hash,io.seed_all
METRICS=previous.METRICS;EV=previous.EV;probability,patient_rows,cluster=previous.probability,previous.patient_rows,previous.cluster
ARMS=['C2_REFERENCE','P_WARM_ONLY']+MODEL.ARMS;NEW=MODEL.ARMS
CRITICAL=['PROTOCOL_LOCK.json','common.py','model.py','test_model.py','admit.py','train.py','verify_internal.py','run_internal.ps1']


def setup():
    ROOT.mkdir(exist_ok=True);PUB.mkdir(exist_ok=True);own=sys.modules[__name__]
    try:old,core,fac,records,rec,banks,_=previous.setup()
    finally:sys.modules['common']=own;sys.modules['model']=MODEL;sys.path.insert(0,str(HERE))
    assert sha(BASE_ROOT/'MODEL_FREEZE_PRIVATE.json')=='c4aaecb2be9dbaa6f0b0da0351a7fa0b3eec13057a763c5b863ed5b613192e10'
    assert sha(BASE_ROOT/'final_stage_v1/OUTER_SCORE_SEAL_PRIVATE.json')=='9f5beef8b232cff825cf1a3b560d9a54f10bdfd4ee9bdf27f3275b3bb7774e6c'
    assert jr(BASE_ROOT/'public/INDEPENDENT_INTERNAL_VERIFICATION.json')['status']=='PASS'
    binding=dict(source={n:sha(HERE/n) for n in CRITICAL},previous_input=digest(old),previous_model_freeze=sha(BASE_ROOT/'MODEL_FREEZE_PRIVATE.json'),
        old_Outer_protection=sha(BASE_ROOT/'final_stage_v1/OUTER_SCORE_SEAL_PRIVATE.json'),raw_registry=io.EXPECTED_REG,reconstruction=io.EXPECTED_REC,
        torch=torch.__version__,numpy=np.__version__)
    freeze(ROOT/'INPUT_FREEZE_PRIVATE.json',binding)
    return binding,core,fac,records,rec,banks,jr(BASE_ROOT/'MODEL_FREEZE_PRIVATE.json')


def make(core,fac,records,fold,arm):
    source=legacy.template(core,fac,records,fold)
    # Only an untrained physiological initializer is copied, never a trained D1.
    m=MODEL.Unified(arm,source).cuda();del source;return m


def bank(fold,banks):return previous.bank(fold,banks)
def scale(fold):return previous.scale(fold)
def store():
    q=legacy.OLD.P3.Inputs(legacy.OLD.BASELINE,Path(r'D:\dgm_net_seed42_v1_runtime'),ROOT);q.arm='M1_SCALED_DGM';return q
def predict(m,d,ix,inputs,ss):return previous.predict(m,d,ix,inputs,ss)


def reference(fr,fold,core,fac,records,d,inputs,ss):
    r=next(r for r in fr['cells'] if r['arm']=='C2_INTERACTION' and r['fold']==fold)
    assert sha(r['checkpoint'])==r['checkpoint_sha256'] and sha(r['validation'])==r['validation_sha256']
    state=torch.load(r['checkpoint'],map_location='cpu',weights_only=False);q=read(r['validation'])
    m=previous.make(core,fac,records,fold,'C2_INTERACTION');assert state_hash(m.state_dict())==q['binding']['initial']
    m.load_state_dict(state['model']);p=predict(m,d,d['val'],inputs,ss);error=float(np.max(np.abs(p-q['logit'])));assert error<=1e-6
    assert np.array_equal(q['archive_indices'],d['archive_indices'][d['val']])
    grid=np.round(np.arange(0,1.0001,.005),3);assert np.array_equal(probability(p)[:,None]<grid,probability(q['logit'])[:,None]<grid)
    sel=core.select_threshold(d['y'][d['val']],probability(p),d['patient'][d['val']])
    for k,v in sel.items():assert abs(float(v)-float(state[k]))<1e-12
    del m;torch.cuda.empty_cache();return r,state,q,p,error


def choose(frame):
    scores=[]
    for arm,g in frame.groupby('arm'):
        sd=float(g.groupby('fold').macro_f1.mean().std(ddof=1))
        scores.append(dict(arm=arm,macro=float(g.macro_f1.mean()),auroc=float(g.auroc.mean()),auprc=float(g.auprc.mean()),fold_sd=sd,penalized=float(g.macro_f1.mean()-.1*sd)))
    base=next(r for r in scores if r['arm']=='C2_REFERENCE')
    eligible=[r for r in scores if r['arm'] in NEW and r['macro']>=base['macro']+.003 and r['auroc']>=base['auroc']-.01 and r['auprc']>=base['auprc']-.01]
    selected=min(eligible,key=lambda r:(-r['penalized'],ARMS.index(r['arm'])))['arm'] if eligible else 'C2_REFERENCE'
    return dict(scores=scores,eligible=[r['arm'] for r in eligible],selected=selected,new_qualified=bool(eligible))
