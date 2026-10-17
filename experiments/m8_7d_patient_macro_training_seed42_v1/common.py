"""Original seven measurements, same roles and raw; prospectively changed pool."""
import importlib.util
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

HERE=Path(__file__).resolve().parent
ROOT=Path(r'D:\m8_7d_patient_macro_training_seed42_v1_runtime');PUB=ROOT/'public'
BASE_CODE=Path(r'D:\m8_7d_balanced_single_head_seed42_v1')
BASE_ROOT=Path(r'D:\m8_7d_balanced_single_head_seed42_v1_runtime')
BASE_SEAL=BASE_ROOT/'final_stage_v1/OUTER_SCORE_SEAL_PRIVATE.json'


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m


previous=module('patient_macro_training_previous',BASE_CODE/'common.py')
MODEL=module('patient_macro_training_model',HERE/'model.py')
sys.path.insert(0,str(HERE))
io=previous.io;legacy=previous.legacy
sha,read,save,write,freeze,jr,digest=io.sha,io.read,io.save,io.write,io.freeze,io.jr,io.digest
atomic_torch,state_hash,seed_all=io.atomic_torch,io.state_hash,io.seed_all
ARMS=MODEL.ARMS;NEW=ARMS[1:];METRICS=previous.METRICS;EV=previous.EV
CRITICAL=['PROTOCOL_LOCK.json','common.py','model.py','test_model.py','admit.py','train.py','verify_internal.py','run_internal.ps1']


def setup():
    ROOT.mkdir(exist_ok=True);PUB.mkdir(exist_ok=True);own=sys.modules[__name__]
    try:prior,core,fac,records,rec,banks,_=previous.setup()
    finally:sys.modules['common']=own;sys.modules['model']=MODEL;sys.path.insert(0,str(HERE))
    assert sha(BASE_ROOT/'MODEL_FREEZE_PRIVATE.json')=='c4aaecb2be9dbaa6f0b0da0351a7fa0b3eec13057a763c5b863ed5b613192e10'
    assert sha(BASE_SEAL)=='9f5beef8b232cff825cf1a3b560d9a54f10bdfd4ee9bdf27f3275b3bb7774e6c'
    assert jr(BASE_ROOT/'public/INDEPENDENT_INTERNAL_VERIFICATION.json')['status']=='PASS'
    binding=dict(source={n:sha(HERE/n) for n in CRITICAL},prior_input_sha256=digest(prior),
        reference_model_freeze=sha(BASE_ROOT/'MODEL_FREEZE_PRIVATE.json'),
        protected_Outer_seal=sha(BASE_SEAL),
        reconstruction=io.EXPECTED_REC,raw_registry=io.EXPECTED_REG,torch=torch.__version__,numpy=np.__version__)
    freeze(ROOT/'INPUT_FREEZE_PRIVATE.json',binding)
    return binding,core,fac,records,rec,banks,jr(BASE_ROOT/'MODEL_FREEZE_PRIVATE.json')


def make(core,fac,records,fold,arm):
    source=legacy.template(core,fac,records,fold)
    m=MODEL.SingleHead(arm,source.cnn,source.gate,type(source).features,42+1009*fold+170003).cuda()
    del source;return m


def bank(fold,banks):return previous.bank(fold,banks)
def scale(fold):return previous.scale(fold)
def store():
    q=legacy.OLD.P3.Inputs(legacy.OLD.BASELINE,Path(r'D:\dgm_net_seed42_v1_runtime'),ROOT)
    q.arm='M1_SCALED_DGM';return q
def predict(m,d,ix,inputs,ss):return previous.predict(m,d,ix,inputs,ss)
probability=previous.probability;patient_rows=previous.patient_rows;cluster=previous.cluster


def control(fr,fold):
    r=next(r for r in fr['cells'] if r['arm']=='C2_INTERACTION' and r['fold']==fold)
    assert sha(r['checkpoint'])==r['checkpoint_sha256'] and sha(r['validation'])==r['validation_sha256']
    state=torch.load(r['checkpoint'],map_location='cpu',weights_only=False);q=read(r['validation'])
    assert q['binding']==state['binding'];return r,state,q


def choose(frame):
    scores=[]
    for arm,g in frame.groupby('arm'):
        sd=float(g.groupby('fold').macro_f1.mean().std(ddof=1))
        scores.append(dict(arm=arm,macro=float(g.macro_f1.mean()),auroc=float(g.auroc.mean()),
            auprc=float(g.auprc.mean()),fold_sd=sd,penalized=float(g.macro_f1.mean()-.1*sd)))
    base=next(r for r in scores if r['arm']=='C2_REFERENCE')
    eligible=[r for r in scores if r['arm'] in NEW and r['macro']>=base['macro']+.003
        and r['auroc']>=base['auroc']-.01 and r['auprc']>=base['auprc']-.01]
    selected=min(eligible,key=lambda r:(-r['penalized'],ARMS.index(r['arm'])))['arm'] if eligible else 'C2_REFERENCE'
    return dict(scores=scores,selected=selected,new_qualified=bool(eligible),eligible=[r['arm'] for r in eligible])
