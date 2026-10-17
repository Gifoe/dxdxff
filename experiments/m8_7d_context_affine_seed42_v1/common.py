"""Exact frozen C2 body, original7D and original ownFIT/Development roles."""
import importlib.util
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

HERE=Path(__file__).resolve().parent
ROOT=Path(r'D:\m8_7d_context_affine_seed42_v1_runtime');PUB=ROOT/'public'
BASE_CODE=Path(r'D:\m8_7d_balanced_single_head_seed42_v1');BASE_ROOT=Path(r'D:\m8_7d_balanced_single_head_seed42_v1_runtime')
BASE_SEAL=BASE_ROOT/'final_stage_v1/OUTER_SCORE_SEAL_PRIVATE.json'


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m


previous=module('context_affine_exact_inputs',BASE_CODE/'common.py')
MODEL=module('context_affine_model',HERE/'model.py');sys.path.insert(0,str(HERE))
io,legacy=previous.io,previous.legacy
sha,read,save,write,freeze,jr,digest=io.sha,io.read,io.save,io.write,io.freeze,io.jr,io.digest
atomic_torch,state_hash,seed_all=io.atomic_torch,io.state_hash,io.seed_all
METRICS,EV=previous.METRICS,previous.EV
probability,patient_rows,cluster=previous.probability,previous.patient_rows,previous.cluster
ARMS=['C2_REFERENCE']+MODEL.ARMS;NEW=MODEL.ARMS
CRITICAL=['PROTOCOL_LOCK.json','common.py','model.py','test_model.py','prepare.py','admit.py','train.py','verify_internal.py','run_internal.ps1']


def setup():
    ROOT.mkdir(exist_ok=True);PUB.mkdir(exist_ok=True);own=sys.modules[__name__]
    try:old,core,fac,records,rec,banks,_=previous.setup()
    finally:sys.modules['common']=own;sys.modules['model']=MODEL;sys.path.insert(0,str(HERE))
    assert sha(BASE_ROOT/'MODEL_FREEZE_PRIVATE.json')=='c4aaecb2be9dbaa6f0b0da0351a7fa0b3eec13057a763c5b863ed5b613192e10'
    assert sha(BASE_ROOT/'final_stage_v1/OUTER_SCORE_SEAL_PRIVATE.json')=='9f5beef8b232cff825cf1a3b560d9a54f10bdfd4ee9bdf27f3275b3bb7774e6c'
    assert jr(BASE_ROOT/'public/INDEPENDENT_INTERNAL_VERIFICATION.json')['status']=='PASS'
    binding=dict(source={n:sha(HERE/n) for n in CRITICAL},previous_input=digest(old),body_models=sha(BASE_ROOT/'MODEL_FREEZE_PRIVATE.json'),
        protected_Outer_seal=sha(BASE_ROOT/'final_stage_v1/OUTER_SCORE_SEAL_PRIVATE.json'),raw_registry=io.EXPECTED_REG,reconstruction=io.EXPECTED_REC,torch=torch.__version__,numpy=np.__version__)
    freeze(ROOT/'INPUT_FREEZE_PRIVATE.json',binding)
    return binding,core,fac,records,rec,banks,jr(BASE_ROOT/'MODEL_FREEZE_PRIVATE.json')


def bank(fold,banks):return previous.bank(fold,banks)
def scale(fold):return previous.scale(fold)
def store():
    q=legacy.OLD.P3.Inputs(legacy.OLD.BASELINE,Path(r'D:\dgm_net_seed42_v1_runtime'),ROOT);q.arm='M1_SCALED_DGM';return q


def body(fr,fold,core,fac,records):
    r=next(r for r in fr['cells'] if r['arm']=='C2_INTERACTION' and r['fold']==fold)
    assert sha(r['checkpoint'])==r['checkpoint_sha256'] and sha(r['validation'])==r['validation_sha256']
    state=torch.load(r['checkpoint'],map_location='cpu',weights_only=False);q=read(r['validation'])
    m=previous.make(core,fac,records,fold,'C2_INTERACTION');assert state_hash(m.state_dict())==q['binding']['initial']
    m.load_state_dict(state['model']);m.eval()
    for p in m.parameters():p.requires_grad_(False)
    return m,r,state,q


def make(kind,readout,fold):return MODEL.Decision(kind,readout,42+1009*fold+193003).cuda()


def cached(fold,d):
    path=ROOT/'banks_private'/f'fold{fold}.pkl';seal=jr(ROOT/'REPRESENTATION_SEAL_PRIVATE.json')
    assert sha(path)==seal['banks'][str(fold)];q=read(path)
    assert q['binding']['input']==jr(ROOT/'INPUT_FREEZE_PRIVATE.json')
    assert np.array_equal(q['archive_indices'],d['archive_indices']) and q['hidden'].shape==(len(d['patient']),64)
    return q


def predict(head,h,d,ix):
    head.eval();out=np.empty(len(ix),np.float32)
    with torch.inference_mode():
        for pid in sorted(set(d['patient'][ix])):
            j=np.flatnonzero(d['patient'][ix]==pid);ii=ix[j]
            out[j]=head(torch.as_tensor(h[ii],device='cuda')).cpu().numpy()
    assert np.isfinite(out).all();return out


def choose(frame):
    scores=[]
    for arm,g in frame.groupby('arm'):
        sd=float(g.groupby('fold').macro_f1.mean().std(ddof=1));scores.append(dict(arm=arm,macro=float(g.macro_f1.mean()),auroc=float(g.auroc.mean()),auprc=float(g.auprc.mean()),fold_sd=sd,penalized=float(g.macro_f1.mean()-.1*sd)))
    candidate=next(r for r in scores if r['arm']=='G1_PATIENT');controls=[r for r in scores if r['arm']!='G1_PATIENT']
    eligible=bool(candidate['macro']>=max(r['macro'] for r in controls)+.003 and candidate['auroc']>=max(r['auroc'] for r in controls)-.01 and candidate['auprc']>=max(r['auprc'] for r in controls)-.01)
    return dict(scores=scores,selected='G1_PATIENT' if eligible else 'C2_REFERENCE',new_qualified=eligible,eligible=['G1_PATIENT'] if eligible else [])
