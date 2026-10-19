"""Pinned previous source, legal original cohort, no outcome-dependent selection."""
import csv
import gc
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np
import torch

OWN=sys.modules[__name__]
HERE=Path(__file__).resolve().parent
ROOT=Path(r'D:\m8_6d_no_interaction_ablation_seed42_v1_runtime');PUB=ROOT/'public'
PREVIOUS=Path(r'D:\m8_6d_no_variance_ablation_baselines_seed42_v1')
PREVROOT=Path(r'D:\m8_6d_no_variance_ablation_baselines_seed42_v1_runtime')
FILES=['PROTOCOL_LOCK.json','common.py','model.py','test_model.py','admit.py','train.py','verify_internal.py','evaluate_outer.py','report_outer.py','verify_outer.py','run_all.ps1','package_public.py']

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m

previous=module('ni_previous_common',PREVIOUS/'common.py')
MODEL=module('ni_current_math',HERE/'model.py')
old,io,legacy,EV=previous.old,previous.io,previous.legacy,previous.EV
sha,read,save,write,freeze,jr,digest=io.sha,io.read,io.save,io.write,io.freeze,io.jr,io.digest
atomic_torch,state_hash,seed_all=io.atomic_torch,io.state_hash,io.seed_all
ARMS=MODEL.ARMS;REF_NAMES={'FULL_F0':'WO_HEAD_INTERACTION','RAW_ONLY':'RAW_ONLY','PHYSIOLOGY_ONLY':'PHYSIOLOGY_ONLY'}
NEW=[a for a in ARMS if a not in REF_NAMES];METRICS=previous.METRICS
probability=previous.probability
sys.path.insert(0,str(HERE));sys.modules['common']=OWN;sys.modules['model']=MODEL

def sources():return {n:sha(HERE/n) for n in FILES}

def setup(freeze_input=True):
    ROOT.mkdir(exist_ok=True);PUB.mkdir(exist_ok=True)
    try:prior,core,fac,records,rec,banks=previous.setup(False)
    finally:sys.modules['common']=OWN;sys.modules['model']=MODEL;sys.path.insert(0,str(HERE))
    lock=jr(HERE/'PROTOCOL_LOCK.json');assert lock['arms']==ARMS and len(NEW)==11 and len(ARMS)==14
    assert sha(PREVROOT/'FROZEN_BEFORE_OUTER_PRIVATE.json')==lock['parent_freeze_sha256']
    assert sha(PREVROOT/'OUTER_SCORE_SEAL_PRIVATE.json')==lock['parent_score_seal_sha256']
    priorfreeze=jr(PREVROOT/'FROZEN_BEFORE_OUTER_PRIVATE.json')
    for n,h in priorfreeze['source'].items():assert sha(PREVIOUS/n)==h
    assert jr(PREVROOT/'public/OUTER_STATUS.json')['status']=='COMPLETE_VERIFIED_F0_6D_30_GROUP_FIVE_FOLD_OUTER'
    assert len(rec['meta']['patient'])==7635 and len(set(rec['meta']['patient']))==80
    assert sum(len(q['records']) for q in rec['ledgers'])==256
    binding=dict(source=sources(),prior_input=digest(prior),parent_freeze=lock['parent_freeze_sha256'],parent_seal=lock['parent_score_seal_sha256'],
        reconstruction=io.EXPECTED_REC,raw_registry=io.EXPECTED_REG,torch=torch.__version__,numpy=np.__version__)
    if freeze_input:freeze(ROOT/'INPUT_FREEZE_PRIVATE.json',binding)
    return binding,core,fac,records,rec,banks

def bank(fold,banks,arm):return previous.bank(fold,banks,arm)
def scale(fold):return previous.scale(fold)
def preprocessor(d,ss):return previous.preprocessor(d,ss)

def make(core,fac,records,fold,arm):
    source=old.make(core,fac,records,fold,'F0_AFTER_GELU')
    m=MODEL.Ablation(source,arm).cuda();del source;return m

def reference(arm,fold):
    assert arm in REF_NAMES
    fr=jr(PREVROOT/'FROZEN_BEFORE_OUTER_PRIVATE.json');name=REF_NAMES[arm]
    r=dict(next(q for q in fr['cells'] if q['arm']==name and q['fold']==fold))
    assert sha(r['checkpoint'])==r['checkpoint_sha256'] and sha(r['validation'])==r['validation_sha256']
    state=torch.load(r['checkpoint'],map_location='cpu',weights_only=False);v=read(r['validation'])
    assert state['binding']==v['binding'] and state['threshold']==r['threshold']
    return dict(r,arm=arm,reused=True,reference_arm=name),state,v

def store():
    q=legacy.OLD.P3.Inputs(legacy.OLD.BASELINE,Path(r'D:\dgm_net_seed42_v1_runtime'),ROOT)
    q.arm='M1_SCALED_DGM';q.ablation_maps={};return q

def mapping(inputs,fold,pid,channels,mask,binding):
    key=(fold,str(pid),tuple(channels))
    if key not in inputs.ablation_maps:
        mm,counts=MODEL.channel_mapping(mask,channels,pid,fold)
        b=dict(input=digest(binding),fold=fold,mask=legacy.array_sha(mask.cpu().numpy()),channels=digest(channels))
        path=ROOT/'mismatch_maps_private'/digest(binding)[:16]/f'fold{fold}'/(hashlib.sha256(str(pid).encode()).hexdigest()+'.pkl')
        result=dict(binding=b,mapping=mm.cpu().numpy(),counts=counts)
        if path.exists():
            q=read(path);assert q['binding']==b and np.array_equal(q['mapping'],result['mapping']) and q['counts']==counts
        else:save(path,result)
        inputs.ablation_maps[key]=(mm,counts,sha(path))
    return inputs.ablation_maps[key]

def forward(m,d,ii,inputs,ss,fold,binding):
    x=torch.as_tensor(d['x'][ii],device='cuda')
    if m.branch=='phys':return m(x)['logit'],dict(valid_windows=0,mapping_sha256=None)
    pid=str(d['patient'][ii[0]]);channels=d['channel'][ii].astype(str).tolist()
    raw,mask,times,delta,_=inputs.get(pid,channels,'cuda');perm=None;ph=None;counts={}
    if m.arm=='EARLY_FILM_MISMATCH':perm,counts,ph=mapping(inputs,fold,pid,channels,mask,binding)
    logit=m(x,raw,mask,times,delta,ss,perm)['logit']
    return logit,dict(valid_windows=int(mask.sum()),mapping_sha256=ph,**counts)

def predict(m,d,ix,inputs,ss,fold,binding):
    m.eval();out=np.empty(len(ix),np.float32)
    with torch.inference_mode():
        for pid in sorted(set(d['patient'][ix])):
            j=np.flatnonzero(d['patient'][ix]==pid);ii=ix[j]
            logit,_=forward(m,d,ii,inputs,ss,fold,binding);out[j]=logit.cpu().numpy()
    assert np.isfinite(out).all();return out

def csv_out(name,rows):
    assert rows and all(set(r)==set(rows[0]) for r in rows)
    path=PUB/(name+'.csv');tmp=path.with_suffix('.csv.tmp')
    with tmp.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    tmp.replace(path)

def frozen():
    fr=jr(ROOT/'FROZEN_BEFORE_OUTER_PRIVATE.json')
    assert fr['source']==sources() and fr['input']==jr(ROOT/'INPUT_FREEZE_PRIVATE.json')
    assert len(fr['cells'])==70 and sha(PREVROOT/'OUTER_SCORE_SEAL_PRIVATE.json')==fr['input']['parent_seal']
    assert jr(PUB/'INTERNAL_VERIFICATION.json')['status']=='PASS'
    assert sha(PUB/'INTERNAL_VERIFICATION.json')==fr['internal_verification']
    for r in fr['cells']:
        assert sha(r['checkpoint'])==r['checkpoint_sha256'] and sha(r['validation'])==r['validation_sha256']
    return fr

def vector(rows):return np.array([[r[m] for m in METRICS] for r in rows],dtype=np.float64)
def interval(values,draw,levels=(.025,.975)):
    assert values.shape==(80,9) and np.isfinite(values).all()
    samples=np.concatenate([values[draw[k:k+250]].mean(1) for k in range(0,10000,250)])
    return values.mean(0),np.quantile(samples,levels,axis=0)
