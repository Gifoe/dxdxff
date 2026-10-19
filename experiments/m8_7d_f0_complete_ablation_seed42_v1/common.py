"""Frozen original inputs, exact reference reuse, no outcome-based arm selection."""
import csv
import gc
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np
import scipy
import sklearn
import torch

HERE=Path(__file__).resolve().parent;ROOT=Path(r'D:\m8_7d_f0_complete_ablation_seed42_v1_runtime');PUB=ROOT/'public'
FILM=Path(r'D:\m8_7d_raw_film_seed42_v1');FILMROOT=Path(r'D:\m8_7d_raw_film_seed42_v1_runtime')
OLDROOT=Path(r'D:\m8_7d_frozen_candidate_outer_audit_seed42_v1_runtime')
NOPR=Path(r'D:\d1_m8_progressive_ablation_seed42_v1_runtime\A1')
FILES=['PROTOCOL_LOCK.json','model.py','common.py','test_model.py','admit.py','train.py','verify_internal.py','evaluate_outer.py','report_outer.py','verify_outer.py','run_all.ps1']

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

old=module('complete_ablation_film',FILM/'common.py');MODEL=module('complete_ablation_model',HERE/'model.py')
io,legacy,EV=old.io,old.legacy,old.EV
sha,read,save,write,freeze,jr,digest=io.sha,io.read,io.save,io.write,io.freeze,io.jr,io.digest
atomic_torch,state_hash,seed_all=io.atomic_torch,io.state_hash,io.seed_all
ARMS=MODEL.ARMS;NEW=ARMS[2:];METRICS=old.METRICS
REF_NAMES={'FULL_F0':'F0_AFTER_GELU','WO_EARLY_FILM':'C2_REFERENCE'}
probability=old.probability;patient_rows=old.patient_rows
sys.path.insert(0,str(HERE));sys.modules['common']=sys.modules[__name__];sys.modules['model']=MODEL

def sources():return {n:sha(HERE/n) for n in FILES}

def setup(freeze_input=True):
    ROOT.mkdir(exist_ok=True);PUB.mkdir(exist_ok=True);own=sys.modules[__name__]
    try:prior,core,fac,records,rec,banks,_=old.setup()
    finally:sys.modules['common']=own;sys.modules['model']=MODEL;sys.path.insert(0,str(HERE))
    assert jr(HERE/'PROTOCOL_LOCK.json')['arms']==ARMS and len(NEW)==16
    assert sha(FILMROOT/'MODEL_FREEZE_PRIVATE.json')=='83ffe0516ce3f7ea8b13448b498948f5b04b3a630ad538a0037134b1243721f7'
    assert sha(OLDROOT/'FROZEN_BEFORE_OUTER_PRIVATE.json')=='f8205e35ca983e49c69265e5c07921d0b9e10eff46c7b89d524bb41f43a81388'
    assert sha(OLDROOT/'OUTER_SCORE_SEAL_PRIVATE.json')=='b345ac3a2aa934c97782280275a149737d96d870b5e54eadd0b629f77ef44bf6'
    assert jr(FILMROOT/'public/INDEPENDENT_INTERNAL_VERIFICATION.json')['status']=='PASS'
    fr=jr(FILMROOT/'MODEL_FREEZE_PRIVATE.json')
    for n,h in fr['input']['source'].items():assert sha(FILM/n)==h
    assert sum(len(q['records']) for q in rec['ledgers'])==256
    no_pr={str(f):sha(NOPR/f'fold{f}/BANK_PRIVATE.pt') for f in range(1,6)}
    binding=dict(source=sources(),prior_input=digest(prior),film_model_freeze=sha(FILMROOT/'MODEL_FREEZE_PRIVATE.json'),
        old_outer_freeze=sha(OLDROOT/'FROZEN_BEFORE_OUTER_PRIVATE.json'),old_outer_seal=sha(OLDROOT/'OUTER_SCORE_SEAL_PRIVATE.json'),
        no_pr_banks=no_pr,reconstruction=io.EXPECTED_REC,raw_registry=io.EXPECTED_REG,torch=torch.__version__,numpy=np.__version__)
    if freeze_input:freeze(ROOT/'INPUT_FREEZE_PRIVATE.json',binding)
    return binding,core,fac,records,rec,banks

def bank(fold,banks,arm):
    d=old.bank(fold,banks)
    if arm=='WO_FEATURE_PR':
        q=torch.load(NOPR/f'fold{fold}/BANK_PRIVATE.pt',weights_only=False,map_location='cpu')
        for k in ['patient','channel','y','train','val']:assert np.array_equal(q[k],d[k])
        assert q['pre']['patient_z'] is False;d=dict(q,archive_indices=d['archive_indices'],roles=d['roles'])
    assert set(d['pre']['fit_patient_ids'])==d['roles']['fit'] and set(d['patient']).isdisjoint(d['roles']['test'])
    assert d['x'].shape[1]==14 and np.isfinite(d['x']).all()
    return d

def scale(fold):return old.scale(fold)
def make(core,fac,records,fold,arm):
    source=old.make(core,fac,records,fold,'F0_AFTER_GELU')
    m=MODEL.Ablation(source,arm).cuda();del source;return m

def reference(arm,fold):
    frozen=jr(OLDROOT/'FROZEN_BEFORE_OUTER_PRIVATE.json');name=REF_NAMES[arm]
    r=dict(next(r for r in frozen['models'] if r['arm']==name and r['fold']==fold))
    assert sha(r['checkpoint'])==r['checkpoint_sha256'] and sha(r['validation'])==r['validation_sha256']
    state=torch.load(r['checkpoint'],map_location='cpu',weights_only=False);v=read(r['validation'])
    assert state['binding']==v['binding'] and state['threshold']==r['threshold']
    return dict(r,arm=arm,reused=True,reference_arm=name),state,v

def store():
    q=legacy.OLD.P3.Inputs(legacy.OLD.BASELINE,Path(r'D:\dgm_net_seed42_v1_runtime'),ROOT);q.arm='M1_SCALED_DGM';q.ablation_maps={};return q

def mapping(inputs,fold,pid,channels,mask,binding):
    key=(fold,str(pid),tuple(channels))
    if key not in inputs.ablation_maps:
        mm,counts=MODEL.channel_mapping(mask,channels,pid,fold)
        b=dict(input=digest(binding),fold=fold,mask=legacy.array_sha(mask.cpu().numpy()),channels=digest(channels))
        path=ROOT/'mismatch_maps_private'/digest(binding)[:16]/f'fold{fold}'/(hashlib.sha256(str(pid).encode()).hexdigest()+'.pkl')
        result=dict(binding=b,mapping=mm.cpu().numpy(),counts=counts)
        if path.exists():
            oldq=read(path);assert oldq['binding']==b and np.array_equal(oldq['mapping'],result['mapping']) and oldq['counts']==counts
        else:save(path,result)
        inputs.ablation_maps[key]=(mm,counts,sha(path))
    return inputs.ablation_maps[key]

def forward(m,d,ii,inputs,ss,fold,binding):
    x=torch.as_tensor(d['x'][ii],device='cuda')
    if m.branch=='phys':return m(x)['logit'],dict(valid_windows=0,mapping_sha256=None)
    pid=str(d['patient'][ii[0]]);channels=d['channel'][ii].astype(str).tolist()
    raw,mask,times,delta,_=inputs.get(pid,channels,'cuda');perm=None;ph=None;counts={}
    if m.arm=='EARLY_FILM_MISMATCH':perm,counts,ph=mapping(inputs,fold,pid,channels,mask,binding)
    out=m(x,raw,mask,times,delta,ss,perm)['logit']
    return out,dict(valid_windows=int(mask.sum()),mapping_sha256=ph,**counts)

def predict(m,d,ix,inputs,ss,fold,binding):
    m.eval();out=np.empty(len(ix),np.float32)
    with torch.inference_mode():
        for pid in sorted(set(d['patient'][ix])):
            j=np.flatnonzero(d['patient'][ix]==pid);ii=ix[j];logit,_=forward(m,d,ii,inputs,ss,fold,binding);out[j]=logit.cpu().numpy()
    assert np.isfinite(out).all();return out

def preprocessor(d,ss):
    return dict(pre={k:legacy.array_sha(np.asarray(v)) for k,v in d['pre'].items() if k!='fit_patient_ids'},
        fit_patients=digest(sorted(d['pre']['fit_patient_ids'])),scale=legacy.array_sha(ss.cpu().numpy()))

def csv_out(name,rows):
    assert rows and all(set(r)==set(rows[0]) for r in rows)
    path=PUB/(name+'.csv');tmp=path.with_suffix('.csv.tmp')
    with tmp.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    tmp.replace(path)

def frozen():
    fr=jr(ROOT/'FROZEN_BEFORE_OUTER_PRIVATE.json');assert fr['source']==sources() and fr['input']==jr(ROOT/'INPUT_FREEZE_PRIVATE.json')
    assert len(fr['cells'])==90 and sha(OLDROOT/'OUTER_SCORE_SEAL_PRIVATE.json')==fr['input']['old_outer_seal']
    assert jr(PUB/'INTERNAL_VERIFICATION.json')['status']=='PASS' and sha(PUB/'INTERNAL_VERIFICATION.json')==fr['internal_verification']
    for r in fr['cells']:
        assert sha(r['checkpoint'])==r['checkpoint_sha256'] and sha(r['validation'])==r['validation_sha256']
    return fr

def vector(rows):return np.array([[r[m] for m in METRICS] for r in rows],dtype=np.float64)
def interval(values,draw,levels=(.025,.975)):
    assert values.shape==(80,9) and np.isfinite(values).all()
    samples=np.concatenate([values[draw[k:k+250]].mean(1) for k in range(0,10000,250)])
    return values.mean(0),np.quantile(samples,levels,axis=0)
