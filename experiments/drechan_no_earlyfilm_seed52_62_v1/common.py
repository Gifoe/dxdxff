"""Frozen parent roles/cache/math; fresh initialization at seeds52 and62 only."""
import csv
import gc
import hashlib
import importlib.util
from pathlib import Path
import sys
import numpy as np
import torch
OWN=sys.modules[__name__];HERE=Path(__file__).resolve().parent
ROOT=Path(r'D:\drechan_no_earlyfilm_seed52_62_v1_runtime');PUB=ROOT/'public'
PARENT=Path(r'D:\drechan_f0_no_lategate_ablation_seed42_v1')
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);sys.modules[name]=value;spec.loader.exec_module(value)
    return value
parent=module('noearly5262_parent_common',PARENT/'common.py')
original=parent.parent
MODEL=module('noearly5262_model',HERE/'model.py')
io,legacy,EV=parent.io,parent.legacy,parent.EV
sha,read,save,write,freeze,jr,digest=io.sha,io.read,io.save,io.write,io.freeze,io.jr,io.digest
atomic_torch,state_hash,seed_all=io.atomic_torch,io.state_hash,io.seed_all
probability,METRICS=parent.probability,parent.METRICS
ARMS,SEEDS=MODEL.ARMS,[52,62]
FILES=['PROTOCOL_LOCK.json','model.py','common.py','test_model.py','admit.py','train.py','verify_internal.py',
       'evaluate_outer.py','report_outer.py','verify_outer.py','run_all.ps1','affinity_entry.py','package_public.py']
CONTEXT={}
sys.modules['common']=OWN;sys.modules['model']=MODEL
def sources():return {n:sha(HERE/n) for n in FILES}
def setup(freeze_input=True):
    ROOT.mkdir(exist_ok=True);PUB.mkdir(exist_ok=True)
    try:previous,core,fac,records,rec,banks=parent.setup(False)
    finally:
        sys.modules['common']=OWN;sys.modules['model']=MODEL;sys.path.insert(0,str(HERE))
    lock=jr(HERE/'PROTOCOL_LOCK.json')
    assert lock['arms']==ARMS and lock['seeds']==SEEDS and lock['parameters']==MODEL.PARAMETERS
    assert parent.sources()==lock['parent_source']
    assert sha(parent.ROOT/'FROZEN_BEFORE_OUTER_PRIVATE.json')==lock['parent_freeze_sha256']
    assert sha(parent.ROOT/'OUTER_SCORE_SEAL_PRIVATE.json')==lock['parent_score_seal_sha256']
    assert jr(parent.PUB/'INDEPENDENT_OUTER_VERIFICATION.json')['status']=='PASS'
    assert rec['R14'].shape==(7635,14) and len(set(rec['meta']['patient']))==80
    assert sum(len(r['records']) for r in rec['ledgers'])==256
    binding=dict(source=sources(),parent_source=parent.sources(),parent_input=digest(previous),
        parent_freeze=lock['parent_freeze_sha256'],parent_seal=lock['parent_score_seal_sha256'],
        reconstruction=io.EXPECTED_REC,raw_registry=io.EXPECTED_REG,torch=torch.__version__,numpy=np.__version__,
        scientific_status='EXPLORATORY_REPEATED_TEST')
    if freeze_input:freeze(ROOT/'INPUT_FREEZE_PRIVATE.json',binding)
    CONTEXT.update(binding=binding,rec=rec)
    return binding,core,fac,records,rec,banks
def bank(fold,banks,arm):
    assert arm in ARMS and CONTEXT
    return parent.bank(fold,banks,'WO_EARLY_FILM')
def query_features(raw,patients,pre,arm,core):
    assert arm in ARMS
    return parent.query_features(raw,patients,pre,'WO_EARLY_FILM',core)
scale=parent.scale
preprocessor=parent.preprocessor
def source_for(core,fac,records,fold,seed):
    assert seed in SEEDS and 1<=fold<=5
    s=seed+1009*fold
    core.seed_all(s)
    template=legacy.OLD.P3.MorphCorrection(legacy.OLD.CompactDelta(fac(s).network,records)).cuda()
    source=original.legacy_parent.old.MODEL.SingleHead('F0_AFTER_GELU',template.cnn,template.gate,
                                                       type(template).features,s+170003).cuda()
    del template
    return source
def make(core,fac,records,fold,arm,seed):
    source=source_for(core,fac,records,fold,seed);m=MODEL.NoEarlyF0(source,arm).cuda();del source
    return m
def reference(arm,seed,fold,optional=False):
    assert arm in ARMS and seed in SEEDS
    return None
def store():
    result=legacy.OLD.P3.Inputs(legacy.OLD.BASELINE,Path(r'D:\dgm_net_seed42_v1_runtime'),ROOT)
    result.arm='M1_SCALED_DGM'
    return result
def forward(model,d,ix,inputs,ss,fold,binding):
    assert model.arm=='WO_EARLY_FILM' and not model.early and not model.late
    return parent.forward(model,d,ix,inputs,ss,fold,binding)
def predict(model,d,ix,inputs,ss,fold,binding):
    model.eval();result=np.empty(len(ix),np.float32)
    with torch.inference_mode():
        for pid in sorted(set(d['patient'][ix])):
            jj=np.flatnonzero(d['patient'][ix]==pid)
            logits,_=forward(model,d,ix[jj],inputs,ss,fold,binding);result[jj]=logits.cpu().numpy()
    assert np.isfinite(result).all()
    return result
def expected_cells():return {(a,s,f) for a in ARMS for s in SEEDS for f in range(1,6)}
def key_for(arm,seed,fold):return f'{arm}_seed{seed}_fold{fold}'
def csv_out(name,rows):
    assert rows and all(set(r)==set(rows[0]) for r in rows)
    p=PUB/(name+'.csv');tmp=p.with_suffix('.csv.tmp')
    with tmp.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    tmp.replace(p)
def frozen():
    q=jr(ROOT/'FROZEN_BEFORE_OUTER_PRIVATE.json')
    assert q['source']==sources() and q['input']==jr(ROOT/'INPUT_FREEZE_PRIVATE.json')
    assert {(r['arm'],r['seed'],r['fold']) for r in q['cells']}==expected_cells()
    assert jr(PUB/'INTERNAL_VERIFICATION.json')['status']=='PASS'
    assert sha(PUB/'INTERNAL_VERIFICATION.json')==q['internal_verification']
    for r in q['cells']:
        assert not r['reused']
        assert sha(r['checkpoint'])==r['checkpoint_sha256'] and sha(r['validation'])==r['validation_sha256']
    return q
def vector(rows):return np.asarray([[r[m] for m in METRICS] for r in rows],np.float64)
def interval(values,draw,levels=(.025,.975)):
    assert values.shape==(80,9) and np.isfinite(values).all()
    samples=np.concatenate([values[draw[k:k+250]].mean(1) for k in range(0,10000,250)])
    return values.mean(0),np.quantile(samples,levels,axis=0)
