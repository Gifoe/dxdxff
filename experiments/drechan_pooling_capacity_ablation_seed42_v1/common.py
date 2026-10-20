"""Reuse frozen no-Late roles/preprocessing/IO without mutating parent sources."""
import csv
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import torch
OWN=sys.modules[__name__];HERE=Path(__file__).resolve().parent
ROOT=Path(r'D:\drechan_pooling_capacity_ablation_seed42_v1_runtime');PUB=ROOT/'public'
PARENT=Path(r'D:\drechan_f0_no_lategate_ablation_seed42_v1')
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec)
    sys.modules[name]=value;spec.loader.exec_module(value);return value
parent=module('poolcap_parent_common',PARENT/'common.py')
sys.path.insert(0,str(HERE));MODEL=module('poolcap_model',HERE/'model.py')
io,legacy,EV=parent.io,parent.legacy,parent.EV
sha,read,save,write,freeze,jr,digest=io.sha,io.read,io.save,io.write,io.freeze,io.jr,io.digest
atomic_torch,state_hash,seed_all=io.atomic_torch,io.state_hash,io.seed_all
probability,METRICS=parent.probability,parent.METRICS
ARMS,SEEDS=MODEL.ARMS,[42]
FILES=['PROTOCOL_LOCK.json','model.py','common.py','test_model.py','admit.py','train.py','verify_internal.py',
       'evaluate_outer.py','report_outer.py','verify_outer.py','run_all.ps1','affinity_entry.py','package_public.py']
CONTEXT={};sys.modules['common']=OWN;sys.modules['model']=MODEL
def sources():return {n:sha(HERE/n) for n in FILES}
def setup(freeze_input=True):
    ROOT.mkdir(exist_ok=True);PUB.mkdir(exist_ok=True)
    try:previous,core,fac,records,rec,banks=parent.setup(False)
    finally:sys.modules['common']=OWN;sys.modules['model']=MODEL;sys.path.insert(0,str(HERE))
    lock=jr(HERE/'PROTOCOL_LOCK.json')
    assert lock['arms']==ARMS and lock['parameters']==MODEL.PARAMETERS and lock['seeds']==SEEDS
    assert sha(parent.ROOT/'FROZEN_BEFORE_OUTER_PRIVATE.json')==lock['parent_freeze_sha256']
    assert sha(parent.ROOT/'OUTER_SCORE_SEAL_PRIVATE.json')==lock['parent_score_seal_sha256']
    assert jr(parent.PUB/'INDEPENDENT_OUTER_VERIFICATION.json')['status']=='PASS'
    previous_fr=parent.frozen()
    binding=dict(source=sources(),parent_source=parent.sources(),parent_input=digest(previous),
        parent_freeze=lock['parent_freeze_sha256'],parent_seal=lock['parent_score_seal_sha256'],
        reconstruction=io.EXPECTED_REC,raw_registry=io.EXPECTED_REG,torch=torch.__version__,numpy=np.__version__,
        scientific_status='EXPLORATORY_REPEATED_TEST')
    if freeze_input:freeze(ROOT/'INPUT_FREEZE_PRIVATE.json',binding)
    CONTEXT.update(binding=binding,rec=rec,parent_freeze=previous_fr)
    return binding,core,fac,records,rec,banks
def input_arm(arm):return 'WO_EARLY_FILM' if arm in ['WO_EARLY_FILM','A2_CAPACITY_MATCHED'] else 'FULL'
def bank(fold,banks,arm):return parent.bank(fold,banks,input_arm(arm))
def query_features(raw,patients,pre,arm,core):return parent.query_features(raw,patients,pre,input_arm(arm),core)
scale,preprocessor,source_for=parent.scale,parent.preprocessor,parent.source_for
def make(core,fac,records,fold,arm,seed):
    source=source_for(core,fac,records,fold,seed)
    m=MODEL.PoolingCapacity(source,arm,seed+1009*fold+170003).cuda();del source;return m
def reference(arm,seed,fold,optional=False):
    assert seed==42
    if arm not in MODEL.REUSE:return None
    original=MODEL.REUSE[arm]
    r=next(r for r in CONTEXT['parent_freeze']['cells'] if r['arm']==original and r['seed']==seed and r['fold']==fold)
    assert sha(r['checkpoint'])==r['checkpoint_sha256'] and sha(r['validation'])==r['validation_sha256']
    state=torch.load(r['checkpoint'],map_location='cpu',weights_only=False);v=read(r['validation'])
    assert state['binding']==v['binding']
    return dict(r,arm=arm,seed=seed,epochs=len(v['history']),reused=True,reuse_kind='STRICT_NO_LATE_PARENT42',
                source_arm=original,parameters=MODEL.PARAMETERS[arm]),state,v
def reference_score(r):
    path=parent.ROOT/'scores_private'/(parent.key_for(r['source_arm'],42,r['fold'])+'.pkl')
    assert sha(path)==jr(parent.ROOT/'OUTER_SCORE_SEAL_PRIVATE.json')['files'][parent.key_for(r['source_arm'],42,r['fold'])]
    q=read(path);assert q['checkpoint']==r['checkpoint_sha256'] and q['threshold']==r['threshold']
    return q,sha(path),parent.ROOT
def store():
    result=legacy.OLD.P3.Inputs(legacy.OLD.BASELINE,Path(r'D:\dgm_net_seed42_v1_runtime'),ROOT)
    result.arm='M1_SCALED_DGM';return result
def forward(model,d,ix,inputs,ss,fold,binding):return parent.forward(model,d,ix,inputs,ss,fold,binding)
def predict(model,d,ix,inputs,ss,fold,binding):return parent.predict(model,d,ix,inputs,ss,fold,binding)
def expected_cells():return {(a,s,f) for a in ARMS for s in SEEDS for f in range(1,6)}
key_for,csv_out,vector,interval=parent.key_for,parent.csv_out,parent.vector,parent.interval
# Parent csv_out's globals point at its public directory: never use it to write.
def csv_out(name,rows):
    assert rows and all(set(r)==set(rows[0]) for r in rows)
    path=PUB/(name+'.csv');temp=path.with_suffix('.csv.tmp')
    with temp.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    temp.replace(path)
def frozen():
    value=jr(ROOT/'FROZEN_BEFORE_OUTER_PRIVATE.json')
    assert value['source']==sources() and value['input']==jr(ROOT/'INPUT_FREEZE_PRIVATE.json')
    assert {(r['arm'],r['seed'],r['fold']) for r in value['cells']}==expected_cells()
    assert jr(PUB/'INTERNAL_VERIFICATION.json')['status']=='PASS'
    assert sha(PUB/'INTERNAL_VERIFICATION.json')==value['internal_verification']
    for r in value['cells']:
        assert sha(r['checkpoint'])==r['checkpoint_sha256'] and sha(r['validation'])==r['validation_sha256']
    return value
PAIRS={'FULL_MINUS_'+a:{'FULL':1,a:-1} for a in ARMS if a!='FULL'}
PAIRS['A2_MINUS_UNMATCHED_NO_EARLY']={'A2_CAPACITY_MATCHED':1,'WO_EARLY_FILM':-1}
INTERACTIONS={}
