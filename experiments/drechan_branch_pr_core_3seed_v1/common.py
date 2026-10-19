"""Disjoint core-study namespace; read-only admitted parent source/roles/anchors."""
import csv
import gc
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np
import torch

OWN=sys.modules[__name__];HERE=Path(__file__).resolve().parent
ROOT=Path(r'D:\drechan_branch_pr_core_3seed_v1_runtime');PUB=ROOT/'public'
PARENT=Path(r'D:\drechan_branch_pr_residual_3seed_v1')

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

parent=module('core_admitted_parent_common',PARENT/'common.py');component=parent.component;prior=parent.prior
sys.path.insert(0,str(HERE));MODEL=module('core_switches',HERE/'model.py')
io,legacy,EV=parent.io,parent.legacy,parent.EV
sha,read,save,write,freeze,jr,digest=io.sha,io.read,io.save,io.write,io.freeze,io.jr,io.digest
atomic_torch,state_hash,seed_all=io.atomic_torch,io.state_hash,io.seed_all
ARMS=MODEL.ARMS;NEW=ARMS[1:];SEEDS=[42,43,44];METRICS=parent.METRICS;probability=parent.probability
FILES=['PROTOCOL_LOCK.json','model.py','common.py','test_model.py','admit.py','train.py','finish.py','assess.py','verify.py','run_all.ps1','affinity_entry.py','package_public.py']
sys.modules['common']=OWN;sys.modules['model']=MODEL

def sources():return {n:sha(HERE/n) for n in FILES}

def setup(freeze_input=True):
    ROOT.mkdir(exist_ok=True);PUB.mkdir(exist_ok=True)
    try:pb,core,fac,records,rec,banks=parent.setup(False)
    finally:sys.modules['common']=OWN;sys.modules['model']=MODEL;sys.path.insert(0,str(HERE))
    lock=jr(HERE/'PROTOCOL_LOCK.json');assert lock['arms']==ARMS and lock['seeds']==SEEDS and lock['parameters']==MODEL.PARAMETERS
    assert sha(parent.ROOT/'INPUT_FREEZE_PRIVATE.json')==lock['parent_input_sha256'] and pb==jr(parent.ROOT/'INPUT_FREEZE_PRIVATE.json')
    assert sha(parent.PUB/'ADMISSION.json')==lock['parent_admission_sha256'] and jr(parent.PUB/'ADMISSION.json')['status']=='PASS'
    assert jr(parent.PUB/'ADMISSION.json')['source']==parent.sources()
    anchors={f'{f}_{s}':sha(parent.anchor_path(f,s)) for f in range(1,6) for s in SEEDS}
    binding=dict(source=sources(),parent_source=parent.sources(),parent_input=digest(pb),parent_admission=lock['parent_admission_sha256'],parent_anchors=anchors,
                 torch=torch.__version__,numpy=np.__version__,scientific_status='EXPLORATORY_REPEATED_TEST')
    if freeze_input:freeze(ROOT/'INPUT_FREEZE_PRIVATE.json',binding)
    return binding,core,fac,records,rec,banks

def store():
    q=legacy.OLD.P3.Inputs(legacy.OLD.BASELINE,Path(r'D:\dgm_net_seed42_v1_runtime'),ROOT);q.arm='M1_SCALED_DGM';q.core_maps={};return q

def data(fold,core,rec,banks,inputs,binding):
    d,scale=parent.data(fold,core,rec,banks,inputs,jr(parent.ROOT/'INPUT_FREEZE_PRIVATE.json'))
    return dict(d,_core_binding=binding),scale

def make(core,fac,records,fold,arm,seed):
    m=MODEL.configure(parent.make(core,fac,records,fold,'FULL',seed),arm)
    assert sum(p.numel() for p in m.parameters())==MODEL.PARAMETERS[arm]
    assert sum(isinstance(q,torch.nn.Linear) and q.out_features==1 for q in m.modules())==1
    return m

def preprocessor(d,scale,arm,fold,seed):return parent.preprocessor(d,scale,'FULL',fold,seed)

def mapping(inputs,fold,pid,channels,mask,binding):
    key=(fold,str(pid),tuple(channels))
    if key not in inputs.core_maps:
        mm,counts=parent.component.MODEL.BASE.channel_mapping(mask,channels,pid,fold)
        bind=dict(input=digest(binding),fold=fold,mask=legacy.array_sha(mask.cpu().numpy()),channels=digest(channels))
        path=ROOT/'mismatch_maps_private'/f'fold{fold}'/(hashlib.sha256(str(pid).encode()).hexdigest()+'.pkl')
        q=dict(binding=bind,mapping=mm.cpu().numpy(),counts=counts)
        if path.exists():
            old=read(path);assert old['binding']==bind and np.array_equal(old['mapping'],q['mapping']) and old['counts']==counts
        else:save(path,q)
        inputs.core_maps[key]=(mm,counts)
    return inputs.core_maps[key]

def forward(m,d,ii,inputs,scale,collect=False):
    x=torch.as_tensor(d['x'][ii],device='cuda');pid=str(d['patient'][ii[0]]);channels=d['channel'][ii].astype(str).tolist()
    raw,mask,times,delta,_=inputs.get(pid,channels,'cuda');perm=None;stats={}
    if m.core_arm=='EARLY_FILM_MISMATCH':perm,stats=mapping(inputs,d['fold'],pid,channels,mask,d['_core_binding'])
    out=m(x,raw,mask,times,delta,scale,perm)
    return out['logit'],dict(valid_windows=int(mask.sum()),**stats)

def predict(m,d,ix,inputs,scale,collect=False):
    m.eval();out=np.empty(len(ix),np.float32);audit={}
    with torch.no_grad():
        for pid in sorted(set(d['patient'][ix])):
            j=np.flatnonzero(d['patient'][ix]==pid);logit,stats=forward(m,d,ix[j],inputs,scale,collect);out[j]=logit.cpu().numpy()
            for k,v in stats.items():audit[k]=audit.get(k,0)+v
    assert np.isfinite(out).all();return out,audit

def csv_out(name,rows):
    assert rows and all(set(r)==set(rows[0]) for r in rows)
    path=PUB/(name+'.csv');path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.csv.tmp')
    with tmp.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    tmp.replace(path)

def cells_expected():return {(a,s,f) for a in ARMS for s in SEEDS for f in range(1,6)}
def vector(rows):return np.array([[r[m] for m in METRICS] for r in rows],dtype=np.float64)

def frozen():
    q=jr(ROOT/'FROZEN_BEFORE_ASSESS_PRIVATE.json');assert q['source']==sources() and q['input']==jr(ROOT/'INPUT_FREEZE_PRIVATE.json')
    assert {(r['arm'],r['seed'],r['fold']) for r in q['cells']}==cells_expected()
    for r in q['cells']:assert sha(r['checkpoint'])==r['checkpoint_sha256'] and sha(r['calibration'])==r['calibration_sha256']
    assert sha(parent.ROOT/'FROZEN_BEFORE_ASSESS_PRIVATE.json')==q['parent_freeze'] and sha(parent.ROOT/'SCORE_SEAL_PRIVATE.json')==q['parent_seal']
    assert sha(parent.PUB/'QUALIFICATION_GATE.json')==q['parent_gate']
    return q

CONTRASTS={
 'EARLY_FILM':{'FULL':1,'WO_EARLY_FILM':-1},'LATE_GATE':{'FULL':1,'WO_LATE_GATE':-1},
 'BOTH_GUIDANCE':{'FULL':1,'WO_BOTH_GUIDANCE':-1},'PRE_REFERENCE':{'FULL':1,'WO_RAW_PRE_REFERENCE':-1},
 'CORRECT_EARLY_CONTEXT':{'FULL':1,'EARLY_FILM_MISMATCH':-1},
 'GUIDANCE_INTERACTION':{'FULL':1,'WO_EARLY_FILM':-1,'WO_LATE_GATE':-1,'WO_BOTH_GUIDANCE':1}}
PRIMARY_CONTRASTS=['EARLY_FILM','LATE_GATE','PRE_REFERENCE','CORRECT_EARLY_CONTEXT'];PRIMARY_METRICS=['auroc','macro_f1']
