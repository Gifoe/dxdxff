"""New candidate namespace; exact old roles and TRAIN-only scale preconditioning."""
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
ROOT=Path(r'D:\drechan_pruned_branch_fusion_3seed_v1_runtime');PUB=ROOT/'public'
COMPONENT=Path(r'D:\drechan_pr_component_factorial_3seed_v1')

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

component=module('pbf_completed_component_common',COMPONENT/'common.py')
prior=component.prior
sys.path.insert(0,str(HERE));MODEL=module('pbf_candidate_model',HERE/'model.py')
io,legacy,EV=component.io,component.legacy,component.EV
sha,read,save,write,freeze,jr,digest=io.sha,io.read,io.save,io.write,io.freeze,io.jr,io.digest
atomic_torch,state_hash,seed_all=io.atomic_torch,io.state_hash,io.seed_all
ARMS=MODEL.ARMS;SEEDS=[42,43,44];METRICS=component.METRICS;probability=component.probability
FILES=['PROTOCOL_LOCK.json','fusion_math.py','model.py','common.py','test_model.py','admit.py','train.py','assess.py','verify.py','run_all.ps1','affinity_entry.py','package_public.py']
sys.modules['common']=OWN;sys.modules['model']=MODEL

def sources():return {n:sha(HERE/n) for n in FILES}

def setup(freeze_input=True):
    ROOT.mkdir(exist_ok=True);PUB.mkdir(exist_ok=True)
    try:previous,core,fac,records,rec,banks=component.setup(False)
    finally:sys.modules['common']=OWN;sys.modules['model']=MODEL;sys.path.insert(0,str(HERE))
    lock=jr(HERE/'PROTOCOL_LOCK.json');assert lock['arms']==ARMS and lock['seeds']==SEEDS and lock['parameters']==MODEL.PARAMETERS
    assert jr(component.PUB/'INDEPENDENT_VERIFICATION.json')['status']=='PASS'
    assert sha(component.ROOT/'FROZEN_BEFORE_ASSESS_PRIVATE.json')==lock['component_freeze_sha256']
    assert sha(component.ROOT/'SCORE_SEAL_PRIVATE.json')==lock['component_score_seal_sha256']
    assert previous==jr(component.ROOT/'INPUT_FREEZE_PRIVATE.json')
    binding=dict(source=sources(),component_input=digest(previous),component_freeze=lock['component_freeze_sha256'],component_seal=lock['component_score_seal_sha256'],
                 torch=torch.__version__,numpy=np.__version__,scientific_status='EXPLORATORY_REPEATED_TEST')
    if freeze_input:freeze(ROOT/'INPUT_FREEZE_PRIVATE.json',binding)
    return binding,core,fac,records,rec,banks

def store():
    q=legacy.OLD.P3.Inputs(legacy.OLD.BASELINE,Path(r'D:\dgm_net_seed42_v1_runtime'),ROOT);q.arm='M1_SCALED_DGM';return q

def data(fold,core,rec,banks,inputs,binding):
    previous=jr(component.ROOT/'INPUT_FREEZE_PRIVATE.json')
    path=component.ROOT/'banks_private'/digest(previous)[:16]/f'fold{fold}.pkl';assert path.exists()
    q=read(path);assert q['binding']==previous;d=q['bank']
    original=jr(component.prior.ROOT/'INPUT_FREEZE_PRIVATE.json')
    base=read(component.prior.ROOT/'banks_private'/digest(original)[:16]/'internal'/f'fold{fold}.pkl')
    assert base['binding']==original
    roles=component.prior.roles_for(component.prior.parent.bank(fold,banks,'FULL_F0'),fold,'internal')
    assert roles==d['roles'] and d['roles']==base['bank']['roles']
    assert set(d['pre']['fit_patient_ids'])==set(d['pre_no_pr']['fit_patient_ids'])==roles['train']
    assert set(d['patient']).isdisjoint(roles['test'])
    return d,torch.tensor(base['scale'],device='cuda')

def initialize(core,fac,records,fold,arm,seed):
    base=component.make(core,fac,records,fold,'FULL',seed)
    if arm=='B0':return base
    m=MODEL.PrunedFusion(base,arm).cuda();del base
    assert sum(p.numel() for p in m.parameters())==9329
    assert sum(isinstance(q,torch.nn.Linear) and q.out_features==1 for q in m.modules())==1
    return m

def anchor_path(fold,seed):return ROOT/'anchors_private'/f'fold{fold}'/f'seed{seed}.pkl'

def fit_anchor(fold,seed,d,scale,inputs,binding,core,fac,records):
    m=initialize(core,fac,records,fold,'FULL',seed);initial=state_hash(m.state_dict())
    bind=dict(input=digest(binding),fold=fold,seed=seed,initial=initial,train_patients=digest(sorted(d['roles']['train'])),
              train_indices=legacy.array_sha(d['archive_indices'][d['train']]))
    path=anchor_path(fold,seed)
    if path.exists():
        q=read(path);assert q['binding']==bind
    else:
        arrays=[];windows=0;m.eval();m.raw_pr=False;m.collect_anchor=True
        with torch.no_grad():
            for pid in sorted(d['roles']['train']):
                ii=d['train'][d['patient'][d['train']]==pid]
                raw,mask,times,delta,_=inputs.get(str(pid),d['channel'][ii].astype(str).tolist(),'cuda')
                vector,_=m.features(raw,mask,times,delta,scale)
                arrays.append(vector.detach().double().cpu());windows+=int(mask.sum());del raw,delta,vector
        x=torch.cat(arrays);mean=x.mean(0);sd=x.std(0,unbiased=False)
        scale_out=torch.where(sd>1e-8,sd,torch.ones_like(sd))
        q=dict(binding=bind,mean=mean,scale=scale_out,channels=len(x),patients=len(d['roles']['train']),valid_windows=windows,
               constant_coordinates=int((sd<=1e-8).sum()),initial_only_not_current_reference=True)
        save(path,q)
    assert q['mean'].shape==q['scale'].shape==(96,) and torch.isfinite(q['mean']).all() and torch.isfinite(q['scale']).all()
    assert q['channels']==len(d['train']) and q['patients']==len(d['roles']['train'])
    del m;gc.collect();torch.cuda.empty_cache();return q

def make(core,fac,records,fold,arm,seed):
    m=initialize(core,fac,records,fold,arm,seed)
    if arm!='B0':m.set_anchor(read(anchor_path(fold,seed)))
    return m

def preprocessor(d,scale,arm,fold,seed):
    feature_pr=arm not in ['WO_FEATURE_PR','WO_BOTH_PR']
    p=component.preprocessor(d,scale,'FULL' if feature_pr else 'WO_FEATURE_PR')
    return dict(summary=p,initial_TRAIN_raw_anchor_sha256=sha(anchor_path(fold,seed)) if arm!='B0' else None)

def forward(m,d,ii,inputs,scale,collect=False):
    x=torch.as_tensor((d['x'] if m.feature_pr else d['x_no_pr'])[ii],device='cuda')
    pid=str(d['patient'][ii[0]]);raw,mask,times,delta,_=inputs.get(pid,d['channel'][ii].astype(str).tolist(),'cuda')
    out=m(x,raw,mask,times,delta,scale);stats=dict(valid_windows=int(mask.sum()))
    if collect:
        z=out['raw_summary'].detach().double()
        stats.update(raw_summary_coordinates=z.numel(),raw_summary_sum_abs=float(z.abs().sum()),raw_summary_sum_square=float(z.square().sum()))
    return out['logit'],stats

def predict(m,d,ix,inputs,scale,collect=False):
    m.eval();out=np.empty(len(ix),np.float32);audit={}
    with torch.no_grad():
        for pid in sorted(set(d['patient'][ix])):
            j=np.flatnonzero(d['patient'][ix]==pid);logit,stats=forward(m,d,ix[j],inputs,scale,collect);out[j]=logit.cpu().numpy()
            for k,v in stats.items():audit[k]=audit.get(k,0)+v
    assert np.isfinite(out).all();return out,audit

def previous_cell(seed,fold):return component.previous_cell(seed,fold)

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
    return q

CONTRASTS={
 'FULL_MINUS_B0':{'FULL':1,'B0':-1},
 'PRUNING_EFFECT':{'JOINT_PRUNED':1,'B0':-1},
 'FUSION_EFFECT':{'FULL':1,'JOINT_PRUNED':-1},
 'FEATURE_PR_WITH_RAW':{'FULL':1,'WO_FEATURE_PR':-1},
 'RAW_PR_WITH_FEATURE':{'FULL':1,'WO_RAW_PR':-1},
 'BOTH_PR_JOINT':{'FULL':1,'WO_BOTH_PR':-1},
 'PRE_REFERENCE':{'FULL':1,'WO_PRE_REFERENCE':-1},
 'PR_INTERACTION':{'FULL':1,'WO_FEATURE_PR':-1,'WO_RAW_PR':-1,'WO_BOTH_PR':1}}
PRIMARY_CONTRASTS=['FULL_MINUS_B0','FUSION_EFFECT','FEATURE_PR_WITH_RAW','RAW_PR_WITH_FEATURE','PRE_REFERENCE']
PRIMARY_METRICS=['auroc','macro_f1']
