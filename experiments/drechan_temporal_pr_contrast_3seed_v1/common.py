"""Original contracts, honest new within-FIT roles and three matched initializers."""
import csv
import gc
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
import numpy as np
import torch

OWN=sys.modules[__name__];HERE=Path(__file__).resolve().parent
ROOT=Path(r'D:\drechan_temporal_pr_contrast_3seed_v1_runtime');PUB=ROOT/'public'
PARENT=Path(r'D:\m8_6d_no_interaction_ablation_seed42_v1');PARENTROOT=Path(r'D:\m8_6d_no_interaction_ablation_seed42_v1_runtime')
FILES=['PROTOCOL_LOCK.json','contrast.py','model.py','common.py','test_contrast.py','test_model.py','admit.py','train.py','assess.py','verify.py','final_outer.py','run_all.ps1','package_public.py']

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

parent=module('tprc_original_common',PARENT/'common.py')
sys.path.insert(0,str(HERE));MODEL=module('tprc_current_model',HERE/'model.py')
old,io,legacy,EV=parent.old,parent.io,parent.legacy,parent.EV
sha,read,save,write,freeze,jr,digest=io.sha,io.read,io.save,io.write,io.freeze,io.jr,io.digest
atomic_torch,state_hash,seed_all=io.atomic_torch,io.state_hash,io.seed_all
ARMS=MODEL.ARMS;SEEDS=[42,43,44];METRICS=parent.METRICS;PRIMARY=['auroc','auprc','macro_f1','f1'];probability=parent.probability
sys.modules['common']=OWN;sys.modules['model']=MODEL

def sources():return {n:sha(HERE/n) for n in FILES}

def setup(freeze_input=True):
    ROOT.mkdir(exist_ok=True);PUB.mkdir(exist_ok=True)
    try:prior,core,fac,records,rec,banks=parent.setup(False)
    finally:sys.modules['common']=OWN;sys.modules['model']=MODEL;sys.path.insert(0,str(HERE))
    lock=jr(HERE/'PROTOCOL_LOCK.json');assert lock['arms']==ARMS and lock['seeds']==SEEDS
    assert sha(PARENTROOT/'FROZEN_BEFORE_OUTER_PRIVATE.json')==lock['parent_freeze_sha256']
    assert sha(PARENTROOT/'OUTER_SCORE_SEAL_PRIVATE.json')==lock['parent_score_seal_sha256']
    original=jr(PARENTROOT/'FROZEN_BEFORE_OUTER_PRIVATE.json')
    for n,h in original['source'].items():assert sha(PARENT/n)==h
    binding=dict(source=sources(),prior_input=digest(prior),parent_freeze=lock['parent_freeze_sha256'],parent_seal=lock['parent_score_seal_sha256'],
        reconstruction=io.EXPECTED_REC,raw_registry=io.EXPECTED_REG,torch=torch.__version__,numpy=np.__version__)
    if freeze_input:freeze(ROOT/'INPUT_FREEZE_PRIVATE.json',binding)
    return binding,core,fac,records,rec,banks

def ranked(values,salt):return sorted(values,key=lambda p:hashlib.sha256(f'{salt}|{p}'.encode()).digest())

def stratified_take(ids,centers,n,salt):
    groups={c:[p for p in ids if centers[p]==c] for c in sorted(set(centers[p] for p in ids))}
    desired={c:len(v)*n/len(ids) for c,v in groups.items()};counts={c:int(math.floor(v)) for c,v in desired.items()}
    order=sorted(groups,key=lambda c:(-(desired[c]-counts[c]),hashlib.sha256(f'{salt}|{c}'.encode()).digest()))
    for c in order[:n-sum(counts.values())]:counts[c]+=1
    selected=set()
    for c,v in groups.items():selected.update(ranked(v,salt)[:counts[c]])
    assert len(selected)==n;return selected

def roles_for(original,fold,stage):
    outer=original['roles'];fit=set(outer['fit'])
    if stage=='final':return dict(train=fit,cal=set(outer['validation']),assessment=set(),test=set(outer['test']))
    assert stage=='internal'
    centers={str(p):str(original['center'][np.flatnonzero(original['patient']==p)[0]]) for p in fit}
    n=int(math.ceil(.15*len(fit)));assessment=stratified_take(fit,centers,n,f'42|{fold}|ASSESS')
    cal=stratified_take(fit-assessment,centers,n,f'42|{fold}|CAL');train=fit-assessment-cal
    assert all(a.isdisjoint(b) for a,b in [(train,cal),(train,assessment),(cal,assessment)])
    assert train|cal|assessment==fit and not (train|cal|assessment)&set(outer['test'])
    return dict(train=train,cal=cal,assessment=assessment,test=set(outer['test']))

def store():
    q=legacy.OLD.P3.Inputs(legacy.OLD.BASELINE,Path(r'D:\dgm_net_seed42_v1_runtime'),ROOT);q.arm='M1_SCALED_DGM';return q

def data(fold,stage,core,rec,banks,inputs,binding):
    path=ROOT/'banks_private'/digest(binding)[:16]/stage/f'fold{fold}.pkl'
    if path.exists():
        q=read(path);assert q['binding']==binding;return q['bank'],torch.tensor(q['scale'],device='cuda')
    orig=parent.bank(fold,banks,'FULL_F0');roles=roles_for(orig,fold,stage)
    include=roles['train']|roles['cal']|roles['assessment'];take=np.flatnonzero(np.isin(orig['patient'],list(include)))
    d={k:orig[k][take].copy() for k in ['patient','channel','center','y','archive_indices']}
    d['train']=np.flatnonzero(np.isin(d['patient'],list(roles['train'])));d['val']=np.flatnonzero(np.isin(d['patient'],list(roles['cal'])))
    d['assessment']=np.flatnonzero(np.isin(d['patient'],list(roles['assessment'])))
    d['roles']=roles;d['fold']=fold;d['stage']=stage
    d['x'],d['pre']=core.prepare(rec['R14'][d['archive_indices']],d['patient'],d['train'])
    assert set(d['pre']['fit_patient_ids'])==roles['train'] and np.isfinite(d['x']).all() and d['x'].shape[1]==14
    magnitudes=[]
    for pid in sorted(roles['train']):
        ii=d['train'][d['patient'][d['train']]==pid]
        raw,mask,times,delta,_=inputs.get(str(pid),d['channel'][ii].astype(str).tolist(),'cpu')
        magnitudes.append(np.median(np.abs(delta[mask].numpy()),axis=0));del raw,delta
    scale=np.maximum(np.median(np.stack(magnitudes),axis=0),1e-30).astype(np.float32)
    if stage=='final':
        assert np.array_equal(d['archive_indices'],orig['archive_indices']) and np.array_equal(d['x'],orig['x'])
        assert np.array_equal(scale,parent.scale(fold).cpu().numpy())
    save(path,dict(binding=binding,bank=d,scale=scale))
    return d,torch.tensor(scale,device='cuda')

def make(core,fac,records,fold,arm,seed):
    s=seed+1009*fold;core.seed_all(s)
    template=legacy.OLD.P3.MorphCorrection(legacy.OLD.CompactDelta(fac(s).network,records)).cuda()
    source=old.MODEL.SingleHead('F0_AFTER_GELU',template.cnn,template.gate,type(template).features,s+170003).cuda()
    m=MODEL.Temporal(source,arm,fold,s).cuda();del source,template
    assert sum(isinstance(q,torch.nn.Linear) and q.out_features==1 for q in m.modules())==1
    assert sum(p.numel() for p in m.parameters())==(11425 if arm=='B0' else 13201)
    return m

def preprocessor(d,scale):
    return dict(pre={k:legacy.array_sha(np.asarray(v)) for k,v in d['pre'].items() if k!='fit_patient_ids'},
        fit_patients=digest(sorted(d['pre']['fit_patient_ids'])),scale=legacy.array_sha(scale.cpu().numpy()),
        roles={k:digest(sorted(v)) for k,v in d['roles'].items()})

def reference_copy(ref):return {k:v.detach().cpu().clone() if isinstance(v,torch.Tensor) else v for k,v in ref.items()} if ref is not None else None
def reference_hash(ref):return state_hash(ref) if ref is not None else None

def forward(m,d,ii,inputs,scale,collect=False):
    x=torch.as_tensor(d['x'][ii],device='cuda');pid=str(d['patient'][ii[0]])
    raw,mask,times,delta,_=inputs.get(pid,d['channel'][ii].astype(str).tolist(),'cuda')
    m.collect=collect;out=m(x,raw,mask,times,delta,scale)['logit']
    return out,dict(valid_windows=int(mask.sum()),**(m.last_audit if collect else {}))

def predict(m,d,ix,inputs,scale,collect=False):
    m.eval();out=np.empty(len(ix),np.float32);audit={}
    with torch.no_grad():
        for pid in sorted(set(d['patient'][ix])):
            j=np.flatnonzero(d['patient'][ix]==pid);logit,stats=forward(m,d,ix[j],inputs,scale,collect)
            out[j]=logit.cpu().numpy()
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

def frozen(stage):
    fr=jr(ROOT/f'{stage}_FROZEN_BEFORE_SCORES_PRIVATE.json');assert fr['source']==sources()
    assert fr['input']==jr(ROOT/'INPUT_FREEZE_PRIVATE.json') and {(r['arm'],r['seed'],r['fold']) for r in fr['cells']}==cells_expected()
    for r in fr['cells']:
        assert sha(r['checkpoint'])==r['checkpoint_sha256'] and sha(r['calibration'])==r['calibration_sha256']
    return fr
