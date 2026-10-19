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
OWN=sys.modules[__name__]

HERE=Path(__file__).resolve().parent;ROOT=Path(r'D:\m8_6d_no_variance_ablation_baselines_seed42_v1_runtime');PUB=ROOT/'public'
FILM=Path(r'D:\m8_7d_raw_film_seed42_v1');FILMROOT=Path(r'D:\m8_7d_raw_film_seed42_v1_runtime')
OLDROOT=Path(r'D:\m8_7d_frozen_candidate_outer_audit_seed42_v1_runtime')
NOPR=Path(r'D:\d1_m8_progressive_ablation_seed42_v1_runtime\A1')
FILES=['BASELINE_CONFIGS.json','tabular.py','seal_all.py','PROTOCOL_LOCK.json','model.py','common.py','test_model.py','admit.py','train.py','verify_internal.py','evaluate_outer.py','report_outer.py','verify_outer.py','run_all.ps1']

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

PARENT=Path(r'D:\m8_7d_f0_complete_ablation_seed42_v1');PARENTROOT=Path(r'D:\m8_7d_f0_complete_ablation_seed42_v1_runtime')
parent=module('sixd_frozen_parent',PARENT/'common.py');old=parent.old
OLDROOT=PARENTROOT
MODEL=module('sixd_ablation_model',HERE/'model.py')
io,legacy,EV=old.io,old.legacy,old.EV
sha,read,save,write,freeze,jr,digest=io.sha,io.read,io.save,io.write,io.freeze,io.jr,io.digest
atomic_torch,state_hash,seed_all=io.atomic_torch,io.state_hash,io.seed_all
DEEP_ARMS=MODEL.ARMS;REF_NAMES={'FULL_F0':'DROP_VARIANCE','RAW_ONLY':'RAW_ONLY'}
NEW=[a for a in DEEP_ARMS if a not in REF_NAMES];ARMS=DEEP_ARMS+MODEL.TABULAR_ARMS;METRICS=old.METRICS
KEEP12=np.array([i for i in range(14) if i not in [8,9]])
probability=old.probability;patient_rows=old.patient_rows
sys.path.insert(0,str(HERE));sys.modules['common']=OWN;sys.modules['model']=MODEL

def sources():return {n:sha(HERE/n) for n in FILES}

def setup(freeze_input=True):
    ROOT.mkdir(exist_ok=True);PUB.mkdir(exist_ok=True);own=OWN
    try:prior,core,fac,records,rec,banks=parent.setup()
    finally:sys.modules['common']=own;sys.modules['model']=MODEL;sys.path.insert(0,str(HERE))
    assert jr(HERE/'PROTOCOL_LOCK.json')['arms']==ARMS and len(NEW)==14
    assert sha(PARENTROOT/'FROZEN_BEFORE_OUTER_PRIVATE.json')=='2bee5ce863af8774491b6e53f58cf0fd5775d562cf3644e107809604abdf15c9'
    assert sha(PARENTROOT/'OUTER_SCORE_SEAL_PRIVATE.json')=='6dc73306f04ffefa9ffee471470786b3e190d16a304a81b8be5b9b2225a1f033'
    assert jr(PARENTROOT/'public/OUTER_STATUS.json')['status']=='COMPLETE_VERIFIED_F0_18_ARM_FIVE_FOLD_OUTER'
    cfg=jr(HERE/'BASELINE_CONFIGS.json')
    for path,expected in cfg['dependencies'].items():assert sha(Path(path))==expected
    for group,expected in cfg['source_tables'].items():
        location=Path(cfg['server_tables'][group]);assert sha(location)==expected
    binding=dict(source=sources(),prior_input=digest(prior),parent_freeze=sha(PARENTROOT/'FROZEN_BEFORE_OUTER_PRIVATE.json'),
        parent_seal=sha(PARENTROOT/'OUTER_SCORE_SEAL_PRIVATE.json'),reconstruction=io.EXPECTED_REC,raw_registry=io.EXPECTED_REG,
        baseline_configs=sha(HERE/'BASELINE_CONFIGS.json'),torch=torch.__version__,numpy=np.__version__)
    if freeze_input:freeze(ROOT/'INPUT_FREEZE_PRIVATE.json',binding)
    return binding,core,fac,records,rec,banks

def bank(fold,banks,arm):
    d=old.bank(fold,banks)
    if arm=='WO_BOTH_PR':
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
    fr=jr(PARENTROOT/'FROZEN_BEFORE_OUTER_PRIVATE.json');name=REF_NAMES[arm]
    r=dict(next(q for q in fr['cells'] if q['arm']==name and q['fold']==fold))
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
    fr=jr(ROOT/'FROZEN_BEFORE_OUTER_PRIVATE.json')
    assert fr['source']==sources() and fr['input']==jr(ROOT/'INPUT_FREEZE_PRIVATE.json')
    assert len(fr['cells'])==80 and sha(PARENTROOT/'OUTER_SCORE_SEAL_PRIVATE.json')==fr['input']['parent_seal']
    assert jr(PUB/'INTERNAL_VERIFICATION.json')['status']=='PASS'
    assert sha(PUB/'INTERNAL_VERIFICATION.json')==fr['internal_verification']
    tab=read(ROOT/'TABULAR_FREEZE_PRIVATE.pkl')
    assert sha(ROOT/'TABULAR_FREEZE_PRIVATE.pkl')==fr['tabular_freeze_sha256'] and len(tab['cells'])==70
    for r in fr['cells']+tab['cells']:
        assert sha(r['checkpoint'])==r['checkpoint_sha256'] and sha(r['validation'])==r['validation_sha256']
    return fr

def all_cells():
    fr=frozen();return fr['cells']+read(ROOT/'TABULAR_FREEZE_PRIVATE.pkl')['cells']

def probability_ez(q):return q['probability_ez'] if 'probability_ez' in q else 1-q['score_nez'].astype(float)
def predicted_ez(q,r):
    if 'prediction_ez' in q:
        assert r['arm'] in MODEL.TABULAR_ARMS
        assert np.array_equal(q['prediction_ez'],q['probability_ez']>=r['threshold'])
        return q['prediction_ez']
    return q['score_nez']<r['threshold']

def vector(rows):return np.array([[r[m] for m in METRICS] for r in rows],dtype=np.float64)
def interval(values,draw,levels=(.025,.975)):
    assert values.shape==(80,9) and np.isfinite(values).all()
    samples=np.concatenate([values[draw[k:k+250]].mean(1) for k in range(0,10000,250)])
    return values.mean(0),np.quantile(samples,levels,axis=0)

