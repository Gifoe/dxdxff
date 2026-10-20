"""Own fixed17-arm execution over immutable original data, roles and model math."""
import csv
import gc
import hashlib
import importlib.util
from pathlib import Path
import sys
import numpy as np
import torch
OWN=sys.modules[__name__];HERE=Path(__file__).resolve().parent
ROOT=Path(r'D:\drechan_no_guidance_ablation_42_44_62_v1_runtime');PUB=ROOT/'public'
PARENT=Path(r'D:\drechan_no_earlyfilm_seed52_62_v1')
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);sys.modules[name]=value;spec.loader.exec_module(value)
    return value
parent=module('nga_parent_common',PARENT/'common.py');original=parent.original
MODEL=module('nga_model',HERE/'model.py')
io,legacy,EV=parent.io,parent.legacy,parent.EV
sha,read,save,write,freeze,jr,digest=io.sha,io.read,io.save,io.write,io.freeze,io.jr,io.digest
atomic_torch,state_hash,seed_all=io.atomic_torch,io.state_hash,io.seed_all
probability,METRICS=parent.probability,parent.METRICS
feature_math=original.feature_math
ARMS,SEEDS=MODEL.ARMS,[42,44,62]
FILES=['PROTOCOL_LOCK.json','model.py','common.py','test_model.py','test_reporting.py','admit.py','train.py',
       'verify_internal.py','evaluate_outer.py','report_outer.py','verify_outer.py','run_all.ps1','affinity_entry.py','package_public.py']
CONTEXT={};sys.modules['common']=OWN;sys.modules['model']=MODEL
def sources():return {n:sha(HERE/n) for n in FILES}
def setup(freeze_input=True):
    ROOT.mkdir(exist_ok=True);PUB.mkdir(exist_ok=True)
    try:previous,core,fac,records,rec,banks=parent.setup(False)
    finally:sys.modules['common']=OWN;sys.modules['model']=MODEL;sys.path.insert(0,str(HERE))
    lock=jr(HERE/'PROTOCOL_LOCK.json')
    assert lock['arms']==ARMS and lock['seeds']==SEEDS and lock['parameters']==MODEL.PARAMETERS and lock['aliases']==MODEL.ALIASES
    assert parent.sources()==lock['parent_source']
    roots={'parent':parent.ROOT,'complete3':original.ROOT,'poolcap':Path(r'D:\drechan_pooling_capacity_ablation_seed42_v1_runtime')}
    freezes={}
    for label,root in roots.items():
        assert sha(root/'FROZEN_BEFORE_OUTER_PRIVATE.json')==lock[label+'_freeze_sha256']
        assert sha(root/'OUTER_SCORE_SEAL_PRIVATE.json')==lock[label+'_score_seal_sha256']
        assert jr(root/'public/INDEPENDENT_OUTER_VERIFICATION.json')['status']=='PASS'
        freezes[label]=jr(root/'FROZEN_BEFORE_OUTER_PRIVATE.json')
    assert rec['R14'].shape==(7635,14) and len(set(rec['meta']['patient']))==80
    assert sum(len(r['records']) for r in rec['ledgers'])==256
    binding=dict(source=sources(),parent_source=parent.sources(),parent_input=digest(previous),
        freezes={k:lock[k+'_freeze_sha256'] for k in roots},seals={k:lock[k+'_score_seal_sha256'] for k in roots},
        reconstruction=io.EXPECTED_REC,raw_registry=io.EXPECTED_REG,torch=torch.__version__,numpy=np.__version__,
        scientific_status=lock['scientific_status'])
    if freeze_input:freeze(ROOT/'INPUT_FREEZE_PRIVATE.json',binding)
    CONTEXT.update(binding=binding,rec=rec,roots=roots,freezes=freezes)
    return binding,core,fac,records,rec,banks
def input_arm(arm):
    assert arm in ARMS
    return 'WO_BOTH_PR' if arm in ['WO_FEATURE_PR','WO_BOTH_PR'] else 'RAW_ONLY' if arm=='RAW_ONLY' else arm if arm in MODEL.DROP else 'FULL'
def bank(fold,banks,arm):
    assert arm in ARMS and CONTEXT
    if arm not in MODEL.DROP:return original.bank(fold,banks,input_arm(arm))
    full=original.bank(fold,banks,'FULL')
    path=ROOT/'banks_private'/digest(CONTEXT['binding'])[:16]/f'{arm}_fold{fold}.pkl'
    if path.exists():
        q=read(path);assert q['binding']==CONTEXT['binding'];d=q['bank']
    else:
        raw=CONTEXT['rec']['R14'][full['archive_indices']].copy();j=MODEL.DROP[arm];raw[:,2*j:2*j+2]=0.
        x,pre=feature_math.prepare(raw,full['patient'],full['train'])
        active=[k for k in range(14) if k not in [2*j,2*j+1]]
        assert np.array_equal(x[:,active],full['x'][:,active]) and not x[:,2*j:2*j+2].any()
        d=dict(full,x=x,pre=pre);save(path,dict(binding=CONTEXT['binding'],bank=d))
    for key in ['patient','channel','y','train','val','archive_indices']:assert np.array_equal(d[key],full[key])
    assert d['roles']==full['roles'] and set(d['pre']['fit_patient_ids'])==d['roles']['fit']
    assert set(d['patient']).isdisjoint(d['roles']['test']) and np.isfinite(d['x']).all()
    return d
def query_features(raw,patients,pre,arm,core):
    return original.query_features(raw,patients,pre,input_arm(arm),core)
scale=parent.scale
preprocessor=parent.preprocessor
def source_for(core,fac,records,fold,seed):
    assert seed in SEEDS and 1<=fold<=5
    s=seed+1009*fold;core.seed_all(s)
    template=legacy.OLD.P3.MorphCorrection(legacy.OLD.CompactDelta(fac(s).network,records)).cuda()
    source=original.legacy_parent.old.MODEL.SingleHead('F0_AFTER_GELU',template.cnn,template.gate,type(template).features,s+170003).cuda()
    del template
    return source
def make(core,fac,records,fold,arm,seed):
    source=source_for(core,fac,records,fold,seed)
    m=MODEL.NoGuidanceF0(source,arm,seed+1009*fold+170003).cuda();del source
    return m
def reference_spec(arm,seed):
    if arm=='FULL':return ('parent','WO_EARLY_FILM') if seed==62 else ('complete3','WO_BOTH_GUIDANCE')
    if seed in [42,44] and arm in ['RAW_ONLY','PHYSIOLOGY_ONLY']:return 'complete3',arm
    if seed in [42,44] and arm=='EARLY_FILM_RESTORED':return 'complete3','WO_LATE_GATE'
    if seed==42 and arm=='CAPACITY_MATCHED':return 'poolcap','A2_CAPACITY_MATCHED'
    return None
def reference(arm,seed,fold,optional=False):
    assert arm in ARMS and seed in SEEDS
    spec=reference_spec(arm,seed)
    if spec is None:return None
    kind,source_arm=spec
    r=next(r for r in CONTEXT['freezes'][kind]['cells'] if r['arm']==source_arm and r['seed']==seed and r['fold']==fold)
    assert sha(r['checkpoint'])==r['checkpoint_sha256'] and sha(r['validation'])==r['validation_sha256']
    state=torch.load(r['checkpoint'],weights_only=False,map_location='cpu');v=read(r['validation'])
    assert state['binding']==v['binding']
    return dict(r,arm=arm,seed=seed,epochs=len(v['history']),parameters=MODEL.PARAMETERS[arm],
                reused=True,reuse_kind=kind,source_arm=source_arm),state,v
def reference_score(r):
    root=CONTEXT['roots'][r['reuse_kind']]
    key=f"{r['source_arm']}_seed{r['seed']}_fold{r['fold']}";path=root/'scores_private'/(key+'.pkl')
    assert sha(path)==jr(root/'OUTER_SCORE_SEAL_PRIVATE.json')['files'][key]
    q=read(path);assert q['checkpoint']==r['checkpoint_sha256'] and q['threshold']==r['threshold']
    return q,sha(path),root
def store():
    value=legacy.OLD.P3.Inputs(legacy.OLD.BASELINE,Path(r'D:\dgm_net_seed42_v1_runtime'),ROOT);value.arm='M1_SCALED_DGM'
    return value
def forward(model,d,ix,inputs,ss,fold,binding):
    assert len(set(d['patient'][ix]))==1
    x=torch.as_tensor(d['x'][ix],device='cuda')
    if model.branch=='phys':return model(x)['logit'],dict(valid_windows=0)
    pid=str(d['patient'][ix[0]]);channels=d['channel'][ix].astype(str).tolist()
    raw,mask,times,delta,_=inputs.get(pid,channels,'cuda')
    out=model(x,raw,mask,times,delta,ss)['logit'];info=dict(model.last_guidance_audit)
    if 'singleton_valid_windows' in info:info['relative_singleton_valid_windows']=info.pop('singleton_valid_windows')
    if 'valid_windows' in info:assert info['valid_windows']==int(mask.sum())
    info['valid_windows']=int(mask.sum())
    return out,info
def predict(model,d,ix,inputs,ss,fold,binding):
    model.eval();result=np.empty(len(ix),np.float32)
    with torch.inference_mode():
        for pid in sorted(set(d['patient'][ix])):
            jj=np.flatnonzero(d['patient'][ix]==pid);logit,_=forward(model,d,ix[jj],inputs,ss,fold,binding)
            result[jj]=logit.cpu().numpy()
    assert np.isfinite(result).all()
    return result
def expected_cells():return {(a,s,f) for a in ARMS for s in SEEDS for f in range(1,6)}
def key_for(arm,seed,fold):return f'{arm}_seed{seed}_fold{fold}'
def csv_out(name,rows):
    assert rows and all(set(r)==set(rows[0]) for r in rows)
    path=PUB/(name+'.csv');tmp=path.with_suffix('.csv.tmp')
    with tmp.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    tmp.replace(path)
def frozen():
    q=jr(ROOT/'FROZEN_BEFORE_OUTER_PRIVATE.json')
    assert q['source']==sources() and q['input']==jr(ROOT/'INPUT_FREEZE_PRIVATE.json')
    assert {(r['arm'],r['seed'],r['fold']) for r in q['cells']}==expected_cells()
    assert jr(PUB/'INTERNAL_VERIFICATION.json')['status']=='PASS' and sha(PUB/'INTERNAL_VERIFICATION.json')==q['internal_verification']
    for r in q['cells']:assert sha(r['checkpoint'])==r['checkpoint_sha256'] and sha(r['validation'])==r['validation_sha256']
    return q
def vector(rows):return np.asarray([[r[m] for m in METRICS] for r in rows],np.float64)
def interval(values,draw,levels=(.025,.975)):
    assert values.shape==(80,9) and np.isfinite(values).all()
    samples=np.concatenate([values[draw[k:k+250]].mean(1) for k in range(0,10000,250)])
    return values.mean(0),np.quantile(samples,levels,axis=0)
PAIRS={'FULL_MINUS_'+a:{'FULL':1,a:-1} for a in ARMS if a!='FULL'}
INTERACTIONS={'PR_INTERACTION':{'FULL':1,'WO_FEATURE_PR':-1,'WO_RAW_PR':-1,'WO_BOTH_PR':1}}
