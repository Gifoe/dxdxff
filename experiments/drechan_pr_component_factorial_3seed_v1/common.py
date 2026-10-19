"""Read-only exact prior roles and dependencies; new binding/output namespace."""
import csv
import gc
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

OWN=sys.modules[__name__];HERE=Path(__file__).resolve().parent
ROOT=Path(r'D:\drechan_pr_component_factorial_3seed_v1_runtime');PUB=ROOT/'public'
PREVIOUS=Path(r'D:\drechan_temporal_pr_contrast_3seed_v1')

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

prior=module('prf_completed_temporal_common',PREVIOUS/'common.py')
MODEL=module('prf_current_model',HERE/'model.py')
old,io,legacy,EV=prior.old,prior.io,prior.legacy,prior.EV
sha,read,save,write,freeze,jr,digest=io.sha,io.read,io.save,io.write,io.freeze,io.jr,io.digest
atomic_torch,state_hash,seed_all=io.atomic_torch,io.state_hash,io.seed_all
ARMS=MODEL.ARMS;SEEDS=[42,43,44];METRICS=prior.METRICS;probability=prior.probability
FILES=['PROTOCOL_LOCK.json','model.py','common.py','test_model.py','admit.py','train.py','assess.py','verify.py','run_all.ps1','affinity_entry.py','package_public.py']
sys.path.insert(0,str(HERE));sys.modules['common']=OWN;sys.modules['model']=MODEL

def sources():return {n:sha(HERE/n) for n in FILES}

def setup(freeze_input=True):
    ROOT.mkdir(exist_ok=True);PUB.mkdir(exist_ok=True)
    try:previous_binding,core,fac,records,rec,banks=prior.setup(False)
    finally:sys.modules['common']=OWN;sys.modules['model']=MODEL;sys.path.insert(0,str(HERE))
    lock=jr(HERE/'PROTOCOL_LOCK.json');assert lock['arms']==ARMS and lock['seeds']==SEEDS
    assert jr(prior.PUB/'internal/INDEPENDENT_VERIFICATION.json')['status']=='PASS'
    assert sha(prior.ROOT/'internal_FROZEN_BEFORE_SCORES_PRIVATE.json')==lock['previous_internal_freeze_sha256']
    assert sha(prior.ROOT/'internal_SCORE_SEAL_PRIVATE.json')==lock['previous_internal_seal_sha256']
    assert previous_binding==jr(prior.ROOT/'INPUT_FREEZE_PRIVATE.json')
    fr=jr(prior.ROOT/'internal_FROZEN_BEFORE_SCORES_PRIVATE.json')
    assert fr['source']==prior.sources() and fr['input']==previous_binding
    binding=dict(source=sources(),previous_input=digest(previous_binding),previous_freeze=lock['previous_internal_freeze_sha256'],previous_seal=lock['previous_internal_seal_sha256'],
                 torch=torch.__version__,numpy=np.__version__,scientific_status='EXPLORATORY_REPEATED_TEST')
    if freeze_input:freeze(ROOT/'INPUT_FREEZE_PRIVATE.json',binding)
    return binding,core,fac,records,rec,banks

def store():
    q=legacy.OLD.P3.Inputs(legacy.OLD.BASELINE,Path(r'D:\dgm_net_seed42_v1_runtime'),ROOT);q.arm='M1_SCALED_DGM';return q

def fit_no_feature_pr(raw,patients,train):
    raw=np.asarray(raw,dtype=np.float64)
    imputer=SimpleImputer();scaler=StandardScaler()
    imputer.fit(raw[train]);scaler.fit(imputer.transform(raw[train]))
    result=scaler.transform(imputer.transform(raw)).astype(np.float32)
    assert result.shape==raw.shape and np.isfinite(result).all()
    return result,dict(imputer_statistics=imputer.statistics_,mean=scaler.mean_,scale=scaler.scale_,var=scaler.var_,fit_patient_ids=sorted(set(patients[train])))

def data(fold,core,rec,banks,inputs,binding):
    # Reuse only an already existing private input bank; never write old experiments.
    prev=jr(prior.ROOT/'INPUT_FREEZE_PRIVATE.json')
    oldpath=prior.ROOT/'banks_private'/digest(prev)[:16]/'internal'/f'fold{fold}.pkl';assert oldpath.exists()
    q=read(oldpath);assert q['binding']==prev
    original=q['bank'];scale=torch.tensor(q['scale'],device='cuda')
    roles=prior.roles_for(prior.parent.bank(fold,banks,'FULL_F0'),fold,'internal')
    assert roles==original['roles'] and set(original['pre']['fit_patient_ids'])==roles['train']
    path=ROOT/'banks_private'/digest(binding)[:16]/f'fold{fold}.pkl'
    if path.exists():
        cached=read(path);assert cached['binding']==binding;d=cached['bank']
    else:
        d=dict(original)
        d['x_no_pr'],d['pre_no_pr']=fit_no_feature_pr(rec['R14'][d['archive_indices']],d['patient'],d['train'])
        save(path,dict(binding=binding,bank=d))
    for k in ['x','archive_indices','train','val','assessment','patient','channel','y']:
        assert np.array_equal(d[k],original[k])
    assert d['roles']==roles and set(d['pre_no_pr']['fit_patient_ids'])==roles['train']
    assert set(d['patient']).isdisjoint(roles['test'])
    return d,scale

def preprocessor(d,scale,arm):
    active=dict(d,pre=d['pre'] if arm in ['FULL','WO_RAW_PR'] else d['pre_no_pr'])
    return prior.preprocessor(active,scale)

def make(core,fac,records,fold,arm,seed):
    s=seed+1009*fold;core.seed_all(s)
    template=legacy.OLD.P3.MorphCorrection(legacy.OLD.CompactDelta(fac(s).network,records)).cuda()
    source=old.MODEL.SingleHead('F0_AFTER_GELU',template.cnn,template.gate,type(template).features,s+170003).cuda()
    m=MODEL.PRFactorial(source,arm).cuda();del source,template
    assert sum(p.numel() for p in m.parameters())==11425
    assert sum(isinstance(q,torch.nn.Linear) and q.out_features==1 for q in m.modules())==1
    return m

def forward(m,d,ii,inputs,scale,collect=False):
    feature=d['x'] if m.feature_pr else d['x_no_pr']
    x=torch.as_tensor(feature[ii],device='cuda');pid=str(d['patient'][ii[0]])
    raw,mask,times,delta,_=inputs.get(pid,d['channel'][ii].astype(str).tolist(),'cuda')
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

def previous_cell(seed,fold):
    fr=jr(prior.ROOT/'internal_FROZEN_BEFORE_SCORES_PRIVATE.json')
    r=next(q for q in fr['cells'] if (q['arm'],q['seed'],q['fold'])==('B0',seed,fold))
    assert sha(r['checkpoint'])==r['checkpoint_sha256'] and sha(r['calibration'])==r['calibration_sha256']
    return r

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
 'FEATURE_PR_WITH_RAW':{'FULL':1,'WO_FEATURE_PR':-1},
 'FEATURE_PR_WITHOUT_RAW':{'WO_RAW_PR':1,'WO_BOTH_PR':-1},
 'RAW_PR_WITH_FEATURE':{'FULL':1,'WO_RAW_PR':-1},
 'RAW_PR_WITHOUT_FEATURE':{'WO_FEATURE_PR':1,'WO_BOTH_PR':-1},
 'BOTH_PR_JOINT':{'FULL':1,'WO_BOTH_PR':-1},
 'PR_INTERACTION':{'FULL':1,'WO_FEATURE_PR':-1,'WO_RAW_PR':-1,'WO_BOTH_PR':1},
 'FEATURE_PR_AVERAGE_MAIN_EFFECT':{'FULL':.5,'WO_FEATURE_PR':-.5,'WO_RAW_PR':.5,'WO_BOTH_PR':-.5},
 'RAW_PR_AVERAGE_MAIN_EFFECT':{'FULL':.5,'WO_FEATURE_PR':.5,'WO_RAW_PR':-.5,'WO_BOTH_PR':-.5}}
PRIMARY_CONTRASTS=['FEATURE_PR_WITH_RAW','RAW_PR_WITH_FEATURE'];PRIMARY_METRICS=['auroc','macro_f1']
