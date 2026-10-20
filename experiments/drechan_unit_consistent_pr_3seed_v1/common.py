"""New derived physiology bank only; immutable Raw/ref bodies and original roles."""
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
ROOT=Path(r'D:\drechan_unit_consistent_pr_3seed_v1_runtime');PUB=ROOT/'public'
PARENT=Path(r'D:\drechan_raw_dualview_3seed_v1')


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m


sys.path.insert(0,str(PARENT));parent=module('ucp_completed_rdv_common',PARENT/'common.py')
component,prior=parent.component,parent.prior
sys.path.insert(0,str(HERE));MODEL=module('ucp_current_model',HERE/'model.py')
from unit_pr import prepare as stable_prepare
io,legacy,EV=parent.io,parent.legacy,parent.EV
sha,read,save,write,freeze,jr,digest=io.sha,io.read,io.save,io.write,io.freeze,io.jr,io.digest
atomic_torch,state_hash,seed_all=io.atomic_torch,io.state_hash,io.seed_all
ARMS=MODEL.ARMS;REFERENCES=MODEL.REFERENCES;NEW=[a for a in ARMS if a not in REFERENCES]
SEEDS=[42,43,44];METRICS=parent.METRICS;probability=parent.probability
FILES=['PROTOCOL_LOCK.json','unit_pr.py','model.py','common.py','test_preprocessing.py','admit.py','train.py','assess.py','verify.py','run_all.ps1','affinity_entry.py','package_public.py']
sys.modules['common']=OWN;sys.modules['model']=MODEL


def sources():return {n:sha(HERE/n) for n in FILES}


def setup(freeze_input=True):
    ROOT.mkdir(exist_ok=True);PUB.mkdir(exist_ok=True)
    try:previous,core,fac,records,rec,banks=parent.setup(False)
    finally:sys.modules['common']=OWN;sys.modules['model']=MODEL;sys.path.insert(0,str(HERE))
    lock=jr(HERE/'PROTOCOL_LOCK.json');assert lock['arms']==ARMS and lock['references']==REFERENCES and lock['parameters']==MODEL.PARAMETERS
    assert sha(parent.ROOT/'FROZEN_BEFORE_ASSESS_PRIVATE.json')==lock['parent_freeze_sha256'] and sha(parent.ROOT/'SCORE_SEAL_PRIVATE.json')==lock['parent_seal_sha256']
    assert jr(parent.PUB/'INDEPENDENT_VERIFICATION.json')['status']=='PASS' and parent.frozen()['input']==previous
    evidence=Path(r'D:\drechan_dualview_traincal_diagnostic_v1_runtime\public\feature_numeric_extension\INPUT_NUMERIC_VERIFICATION.json')
    assert sha(evidence)==lock['diagnostic_numeric_verification_sha256'] and jr(evidence)['status']=='PASS' and not jr(evidence)['outcome_labels_used']
    binding=dict(source=sources(),parent_source=parent.sources(),parent_input=digest(previous),parent_freeze=lock['parent_freeze_sha256'],
                 parent_seal=lock['parent_seal_sha256'],numeric_evidence=sha(evidence),torch=torch.__version__,numpy=np.__version__,scientific_status='EXPLORATORY_REPEATED_TEST')
    if freeze_input:freeze(ROOT/'INPUT_FREEZE_PRIVATE.json',binding)
    return binding,core,fac,records,rec,banks


def store():
    q=legacy.OLD.P3.Inputs(legacy.OLD.BASELINE,Path(r'D:\dgm_net_seed42_v1_runtime'),ROOT);q.arm='M1_SCALED_DGM';return q


def data(fold,core,rec,banks,inputs,binding):
    previous=jr(parent.ROOT/'INPUT_FREEZE_PRIVATE.json');old,scale=parent.data(fold,core,rec,banks,inputs,previous)
    path=ROOT/'banks_private'/digest(binding)[:16]/f'fold{fold}.pkl'
    if path.exists():
        q=read(path);assert q['binding']==binding;d=q['bank']
    else:
        raw=rec['R14'][old['archive_indices']]
        x,pre=stable_prepare(raw,old['patient'],old['train'])
        assert np.array_equal(pre['TRAIN_unit_scale'],old['pre_no_pr']['scale']) and np.array_equal(pre['TRAIN_unit_variance'],old['pre_no_pr']['var'])
        d=dict(old,x_legacy=old['x'],pre_legacy=old['pre'],x=x,pre=pre)
        save(path,dict(binding=binding,bank=d))
    for k in ['patient','channel','center','y','archive_indices','train','val','assessment','x_no_pr']:
        assert np.array_equal(d[k],old[k])
    assert np.array_equal(d['x_legacy'],old['x']) and set(d['pre_legacy'])==set(old['pre'])
    for k in old['pre']:
        if k=='fit_patient_ids':assert d['pre_legacy'][k]==old['pre'][k]
        else:assert np.array_equal(d['pre_legacy'][k],old['pre'][k])
    assert d['roles']==old['roles'] and set(d['pre']['fit_patient_ids'])==d['roles']['train']
    assert set(d['pre_no_pr']['fit_patient_ids'])==d['roles']['train'] and set(d['patient']).isdisjoint(d['roles']['test'])
    assert np.array_equal(d['pre']['TRAIN_unit_scale'],old['pre_no_pr']['scale'])
    return d,scale


def legacy_data(d):return dict(d,x=d['x_legacy'],pre=d['pre_legacy'])


def reference_cell(arm,seed,fold):
    assert arm in REFERENCES
    fr=parent.frozen();r=next(r for r in fr['cells'] if (r['arm'],r['seed'],r['fold'])==(REFERENCES[arm],seed,fold))
    assert sha(r['checkpoint'])==r['checkpoint_sha256'] and sha(r['calibration'])==r['calibration_sha256'];return r


def initialize(core,fac,records,fold,arm,seed):
    if arm in REFERENCES:m=parent.initialize(core,fac,records,fold,REFERENCES[arm],seed)
    else:m=parent.initialize(core,fac,records,fold,arm,seed)
    m.ucp_legacy_feature_input=arm in REFERENCES
    assert sum(p.numel() for p in m.parameters())==MODEL.PARAMETERS[arm];return m


def anchor_kind(arm):return 'no_pre' if arm=='WO_PRE_REFERENCE' else 'pre'
def anchor_path(fold,seed,kind='pre'):return parent.anchor_path(fold,seed,kind)


def make(core,fac,records,fold,arm,seed):
    if arm in REFERENCES:
        m=parent.make(core,fac,records,fold,REFERENCES[arm],seed);m.ucp_legacy_feature_input=True
    else:
        m=initialize(core,fac,records,fold,arm,seed);m.set_anchor(read(anchor_path(fold,seed,anchor_kind(arm))))
    return m


def preprocessor(d,scale,arm,fold,seed):
    if arm in REFERENCES:return parent.preprocessor(legacy_data(d),scale,REFERENCES[arm],fold,seed)
    if arm in ['WO_FEATURE_PR','WO_BOTH_PR']:return parent.preprocessor(legacy_data(d),scale,arm,fold,seed)
    summary=prior.preprocessor(d,scale)
    return dict(summary=summary,initial_TRAIN_raw_anchor_sha256=sha(anchor_path(fold,seed,anchor_kind(arm))),
                anchor_kind=anchor_kind(arm),raw_view_width=192,FeaturePR_policy='TRAIN_UNIT_RELATIVE_EPS_ZERO_DEGENERATE_V1')


def feature_input(m,d):
    if not m.feature_pr:return d['x_no_pr']
    return d['x_legacy'] if getattr(m,'ucp_legacy_feature_input',False) else d['x']


def forward(m,d,ii,inputs,scale,collect=False):
    x=torch.as_tensor(feature_input(m,d)[ii],device='cuda');pid=str(d['patient'][ii[0]])
    raw,mask,times,delta,_=inputs.get(pid,d['channel'][ii].astype(str).tolist(),'cuda')
    out=m(x,raw,mask,times,delta,scale);stats=dict(valid_windows=int(mask.sum()))
    if collect:
        z=out['raw_summary'].detach().double();stats.update(raw_summary_coordinates=z.numel(),raw_summary_sum_abs=float(z.abs().sum()),raw_summary_sum_square=float(z.square().sum()))
    return out['logit'],stats


def predict(m,d,ix,inputs,scale,collect=False):
    m.eval();out=np.empty(len(ix),np.float32);audit={}
    with torch.no_grad():
        for pid in sorted(set(d['patient'][ix])):
            j=np.flatnonzero(d['patient'][ix]==pid);logit,stats=forward(m,d,ix[j],inputs,scale,collect);out[j]=logit.cpu().numpy()
            for k,v in stats.items():audit[k]=audit.get(k,0)+v
    assert np.isfinite(out).all();return out,audit


def csv_out(name,rows):
    assert rows and all(set(r)==set(rows[0]) for r in rows)
    path=PUB/(name+'.csv');tmp=path.with_suffix('.csv.tmp')
    with tmp.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    tmp.replace(path)


def cells_expected():return {(a,s,f) for a in ARMS for s in SEEDS for f in range(1,6)}
def vector(rows):return np.array([[r[m] for m in METRICS] for r in rows],dtype=np.float64)


def frozen():
    q=jr(ROOT/'FROZEN_BEFORE_ASSESS_PRIVATE.json');assert q['source']==sources() and q['input']==jr(ROOT/'INPUT_FREEZE_PRIVATE.json')
    assert {(r['arm'],r['seed'],r['fold']) for r in q['cells']}==cells_expected()
    assert sha(parent.ROOT/'FROZEN_BEFORE_ASSESS_PRIVATE.json')==q['input']['parent_freeze'] and sha(parent.ROOT/'SCORE_SEAL_PRIVATE.json')==q['input']['parent_seal']
    for r in q['cells']:assert sha(r['checkpoint'])==r['checkpoint_sha256'] and sha(r['calibration'])==r['calibration_sha256']
    return q


CONTRASTS={'FULL_MINUS_B0':{'FULL':1,'B0':-1},'FULL_MINUS_PRUNED_REF':{'FULL':1,'PRUNED_REF':-1},
           'UNIT_CONSISTENCY':{'FULL':1,'LEGACY_DUALVIEW':-1},'FEATURE_PR_WITH_RAW':{'FULL':1,'WO_FEATURE_PR':-1},
           'RAW_PR_WITH_FEATURE':{'FULL':1,'WO_RAW_PR':-1},'BOTH_PR_JOINT':{'FULL':1,'WO_BOTH_PR':-1},
           'GLOBAL_VIEW':{'FULL':1,'WO_GLOBAL_VIEW':-1},'PRE_REFERENCE':{'FULL':1,'WO_PRE_REFERENCE':-1},
           'PR_INTERACTION':{'FULL':1,'WO_FEATURE_PR':-1,'WO_RAW_PR':-1,'WO_BOTH_PR':1}}
PRIMARY_CONTRASTS=['FULL_MINUS_B0','FULL_MINUS_PRUNED_REF','UNIT_CONSISTENCY','FEATURE_PR_WITH_RAW','RAW_PR_WITH_FEATURE','GLOBAL_VIEW','PRE_REFERENCE']
PRIMARY_METRICS=['auroc','macro_f1']
