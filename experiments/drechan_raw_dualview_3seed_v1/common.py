"""New dual-view binding; immutable roles,references and complete TRAIN anchors."""
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
ROOT=Path(r'D:\drechan_raw_dualview_3seed_v1_runtime');PUB=ROOT/'public'
PARENT=Path(r'D:\drechan_pruned_branch_fusion_3seed_v1')


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m


parent=module('rdv_completed_parent_common',PARENT/'common.py');component=parent.component;prior=parent.prior
sys.path.insert(0,str(HERE));MODEL=module('rdv_current_model',HERE/'model.py')
io,legacy,EV=parent.io,parent.legacy,parent.EV
sha,read,save,write,freeze,jr,digest=io.sha,io.read,io.save,io.write,io.freeze,io.jr,io.digest
atomic_torch,state_hash,seed_all=io.atomic_torch,io.state_hash,io.seed_all
ARMS=MODEL.ARMS;REFERENCES=MODEL.REFERENCES;NEW=[a for a in ARMS if a not in REFERENCES]
SEEDS=[42,43,44];METRICS=parent.METRICS;probability=parent.probability
FILES=['PROTOCOL_LOCK.json','view_math.py','model.py','common.py','test_model.py','admit.py','train.py','assess.py','verify.py','run_all.ps1','affinity_entry.py','package_public.py']
sys.modules['common']=OWN;sys.modules['model']=MODEL


def sources():return {n:sha(HERE/n) for n in FILES}


def setup(freeze_input=True):
    ROOT.mkdir(exist_ok=True);PUB.mkdir(exist_ok=True)
    try:pb,core,fac,records,rec,banks=parent.setup(False)
    finally:sys.modules['common']=OWN;sys.modules['model']=MODEL;sys.path.insert(0,str(HERE))
    lock=jr(HERE/'PROTOCOL_LOCK.json')
    assert lock['arms']==ARMS and lock['seeds']==SEEDS and lock['parameters']==MODEL.PARAMETERS and lock['references']==REFERENCES
    assert jr(parent.PUB/'INDEPENDENT_VERIFICATION.json')['status']=='PASS'
    assert sha(parent.ROOT/'FROZEN_BEFORE_ASSESS_PRIVATE.json')==lock['parent_freeze_sha256']
    assert sha(parent.ROOT/'SCORE_SEAL_PRIVATE.json')==lock['parent_seal_sha256']
    assert parent.frozen()['input']==pb==jr(parent.ROOT/'INPUT_FREEZE_PRIVATE.json')
    binding=dict(source=sources(),parent_source=parent.sources(),parent_input=digest(pb),parent_freeze=lock['parent_freeze_sha256'],parent_seal=lock['parent_seal_sha256'],
                 torch=torch.__version__,numpy=np.__version__,scientific_status='EXPLORATORY_REPEATED_TEST')
    if freeze_input:freeze(ROOT/'INPUT_FREEZE_PRIVATE.json',binding)
    return binding,core,fac,records,rec,banks


def store():
    q=legacy.OLD.P3.Inputs(legacy.OLD.BASELINE,Path(r'D:\dgm_net_seed42_v1_runtime'),ROOT);q.arm='M1_SCALED_DGM';return q


def data(fold,core,rec,banks,inputs,binding):return parent.data(fold,core,rec,banks,inputs,binding)


def reference_cell(arm,seed,fold):
    assert arm in REFERENCES
    fr=jr(parent.ROOT/'FROZEN_BEFORE_ASSESS_PRIVATE.json')
    assert sha(parent.ROOT/'FROZEN_BEFORE_ASSESS_PRIVATE.json')==jr(HERE/'PROTOCOL_LOCK.json')['parent_freeze_sha256']
    r=next(q for q in fr['cells'] if (q['arm'],q['seed'],q['fold'])==(REFERENCES[arm],seed,fold))
    assert sha(r['checkpoint'])==r['checkpoint_sha256'] and sha(r['calibration'])==r['calibration_sha256']
    return r


def initialize(core,fac,records,fold,arm,seed):
    if arm in REFERENCES:return parent.make(core,fac,records,fold,REFERENCES[arm],seed)
    source=component.make(core,fac,records,fold,'FULL',seed)
    m=MODEL.DualView(source,arm).cuda();del source
    assert sum(p.numel() for p in m.parameters())==MODEL.PARAMETERS[arm]
    assert sum(isinstance(q,torch.nn.Linear) and q.out_features==1 for q in m.modules())==1
    return m


def anchor_kind(arm):return 'no_pre' if arm=='WO_PRE_REFERENCE' else 'pre'
def anchor_path(fold,seed,kind='pre'):return ROOT/'anchors_private'/kind/f'fold{fold}'/f'seed{seed}.pkl'


def fit_anchor(fold,seed,kind,d,scale,inputs,binding,core,fac,records):
    assert kind in ['pre','no_pre'];arm='FULL' if kind=='pre' else 'WO_PRE_REFERENCE'
    m=initialize(core,fac,records,fold,arm,seed);initial=state_hash(m.state_dict())
    bind=dict(input=digest(binding),fold=fold,seed=seed,kind=kind,initial=initial,
              train_patients=digest(sorted(d['roles']['train'])),train_indices=legacy.array_sha(d['archive_indices'][d['train']]))
    path=anchor_path(fold,seed,kind)
    if path.exists():q=read(path);assert q['binding']==bind
    else:
        arrays=[];windows=0;m.eval();m.collect_anchor=True
        with torch.no_grad():
            for pid in sorted(d['roles']['train']):
                ii=d['train'][d['patient'][d['train']]==pid]
                raw,mask,times,delta,_=inputs.get(str(pid),d['channel'][ii].astype(str).tolist(),'cuda')
                vector,_=m.features(raw,mask,times,delta,scale);assert vector.shape==(len(ii),96)
                arrays.append(vector.double().cpu());windows+=int(mask.sum());del raw,delta,vector
        x=torch.cat(arrays);mu=x.mean(0);sd=x.std(0,unbiased=False)
        q=dict(binding=bind,mean=mu,scale=torch.where(sd>1e-8,sd,torch.ones_like(sd)),channels=len(x),patients=len(d['roles']['train']),valid_windows=windows,
               constant_coordinates=int((sd<=1e-8).sum()),initial_only_not_current_reference=True)
        save(path,q)
    assert q['mean'].shape==q['scale'].shape==(96,) and torch.isfinite(q['mean']).all() and torch.isfinite(q['scale']).all()
    assert q['channels']==len(d['train']) and q['patients']==len(d['roles']['train'])
    del m;gc.collect();torch.cuda.empty_cache();return q


def make(core,fac,records,fold,arm,seed):
    m=initialize(core,fac,records,fold,arm,seed)
    if arm not in REFERENCES:m.set_anchor(read(anchor_path(fold,seed,anchor_kind(arm))))
    return m


def preprocessor(d,scale,arm,fold,seed):
    if arm in REFERENCES:return parent.preprocessor(d,scale,REFERENCES[arm],fold,seed)
    p=component.preprocessor(d,scale,'WO_FEATURE_PR' if arm in ['WO_FEATURE_PR','WO_BOTH_PR'] else 'FULL')
    return dict(summary=p,initial_TRAIN_raw_anchor_sha256=sha(anchor_path(fold,seed,anchor_kind(arm))),anchor_kind=anchor_kind(arm),raw_view_width=192)


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
    assert sha(parent.ROOT/'FROZEN_BEFORE_ASSESS_PRIVATE.json')==q['input']['parent_freeze'] and sha(parent.ROOT/'SCORE_SEAL_PRIVATE.json')==q['input']['parent_seal']
    for r in q['cells']:assert sha(r['checkpoint'])==r['checkpoint_sha256'] and sha(r['calibration'])==r['calibration_sha256']
    return q


CONTRASTS={
 'FULL_MINUS_B0':{'FULL':1,'B0':-1},'FULL_MINUS_PRUNED_REF':{'FULL':1,'PRUNED_REF':-1},
 'FEATURE_PR_WITH_RAW':{'FULL':1,'WO_FEATURE_PR':-1},'RAW_PR_WITH_FEATURE':{'FULL':1,'WO_RAW_PR':-1},
 'BOTH_PR_JOINT':{'FULL':1,'WO_BOTH_PR':-1},'GLOBAL_VIEW':{'FULL':1,'WO_GLOBAL_VIEW':-1},
 'PRE_REFERENCE':{'FULL':1,'WO_PRE_REFERENCE':-1},'PR_INTERACTION':{'FULL':1,'WO_FEATURE_PR':-1,'WO_RAW_PR':-1,'WO_BOTH_PR':1}}
PRIMARY_CONTRASTS=['FULL_MINUS_B0','FULL_MINUS_PRUNED_REF','FEATURE_PR_WITH_RAW','RAW_PR_WITH_FEATURE','GLOBAL_VIEW','PRE_REFERENCE']
PRIMARY_METRICS=['auroc','macro_f1']
