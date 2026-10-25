"""New source-held-out roles; immutable readers/math, never old fitted banks."""
import csv
import gc
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import sys
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
BASE_ROOT=Path(r'D:\drechan_loco_five_models_8seed_v1_runtime')
TRAIN_SEED=int(os.environ.get('LOCO_TRAIN_SEED','45'));assert TRAIN_SEED in [42,43,44,45,46,47,48,49]
ROOT=BASE_ROOT/f'seed{TRAIN_SEED}';PUB=ROOT/'public'
SOURCES=['hup','lzu','multicenter','pediatric']
ARMS=['OURS_RAW_FEATURE','OURS_RAW','OURS_FEATURE','LINEAR_SVM_FEATURE_PR','TIMECONVCNN_RAW']
OURS=dict(OURS_RAW_FEATURE='FULL',OURS_RAW='RAW_ONLY',OURS_FEATURE='PHYSIOLOGY_ONLY')
METRICS=['macro_f1','auroc','auprc','f1','nez_f1','accuracy','mrr','hit_at_1','ez_fraction_bias']
KEEP=np.array([i for i in range(14) if i not in [8,9]])
ISOLATED_DEPS=Path(r'D:\drechan_loco_five_models_seed42_v1_deps')
sys.path.insert(0,str(ISOLATED_DEPS))
ARCHIVE=Path(r'C:\a0_patient_local_label_reliability_seed42_runtime\gate\FEATURES_PRIVATE.npz')
ARCHIVE_SHA='1bdcc479a2bd148bdc94614c8253bea67e4008bd98bc6dad9e8030df827db9bd'

def module(name,path,**kwargs):
    s=importlib.util.spec_from_file_location(name,path,**kwargs);m=importlib.util.module_from_spec(s)
    sys.modules[name]=m;s.loader.exec_module(m);return m

# Import immutable constructors only. DO NOT invoke legacy.setup()/context().
OWNER=sys.modules[__name__]
LEG=module('loco_immutable_parent',Path(r'D:\drechan_no_guidance_ablation_42_44_62_v1\common.py'))
sys.modules['common']=OWNER;sys.path.insert(0,str(HERE))
io=LEG.io;sha,read,save,write,freeze,jr,digest=io.sha,io.read,io.save,io.write,io.freeze,io.jr,io.digest
seed_all,atomic_torch,state_hash=io.seed_all,io.atomic_torch,io.state_hash
FM=LEG.feature_math;EV=LEG.EV
SOURCE=Path(r'E:\DRE-nips\new-pipeline\7-11')
SA=module('loco_untrained_source',SOURCE/'a0_source_posterior_decoder_seed42_v1/code/model.py')
CORE=module('loco_original_selector',SOURCE/'pr_uncertainty_aware_supervision_seed42_v1/code/uas_core.py')
COMPACT=sys.modules[LEG.legacy.OLD.CompactDelta.__module__]
MANIFEST=Path(r'C:\d2_ictal_ablation_seed42_v1_runtime\public\FEATURE_GROUP_MANIFEST.json')
TCROOT=Path(r'D:\m8_raw_backbones_pr_seed42_v1')
module('patient_relative',TCROOT/'patient_relative.py')
module('loco_tc_models',TCROOT/'models/__init__.py',submodule_search_locations=[str(TCROOT/'models')])
TC=importlib.import_module('loco_tc_models.backbones')
TCLOCK=jr(TCROOT/'configs/PROTOCOL_LOCK.json')
sys.modules['common']=OWNER;sys.path.insert(0,str(HERE))
CTX={}

def sources():
    return {p.name:sha(p) for p in sorted(HERE.iterdir()) if p.suffix in ['.py','.json','.ps1'] and p.name!='release_check.py'}

def dependencies():
    result={}
    for m in list(sys.modules.values()):
        p=getattr(m,'__file__',None)
        if not p:continue
        p=Path(p)
        code_path=(p.drive.upper()=='D:' and len(p.parts)>2 and ('seed' in p.parts[1] or p.parts[1]=='drechan_no_guidance_ablation_42_44_62_v1') and 'runtime' not in p.parts[1] and '_env' not in p.parts[1] and '_deps' not in p.parts[1]) or str(p).startswith(str(SOURCE))
        if p.suffix=='.py' and p.is_file() and p.parent!=HERE and code_path:
            # Code paths only, never private runtime artifacts or identities.
            result[str(p)]=sha(p)
    return result

def split_roles(patient,center,k):
    patient=np.asarray(patient).astype(str);center=np.asarray(center).astype(str)
    groups={z:sorted(set(patient[center==z])) for z in SOURCES}
    assert [len(groups[z]) for z in SOURCES]==[36,21,15,8]
    assert sum(map(len,groups.values()))==len(set(patient))==80
    held=SOURCES[k-1];roles={'fit':set(),'validation':set(),'test':set(groups[held])}
    for j,z in enumerate(SOURCES,1):
        if z==held:continue
        ids=np.asarray(groups[z]);order=np.random.default_rng(42+1009*k+53*j).permutation(len(ids))
        n=math.ceil(.2*len(ids));roles['validation'].update(ids[order[:n]]);roles['fit'].update(ids[order[n:]])
    assert [len(roles[r]) for r in ['fit','validation','test']]==[[34,10,36],[46,13,21],[50,15,15],[56,16,8]][k-1]
    assert not any(roles[a]&roles[b] for a,b in [('fit','validation'),('fit','test'),('validation','test')])
    return roles

def labels(indices):
    assert sha(ARCHIVE)==ARCHIVE_SHA
    with np.load(ARCHIVE,allow_pickle=True) as z:return z['y'][indices].astype(np.float32)

def feature_fit(raw,pid,tr):
    raw=np.asarray(raw).copy();raw[:,8:10]=0.
    return FM.prepare(raw,pid,tr)

def feature_query(raw,pid,pre):
    raw=np.asarray(raw).copy();raw[:,8:10]=0.
    return FM.query(raw,pid,pre)

def setup():
    ROOT.mkdir(parents=True,exist_ok=True);PUB.mkdir(exist_ok=True)
    import torchvision
    import PIL
    env=jr(HERE/'ENVIRONMENT_LOCK.json')
    assert torch.__version__==env['torch'] and torchvision.__version__==env['torchvision'] and PIL.__version__==env['pillow']
    files=sorted(p for p in ISOLATED_DEPS.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc')
    tree=[[p.relative_to(ISOLATED_DEPS).as_posix(),sha(p)] for p in files]
    assert len(tree)==env['isolated_dependency_files'] and hashlib.sha256(json.dumps(tree,separators=(',',':')).encode()).hexdigest()==env['isolated_dependency_tree_sha256']
    assert sha(io.BASE/'RECONSTRUCTION_PRIVATE.pkl')==io.EXPECTED_REC
    assert sha(io.BASE/'RAW_DYNAMIC_REGISTRY_PRIVATE.pkl')==io.EXPECTED_REG
    assert sha(ARCHIVE)==ARCHIVE_SHA
    assert sha(MANIFEST)=='f7869e994e8494397761e4efa0a2a2057c2a13541331e5933e94c6f7b82a065f'
    rec=read(io.BASE/'RECONSTRUCTION_PRIVATE.pkl');meta=rec['meta']
    assert rec['binding']['archive']==ARCHIVE_SHA and rec['R14'].shape==(7635,14)
    assert len(set(zip(meta['patient'],meta['channel'])))==7635 and len(set(meta['patient']))==80
    assert sum(len(q['records']) for q in rec['ledgers'])==256
    with np.load(ARCHIVE,allow_pickle=True) as z:
        for name in ['patient','channel','center']:assert np.array_equal(z[name],meta[name])
    lock=jr(HERE/'PROTOCOL_LOCK.json');assert lock['arms']==ARMS and lock['sources']==SOURCES
    torch.set_num_threads(2);torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    binding=dict(train_seed=TRAIN_SEED,split_seed=42,source=sources(),dependencies=dependencies(),reconstruction=io.EXPECTED_REC,
                 raw_registry=io.EXPECTED_REG,archive=ARCHIVE_SHA,torch=torch.__version__,numpy=np.__version__)
    freeze(ROOT/'INPUT_FREEZE_PRIVATE.json',binding)
    CTX.update(binding=binding,rec=rec,records=COMPACT.records_from_manifest(jr(MANIFEST)))
    roles=[];banks=[]
    for k in range(1,5):
        r=split_roles(meta['patient'],meta['center'],k);roles.append(r)
        ix=np.flatnonzero(np.isin(meta['patient'],sorted(r['fit']|r['validation'])))
        pid=meta['patient'][ix];tr=np.flatnonzero(np.isin(pid,sorted(r['fit'])));va=np.flatnonzero(np.isin(pid,sorted(r['validation'])))
        x,pre=feature_fit(rec['R14'][ix],pid,tr)
        assert set(pre['fit_patient_ids'])==r['fit'] and not x[:,8:10].any()
        banks.append(dict(patient=pid,channel=meta['channel'][ix],center=meta['center'][ix],
                          archive_indices=ix,x=x,pre=pre,y=labels(ix),train=tr,val=va,roles=r,raw=rec['R14'][ix]))
    role_export=[{a:sorted(v) for a,v in r.items()} for r in roles]
    freeze(ROOT/'ROLES_PRIVATE.json',dict(binding=binding,roles=role_export))
    CTX.update(roles=roles,banks=banks)
    return binding,rec,banks

def make_source(k):
    s=TRAIN_SEED+1009*k;seed_all(s)
    template=LEG.legacy.OLD.P3.MorphCorrection(COMPACT.CompactDelta(SA.SourceA0(s).network,CTX['records'])).cuda()
    source=LEG.original.legacy_parent.old.MODEL.SingleHead('F0_AFTER_GELU',template.cnn,template.gate,type(template).features,s+170003).cuda()
    del template;return source

def make(arm,k):
    if arm=='TIMECONVCNN_RAW':
        seed_all(TRAIN_SEED+1009*k);return TC.RawBackbone('TimeConvCNN',False,TCLOCK).cuda()
    assert arm in OURS
    source=make_source(k);m=LEG.MODEL.NoGuidanceF0(source,OURS[arm],TRAIN_SEED+1009*k+170003).cuda();del source
    assert not m.early and not m.late and not m.interaction and m.head[0].out_features==64
    return m

def rng_state():return dict(py=random.getstate(),np=np.random.get_state(),cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state())
def restore_rng(q):
    random.setstate(q['py']);np.random.set_state(q['np']);torch.set_rng_state(q['cpu'].cpu());torch.cuda.set_rng_state(q['cuda'].cpu())

def forward(m,arm,d,ii,store,train=False,pos=None):
    from timeconv_engine import forward as tc_forward
    assert len(set(d['patient'][ii]))==1
    if arm=='OURS_FEATURE':return m(torch.as_tensor(d['x'][ii],device='cuda'))['logit']
    raw,mask,times=store.get(str(d['patient'][ii[0]]),d['channel'][ii].astype(str).tolist(),'cuda')
    if arm=='TIMECONVCNN_RAW':
        target=torch.as_tensor(d['y'][ii],device='cuda',dtype=torch.float32) if train else None
        return tc_forward(m,raw,mask,128,train,target,pos)
    return m(torch.as_tensor(d['x'][ii],device='cuda'),raw,mask,times,None,None)['logit']

def predict(m,arm,d,ix,store):
    m.eval();score=np.empty(len(ix),np.float32)
    with torch.inference_mode():
        for pid in sorted(set(d['patient'][ix])):
            jj=np.flatnonzero(d['patient'][ix]==pid);v=forward(m,arm,d,ix[jj],store)
            if arm=='TIMECONVCNN_RAW':v=v[0]
            score[jj]=v.cpu().numpy()
    assert np.isfinite(score).all();return score

def pn(logit):return torch.sigmoid(torch.as_tensor(logit,dtype=torch.float32)).numpy()
def select(arm,d,logit):
    va=d['val']
    if arm in OURS:
        q=CORE.select_threshold(d['y'][va],pn(logit),d['patient'][va])
        return dict(threshold=float(q['threshold']),macro_f1=float(q['macro_f1']),ez_f1=float(q['ez_f1']),rule='NEZ_LT')
    p=logit if arm=='LINEAR_SVM_FEATURE_PR' else 1-pn(logit)
    q=EV.threshold(1-d['y'][va],p,d['patient'][va])
    return dict(threshold=float(q['threshold']),macro_f1=float(q['validation_macro_f1']),ez_f1=float(q['validation_positive_f1']),rule='EZ_GE')

def csv_out(name,rows):
    assert rows and all(set(r)==set(rows[0]) for r in rows)
    p=PUB/(name+'.csv');p.parent.mkdir(exist_ok=True);tmp=p.with_suffix('.csv.partial')
    with tmp.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    tmp.replace(p)

def frozen():
    assert jr(BASE_ROOT/'ALL160_SELECTIONS_FROZEN_PRIVATE.json')['status']=='ALL160_FROZEN_BEFORE_NEW_TEST'
    q=jr(ROOT/'FROZEN_BEFORE_TEST_PRIVATE.json')
    assert q['binding']==CTX['binding'] and len(q['cells'])==20
    assert {(r['arm'],r['held_index']) for r in q['cells']}=={(a,k) for a in ARMS for k in range(1,5)}
    for r in q['cells']:
        assert sha(r['checkpoint'])==r['checkpoint_sha256'] and sha(r['validation'])==r['validation_sha256']
    return q
