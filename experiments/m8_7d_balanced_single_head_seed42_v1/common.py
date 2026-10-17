"""Pinned original seven-feature input and original patient-held-out roles."""
import importlib.util
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

HERE=Path(__file__).resolve().parent
ROOT=Path(r'D:\m8_7d_balanced_single_head_seed42_v1_runtime');PUB=ROOT/'public'
OLD=Path(r'D:\m8_single_head_concat_seed42_v1');OLDROOT=Path(r'D:\m8_single_head_concat_seed42_v1_runtime')


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m


io=module('balanced_single_safe_io',Path(r'D:\raw_latent_contrast_optimization_seed42_v1\common.py'))
assert io.sha(Path(io.__file__))=='c5dd9fb82fab45dd3b1e4ae0911f642a221fbf5bd039a5a642f838a3a5ffa499'
MODEL=module('balanced_single_model',HERE/'model.py')
legacy=module('balanced_single_original',OLD/'common.py')
sys.path.insert(0,str(HERE))
sha,read,save,write,freeze,jr,digest=io.sha,io.read,io.save,io.write,io.freeze,io.jr,io.digest
atomic_torch,state_hash,seed_all=io.atomic_torch,io.state_hash,io.seed_all
ARMS=['C0_CONCAT','C1_BALANCED','C2_INTERACTION'];NEW=ARMS[1:]
METRICS=['macro_f1','auroc','auprc','f1','nez_f1','accuracy','mrr','hit_at_1','ez_fraction_bias']
CRITICAL=['PROTOCOL_LOCK.json','common.py','model.py','test_model.py','admit.py','train.py','verify_internal.py','run_internal.ps1']
EV=module('balanced_single_metrics',Path(r'D:\m8_raw_backbones_pr_seed42_v1\evaluate.py'))


def setup():
    ROOT.mkdir(exist_ok=True);PUB.mkdir(exist_ok=True)
    own=sys.modules[__name__]
    try:_,_,core,fac,_,records,rec,banks,prior=legacy.OLD.setup_new()
    finally:sys.modules['common']=own;sys.modules['model']=MODEL;sys.path.insert(0,str(HERE))
    assert sha(io.BASE/'RECONSTRUCTION_PRIVATE.pkl')==io.EXPECTED_REC and sha(io.BASE/'RAW_DYNAMIC_REGISTRY_PRIVATE.pkl')==io.EXPECTED_REG
    old=legacy.read(OLDROOT/'MODEL_FREEZE_PRIVATE.pkl')
    for name,h in old['binding']['source'].items():assert sha(OLD/name)==h
    assert sha(OLDROOT/'FROZEN_BEFORE_OUTER_PRIVATE.json')=='34d95ce017d4b3993bc284a88834db367c646d3e154d148e2ff2a1eccacb8484'
    assert sha(OLDROOT/'OUTER_PREDICTIONS_SEAL_PRIVATE.json')=='7ad3427cfe4b2c26077c4ac2350c2b87463d592df31d3376ff899a070937b5c9'
    binding=dict(source={n:sha(HERE/n) for n in CRITICAL},old_model_freeze=sha(OLDROOT/'MODEL_FREEZE_PRIVATE.pkl'),
        original_input=old['binding'],prior=prior,raw_registry=io.EXPECTED_REG,reconstruction=io.EXPECTED_REC,
        old_Outer_protected=sha(OLDROOT/'OUTER_PREDICTIONS_SEAL_PRIVATE.json'),torch=torch.__version__,numpy=np.__version__)
    freeze(ROOT/'INPUT_FREEZE_PRIVATE.json',binding)
    assert rec['R14'].shape==(7635,14) and len(set(rec['meta']['patient']))==80
    torch.set_num_threads(2);torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    return binding,core,fac,records,rec,banks,old


def make(core,fac,records,fold,arm):
    # This constructor has no trained state argument; the discarded template is untrained.
    source=legacy.template(core,fac,records,fold)
    m=MODEL.SingleHead(arm,source.cnn,source.gate,type(source).features,42+1009*fold+170003).cuda()
    del source;return m


def bank(fold,banks):return legacy.bank(fold,banks)
def scale(fold):return legacy.scale(fold)
def store():
    q=legacy.OLD.P3.Inputs(legacy.OLD.BASELINE,Path(r'D:\dgm_net_seed42_v1_runtime'),ROOT)
    q.arm='M1_SCALED_DGM';return q


def predict(m,d,ix,inputs,ss):return legacy.predict(m,d,ix,inputs,ss)
def probability(x):return torch.sigmoid(torch.as_tensor(x,dtype=torch.float32)).numpy()


def patient_rows(d,ix,logit,theta,arm,fold):
    pn=probability(logit);rows=[]
    for pid in sorted(set(d['patient'][ix])):
        j=d['patient'][ix]==pid;y=1-d['y'][ix][j];p=1-pn[j].astype(float);pred=pn[j]<theta
        rows.append(dict(arm=arm,fold=fold,patient=str(pid),center=str(d['center'][ix][j][0]),
            **EV.classification(y,p,pred),**EV.ranking(y,p,d['channel'][ix][j])))
    return rows


def cluster(rows):
    f=pd.DataFrame(rows);ids=sorted(f.patient.unique());counts=np.array([int((f.patient==p).sum()) for p in ids])
    sums=np.stack([f[f.patient==p][METRICS].sum().to_numpy(float) for p in ids]);draw=np.random.default_rng(42).integers(0,len(ids),(10000,len(ids)))
    samples=sums[draw].sum(1)/counts[draw].sum(1)[:,None];ci=np.quantile(samples,[.025,.975],axis=0)
    return [dict(metric=m,mean=float(f[m].mean()),low=float(ci[0,j]),high=float(ci[1,j]),appearances=len(f),unique_patients=len(ids),draws=10000) for j,m in enumerate(METRICS)]


def choose(frame):
    scores=[]
    for arm,g in frame.groupby('arm'):
        fold=g.groupby('fold').macro_f1.mean();scores.append(dict(arm=arm,macro=float(g.macro_f1.mean()),auroc=float(g.auroc.mean()),
            auprc=float(g.auprc.mean()),fold_sd=float(fold.std(ddof=1)),penalized=float(g.macro_f1.mean()-.1*fold.std(ddof=1))))
    base=next(r for r in scores if r['arm']=='C0_CONCAT')
    eligible=[r for r in scores if r['arm'] in NEW and r['macro']>=base['macro']+.003 and r['auroc']>=base['auroc']-.01 and r['auprc']>=base['auprc']-.01]
    selected=min(eligible,key=lambda r:(-r['penalized'],ARMS.index(r['arm'])))['arm'] if eligible else 'C0_CONCAT'
    return dict(scores=scores,selected=selected,new_qualified=bool(eligible),eligible=[r['arm'] for r in eligible])
