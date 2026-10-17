"""Pinned ownFIT roles plus all actually available stronger Raw references."""
import importlib.util
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

HERE=Path(__file__).resolve().parent
ROOT=Path(r'D:\raw_fresh_temporal_macro_seed42_v1_runtime');PUB=ROOT/'public'
BASE_CODE=Path(r'D:\raw_fresh_energy_macro_seed42_v1')
BASE=Path(r'D:\raw_fresh_energy_macro_seed42_v1_runtime')
PRESTATE=Path(r'D:\raw_prestate_contrast_macro_seed42_v1_runtime')
FIT_ENGINE=Path(r'D:\raw_joint_moment_macro_seed42_v1\train_internal.py')


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m


previous=module('fresh_temporal_pinned_roles',BASE_CODE/'common.py')
io,parent,wave=previous.io,previous.parent,previous.wave
sha,read,save,write,freeze,jr,digest=io.sha,io.read,io.save,io.write,io.freeze,io.jr,io.digest
state_hash,atomic_torch,seed_all=io.state_hash,io.atomic_torch,io.seed_all
state_cpu,rng,restore=previous.state_cpu,previous.rng,previous.restore
RawStore=io.RawStore;METRICS=previous.METRICS;probability=previous.probability
calibrate,held_rows=previous.calibrate,previous.held_rows
REFERENCES=['NATIVE','FROZEN_PRE','COMPACT_PRE','PRIOR_PRESTATE']
TRAINED=['FRESH_BASE','FRESH_POINTWISE','FRESH_TEMPORAL'];ARMS=REFERENCES+TRAINED
CRITICAL=['PROTOCOL_LOCK.json','common.py','model.py','test_model.py','admit.py','train_internal.py','summarize.py','run_internal.ps1']


def setup():
    ROOT.mkdir(exist_ok=True);PUB.mkdir(exist_ok=True);old=previous.setup()
    assert sha(FIT_ENGINE)=='72d117c68785c5c020eda3d41689ad756a588c0cec6b1006e2b5e11736e6802a'
    assert sha(BASE/'INTERNAL_RESULTS_SEAL_PRIVATE.json')=='7ba9c5fb3c1577fb7db42dd2bf4030687cb391757ce7bf3b5bf6c223f0ca461a'
    assert sha(PRESTATE/'INTERNAL_RESULTS_SEAL_PRIVATE.json')=='a4d05159a96c6fc10aa93f21bf51fa546e7d6402754f3b85c4afb893945e784a'
    assert jr(PRESTATE/'public/INDEPENDENT_INTERNAL_VERIFICATION.json')['status']=='PASS'
    protected=Path(r'D:\raw_joint_moment_macro_seed42_v1_runtime\final_stage_v1')
    assert sha(protected/'OUTER_SCORE_SEAL_PRIVATE.json')=='476a96a1b4d200002238f345f6a716e46ef0be955e1369ea2aaf6d50a3193cc1'
    binding=dict(source={n:sha(HERE/n) for n in CRITICAL},prior_source_input=digest(old),fit_engine_sha256=sha(FIT_ENGINE),
        strong_reference_seal=sha(BASE/'INTERNAL_RESULTS_SEAL_PRIVATE.json'),prestate_reference_seal=sha(PRESTATE/'INTERNAL_RESULTS_SEAL_PRIVATE.json'),
        protected_old_Outer=sha(protected/'OUTER_SCORE_SEAL_PRIVATE.json'),raw_registry=io.EXPECTED_REG,reconstruction=io.EXPECTED_REC,
        role_registry=io.EXPECTED_ENCODERS,pure_source=sha(parent.REFERENCE/'pure_model.py'),
        normalizer_source=sha(previous.WAVE_CODE/'model.py'),torch=torch.__version__,numpy=np.__version__,initialization='RANDOM_NO_WARMSTART')
    freeze(ROOT/'INPUT_FREEZE_PRIVATE.json',binding)
    torch.set_num_threads(2);torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    return binding


def contexts():
    result=previous.contexts();oldseal=jr(BASE/'INTERNAL_RESULTS_SEAL_PRIVATE.json');ps=jr(PRESTATE/'INTERNAL_RESULTS_SEAL_PRIVATE.json')
    for ctx in result:
        d,rr=ctx['data'],ctx['roles'];groups=[set(d['patient'][rr[k]]) for k in ['train','cal','choice','held']]
        assert all(groups[i].isdisjoint(groups[j]) for i in range(4) for j in range(i))
        ctx['extra_references']={}
        for arm,root,seal,oldarm in [('COMPACT_PRE',BASE,oldseal,'PRE_MEAN'),('PRIOR_PRESTATE',PRESTATE,ps,'PRE_DELTA')]:
            matches=[r for r in seal['cells'] if (r['fold'],r['screen'],r['arm'])==(ctx['fold'],ctx['screen'],oldarm)]
            if not matches:continue
            assert len(matches)==1
            path=root/'models_private'/f"fold{ctx['fold']}_screen{ctx['screen']}"/oldarm/'HELD_PRIVATE.pkl'
            assert sha(path)==matches[0]['sha256'];q=read(path)
            assert np.array_equal(q['archive_indices'],d['archive_indices'][rr['held']])
            assert q['binding']['role_representation_sha256']==ctx['representation_sha256']
            ctx['extra_references'][arm]=dict(logit=q['logit_nez'].copy(),threshold=q['summary']['threshold'],score_sha256=sha(path))
        if ctx['screen']==1:assert set(ctx['extra_references'])==set(REFERENCES[2:])
    return result


def reference_names(ctx):return REFERENCES[:2]+[a for a in REFERENCES[2:] if a in ctx['extra_references']]


def reference(ctx,arm):
    if arm=='NATIVE':return ctx['native_logits'],ctx['native_threshold']
    if arm=='FROZEN_PRE':return ctx['pre_logits'],ctx['pre_threshold']
    assert arm in REFERENCES[2:] and arm in ctx['extra_references']
    return ctx['extra_references'][arm]['logit'],ctx['extra_references'][arm]['threshold']
