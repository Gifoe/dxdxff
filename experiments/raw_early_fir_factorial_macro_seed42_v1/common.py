"""Exact ownFIT data, fixed references and genuinely reused short-mean cells."""
import importlib.util
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

HERE=Path(__file__).resolve().parent
ROOT=Path(r'D:\raw_early_fir_factorial_macro_seed42_v1_runtime');PUB=ROOT/'public'
BASE_CODE=Path(r'D:\raw_fresh_energy_macro_seed42_v1');BASE=Path(r'D:\raw_fresh_energy_macro_seed42_v1_runtime')
PRESTATE=Path(r'D:\raw_prestate_contrast_macro_seed42_v1_runtime')
RECENT=Path(r'D:\raw_fresh_temporal_macro_seed42_v1_runtime')
FIT_ENGINE=Path(r'D:\raw_joint_moment_macro_seed42_v1\train_internal.py')


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m


previous=module('early_fir_exact_roles',BASE_CODE/'common.py')
io,parent,wave=previous.io,previous.parent,previous.wave
sha,read,save,write,freeze,jr,digest=io.sha,io.read,io.save,io.write,io.freeze,io.jr,io.digest
state_hash,atomic_torch,seed_all=io.state_hash,io.atomic_torch,io.seed_all
state_cpu,rng,restore=previous.state_cpu,previous.rng,previous.restore
RawStore=io.RawStore;METRICS=previous.METRICS;probability=previous.probability
calibrate,held_rows=previous.calibrate,previous.held_rows
REFERENCES=['NATIVE','FROZEN_PRE','LEGACY_PRE_MEAN','PRIOR_PRESTATE']
FAMILY=['SHORT_MEAN','SHORT_ENERGY','LONG_MEAN','LONG_ENERGY']
NOVEL=['SHORT_ENERGY','LONG_MEAN','LONG_ENERGY'];TRAINED=FAMILY;ARMS=REFERENCES+FAMILY
CRITICAL=['PROTOCOL_LOCK.json','common.py','model.py','test_model.py','admit.py','train_internal.py','summarize.py','run_internal.ps1']


def setup():
    ROOT.mkdir(exist_ok=True);PUB.mkdir(exist_ok=True);old=previous.setup()
    assert sha(FIT_ENGINE)=='72d117c68785c5c020eda3d41689ad756a588c0cec6b1006e2b5e11736e6802a'
    expected=[(BASE,'7ba9c5fb3c1577fb7db42dd2bf4030687cb391757ce7bf3b5bf6c223f0ca461a'),
        (PRESTATE,'a4d05159a96c6fc10aa93f21bf51fa546e7d6402754f3b85c4afb893945e784a'),
        (RECENT,'7ab66d2b6bdbb9826bc57fa67bbc4373b6fd76f1401ddbeb31a4120b8cc19b7e')]
    for root,wanted in expected:
        assert sha(root/'INTERNAL_RESULTS_SEAL_PRIVATE.json')==wanted
        assert jr(root/'public/INDEPENDENT_INTERNAL_VERIFICATION.json')['status']=='PASS'
    protected=Path(r'D:\raw_joint_moment_macro_seed42_v1_runtime\final_stage_v1\OUTER_SCORE_SEAL_PRIVATE.json')
    assert sha(protected)=='476a96a1b4d200002238f345f6a716e46ef0be955e1369ea2aaf6d50a3193cc1'
    binding=dict(source={n:sha(HERE/n) for n in CRITICAL},prior_source_input=digest(old),fit_engine_sha256=sha(FIT_ENGINE),
        prior_internal_seals={r.name:sha(r/'INTERNAL_RESULTS_SEAL_PRIVATE.json') for r,_ in expected},protected_old_Outer=sha(protected),
        raw_registry=io.EXPECTED_REG,reconstruction=io.EXPECTED_REC,role_registry=io.EXPECTED_ENCODERS,
        pure_source=sha(parent.REFERENCE/'pure_model.py'),normalizer_source=sha(previous.WAVE_CODE/'model.py'),
        torch=torch.__version__,numpy=np.__version__,initialization='FRESH_RANDOM_CORE_ZERO_OUTER_EXTENSION_NO_WARMSTART')
    freeze(ROOT/'INPUT_FREEZE_PRIVATE.json',binding)
    torch.set_num_threads(2);torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    return binding


def source_cell(root,seal,ctx,arm):
    matches=[r for r in seal['cells'] if (r['fold'],r['screen'],r['arm'])==(ctx['fold'],ctx['screen'],arm)]
    if not matches:return None
    assert len(matches)==1
    hp=root/'models_private'/f"fold{ctx['fold']}_screen{ctx['screen']}"/arm/'HELD_PRIVATE.pkl';bp=hp.with_name('BEST_PRIVATE.pt')
    assert sha(hp)==matches[0]['sha256'];q=read(hp)
    d,rr=ctx['data'],ctx['roles'];assert np.array_equal(q['archive_indices'],d['archive_indices'][rr['held']])
    assert np.array_equal(q['cal_archive_indices'],d['archive_indices'][rr['cal']])
    assert q['binding']['role_representation_sha256']==ctx['representation_sha256'] and sha(bp)==q['summary']['checkpoint_sha256']
    return dict(hp=hp,bp=bp,q=q,source_arm=arm,score_sha256=sha(hp),checkpoint_sha256=sha(bp))


def contexts():
    result=previous.contexts();seals={r:jr(r/'INTERNAL_RESULTS_SEAL_PRIVATE.json') for r in [BASE,PRESTATE,RECENT]}
    for ctx in result:
        d,rr=ctx['data'],ctx['roles'];groups=[set(d['patient'][rr[k]]) for k in ['train','cal','choice','held']]
        assert all(groups[i].isdisjoint(groups[j]) for i in range(4) for j in range(i))
        ctx['extra_references']={}
        for name,root,oldarm in [('LEGACY_PRE_MEAN',BASE,'PRE_MEAN'),('PRIOR_PRESTATE',PRESTATE,'PRE_DELTA')]:
            cell=source_cell(root,seals[root],ctx,oldarm)
            if cell is not None:ctx['extra_references'][name]=cell
        ctx['reused_short_mean']=source_cell(RECENT,seals[RECENT],ctx,'FRESH_BASE')
        if ctx['screen']==1:
            assert set(ctx['extra_references'])==set(REFERENCES[2:]) and ctx['reused_short_mean'] is not None
    return result


def reference_names(ctx):return REFERENCES[:2]+[a for a in REFERENCES[2:] if a in ctx['extra_references']]


def reference(ctx,arm):
    if arm=='NATIVE':return ctx['native_logits'],ctx['native_threshold']
    if arm=='FROZEN_PRE':return ctx['pre_logits'],ctx['pre_threshold']
    q=ctx['extra_references'][arm]['q'];return q['logit_nez'],q['summary']['threshold']


def factorial_values(sm,se,lm,le):
    return dict(kernel_main=.5*((lm-sm)+(le-se)),energy_main=.5*((se-sm)+(le-lm)),interaction=(le-lm)-(se-sm))
