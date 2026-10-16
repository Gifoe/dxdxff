"""Immutable Raw neural score/role inputs; no physiological feature predictors."""
import importlib.util
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

HERE=Path(__file__).resolve().parent
ROOT=Path(r'D:\raw_patient_decision_macro_seed42_v1_runtime')
PUB=ROOT/'public'
WAVE_CODE=Path(r'D:\raw_precompand_macro_seed42_v1')
WAVE=Path(r'D:\raw_precompand_macro_seed42_v1_runtime')


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m


wave=module('raw_decision_pinned_wave_io',WAVE_CODE/'common.py')
io=wave.io
sha,read,save,write,freeze,jr,digest=io.sha,io.read,io.save,io.write,io.freeze,io.jr,io.digest
METRICS=wave.METRICS
BODIES=['NATIVE','PRECOMPAND_RESIDUAL']
DECODERS=['ORIGINAL','MARGIN_GRID','PATIENT_Z','TOP_FRACTION']
PROCEDURES=[b+'__'+d for b in BODIES for d in DECODERS]
CRITICAL=['PROTOCOL_LOCK.json','common.py','decoders.py','internal.py','summarize.py','test_decoders.py']


def setup():
    ROOT.mkdir(exist_ok=True);PUB.mkdir(exist_ok=True)
    previous=wave.setup()
    assert jr(WAVE/'public/INDEPENDENT_INTERNAL_VERIFICATION.json')['status']=='PASS'
    assert jr(WAVE/'final_stage_v1/public/INDEPENDENT_FINAL_VERIFICATION.json')['status']=='PASS'
    assert sha(WAVE/'INTERNAL_RESULTS_SEAL_PRIVATE.json')=='cdc2932bf0c977533e01f008bdd7b38305a048cd87b88fb2030ca5eec352b3cd'
    assert sha(WAVE/'final_stage_v1/MODEL_FREEZE_PRIVATE.pkl')=='4f936da78565bf823f0905a8e7fb6ba6da09f13c8b873c4cff4719e5c6319a9c'
    assert sha(WAVE/'final_stage_v1/OUTER_SCORE_SEAL_PRIVATE.json')=='2efe5fe40dbe327c8bfcec1869c214b2c41e2294133288a9f6bed5b46b46298f'
    binding=dict(source={n:sha(HERE/n) for n in CRITICAL},body_input_sha256=digest(previous),
        body_internal_seal=sha(WAVE/'INTERNAL_RESULTS_SEAL_PRIVATE.json'),body_final_model_freeze=sha(WAVE/'final_stage_v1/MODEL_FREEZE_PRIVATE.pkl'),
        protected_previous_Outer_seal=sha(WAVE/'final_stage_v1/OUTER_SCORE_SEAL_PRIVATE.json'),
        raw_registry=io.EXPECTED_REG,reconstruction=io.EXPECTED_REC,inner_encoders=io.EXPECTED_ENCODERS,
        threshold_source=sha(wave.parent.corepath),torch=torch.__version__,numpy=np.__version__)
    freeze(ROOT/'INPUT_FREEZE_PRIVATE.json',binding)
    return binding


def contexts():return wave.contexts()


def body_scores(ctx,body):
    d=ctx['data'];rr=ctx['roles'];cal,held=rr['cal'],rr['held']
    if body=='NATIVE':
        return dict(cal=ctx['stored']['raw_logit_nez'][cal].copy(),held=ctx['stored']['raw_logit_nez'][held].copy(),
                    theta=ctx['reference_threshold'],source=dict(representation=ctx['representation_sha256'],checkpoint=ctx['stored']['checkpoint_sha256']))
    seal=jr(WAVE/'INTERNAL_RESULTS_SEAL_PRIVATE.json')
    entry=next(q for q in seal['cells'] if (q['fold'],q['screen'],q['arm'])==(ctx['fold'],ctx['screen'],body))
    cell=WAVE/'matrices_private'/f"fold{ctx['fold']}_screen{ctx['screen']}"/body
    hp=cell/'HELD_PRIVATE.pkl';bp=cell/'BEST_PRIVATE.pt';assert sha(hp)==entry['sha256']
    q=read(hp);complete=jr(cell/'COMPLETE_PRIVATE.json')
    assert complete['held_sha256']==sha(hp) and complete['best_sha256']==sha(bp)
    assert q['binding']['body_checkpoint_sha256']==ctx['stored']['checkpoint_sha256']
    assert np.array_equal(q['archive_indices'],d['archive_indices'][held]) and np.array_equal(q['cal_archive_indices'],d['archive_indices'][cal])
    return dict(cal=q['cal_logit_nez'].copy(),held=q['logit_nez'].copy(),theta=q['summary']['threshold'],
                source=dict(prediction=sha(hp),matrix=sha(bp),checkpoint=ctx['stored']['checkpoint_sha256']))


def probability(logit):return torch.sigmoid(torch.as_tensor(logit,dtype=torch.float32)).numpy()


def interval(rows):return wave.parent.cluster_interval(rows)
