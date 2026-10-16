"""Unchanged Raw and ownFIT role inputs; no manual feature predictor path."""
import importlib.util
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

HERE=Path(__file__).resolve().parent
ROOT=Path(r'D:\raw_temporal_context_macro_seed42_v1_runtime');PUB=ROOT/'public'
WAVE_CODE=Path(r'D:\raw_precompand_macro_seed42_v1');WAVE=Path(r'D:\raw_precompand_macro_seed42_v1_runtime')


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m


wave=module('temporal_fixed_raw_roles',WAVE_CODE/'common.py');parent=wave.parent;io=wave.io
sha,read,save,write,freeze,jr,digest=io.sha,io.read,io.save,io.write,io.freeze,io.jr,io.digest
state_hash,atomic_torch,seed_all=io.state_hash,io.atomic_torch,io.seed_all
RawStore=io.RawStore;METRICS=wave.METRICS
probability,patient_rows=parent.probability,parent.patient_rows
ARMS=['NATIVE','FROZEN_PRE','PRE_TUNE','CHANNEL_MLP','TEMPORAL_CONTEXT'];TRAINED=ARMS[2:]
CRITICAL=['PROTOCOL_LOCK.json','common.py','model.py','test_model.py','admit.py','train_internal.py','summarize.py']


def setup():
    ROOT.mkdir(exist_ok=True);PUB.mkdir(exist_ok=True);previous=wave.setup()
    assert jr(WAVE/'public/INDEPENDENT_INTERNAL_VERIFICATION.json')['status']=='PASS'
    assert sha(WAVE/'INTERNAL_RESULTS_SEAL_PRIVATE.json')=='cdc2932bf0c977533e01f008bdd7b38305a048cd87b88fb2030ca5eec352b3cd'
    protected=Path(r'D:\raw_patient_decision_macro_seed42_v1_runtime\final_stage_v1')
    assert sha(protected/'OUTER_SCORE_SEAL_PRIVATE.json')=='fc5ac982885e57f01d68ed64d95669dc0e5b2920b7b6bd3422a08801572d1353'
    assert jr(protected/'public/INDEPENDENT_FINAL_VERIFICATION.json')['status']=='PASS'
    binding=dict(source={n:sha(HERE/n) for n in CRITICAL},previous_input=digest(previous),
        previous_inner_score_seal=sha(WAVE/'INTERNAL_RESULTS_SEAL_PRIVATE.json'),protected_previous_Outer=sha(protected/'OUTER_SCORE_SEAL_PRIVATE.json'),
        dependencies={n:sha(WAVE_CODE/n) for n in ['common.py','model.py']},raw_registry=io.EXPECTED_REG,
        reconstruction=io.EXPECTED_REC,inner_encoders=io.EXPECTED_ENCODERS,torch=torch.__version__,numpy=np.__version__)
    freeze(ROOT/'INPUT_FREEZE_PRIVATE.json',binding)
    torch.set_num_threads(2);torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    return binding


def contexts():
    result=wave.contexts();seal=jr(WAVE/'INTERNAL_RESULTS_SEAL_PRIVATE.json')
    for ctx in result:
        cell=WAVE/'matrices_private'/f"fold{ctx['fold']}_screen{ctx['screen']}"/'PRECOMPAND_RESIDUAL'
        p=cell/'HELD_PRIVATE.pkl';best=cell/'BEST_PRIVATE.pt';q=read(p)
        expected=next(r for r in seal['cells'] if (r['fold'],r['screen'],r['arm'])==(ctx['fold'],ctx['screen'],'PRECOMPAND_RESIDUAL'))
        assert sha(p)==expected['sha256'] and jr(cell/'COMPLETE_PRIVATE.json')['best_sha256']==sha(best)
        state=torch.load(best,map_location='cpu',weights_only=False)
        assert state['binding']==q['binding'] and q['binding']['body_checkpoint_sha256']==ctx['stored']['checkpoint_sha256']
        d=ctx['data'];rr=ctx['roles']
        assert np.array_equal(q['archive_indices'],d['archive_indices'][rr['held']]) and np.array_equal(q['cal_archive_indices'],d['archive_indices'][rr['cal']])
        ctx.update(matrix=state['matrix'],pre_result=q,pre_matrix_sha256=sha(best),pre_scores_sha256=sha(p))
    return result


def calibrate(d,ix,logit):return wave.calibrate(d,ix,logit)
def held_rows(ctx,logit,theta,arm):return wave.held_rows(ctx,logit,theta,arm)
def state_cpu(model):return {n:v.detach().cpu().clone() for n,v in model.state_dict().items()}
def rng():return torch.get_rng_state(),torch.cuda.get_rng_state_all()
def restore(r):torch.set_rng_state(r[0].cpu());torch.cuda.set_rng_state_all([x.cpu() for x in r[1]])
