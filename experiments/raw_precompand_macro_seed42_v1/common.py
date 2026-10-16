"""Pinned Raw/role reader; only ownFIT metadata and fixed source bodies."""
import importlib.util
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

HERE=Path(__file__).resolve().parent
ROOT=Path(r'D:\raw_precompand_macro_seed42_v1_runtime')
PUB=ROOT/'public'
PARENT=Path(r'D:\raw_macro_f1_head_screen_seed42_v1')


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m


parent=module('precompand_pinned_ownFIT_metadata',PARENT/'common.py')
assert parent.sha(PARENT/'common.py')=='b84bbdb931c704543ec13c0f56f00cc7f668400730c4dfcf8cac4fc1dcf6b039'
io,core=parent.io,parent.core
sha,read,save,write,freeze,jr,digest=io.sha,io.read,io.save,io.write,io.freeze,io.jr,io.digest
state_hash,atomic_torch=io.state_hash,io.atomic_torch
RawStore=io.RawStore
probability,patient_rows=parent.probability,parent.patient_rows
METRICS=parent.METRICS
CRITICAL=['PROTOCOL_LOCK.json','common.py','model.py','prepare.py','train.py','summarize.py','test_model.py']


def setup():
    ROOT.mkdir(exist_ok=True);PUB.mkdir(exist_ok=True)
    assert sha(io.BASE/'RECONSTRUCTION_PRIVATE.pkl')==io.EXPECTED_REC
    assert sha(io.BASE/'RAW_DYNAMIC_REGISTRY_PRIVATE.pkl')==io.EXPECTED_REG
    assert sha(parent.ENC/'ENCODER_REGISTRY_PRIVATE.pkl')==io.EXPECTED_ENCODERS
    target=parent.REFERENCE/'TARGET_DEFINITION.json'
    assert sha(target)=='e069078dcda4352e7026d5de2fbcf0e8e12df93da8869ff088c8cff7f1b3b9e6'
    protected=Path(r'D:\raw_latent_dispersion_macro_seed42_v1_runtime')
    assert jr(protected/'public/INDEPENDENT_INTERNAL_VERIFICATION.json')['status']=='PASS'
    assert sha(protected/'INTERNAL_RESULTS_SEAL_PRIVATE.json')=='8496913d168fae887ae2dab59566efc680cfb2660d3f1b27048705f5408fe803'
    binding=dict(source={n:sha(HERE/n) for n in CRITICAL},raw_registry=io.EXPECTED_REG,reconstruction=io.EXPECTED_REC,
                 inner_encoders=io.EXPECTED_ENCODERS,target=sha(target),protected_previous_internal_seal=sha(protected/'INTERNAL_RESULTS_SEAL_PRIVATE.json'),
                 dependencies=dict(meta=sha(PARENT/'common.py'),io=sha(io.HERE/'common.py'),pure=sha(parent.REFERENCE/'pure_model.py'),
                                   threshold=sha(parent.corepath),metrics=sha(parent.metricpath)),torch=torch.__version__,numpy=np.__version__)
    freeze(ROOT/'INPUT_FREEZE_PRIVATE.json',binding)
    torch.set_num_threads(2);torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    return binding


def contexts():
    result=parent.contexts()
    for ctx in result:
        cp=torch.load(ctx['checkpoint'],map_location='cpu',weights_only=False)
        assert torch.count_nonzero(cp['model']['keep32'])==0
        ctx['reference_state']=cp['model'];ctx['reference_threshold']=cp['threshold']
    return result


def path_for(ctx,pid):return ROOT/'phase_views_private'/f"fold{ctx['fold']}_screen{ctx['screen']}"/(io.private_key(pid)+'.pkl')


def calibrate(d,ix,logit):return core.select_threshold(d['y'][ix],probability(logit),d['patient'][ix])


def held_rows(ctx,logit,theta,arm):
    return [dict(fold=ctx['fold'],screen=ctx['screen'],arm=arm,**r)
            for r in patient_rows(ctx['data'],ctx['roles']['held'],logit,theta)]


def payload(q,device='cuda'):
    return {k:torch.as_tensor(q[k],device=device) for k in ['window','reference','available']}
