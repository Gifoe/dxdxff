"""Pinned legal ownFIT roles and waveform source, never physiological predictors."""
import importlib.util
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

HERE=Path(__file__).resolve().parent
ROOT=Path(r'D:\raw_joint_moment_macro_seed42_v1_runtime');PUB=ROOT/'public'
PREVIOUS_CODE=Path(r'D:\raw_fresh_energy_macro_seed42_v1')
PREVIOUS=Path(r'D:\raw_fresh_energy_macro_seed42_v1_runtime')


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m


previous=module('joint_moments_pinned_roles',PREVIOUS_CODE/'common.py')
io,parent,wave=previous.io,previous.parent,previous.wave
sha,read,save,write,freeze,jr,digest=io.sha,io.read,io.save,io.write,io.freeze,io.jr,io.digest
state_hash,atomic_torch,seed_all=io.state_hash,io.atomic_torch,io.seed_all
state_cpu,rng,restore=previous.state_cpu,previous.rng,previous.restore
RawStore=io.RawStore;METRICS=previous.METRICS;probability=previous.probability
calibrate,held_rows=previous.calibrate,previous.held_rows
REFERENCES=['NATIVE','FROZEN_PRE']
TRAINED=['PRE_MEAN_DUP','PRE_ENERGY_DUP','PRE_JOINT'];ARMS=REFERENCES+TRAINED
CRITICAL=['PROTOCOL_LOCK.json','common.py','model.py','test_model.py','admit.py','train_internal.py','summarize.py']


def setup():
    ROOT.mkdir(exist_ok=True);PUB.mkdir(exist_ok=True);old=previous.setup()
    assert sha(PREVIOUS/'INTERNAL_RESULTS_SEAL_PRIVATE.json')=='7ba9c5fb3c1577fb7db42dd2bf4030687cb391757ce7bf3b5bf6c223f0ca461a'
    # Protection, not a use of old Outer values for selecting new candidates.
    assert sha(PREVIOUS/'final_stage_v1/OUTER_SCORE_SEAL_PRIVATE.json')=='0e8c2454ac21ad3af3b980294da28e06f01327ef26844a09210babeb4a2245ae'
    assert jr(PREVIOUS/'final_stage_v1/public/INDEPENDENT_FINAL_VERIFICATION.json')['status']=='PASS'
    binding=dict(source={n:sha(HERE/n) for n in CRITICAL},prior_source_input=digest(old),
        protected_prior_internal=sha(PREVIOUS/'INTERNAL_RESULTS_SEAL_PRIVATE.json'),
        protected_prior_Outer=sha(PREVIOUS/'final_stage_v1/OUTER_SCORE_SEAL_PRIVATE.json'),
        raw_registry=io.EXPECTED_REG,reconstruction=io.EXPECTED_REC,role_registry=io.EXPECTED_ENCODERS,
        pure_source=sha(parent.REFERENCE/'pure_model.py'),normalizer_source=sha(previous.WAVE_CODE/'model.py'),
        torch=torch.__version__,numpy=np.__version__,initialization='RANDOM_NO_WARMSTART')
    freeze(ROOT/'INPUT_FREEZE_PRIVATE.json',binding)
    torch.set_num_threads(2);torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    return binding


def contexts():
    result=previous.contexts()
    for q in result:
        groups=[set(q['data']['patient'][q['roles'][k]]) for k in ['train','cal','choice','held']]
        assert all(groups[i].isdisjoint(groups[j]) for i in range(4) for j in range(i))
    return result
