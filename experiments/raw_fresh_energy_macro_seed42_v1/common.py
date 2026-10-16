"""Only Raw waveforms and legal ownFIT metadata are available to fresh models."""
import importlib.util
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

HERE=Path(__file__).resolve().parent;ROOT=Path(r'D:\raw_fresh_energy_macro_seed42_v1_runtime');PUB=ROOT/'public'
WAVE_CODE=Path(r'D:\raw_precompand_macro_seed42_v1');WAVE=Path(r'D:\raw_precompand_macro_seed42_v1_runtime')


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m


wave=module('fresh_energy_pinned_legal_roles',WAVE_CODE/'common.py');parent=wave.parent;io=wave.io
sha,read,save,write,freeze,jr,digest=io.sha,io.read,io.save,io.write,io.freeze,io.jr,io.digest
state_hash,atomic_torch,seed_all=io.state_hash,io.atomic_torch,io.seed_all
RawStore=io.RawStore;METRICS=wave.METRICS;probability,patient_rows=parent.probability,parent.patient_rows
REFERENCES=['NATIVE','FROZEN_PRE'];TRAINED=['WINDOW_MEAN','WINDOW_LOGENERGY','PRE_MEAN','PRE_LOGENERGY'];ARMS=REFERENCES+TRAINED
CRITICAL=['PROTOCOL_LOCK.json','common.py','model.py','test_model.py','admit.py','train_internal.py','summarize.py']


def setup():
    ROOT.mkdir(exist_ok=True);PUB.mkdir(exist_ok=True);previous=wave.setup()
    assert sha(WAVE/'INTERNAL_RESULTS_SEAL_PRIVATE.json')=='cdc2932bf0c977533e01f008bdd7b38305a048cd87b88fb2030ca5eec352b3cd'
    protected=Path(r'D:\raw_temporal_context_macro_seed42_v1_runtime')
    assert jr(protected/'public/INDEPENDENT_INTERNAL_VERIFICATION.json')['status']=='PASS'
    assert sha(protected/'INTERNAL_RESULTS_SEAL_PRIVATE.json')=='2c1edbac1dbc1c85e44cb3b7fb330f25545e4f1a1a7729fcae58c5a3ce5d808b'
    binding=dict(source={n:sha(HERE/n) for n in CRITICAL},raw_registry=io.EXPECTED_REG,reconstruction=io.EXPECTED_REC,
        inner_role_registry=io.EXPECTED_ENCODERS,dependency_input=digest(previous),prior_temporal_seal=sha(protected/'INTERNAL_RESULTS_SEAL_PRIVATE.json'),
        prior_pre_scores=sha(WAVE/'INTERNAL_RESULTS_SEAL_PRIVATE.json'),pure_source=sha(parent.REFERENCE/'pure_model.py'),
        pre_normalization_source=sha(WAVE_CODE/'model.py'),torch=torch.__version__,numpy=np.__version__,initialization='RANDOM_NO_TRAINED_STATE_ARGUMENT')
    freeze(ROOT/'INPUT_FREEZE_PRIVATE.json',binding)
    torch.set_num_threads(2);torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    return binding


def contexts():
    cc=parent.contexts();seal=jr(WAVE/'INTERNAL_RESULTS_SEAL_PRIVATE.json');out=[]
    for old in cc:
        f,s=old['fold'],old['screen'];d,rr=old['data'],old['roles'];cp=torch.load(old['checkpoint'],map_location='cpu',weights_only=False)
        assert torch.count_nonzero(cp['model']['keep32'])==0
        p=WAVE/'matrices_private'/f'fold{f}_screen{s}'/'PRECOMPAND_RESIDUAL'/'HELD_PRIVATE.pkl'
        expected=next(r['sha256'] for r in seal['cells'] if (r['fold'],r['screen'],r['arm'])==(f,s,'PRECOMPAND_RESIDUAL'))
        assert sha(p)==expected;q=read(p)
        assert q['binding']['body_checkpoint_sha256']==old['stored']['checkpoint_sha256']
        assert np.array_equal(q['archive_indices'],d['archive_indices'][rr['held']])
        out.append(dict(fold=f,screen=s,data=d,roles=rr,native_logits=old['stored']['raw_logit_nez'][rr['held']].copy(),
            native_threshold=cp['threshold'],pre_logits=q['logit_nez'].copy(),pre_threshold=q['summary']['threshold'],
            reference_checkpoint=str(old['checkpoint']),reference_checkpoint_sha256=old['stored']['checkpoint_sha256'],
            representation_sha256=old['representation_sha256'],pre_score_sha256=sha(p)))
    return out


def calibrate(d,ix,logit):return wave.calibrate(d,ix,logit)
def held_rows(ctx,logit,theta,arm):return wave.held_rows(ctx,logit,theta,arm)
def state_cpu(m):return {n:v.detach().cpu().clone() for n,v in m.state_dict().items()}
def rng():return torch.get_rng_state(),torch.cuda.get_rng_state_all()
def restore(q):torch.set_rng_state(q[0].cpu());torch.cuda.set_rng_state_all([x.cpu() for x in q[1]])
