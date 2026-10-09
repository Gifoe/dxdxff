"""Pretraining server synthetic tests and real FIT-only identity/gradient checks."""
import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import torch
from base import sha,json_write,seed_all,state_hash,PRMLP
from model import Temporal
from prepare import binding


def main():
    p=argparse.ArgumentParser(); p.add_argument('--runtime',type=Path,required=True)
    p.add_argument('--protocol',type=Path,required=True); p.add_argument('--device',default='cuda')
    a=p.parse_args(); torch.set_num_threads(2); root=a.runtime; pub=root/'public'
    tests=Path(__file__).resolve().parents[1]/'tests/test_temporal.py'
    subprocess.run([sys.executable,'-m','pytest',str(tests),'-q'],check=True)
    d=torch.load(root/'fold1/DEVELOPMENT_PRIVATE.pt',weights_only=False)
    pids=sorted(set(d['patient'][d['train']]))[:3]; checks=[]
    for mode in ['T1','T2']:
        seed_all(42); model=Temporal(mode).to(a.device); initial=state_hash(model.state_dict())
        opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-3)
        frozen=PRMLP().to(a.device).requires_grad_(False).eval()
        snap=torch.load(root/'fold1/A0_FROZEN_PRIVATE.pt',weights_only=False); frozen.load_state_dict(snap['state'])
        pos=torch.tensor((d['y'][d['train']]==0).sum()/(d['y'][d['train']]==1).sum(),device=a.device)
        maximum=0.
        for index,pid in enumerate(pids):
            ix=np.flatnonzero(d['patient']==pid); bank=d['banks'][pid]
            z,t,m=[torch.as_tensor(bank[k],device=a.device) for k in ['z','time','mask']]
            model.eval()
            if index==0:
                delta=model(z,t,m); assert torch.equal(delta,torch.zeros_like(delta))
                bz=torch.as_tensor(d['base_logits'][ix],device=a.device)
                assert torch.equal(bz+delta,bz)
            model.train(); opt.zero_grad(set_to_none=True)
            logits=torch.as_tensor(d['base_logits'][ix],device=a.device)+model(z,t,m)
            loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,torch.as_tensor(d['y'][ix],dtype=torch.float32,device=a.device),pos_weight=pos)
            loss.backward(); assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
            assert all(p.grad is None for p in frozen.parameters())
            opt.step(); maximum=max(maximum,float(abs(model(z,t,m)).max().detach().cpu()))
        checks.append({'arm':mode,'initial_hash':initial,'parameters':sum(p.numel() for p in model.parameters()),
            'real_FIT_patient_updates':3,'epoch0_exact':True,'finite_gradient':True,'A0_gradient_absent':True,
            'residual_after_smoke_max_abs':maximum})
    assert checks[0]['initial_hash']==checks[1]['initial_hash']
    assert checks[0]['parameters']==checks[1]['parameters']==5797
    json_write(pub/'PARAMETER_PARITY_AUDIT.json',{'status':'PASS','arms':checks,'temporal_parameters':5797,
        'frozen_A0_parameters':8817,'trainable_A0_parameters':0,'identical_input_bank':True})
    json_write(pub/'COTAR_PARITY_AUDIT.json',{'status':'PASS','unmasked_official_copied_weights':True,
        'exact_float32_forward':True,'official_source_hash':json.loads(a.protocol.read_text())['official_layer_sha256'],
        'all_missing_finite':True,'masked_token_perturbation_invariant':True})
    json_write(pub/'TEST_AND_SMOKE_AUDIT.json',{'status':'PASS','binding':binding(a.protocol),'synthetic_groups':7,
        'tests_sha256':sha(tests),'real_fit_checks':checks,'runtime':{'torch':torch.__version__,'numpy':np.__version__},
        'no_outer_prediction_evaluation':True})
    print('TEMPORAL_TEST_SMOKE_PASS',flush=True)


if __name__=='__main__':main()
