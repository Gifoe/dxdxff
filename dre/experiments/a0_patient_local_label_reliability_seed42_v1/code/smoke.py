"""Synthetic plus actual FIT-only A0 parity and weighted-gradient smoke gate."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
import torch.nn.functional as F
from baseline import PRMLP, seed_all, sha, json_write, state_hash, train_model
from train import train_weighted
from prepare import binding


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--runtime',type=Path,required=True)
    parser.add_argument('--protocol',type=Path,required=True); parser.add_argument('--device',default='cuda')
    a=parser.parse_args(); torch.set_num_threads(2)
    feasible=json.loads((a.runtime/'public/PLLR_FEASIBILITY_AUDIT.json').read_text())
    assert feasible['pass'], 'No training smoke on infeasible PLLR'
    assert feasible['binding']==binding(a.protocol)
    tests=Path(__file__).resolve().parents[1]/'tests'
    result=subprocess.run([sys.executable,'-m','pytest',str(tests),'-q'],capture_output=True,text=True)
    print(result.stdout,flush=True)
    if result.returncode: raise RuntimeError('Synthetic test failure; preserve private stderr')
    d=torch.load(a.runtime/'fold1/DEVELOPMENT_PRIVATE.pt',weights_only=False)
    w=torch.load(a.runtime/'fold1/PLLR_FROZEN_PRIVATE.pt',weights_only=False)
    tr=d['train']; ids=sorted(set(d['patient'][tr]))[:3]
    fit_take=tr[np.isin(d['patient'][tr],ids)]
    x,y,pid=[d[k][fit_take] for k in ['x','y','patient']]
    train=np.flatnonzero(np.isin(pid,ids[:2])); val=np.flatnonzero(pid==ids[2])
    assert not set(pid)&set(d['patient'][d['val']])
    seed_all(1051); initial=copy.deepcopy(PRMLP().state_dict())
    old,ob,_=train_model(x,y,pid,train,val,a.runtime/'smoke/original',1051,initial,'FIT_smoke',a.device,max_epochs=1)
    new,nb,_=train_weighted(x,y,pid,train,val,a.runtime/'smoke/ones',1051,initial,
        'FIT_smoke',a.device,'A0',np.ones(len(x)),max_epochs=1)
    assert state_hash(old.state_dict())==state_hash(new.state_dict())
    assert ob['threshold']==nb['threshold'] and ob['epoch']==nb['epoch']
    model=PRMLP().to(a.device); model.load_state_dict(initial)
    logits=model(torch.as_tensor(d['x'][tr],device=a.device))
    labels=torch.as_tensor(d['y'][tr],dtype=torch.float32,device=a.device)
    pos=torch.tensor(float((d['y'][tr]==0).sum()/(d['y'][tr]==1).sum()),device=a.device)
    for arm in ['B1','B2']:
        loss=F.binary_cross_entropy_with_logits(logits,labels,
            weight=torch.tensor(w['artifact'][arm],dtype=torch.float32,device=a.device),pos_weight=pos)
        grads=torch.autograd.grad(loss,tuple(model.parameters()),retain_graph=True)
        assert all(torch.isfinite(g).all() for g in grads)
    audit={'status':'PASS','binding':binding(a.protocol),'tests_sha256':sha(tests/'test_pllr.py'),
        'synthetic_test_groups':6,'real_FIT_patients':3,'original_vs_ones_parameter_max_drift':0.,
        'original_vs_ones_parameter_hash':state_hash(new.state_dict()),
        'selected_threshold_exact_match':True,'real_B1_B2_finite_gradients':True,
        'outer_VAL_TEST_labels_used':False,'runtime':{'torch':torch.__version__,'numpy':np.__version__}}
    json_write(a.runtime/'public/TEST_AND_SMOKE_AUDIT.json',audit)
    print('PLLR_TEST_AND_REAL_FIT_SMOKE_PASS',flush=True)


if __name__=='__main__':main()
