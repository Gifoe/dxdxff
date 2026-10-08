"""Synthetic tests; source/data parity checks are additionally run on the server."""
import sys
import os
import json
import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, f1_score

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from uas_core import (PRMLP, soft_target, lambda_epoch, select_threshold, mc_predict,
                      oof_plan, validate_exclusion, seed_all, train_model, state_hash, prepare)


def test_target_and_loss_invariants():
    y=torch.tensor([0.,1.,0.,1.]); q=torch.tensor([.5,.4,.2,.8],requires_grad=True)
    u=torch.tensor([1.,.3,.8,.7],requires_grad=True); z=torch.tensor([-.2,.2,.4,-.4],requires_grad=True)
    assert torch.equal(soft_target(y,q,u,1),y)
    assert torch.equal(soft_target(y,q,torch.zeros_like(u),30),y)
    assert torch.equal(soft_target(y,y,u,30),y)
    assert torch.equal(F.binary_cross_entropy_with_logits(z,soft_target(y,q,u,1),pos_weight=torch.tensor(.3)),
                       F.binary_cross_entropy_with_logits(z,y,pos_weight=torch.tensor(.3)))
    target=soft_target(y,q,u,30)
    assert ((target>=0)&(target<=1)).all()
    assert (abs(target-y)<=.25*u+1e-7).all()
    F.binary_cross_entropy_with_logits(z,target).backward()
    assert torch.isfinite(z.grad).all() and q.grad is None and u.grad is None
    assert [lambda_epoch(i) for i in [1,4,5,6,10,11,30]]==[0.,0.,0.,.05,.25,.25,.25]


def test_model_init_orientation_mc():
    models=[]
    for arm in ['A0','A1','A2']:
        seed_all(42); models.append(PRMLP())
        assert sum(p.numel() for p in models[-1].parameters())==8817
    assert len({state_hash(m.state_dict()) for m in models})==1
    assert torch.allclose(torch.sigmoid(torch.tensor(-2.)),1-torch.sigmoid(torch.tensor(2.)))
    mc=mc_predict(models[0],torch.randn(20,88),42)
    assert np.all((mc['q']>=0)&(mc['q']<=1)) and np.all((mc['u']>=0)&(mc['u']<=1))
    assert (mc['variance']>0).any() and (mc['mutual_information']>=0).all()
    assert not models[0].training and all(not m.training for m in models[0].modules())


def test_teacher_exclusion():
    ids=[f'p{i:02}' for i in range(51)]; plan=oof_plan(ids,1)
    queried=[p for item in plan for p in item['query']]
    assert len(queried)==len(set(queried))==51
    for item in plan:
        validate_exclusion(item['train'],item['validation'],item['query'],ids)
        with pytest.raises(ValueError):
            validate_exclusion(item['train']+item['query'][:1],item['validation'],item['query'],ids)


def reference_threshold(y,s,p):
    candidates=[]
    for t in np.round(np.arange(0,1.0001,.005),3):
        macro=[]; ez=[]
        for pid in np.unique(p):
            take=p==pid; pred=s[take]>=t; a=y[take]
            macro.append(f1_score(a,pred,labels=[0,1],average='macro',zero_division=0))
            ez.append(f1_score(a==0,~pred,zero_division=0))
        ba=balanced_accuracy_score(y,s>=t)
        candidates.append((float(np.mean(macro)),float(np.mean(ez)),ba,-abs(float(t)-.5),-float(t),float(t)))
    return max(candidates)


def test_threshold_exact_parity_and_frozen_application():
    rng=np.random.default_rng(42)
    for k in range(8):
        y=rng.integers(0,2,80); s=np.round(rng.random(80),2) if k%2 else rng.random(80)
        p=np.repeat(np.arange(4),20)
        oracle=json.loads((Path(__file__).parent/'SYNTHETIC_THRESHOLD_ORACLE.json').read_text())[k]
        digest=hashlib.sha256(y.astype('<i8').tobytes()+s.astype('<f8').tobytes()+p.astype('<i8').tobytes()).hexdigest()
        assert digest==oracle['input_sha256']
        old=oracle['key'] if os.environ.get('PR_UAS_SYNTHETIC_REFERENCE_FIXTURE')=='1' else reference_threshold(y,s,p)
        assert np.allclose(old,oracle['key'],rtol=0,atol=1e-15)
        new=select_threshold(y,s,p)
        assert new['threshold']==old[-1]
        assert abs(new['macro_f1']-old[0])<1e-15 and abs(new['ez_f1']-old[1])<1e-15
        t=new['threshold']; heldout=rng.random(50)
        pred=heldout>=t
        assert new['threshold']==t and np.array_equal(pred,heldout>=t)


def test_preprocess_train_only():
    rng=np.random.default_rng(1); x=rng.normal(size=(40,88)); p=np.repeat(['a','b','c','d'],10)
    a,sa=prepare(x,p,np.arange(20)); x[20:]*=100
    b,sb=prepare(x,p,np.arange(20))
    assert np.array_equal(a[:20],b[:20])
    assert np.array_equal(sa['mean'],sb['mean']) and sa['fit_patient_ids']==['a','b']


def test_exact_resume(tmp_path):
    torch.set_num_threads(2); rng=np.random.default_rng(9)
    x=rng.normal(size=(60,88)).astype('f'); y=np.tile([0,1],30)
    p=np.repeat(['a','b','c'],20); tr=np.arange(40); va=np.arange(40,60)
    seed_all(42); initial=PRMLP().state_dict()
    full,bfull,hfull=train_model(x,y,p,tr,va,tmp_path/'full',42,initial,'test','cpu',max_epochs=3)
    train_model(x,y,p,tr,va,tmp_path/'resume',42,initial,'test','cpu',max_epochs=3,stop_after=1)
    resumed,bres,hres=train_model(x,y,p,tr,va,tmp_path/'resume',42,initial,'test','cpu',max_epochs=3)
    assert hfull==hres and state_hash(full.state_dict())==state_hash(resumed.state_dict())
    for key in ['model','optimizer','rng']:
        a=torch.load(tmp_path/'full'/'LAST_PRIVATE.pt',weights_only=False)
        b=torch.load(tmp_path/'resume'/'LAST_PRIVATE.pt',weights_only=False)
        if key=='rng': assert torch.equal(a[key]['torch'],b[key]['torch'])
        elif key=='optimizer':
            for i in a[key]['state']:
                for name in a[key]['state'][i]: assert torch.equal(a[key]['state'][i][name],b[key]['state'][i][name])
    with pytest.raises(AssertionError):
        train_model(x,y,p,tr,va,tmp_path/'resume',42,initial,'changed-source','cpu',max_epochs=3)
