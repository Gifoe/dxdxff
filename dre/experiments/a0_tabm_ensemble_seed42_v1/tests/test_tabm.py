import copy
import os
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from base import PRMLP,prepare,seed_all,state_hash,select_threshold
from model import make,objective,probability
from source_parity import official
from train import train


def test_counts_shape_initialization_and_orientation():
    assert sum(p.numel() for p in PRMLP().parameters())==8817
    for arm,total in [('W1',10167),('W2',10132)]:
        seed_all(42); a=make(arm); seed_all(42); b=make(arm)
        assert sum(p.numel() for p in a.parameters())==total
        assert state_hash(a.state_dict())==state_hash(b.state_dict())
    a.eval(); x=torch.randn(12,88); logits=a(x)
    assert logits.shape==(12,4) and torch.isfinite(logits).all()
    assert not torch.equal(logits[:,0],logits[:,1])
    assert set(a.hidden.r.detach().flatten().tolist())=={-1.,1.}
    assert torch.equal(a.hidden.s,torch.ones_like(a.hidden.s))
    assert all(torch.equal(a.hidden.bias[0],v) for v in a.hidden.bias)
    assert not torch.equal(a.head.weight[0],a.head.weight[1])
    assert probability(torch.tensor([-1.]))<.5 and (1-probability(torch.tensor([-1.])))>.5
    ix=torch.randperm(len(x)); assert torch.equal(a(x[ix]),a(x)[ix])


def test_official_source_initial_output_and_gradient_parity():
    upstream=official(os.environ['TABM_SOURCE'])
    seed_all(51); a=make('W2')
    seed_all(51)
    b=torch.nn.Sequential(torch.nn.LayerNorm(88),upstream.EnsembleView(k=4),
        upstream.LinearBatchEnsemble(88,96,k=4,scaling_init=('random-signs','ones')),
        torch.nn.GELU(),torch.nn.Dropout(.15),upstream.LinearEnsemble(96,1,k=4))
    assert all(torch.equal(x,y) for x,y in zip(a.parameters(),b.parameters()))
    x=torch.randn(15,88); y=torch.randint(0,2,(15,)).float(); pos=torch.tensor(.3)
    a.eval(); b.eval(); assert torch.equal(a(x),b(x).squeeze(-1))
    objective(a(x),y,pos).backward(); objective(b(x).squeeze(-1),y,pos).backward()
    assert all(torch.equal(x.grad,y.grad) for x,y in zip(a.parameters(),b.parameters()))


def test_loss_and_inference_are_not_averaged_logits():
    logits=torch.tensor([[4.,-2.,1.,-4.],[-3.,1.,5.,2.]],requires_grad=True)
    y=torch.tensor([0.,1.]); pos=torch.tensor(.4)
    loss=objective(logits,y,pos)
    expected=torch.stack([torch.nn.functional.binary_cross_entropy_with_logits(logits[:,k],y,pos_weight=pos) for k in range(4)]).mean()
    assert torch.allclose(loss,expected,rtol=0,atol=1e-7)
    assert not torch.allclose(loss,objective(logits.mean(1),y,pos))
    assert torch.equal(probability(logits),torch.sigmoid(logits).mean(1))
    assert not torch.allclose(probability(logits),torch.sigmoid(logits.mean(1)))
    loss.backward(); assert torch.isfinite(logits.grad).all() and (logits.grad.abs().sum(0)>0).all()


def test_each_member_shared_and_private_gradients():
    seed_all(42); a=make('W2').eval(); x=torch.randn(17,88); y=torch.randint(0,2,(17,)).float()
    pieces=[]
    for k in range(4):
        a.zero_grad(); loss=torch.nn.functional.binary_cross_entropy_with_logits(a(x)[:,k],y,pos_weight=torch.tensor(.5)); loss.backward()
        assert a.hidden.weight.grad.abs().sum()>0
        assert a.head.weight.grad[k].abs().sum()>0
        assert sum(a.head.weight.grad[j].abs().sum() for j in range(4) if j!=k)==0
        pieces.append(a.hidden.weight.grad.clone())
    a.zero_grad(); objective(a(x),y,torch.tensor(.5)).backward()
    assert torch.allclose(a.hidden.weight.grad,torch.stack(pieces).mean(0),rtol=1e-5,atol=1e-7)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in a.parameters())


def test_fit_only_preprocessing_and_threshold_ties():
    rng=np.random.default_rng(42); x=rng.normal(size=(32,88)); pid=np.repeat(['a','b','c','d'],8); tr=np.arange(16)
    x[0,1]=np.nan; a,pre=prepare(x,pid,tr); change=x.copy(); change[16:]=rng.normal(80,10,(16,88)); b,post=prepare(change,pid,tr)
    assert np.array_equal(a[:16],b[:16])
    assert all(np.array_equal(pre[k],post[k]) for k in ['imputer_statistics','mean','scale','var'])
    assert pre['fit_patient_ids']==['a','b']
    y=np.array([0,1,0,1]); score=np.array([.1,.9,.2,.8]); patient=np.array(['a','a','b','b'])
    selected=select_threshold(y,score,patient)
    assert selected['macro_f1']==1. and selected['threshold']==.5
    # No member selection or label argument enters either model forward.
    import inspect
    assert list(inspect.signature(make('W2').forward).parameters)==['x']


def test_exact_epoch_boundary_resume_both_arms(tmp_path):
    rng=np.random.default_rng(42); d={'x':rng.normal(size=(40,88)).astype(np.float32),
        'y':np.tile([0,1],20),'patient':np.repeat(['a','b','c','d','e'],8),
        'train':np.arange(24),'val':np.arange(24,40)}
    for arm in ['W1','W2']:
        seed_all(1051); initial=copy.deepcopy(make(arm).state_dict())
        full,best,hist,_=train(d,arm,tmp_path/(arm+'full'),1051,initial,'test','cpu',max_epochs=4)
        train(d,arm,tmp_path/(arm+'resume'),1051,initial,'test','cpu',max_epochs=4,stop_after=2)
        resumed,rbest,rhist,_=train(d,arm,tmp_path/(arm+'resume'),1051,initial,'test','cpu',max_epochs=4)
        assert state_hash(full.state_dict())==state_hash(resumed.state_dict())
        assert hist==rhist and best['epoch']==rbest['epoch'] and best['threshold']==rbest['threshold']
