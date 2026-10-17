"""Five unique loss/model checks,not simulated performance."""
import numpy as np
import torch
from torch import nn
from model import *


def models():
    torch.manual_seed(51)
    cnn=nn.Sequential(nn.Conv1d(1,12,9,4),nn.GELU(),nn.Conv1d(12,24,7,4),nn.GELU(),nn.AdaptiveAvgPool1d(1))
    gate=nn.Sequential(nn.Linear(7,32),nn.GELU(),nn.Linear(32,24))
    return [SingleHead(a,cnn,gate,None,170054) for a in ARMS]


def test_identical_architecture_initial_state_one_head_all14():
    ms=models();z=torch.randn(11,96,requires_grad=True);x=torch.randn(11,14,requires_grad=True)
    outputs=[m.classify(z,x) for m in ms]
    assert all(torch.equal(outputs[0],v) for v in outputs[1:])
    for m in ms:
        assert sum(p.numel() for p in m.parameters())==10378
        assert sum(isinstance(q,nn.Linear) and q.out_features==1 for q in m.modules())==1
        assert torch.equal(m.keep7,torch.ones(7))
        for k,v in ms[0].state_dict().items():assert torch.equal(v,m.state_dict()[k])
    outputs[0].square().sum().backward()
    assert torch.all(x.grad.abs().sum(0)>0) and torch.all(z.grad.abs().sum(0)>0)


def test_exact_three_objective_formulas():
    x=torch.tensor([-.4,.2,.8,-.9],requires_grad=True);y=torch.tensor([0.,1.,1.,1.]);pos=torch.tensor(.7);p=x.sigmoid()
    macro=1-.5*(2*(p*y).sum()/(p.sum()+y.sum())+2*((1-p)*(1-y)).sum()/((1-p).sum()+(1-y).sum()))
    element=F.binary_cross_entropy_with_logits(x,y,reduction='none');weights=(1-y)+pos*y
    expected=[.75*F.binary_cross_entropy_with_logits(x,y,pos_weight=pos)+.25*macro,
        .5*(element*weights).sum()/weights.sum()+.5*macro,
        .25*(element[y==0].mean()+element[y==1].mean())+.5*macro]
    for arm,v in zip(ARMS,expected):assert torch.equal(objective(x,y,pos,arm),v)


def test_local_equals_global_at_patient_class_ratio():
    torch.manual_seed(19);x=torch.randn(19);y=torch.tensor([0.]*5+[1.]*14);pos=(1-y).sum()/y.sum()
    assert torch.allclose(normalized_bce(x,y,pos,False),normalized_bce(x,y,pos,True),atol=1e-7,rtol=1e-6)


def test_local_bce_class_replication_not_soft_f1_invariance():
    x=torch.tensor([-.4,.2,.8,-.9]);y=torch.tensor([0.,0.,1.,1.]);pos=torch.tensor(.3)
    ix=torch.tensor([0,1,0,1,0,1,2,3])
    assert torch.allclose(normalized_bce(x,y,pos,True),normalized_bce(x[ix],y[ix],pos,True))
    assert not torch.allclose(normalized_bce(x,y,pos,False),normalized_bce(x[ix],y[ix],pos,False))


def test_extreme_logits_finite_gradients_and_no_forward_labels():
    for arm in ARMS:
        x=torch.tensor([-1000.,1000.,-.3,.4],requires_grad=True);y=torch.tensor([0.,1.,1.,0.]);v=objective(x,y,torch.tensor(.3),arm)
        assert torch.isfinite(v);v.backward();assert torch.isfinite(x.grad).all()
    import inspect
    assert 'y' not in inspect.signature(SingleHead.forward).parameters
