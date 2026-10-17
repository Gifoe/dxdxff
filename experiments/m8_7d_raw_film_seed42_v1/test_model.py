"""Five unique changed-path implementation tests; no synthetic performance."""
import numpy as np
import torch
from torch import nn
from model import *


def models():
    torch.manual_seed(51)
    cnn=nn.Sequential(nn.Conv1d(1,12,9,4),nn.GELU(),nn.Conv1d(12,24,7,4),nn.GELU(),nn.AdaptiveAvgPool1d(1))
    gate=nn.Sequential(nn.Linear(7,32),nn.GELU(),nn.Linear(32,24))
    return [SingleHead(a,cnn,gate,None,170054) for a in ARMS]


def test_counts_shared_state_one_output():
    ms=models()
    assert [sum(p.numel() for p in m.parameters()) for m in ms]==[10378,11426,11426]
    for m in ms:
        assert sum(isinstance(q,nn.Linear) and q.out_features==1 for q in m.modules())==1
        assert torch.equal(m.keep7,torch.ones(7))
        for a,b in zip(ms[0].head.parameters(),m.head.parameters()):assert torch.equal(a,b)
        for a,b in zip(ms[0].gate.parameters(),m.gate.parameters()):assert torch.equal(a,b)
    for m in ms[1:]:
        for a,b in zip(ms[0].cnn[:4].parameters(),m.cnn.stem.parameters()):assert torch.equal(a,b)
    for a,b in zip(ms[1].cnn.conditioner.parameters(),ms[2].cnn.conditioner.parameters()):assert torch.equal(a,b)


def test_initial_identity_and_shared_gradients():
    ms=models();torch.manual_seed(63);raw=torch.randn(31,1,500);u=torch.randn(31,7)
    out=[ms[0].cnn(raw)]+[m.cnn(raw,u) for m in ms[1:]]
    assert torch.equal(out[0],out[1]) and torch.equal(out[1],out[2])
    for v in out:v.square().sum().backward()
    for m in ms[1:]:
        for a,b in zip(ms[0].cnn[:4].parameters(),m.cnn.stem.parameters()):assert torch.equal(a.grad,b.grad)
        assert m.cnn.conditioner[2].weight.grad.abs().sum()>0


def test_conditioner_learning_and_placement():
    ms=models();torch.manual_seed(78);raw=torch.randn(11,1,500);u=torch.randn(11,7,requires_grad=True)
    for m in ms[1:]:
        opt=torch.optim.AdamW(m.parameters(),lr=.001)
        for step in range(2):
            opt.zero_grad(set_to_none=True);out=m.cnn(raw,u);out.square().sum().backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.cnn.parameters())
            if step:assert m.cnn.conditioner[0].weight.grad.abs().sum()>0
            opt.step()
        assert torch.isfinite(u.grad).all() and u.grad.abs().sum()>0
    for m in ms[1:]:
        with torch.no_grad():
            m.cnn.conditioner[2].weight.zero_();m.cnn.conditioner[2].bias.zero_();m.cnn.conditioner[2].bias[12:]=.8
    h=ms[1].cnn.stem[0](raw);shift=.25*torch.tensor(.8).tanh()
    assert not torch.equal(torch.nn.functional.gelu(h)+shift,torch.nn.functional.gelu(h+shift))


def test_all14_coordinates_and_c2_formula():
    m=models()[2];z=torch.randn(9,96,requires_grad=True);x=torch.randn(9,14,requires_grad=True)
    out=m.classify(z,x);first=m.head[0]
    a=F.linear(z*math.sqrt(110/192),first.weight[:,:96]);b=F.linear(x*math.sqrt(110/28),first.weight[:,96:])
    expected=m.head[2](m.head[1](a+b+first.bias)).squeeze(-1)
    assert torch.equal(out,expected);out.square().sum().backward()
    assert torch.all(z.grad.abs().sum(0)>0) and torch.all(x.grad.abs().sum(0)>0)
    assert m.interaction_gain.grad.abs()>0


def test_original_objective():
    x=torch.tensor([-.4,.2,.8],requires_grad=True);y=torch.tensor([0.,1.,1.]);pos=torch.tensor(.7);p=x.sigmoid()
    macro=.5*(2*(p*y).sum()/(p.sum()+y.sum())+2*((1-p)*(1-y)).sum()/((1-p).sum()+(1-y).sum()))
    assert torch.equal(objective(x,y,pos),.75*F.binary_cross_entropy_with_logits(x,y,pos_weight=pos)+.25*(1-macro))
