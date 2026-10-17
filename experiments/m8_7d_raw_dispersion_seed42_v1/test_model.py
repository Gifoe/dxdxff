"""Five distinct implementation checks, not synthetic performance evidence."""
import numpy as np
import copy
import torch
from torch import nn
from model import *


def source():
    torch.manual_seed(51)
    cnn=nn.Sequential(nn.Conv1d(1,12,9,4),nn.GELU(),nn.Conv1d(12,24,7,4),nn.GELU(),nn.AdaptiveAvgPool1d(1))
    gate=nn.Sequential(nn.Linear(7,32),nn.GELU(),nn.Linear(32,24))
    return cnn,gate


def models():
    cnn,gate=source()
    return [SingleHead(a,cnn,gate,None,170054) for a in ARMS]


def test_counts_one_readout_and_initial_shared_tensors():
    ms=models()
    assert [sum(p.numel() for p in m.parameters()) for m in ms]==[10378,10978,10978]
    for m in ms:
        assert sum(isinstance(q,nn.Linear) and q.out_features==1 for q in m.modules())==1
        assert torch.equal(m.keep7,torch.ones(7))
        for a,b in zip(ms[0].head.parameters(),m.head.parameters()):assert torch.equal(a,b)
        for a,b in zip(ms[0].gate.parameters(),m.gate.parameters()):assert torch.equal(a,b)
    for m in ms[1:]:
        for a,b in zip(ms[0].cnn[:4].parameters(),m.cnn.stem.parameters()):assert torch.equal(a,b)
        assert torch.count_nonzero(m.cnn.projection.weight)==0


def test_initial_raw_function_and_shared_gradient():
    ms=models();torch.manual_seed(63);raw=torch.randn(31,1,500)
    outs=[m.cnn(raw) for m in ms]
    assert torch.equal(outs[0],outs[1]) and torch.equal(outs[1],outs[2])
    for out in outs:out.square().sum().backward()
    for m in ms[1:]:
        for a,b in zip(ms[0].cnn[:4].parameters(),m.cnn.stem.parameters()):
            assert torch.equal(a.grad,b.grad)
        assert m.cnn.projection.weight.grad.abs().sum()>0


def test_dispersion_distinguishes_equal_mean():
    ms=models();x=torch.ones(2,24,30);x[1,:,:15]=0;x[1,:,15:]=2
    for m in ms[1:]:
        with torch.no_grad():m.cnn.projection.weight.copy_(torch.eye(24))
    r0=ms[1].cnn.pool(x);r1=ms[2].cnn.pool(x)
    assert torch.equal(r0[0],r0[1]) and torch.all(r1[1]>r1[0])
    for m in ms[1:]:
        order=torch.arange(29,-1,-1)
        assert torch.equal(m.cnn.pool(x),m.cnn.pool(x[:,:,order]))
        constant=torch.ones(2,24,30,requires_grad=True);m.cnn.pool(constant).sum().backward()
        assert torch.isfinite(constant.grad).all()


def test_all_coordinates_and_single_c2_formula():
    m=models()[2];z=torch.randn(9,96,requires_grad=True);x=torch.randn(9,14,requires_grad=True)
    out=m.classify(z,x);first=m.head[0]
    a=F.linear(z*math.sqrt(110/192),first.weight[:,:96]);b=F.linear(x*math.sqrt(110/28),first.weight[:,96:])
    expected=m.head[2](m.head[1](a+b+first.bias)).squeeze(-1)
    assert torch.equal(out,expected);out.square().sum().backward()
    assert torch.all(z.grad.abs().sum(0)>0) and torch.all(x.grad.abs().sum(0)>0)
    assert m.interaction_gain.grad.abs()>0


def test_exact_original_loss():
    x=torch.tensor([-.4,.2,.8],requires_grad=True);y=torch.tensor([0.,1.,1.]);pos=torch.tensor(.7)
    p=x.sigmoid()
    macro=.5*(2*(p*y).sum()/(p.sum()+y.sum())+2*((1-p)*(1-y)).sum()/((1-p).sum()+(1-y).sum()))
    assert torch.equal(objective(x,y,pos),.75*F.binary_cross_entropy_with_logits(x,y,pos_weight=pos)+.25*(1-macro))
