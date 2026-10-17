"""Five changed-critical-path checks,not performance evidence."""
import numpy as np
import torch
from torch import nn
from model import *


def models():
    torch.manual_seed(51)
    cnn=nn.Sequential(nn.Conv1d(1,12,9,4),nn.GELU(),nn.Conv1d(12,24,7,4),nn.GELU(),nn.AdaptiveAvgPool1d(1))
    gate=nn.Sequential(nn.Linear(7,32),nn.GELU(),nn.Linear(32,24))
    return [SingleHead(a,cnn,gate,None,170054) for a in ARMS]


def test_counts_shared_core_one_readout_and_all7():
    ms=models();assert [sum(p.numel() for p in m.parameters()) for m in ms]==[10378,13450,13450]
    for m in ms:
        assert sum(isinstance(q,nn.Linear) and q.out_features==1 for q in m.modules())==1
        assert torch.equal(m.keep7,torch.ones(7))
        for name in ['cnn','gate','head']:
            for a,b in zip(getattr(ms[0],name).parameters(),getattr(m,name).parameters()):assert torch.equal(a,b)
    assert sum(p.numel() for p in ms[1].mixer.parameters())==sum(p.numel() for p in ms[2].mixer.parameters())==3072


def test_initial_identity_and_shared_core_gradients():
    ms=models();z=torch.randn(11,96);x=torch.randn(11,14);outs=[m.classify(z,x) for m in ms]
    assert torch.equal(outs[0],outs[1]) and torch.equal(outs[1],outs[2])
    for p in outs:p.square().mean().backward()
    for m in ms[1:]:
        for a,b in zip(ms[0].head.parameters(),m.head.parameters()):assert torch.equal(a.grad,b.grad)
        assert torch.equal(ms[0].interaction_gain.grad,m.interaction_gain.grad)
        assert m.mixer.output.weight.grad.abs().sum()>0


def test_permutation_neighbors_self_mask_and_singleton():
    h=torch.randn(11,64);order=torch.tensor([3,1,7,2,9,5,8,0,10,4,6]);ms=models()
    for m in ms[1:]:
        with torch.no_grad():m.mixer.output.weight.fill_(.05)
        out=m.mixer(h);assert torch.allclose(out[order],m.mixer(h[order]),atol=1e-6,rtol=1e-5)
        assert out.abs().max()<=.25
        changed=h.clone();changed[1:]+=2
        if m.variant=='P0_POINTWISE':assert torch.equal(out[0],m.mixer(changed)[0])
        else:
            assert not torch.allclose(out[0],m.mixer(changed)[0])
            a=m.mixer.attention(h);assert torch.equal(a.diag(),torch.zeros(11))
            assert torch.allclose(a.sum(-1),torch.ones(11))
            one=h[:1].clone().requires_grad_();v=m.mixer(one);assert torch.count_nonzero(v)==0
            v.sum().backward();assert torch.isfinite(one.grad).all()


def test_all14_and_mixer_gradient_open():
    for m in models()[1:]:
        z=torch.randn(11,96,requires_grad=True);x=torch.randn(11,14,requires_grad=True)
        opt=torch.optim.AdamW(m.parameters(),lr=.001)
        for step in range(2):
            opt.zero_grad(set_to_none=True);m.classify(z,x).square().mean().backward()
            assert torch.all(z.grad.abs().sum(0)>0) and torch.all(x.grad.abs().sum(0)>0)
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.mixer.parameters())
            if step:assert all(p.grad.abs().sum()>0 for p in m.mixer.parameters())
            opt.step()


def test_exact_original_objective():
    x=torch.tensor([-.4,.2,.8],requires_grad=True);y=torch.tensor([0.,1.,1.]);pos=torch.tensor(.7);p=x.sigmoid()
    macro=.5*(2*(p*y).sum()/(p.sum()+y.sum())+2*((1-p)*(1-y)).sum()/((1-p).sum()+(1-y).sum()))
    assert torch.equal(objective(x,y,pos),.75*F.binary_cross_entropy_with_logits(x,y,pos_weight=pos)+.25*(1-macro))
