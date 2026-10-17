import copy
import math
import numpy as np
import torch
from torch import nn
from model import Decision,Fused,hidden,objective,ARMS


class Body(nn.Module):
    def __init__(self):
        super().__init__();torch.manual_seed(31)
        self.cnn=nn.Sequential(nn.Conv1d(1,12,9,4),nn.GELU(),nn.Conv1d(12,24,7,4),nn.GELU(),nn.AdaptiveAvgPool1d(1))
        self.gate=nn.Sequential(nn.Linear(7,32),nn.GELU(),nn.Linear(32,24));self.head=nn.Sequential(nn.Linear(110,64),nn.GELU(),nn.Linear(64,1))
        self.interaction_gain=nn.Parameter(torch.tensor(.13));self.register_buffer('keep7',torch.ones(7));self.raw_feature_function=lambda *a,**k:None


def test_counts_single_readout_frozen_body_and_zero_identity():
    b=Body();h=torch.randn(11,64)
    for arm in ARMS:
        head=Decision(arm,b.head[2],41);m=Fused(b,head)
        assert sum(p.numel() for p in m.parameters())==(10506 if arm=='G0_GLOBAL' else 12562)
        assert sum(p.numel() for p in m.parameters() if p.requires_grad)==(128 if arm=='G0_GLOBAL' else 2184)
        assert sum(isinstance(q,nn.Linear) and q.out_features==1 for q in m.modules())==1
        assert torch.equal(head(h),b.head[2](h).squeeze(-1)) and torch.equal(m.classify_hidden(h),head(h))


def test_exact_hidden_definition():
    b=Body();z=torch.randn(11,96);x=torch.randn(11,14)
    a=torch.nn.functional.linear(z*math.sqrt(110/192),b.head[0].weight[:,:96]);p=torch.nn.functional.linear(x*math.sqrt(110/28),b.head[0].weight[:,96:])
    expected=b.head[1](a+p+b.head[0].bias+.5*torch.tanh(b.interaction_gain)*a*torch.tanh(p))
    assert torch.equal(hidden(b,z,x),expected)


def test_context_permutation_nonconstant_shift_and_boundedness():
    b=Body();h=torch.randn(11,64);order=torch.tensor([3,1,7,2,9,5,8,0,10,4,6])
    for arm in ARMS:
        m=Decision(arm,b.head[2],41)
        with torch.no_grad():
            if arm=='G0_GLOBAL':m.offset.copy_(torch.linspace(-.5,.5,128))
            else:m.context[2].weight.fill_(.01);m.context[2].bias.copy_(torch.linspace(-.5,.5,128))
        assert torch.allclose(m(h)[order],m(h[order]),atol=1e-6,rtol=1e-5)
        gain,offset=m.affine(h)
        assert gain.shape==offset.shape==(1,64) and gain.abs().max()<=.25 and offset.abs().max()<=.25
        shift=m(h)-b.head[2](h).squeeze(-1);assert torch.max((shift-shift[0]).abs())>1e-5


def test_context_gradients_open_without_readout_updates():
    b=Body();h=torch.randn(11,64);m=Decision('G1_PATIENT',b.head[2],41);before=copy.deepcopy(m.readout.state_dict())
    opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=.0003)
    for step in range(2):
        opt.zero_grad();m(h).square().mean().backward();assert m.context[2].weight.grad.abs().sum()>0
        if step:assert m.context[0].weight.grad.abs().sum()>0
        assert all(p.grad is None for p in m.readout.parameters());opt.step()
    assert all(torch.equal(v,m.readout.state_dict()[k]) for k,v in before.items())


def test_exact_original_macro_objective():
    x=torch.tensor([-.7,.3,.5,-.2],requires_grad=True);y=torch.tensor([0.,1.,1.,0.]);p=x.sigmoid()
    macro=.5*(2*(p*y).sum()/(p.sum()+y.sum())+2*((1-p)*(1-y)).sum()/((1-p).sum()+(1-y).sum()))
    expected=.75*torch.nn.functional.binary_cross_entropy_with_logits(x,y)+.25*(1-macro)
    assert torch.equal(objective(x,y,torch.tensor(1.)),expected)
