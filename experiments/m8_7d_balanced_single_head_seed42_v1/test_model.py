import numpy as np
import torch
from torch import nn
from model import SingleHead,ARMS,objective


def fixture(arm):
    torch.manual_seed(51)
    cnn=nn.Sequential(nn.Conv1d(1,12,9,4),nn.GELU(),nn.Conv1d(12,24,7,4),nn.GELU(),nn.AdaptiveAvgPool1d(1))
    class Gate(nn.Module):
        def __init__(self):super().__init__();self.network=nn.Sequential(nn.Linear(7,32),nn.GELU(),nn.Linear(32,24))
        def forward(self,x):return 1+.25*torch.tanh(self.network(x))
    def feature(self,raw,mask,times,delta,scale,mode='full',perms=None):
        z=self.cnn(raw[0,:,0,None]).flatten(1)*self.gate(delta[0,:,0]/scale)
        return torch.cat([z,z*2,z*3,z*4],1),z
    return SingleHead(arm,cnn,Gate(),feature,111)


def test_one_head_counts_and_shared_initialization():
    ms=[fixture(a) for a in ARMS]
    assert [sum(p.numel() for p in m.parameters()) for m in ms]==[10377,10377,10378]
    assert all(not hasattr(m,'base') and int(m.keep7.sum())==7 for m in ms)
    assert all(torch.equal(ms[0].head[0].weight,m.head[0].weight) for m in ms)
    assert all(sum(isinstance(x,nn.Linear) and x.out_features==1 for x in m.modules())==1 for m in ms)


def test_interaction_initial_function_and_gradients():
    z=torch.randn(8,96);x=torch.randn(8,14);a,b=fixture(ARMS[1]),fixture(ARMS[2])
    assert torch.equal(a.classify(z,x),b.classify(z,x))
    a.classify(z,x).square().mean().backward();b.classify(z,x).square().mean().backward()
    assert all(torch.equal(p.grad,dict(b.named_parameters())[n].grad) for n,p in a.named_parameters() if n.startswith('head.'))
    assert b.interaction_gain.grad.abs()>0


def test_both_modalities_used_and_no_coordinate_mask():
    for arm in ARMS:
        m=fixture(arm);z=torch.randn(8,96,requires_grad=True);x=torch.randn(8,14,requires_grad=True)
        m.classify(z,x).square().mean().backward()
        assert torch.isfinite(z.grad).all() and torch.isfinite(x.grad).all()
        assert torch.all(z.grad.abs().sum(0)>0) and torch.all(x.grad.abs().sum(0)>0)
        assert not torch.equal(m.classify(z,x),m.classify(z,x+1))


def test_exact_original_objective():
    x=torch.tensor([-.7,.3,.5,-.2],requires_grad=True);y=torch.tensor([0.,1.,1.,0.]);p=x.sigmoid()
    macro=.5*(2*(p*y).sum()/(p.sum()+y.sum())+2*((1-p)*(1-y)).sum()/((1-p).sum()+(1-y).sum()))
    expected=.75*torch.nn.functional.binary_cross_entropy_with_logits(x,y)+.25*(1-macro)
    assert torch.equal(objective(x,y,torch.tensor(1.)),expected)
