import types
import numpy as np
import torch
from torch import nn
from model import Unified,ARMS,objective


class Source(nn.Module):
    def __init__(self):
        super().__init__();torch.manual_seed(71)
        self.cnn=nn.Sequential(nn.Conv1d(1,12,9,4),nn.GELU(),nn.Conv1d(12,24,7,4),nn.GELU(),nn.AdaptiveAvgPool1d(1))
        self.gate=nn.Sequential(nn.Linear(7,32),nn.GELU(),nn.Linear(32,24))
        self.base=nn.Module();self.base.first=nn.Linear(14,96);self.base.last=nn.Linear(96,1)
        self.base.register_buffer('multiplicities',torch.tensor([2]*8+[1]*6,dtype=torch.float64));self.base.n0=66
    def features(self,*args,**kwargs):raise AssertionError('Test does not replace the real Raw feature path')


def test_one_output_shared_initial_state_and_parameter_counts():
    source=Source();ms=[Unified(a,source) for a in ARMS]
    assert all(sum(p.numel() for p in m.parameters())==13961 for m in ms)
    assert all(sum(isinstance(q,nn.Linear) and q.out_features==1 for q in m.modules())==1 and not hasattr(m,'base') for m in ms)
    assert all(torch.equal(ms[0].state_dict()[k],m.state_dict()[k]) for m in ms for k in ms[0].state_dict())


def test_exact_duplicate_aware_normalization():
    m=Unified(ARMS[0],Source());x=torch.randn(9,14);weight=m.multiplicities.to(x.dtype)
    mu=(x*weight).sum(-1,keepdim=True)/88;var=(((x-mu).square()*weight).sum(-1,keepdim=True)+66*mu.square())/88
    assert torch.equal(m.normalized(x),(x-mu)/torch.sqrt(var+1e-5))
    # Explicit virtual88 calculation verifies the intended original readout geometry.
    copies=torch.repeat_interleave(x,weight.long(),dim=1);virtual=torch.cat([copies,x.new_zeros(len(x),66)],1)
    assert torch.allclose(mu,virtual.mean(-1,keepdim=True),atol=1e-6,rtol=1e-5)
    assert torch.allclose(var,virtual.var(-1,unbiased=False,keepdim=True),atol=1e-6,rtol=1e-5)


def test_feature_only_identity_and_stage_trainables():
    m=Unified(ARMS[0],Source()).eval();x=torch.randn(7,14);z=torch.randn(7,96)
    assert torch.equal(m.classify(z,x),m.physiology_only(x))
    m.stage('phys');assert sum(p.numel() for p in m.parameters() if p.requires_grad)==1537
    m.stage('fixed');assert sum(p.numel() for p in m.parameters() if p.requires_grad)==12424
    m.stage('joint');assert all(p.requires_grad for p in m.parameters())


def test_zero_raw_projection_opens_gradient_and_fixed_head_preserves_state():
    m=Unified(ARMS[2],Source()).eval();m.stage('fixed');before={n:p.detach().clone() for n,p in m.named_parameters() if not p.requires_grad}
    z=torch.randn(7,96,requires_grad=True);x=torch.randn(7,14)
    opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=.0003)
    opt.zero_grad();m.classify(z,x).square().mean().backward();assert m.raw_projection.weight.grad.abs().sum()>0 and z.grad.abs().sum()==0
    opt.step();z.grad=None;opt.zero_grad();m.classify(z,x).square().mean().backward();assert z.grad.abs().sum()>0
    assert all(torch.equal(p,before[n]) for n,p in m.named_parameters() if n in before)


def test_exact_macro_objective():
    x=torch.tensor([-.7,.3,.5,-.2],requires_grad=True);y=torch.tensor([0.,1.,1.,0.]);p=x.sigmoid()
    macro=.5*(2*(p*y).sum()/(p.sum()+y.sum())+2*((1-p)*(1-y)).sum()/((1-p).sum()+(1-y).sum()))
    expected=.75*torch.nn.functional.binary_cross_entropy_with_logits(x,y)+.25*(1-macro)
    assert torch.equal(objective(x,y,torch.tensor(1.)),expected)
