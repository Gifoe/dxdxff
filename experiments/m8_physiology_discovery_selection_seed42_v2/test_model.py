import copy
from types import SimpleNamespace
import torch
from torch import nn
from model import FeatureHead,M8,Concat


def original():
    cnn=nn.Sequential(nn.Conv1d(1,12,9,4),nn.GELU(),nn.Conv1d(12,24,7,4),nn.GELU(),nn.AdaptiveAvgPool1d(1))
    head=nn.Sequential(nn.Linear(96,64),nn.GELU(),nn.Linear(64,1))
    gate=SimpleNamespace(network=nn.Sequential(nn.Linear(7,32),nn.GELU(),nn.Linear(32,24)))
    with torch.no_grad():
        head[2].weight.zero_();head[2].bias.zero_();gate.network[2].weight.zero_();gate.network[2].bias.zero_()
    return SimpleNamespace(cnn=cnn,head=head,gate=gate)


def data():
    torch.manual_seed(42)
    raw=torch.randn(2,7,6,500)*1e-4;mask=torch.ones(2,7,6,dtype=torch.bool)
    times=torch.tensor([[-3,-2,-1,1,5,10]]*2)
    x=torch.randn(7,64);delta=torch.randn(2,7,6,32);scale=torch.ones(32)
    return x,raw,mask,times,delta,scale


def test_fixed_capacity_and_single_head():
    a=original();base=FeatureHead(42);m=M8(base,a,[5,18],42);s=Concat(a,[5,18],42)
    assert sum(p.numel() for p in m.parameters())==16618
    assert sum(p.numel() for p in s.parameters())==12529 and not hasattr(s,'base')
    assert len([x for x in s.modules() if isinstance(x,nn.Linear)])==2


def test_masked_feature_does_not_affect_trained_state():
    x,raw,mask,times,delta,scale=data();a=original()
    m=M8(FeatureHead(42),a,[5,18],42).eval()
    with torch.no_grad():
        m.head[2].weight.normal_();m.gate.network[2].weight.normal_(std=.01)
        first=m(x,raw,mask,times,delta,scale)[0]
        x[:,0:2]+=100;delta[...,0]+=100
        second=m(x,raw,mask,times,delta,scale)[0]
    torch.testing.assert_close(first,second,rtol=0,atol=0)


def test_raw_only_no_physiology_and_initial_m8_identity():
    x,raw,mask,times,delta,scale=data();a=original()
    s=Concat(a,[],42).eval();m=M8(FeatureHead(42),a,[5],42).eval()
    with torch.no_grad():
        p=s(x,raw,mask,times,delta,scale)[0]
        torch.testing.assert_close(p,s(x+123,raw,mask,times,delta+456,scale)[0],rtol=0,atol=0)
        q=m(x,raw,mask,times,delta,scale)
        torch.testing.assert_close(q[0],q[2],rtol=0,atol=0)
