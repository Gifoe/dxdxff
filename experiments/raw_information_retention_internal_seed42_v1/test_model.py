import copy
import numpy as np
import pytest
import torch
from model import Model,ARMS,normalize,pre_scale,local_receptive_field
from engine import direct,cached


def inputs():
    torch.manual_seed(71);raw=torch.randn(2,5,6,500)*.1
    mask=torch.ones(2,5,6,dtype=torch.bool);mask[1,3:]=False
    times=torch.tensor([[-3.,-1,1,5,9,15]]).repeat(2,1)
    return raw,mask,times


def test_amplitude_retention_and_units():
    raw,mask,times=inputs();post=(times[:,None]>=0)&mask
    scaled=raw.clone();scaled[post]*=2
    assert torch.allclose(normalize(raw,mask,times,'window'),normalize(scaled,mask,times,'window'),atol=1e-6,rtol=1e-6)
    assert not torch.allclose(normalize(raw,mask,times,'pre'),normalize(scaled,mask,times,'pre'))
    for scale in [2.**-3,2.**3]:
        assert torch.equal(normalize(raw,mask,times,'pre'),normalize(raw*scale,mask,times,'pre'))
    assert local_receptive_field(False)==33 and local_receptive_field(True)==535


def test_zero_pre_fallback_and_padding():
    raw,mask,times=inputs();raw[0,0,:2]=0
    scale,info=pre_scale(raw,mask,times);assert info['constant_pre_channels'][0,0] and scale[0,0]>0
    padded=raw.clone();padded[~mask]=1000
    assert torch.equal(normalize(raw,mask,times,'pre'),normalize(padded,mask,times,'pre'))
    allzero=raw.clone();allzero[:,:,:2]=0
    with pytest.raises(AssertionError):pre_scale(allzero,mask,times)


def test_initialization_single_classifier_and_channel_order():
    a=Model('C3_PRE_LONG');b=Model('C4_PRE_LONG_CONTEXT')
    for k,v in a.state_dict().items():assert torch.equal(v,b.state_dict()[k])
    z=torch.randn(5,96);assert torch.equal(a.classify(z),b.classify(z))
    assert not hasattr(b,'base') and not hasattr(b,'D1') and len(b.head)==3
    permutation=torch.tensor([3,0,4,1,2])
    assert torch.allclose(b.classify(z)[permutation],b.classify(z[permutation]),atol=1e-6)


def test_all_arms_full_gradient_cache():
    raw,mask,times=inputs();y=torch.tensor([0.,1.,0.,1.,1.]);pos=torch.tensor(2/3)
    for arm in ARMS:
        a=Model(arm);b=copy.deepcopy(a)
        loss,score,z=direct(a,raw,mask,times,y,pos,chunk=13);loss.backward()
        other,ss,zz=cached(b,raw,mask,times,y,pos,chunk=13)
        assert torch.equal(score,ss) and torch.equal(z,zz) and other==float(loss.detach())
        for (name,p),(nn,q) in zip(a.named_parameters(),b.named_parameters()):
            assert name==nn and (p.grad is None)==(q.grad is None)
            if p.grad is not None:assert torch.allclose(p.grad,q.grad,atol=1e-6,rtol=1e-5), (arm,name)
