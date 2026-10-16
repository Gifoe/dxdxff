"""Unique normalization, single-head/cache and differentiable-pooling checks."""
import copy
import pytest
import torch
from model import Model,reference_wave,objective,pure


def example():
    torch.manual_seed(42);raw=torch.randn(2,5,6,500)
    mask=torch.ones(2,5,6,dtype=torch.bool);times=torch.tensor([[-4.,-2.,0.,2.,8.,10.]]*2)
    return raw,mask,times


def state():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42);m=pure.PureRaw()
    return dict(keep32=torch.zeros(32),**m.state_dict())


def test_normalizations_gain_and_pre_only_reference():
    raw,mask,times=example();body=Model('NATIVE',1,state())
    a,_=reference_wave(raw,mask,times);gain=raw.clone();gain[:,:,2:]*=4
    b,_=reference_wave(gain,mask,times)
    # Pre samples cannot change their normalization when only post samples change.
    index=mask.nonzero()[:,2]<2
    assert torch.equal(a[index],b[index]) and b[~index].abs().mean()>a[~index].abs().mean()
    assert torch.allclose(body.normalized_windows(raw,mask,times),body.normalized_windows(gain,mask,times),atol=1e-6,rtol=1e-5)
    scaled,_=reference_wave(raw*8,mask,times);assert torch.equal(a,scaled)


def test_zero_pre_channels_and_unidentified_stop():
    raw,mask,times=example();raw[:,0,:2]=0
    q,audit=reference_wave(raw,mask,times);assert torch.isfinite(q).all() and audit['zero_pre_channels']==2
    raw[0,:,:2]=0;q,audit=reference_wave(raw,mask,times)
    assert torch.isfinite(q).all() and audit['fallback_seizures']==1
    raw[1,:,:2]=0
    with pytest.raises(AssertionError,match='unidentified'):reference_wave(raw,mask,times)


def test_identity_capacity_and_raw_cached_gradient():
    raw,mask,times=example();ref=state();native=Model('NATIVE',1,ref);control=Model('WINDOW_RESIDUAL',1,ref);m=Model('PRECOMPAND_RESIDUAL',1,ref)
    q,_=m.encode(raw,mask,times)
    assert torch.equal(native.forward_stats(q),control.forward_stats(q)) and torch.equal(native.forward_stats(q),m.forward_stats(q))
    assert sum(p.numel() for p in native.parameters())==12529
    assert sum(p.numel() for p in m.parameters())==sum(p.numel() for p in control.parameters())==13105
    assert sum(p.numel() for p in m.parameters() if p.requires_grad)==576
    with torch.no_grad():m.matrix.copy_(torch.linspace(-.01,.01,576).reshape(24,24))
    other=copy.deepcopy(m);y=torch.tensor([0.,1.,0.,1.,1.]);pos=torch.tensor(2/3)
    a=m(raw,mask,times);b=other.forward_stats(q)
    assert torch.equal(a,b)
    x=objective(a,y,pos);z=objective(b,y,pos);x.backward();z.backward()
    assert torch.equal(m.matrix.grad,other.matrix.grad) and torch.isfinite(m.matrix.grad).all() and m.matrix.grad.abs().sum()>0
    assert all(p.grad is None for p in m.cnn.parameters()) and all(p.grad is None for p in m.head.parameters())


def test_complete_channel_order_and_padding():
    raw,mask,times=example();m=Model('PRECOMPAND_RESIDUAL',1,state())
    with torch.no_grad():m.matrix.copy_(torch.linspace(-.01,.01,576).reshape(24,24))
    a=m(raw,mask,times);order=torch.tensor([2,4,0,3,1]);b=m(raw[:,order],mask[:,order],times)
    assert torch.allclose(a[order],b,atol=1e-6,rtol=1e-5)
    padded=torch.cat([raw,torch.full((2,5,2,500),999.)],2);pm=torch.cat([mask,torch.zeros(2,5,2,dtype=torch.bool)],2)
    pt=torch.cat([times,torch.full((2,2),100.)],1)
    assert torch.allclose(a,m(padded,pm,pt),atol=1e-6,rtol=1e-5)
