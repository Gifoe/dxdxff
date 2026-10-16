"""Outcome-free regression checks, not classification evidence."""
import copy
import torch
from model import Model,ResidualPool,objective,pure


def reference():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42);m=pure.PureRaw()
    return dict(keep32=torch.zeros(32),**m.state_dict())


def test_zero_identity_and_capacity():
    state=reference();a=Model('MEAN_TUNE',1,state);b=Model('MEAN_RESIDUAL',1,state);c=Model('DISPERSION',1,state)
    x=torch.randn(8,1,500)
    assert torch.equal(a.cnn(x),b.cnn(x)) and torch.equal(a.cnn(x),c.cnn(x))
    assert sum(p.numel() for p in a.parameters())==12529
    assert sum(p.numel() for p in b.parameters())==sum(p.numel() for p in c.parameters())==13105
    for k,v in a.state_dict().items():assert torch.equal(v,b.state_dict()[k]) and torch.equal(v,c.state_dict()[k])


def test_dispersion_is_not_only_mean_and_gradients_finite():
    m=ResidualPool('DISPERSION')
    with torch.no_grad():m.mix.weight.copy_(torch.eye(24)[:,:,None])
    x=torch.tensor([-1.,1.,-1.,1.]).repeat(2,24,1).requires_grad_();flat=torch.zeros_like(x)
    assert not torch.equal(m(x),m(flat))
    m(x).sum().backward();assert torch.isfinite(x.grad).all()
    constant=torch.zeros(2,24,4,requires_grad=True);m(constant).sum().backward()
    assert torch.isfinite(constant.grad).all()


def test_pool_padding_channel_order_and_all_channel_gradients():
    torch.manual_seed(42);state=reference();m=Model('DISPERSION',1,state)
    emb=torch.randn(2,5,6,24,requires_grad=True);mask=torch.ones(2,5,6,dtype=torch.bool)
    times=torch.tensor([[-4.,-2.,0.,2.,8.,10.]]*2)
    z=m.aggregate(emb,mask,times)
    order=torch.tensor([2,0,4,1,3]);other=m.aggregate(emb[:,order],mask[:,order],times)
    assert torch.allclose(z[order],other,atol=1e-6,rtol=1e-5)
    pad=torch.cat([emb,torch.full((2,5,2,24),999.)],2)
    pm=torch.cat([mask,torch.zeros(2,5,2,dtype=torch.bool)],2)
    pt=torch.cat([times,torch.full((2,2),100.)],1)
    assert torch.allclose(z,m.aggregate(pad,pm,pt),atol=1e-6,rtol=1e-5)
    m.classify(z).square().sum().backward();assert torch.isfinite(emb.grad).all()
    assert all(emb.grad[:,j].abs().sum()>0 for j in range(5))


def test_macro_loss_uses_both_classes():
    x=torch.tensor([-.7,.3,.5,-.2],requires_grad=True);y=torch.tensor([0.,1.,1.,0.]);pos=torch.tensor(1.)
    a=objective(x,y,pos);b=objective(-x,1-y,pos)
    assert torch.allclose(a,b,atol=1e-7,rtol=1e-6)
    a.backward();assert torch.isfinite(x.grad).all() and (x.grad.abs()>0).all()
