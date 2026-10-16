"""Only unique changed critical paths; not classification evidence."""
import inspect
import numpy as np
import torch
import common as c
from model import Model,MomentPool,aggregate,pure,objective


def test_equal_fresh_initial_states_and_counts():
    mm=[Model(a,1) for a in c.TRAINED]
    assert len({c.state_hash(m.state_dict()) for m in mm})==1
    assert all(sum(p.numel() for p in m.parameters())==14577 for m in mm)
    assert list(inspect.signature(Model).parameters)==['arm','fold']
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42+1009);old=pure.PureRaw()
    for m in mm:
        for name in ['0.weight','0.bias','2.weight','2.bias']:assert torch.equal(m.cnn.state_dict()[name],old.cnn.state_dict()[name])


def test_exact_pool_contents_and_finite_gradients():
    h=torch.tensor([-2.,1.,0.,3.]).repeat(2,24,1).requires_grad_();mu=h.mean(-1,keepdim=True);e=torch.log1p(h.square().mean(-1,keepdim=True))
    expected=[torch.cat([mu,mu],1),torch.cat([e,e],1),torch.cat([mu,e],1)]
    for arm,value in zip(c.TRAINED,expected):assert torch.equal(MomentPool(arm)(h),value)
    expected[-1].sum().backward();assert torch.isfinite(h.grad).all() and h.grad.abs().sum()>0
    z=torch.zeros(2,24,8,requires_grad=True);MomentPool('PRE_JOINT')(z).sum().backward();assert torch.isfinite(z.grad).all()


def test_generalized_aggregation_exact_output_and_gradient():
    torch.manual_seed(42);a=torch.randn(2,5,6,24,requires_grad=True);b=a.detach().clone().requires_grad_()
    mask=torch.ones(2,5,6,dtype=torch.bool);times=torch.tensor([[-4.,-2.,0.,2.,8.,10.]]*2)
    x=pure.PureRaw().aggregate(a,mask,times);y=aggregate(b,mask,times);assert torch.equal(x,y)
    w=torch.randn_like(x);(x*w).sum().backward();(y*w).sum().backward();assert torch.equal(a.grad,b.grad)
    duplicate=aggregate(torch.cat([a.detach(),a.detach()],-1),mask,times).reshape(5,4,48)
    assert torch.equal(duplicate[:,:,:24],duplicate[:,:,24:])


def test_patient_permutation_masked_nan_padding_full_gradients():
    torch.manual_seed(43);raw=torch.randn(2,5,6,500);mask=torch.ones(2,5,6,dtype=torch.bool);times=torch.tensor([[-4.,-2.,0.,2.,8.,10.]]*2)
    order=torch.tensor([2,0,4,1,3])
    for arm in c.TRAINED:
        m=Model(arm,1);a=m(raw,mask,times);b=m(raw[:,order],mask[:,order],times)
        assert torch.allclose(a[order],b,atol=1e-6,rtol=1e-5)
        pad=torch.cat([raw,torch.full((2,5,2,500),999.)],2);pm=torch.cat([mask,torch.zeros(2,5,2,dtype=torch.bool)],2)
        pt=torch.cat([times,torch.full((2,2),float('nan'))],1);assert torch.allclose(a,m(pad,pm,pt),atol=1e-6,rtol=1e-5)
        (a*torch.arange(1,6)).sum().backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
        assert m.cnn[0].weight.grad.abs().sum()>0 and m.head[2].weight.grad.abs().sum()>0


def test_macro_loss_exact_definition():
    x=torch.tensor([-.7,.3,.5,-.2],requires_grad=True);y=torch.tensor([0.,1.,1.,0.]);p=x.sigmoid()
    macro=(2*(p*y).sum()/(p.sum()+y.sum())+2*((1-p)*(1-y)).sum()/((1-p).sum()+(1-y).sum()))/2
    expected=.5*torch.nn.functional.binary_cross_entropy_with_logits(x,y)+.5*(1-macro)
    assert torch.equal(objective(x,y,torch.tensor(1.)),expected)
    expected.backward();assert torch.isfinite(x.grad).all()
