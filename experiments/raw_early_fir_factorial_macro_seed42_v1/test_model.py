"""Five neural paths plus factorial algebra; no synthetic classification claim."""
import inspect
import numpy as np
import torch
import common as c
from model import Model,ExtendedFirstConv,LogEnergy,collapsed_state,local_receptive_field,objective


def fixture():
    torch.manual_seed(43);raw=torch.randn(2,5,6,500);mask=torch.ones(2,5,6,dtype=torch.bool)
    times=torch.tensor([[-4.,-2.,0.,2.,8.,10.]]*2);return raw,mask,times


def test_shared_random_core_counts_and_signature():
    models=[Model(a,1) for a in c.FAMILY]
    assert len({c.state_hash(collapsed_state(m)) for m in models})==1
    assert [sum(p.numel() for p in m.parameters()) for m in models]==[8433,8433,9873,9873]
    assert list(inspect.signature(Model).parameters)==['arm','fold']
    assert all(m.compact and m.head[0].in_features==96 for m in models)
    assert all(torch.count_nonzero(m.cnn[0].outer)==0 for m in models[2:])


def test_exact_pair_initial_logits_and_shared_gradients():
    raw,mask,times=fixture()
    for short,long in [('SHORT_MEAN','LONG_MEAN'),('SHORT_ENERGY','LONG_ENERGY')]:
        a,b=Model(short,1),Model(long,1);x,y=a(raw,mask,times),b(raw,mask,times);assert torch.equal(x,y)
        w=torch.arange(1,6);(x*w).sum().backward();(y*w).sum().backward()
        bg={n:p.grad for n,p in b.named_parameters() if not n.startswith('cnn.0.')}
        bg.update({'cnn.0.weight':b.cnn[0].center.weight.grad,'cnn.0.bias':b.cnn[0].center.bias.grad})
        assert all(torch.equal(p.grad,bg[n]) for n,p in a.named_parameters())
        assert torch.isfinite(b.cnn[0].outer.grad).all() and b.cnn[0].outer.grad.abs().sum()>0


def test_early_support_and_exact_learned_energy():
    assert [local_receptive_field(a) for a in c.FAMILY]==[33,33,153,153]
    x=torch.randn(3,24,30,requires_grad=True)
    assert torch.equal(LogEnergy()(x),torch.log1p(x.square().mean(-1,keepdim=True)))
    LogEnergy()(x).sum().backward();assert torch.isfinite(x.grad).all()
    raw=torch.randn(1,1,500,requires_grad=True);short=Model('SHORT_MEAN',1).cnn[0];long=Model('LONG_MEAN',1).cnn[0]
    assert short(raw).shape==long(raw).shape==(1,12,123) and torch.equal(short(raw),long(raw))
    with torch.no_grad():long.outer[...,-1]=.1
    a=torch.autograd.grad(short(raw)[0,:,0].sum(),raw,retain_graph=True)[0]
    b=torch.autograd.grad(long(raw)[0,:,0].sum(),raw)[0]
    assert a[0,0,68]==0 and b[0,0,68].abs()>0


def test_complete_raw_order_masked_padding_and_gradients():
    raw,mask,times=fixture();order=torch.tensor([2,0,4,1,3])
    for arm in c.FAMILY:
        m=Model(arm,1);a=m(raw,mask,times);b=m(raw[:,order],mask[:,order],times)
        assert torch.allclose(a[order],b,atol=1e-6,rtol=1e-5)
        pad=torch.cat([raw,torch.full((2,5,2,500),float('nan'))],2)
        pm=torch.cat([mask,torch.zeros(2,5,2,dtype=torch.bool)],2);pt=torch.cat([times,torch.full((2,2),float('nan'))],1)
        assert torch.allclose(a,m(pad,pm,pt),atol=1e-6,rtol=1e-5)
        (a*torch.arange(1,6)).sum().backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
        assert m.cnn[0].weight.grad.abs().sum()>0 and m.head[2].weight.grad.abs().sum()>0
        if arm.startswith('LONG'):assert m.cnn[0].outer.grad.abs().sum()>0


def test_exact_macro_objective():
    x=torch.tensor([-.7,.3,.5,-.2],requires_grad=True);y=torch.tensor([0.,1.,1.,0.]);p=x.sigmoid()
    macro=(2*(p*y).sum()/(p.sum()+y.sum())+2*((1-p)*(1-y)).sum()/((1-p).sum()+(1-y).sum()))/2
    expected=.5*torch.nn.functional.binary_cross_entropy_with_logits(x,y)+.5*(1-macro)
    assert torch.equal(objective(x,y,torch.tensor(1.)),expected)
    expected.backward();assert torch.isfinite(x.grad).all()


def test_factorial_contrast_algebra():
    result=c.factorial_values(np.array([0.,2.]),np.array([1.,2.]),np.array([2.,3.]),np.array([4.,3.]))
    assert np.array_equal(result['kernel_main'],[2.5,1.])
    assert np.array_equal(result['energy_main'],[1.5,0.])
    assert np.array_equal(result['interaction'],[1.,0.])
