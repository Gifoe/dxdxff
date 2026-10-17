"""Five unique altered executable paths, not classification evidence."""
import inspect
import numpy as np
import torch
import common as c
from model import Model,ResidualPool,core_state_hash,local_receptive_field,pure,normalizer,objective


def fixture():
    torch.manual_seed(43);raw=torch.randn(2,5,6,500);mask=torch.ones(2,5,6,dtype=torch.bool)
    times=torch.tensor([[-4.,-2.,0.,2.,8.,10.]]*2)
    return raw,mask,times


def test_matched_random_core_and_parameter_counts():
    models=[Model(a,1) for a in c.TRAINED]
    assert len({core_state_hash(m) for m in models})==1
    assert [sum(p.numel() for p in m.parameters()) for m in models]==[8433,9417,9417]
    assert list(inspect.signature(Model).parameters)==['arm','fold']
    assert all(m.compact and m.head[0].in_features==96 and len(m.head)==3 for m in models)
    assert sum(p.numel() for p in models[1].cnn[4].parameters())==sum(p.numel() for p in models[2].cnn[4].parameters())==984
    assert not any('keep32' in m.state_dict() for m in models)


def test_zero_output_identity_and_exact_initial_logits():
    x=torch.randn(3,24,30);expected=torch.nn.functional.adaptive_avg_pool1d(x,1)
    for arm in c.TRAINED[1:]:assert torch.equal(ResidualPool(arm)(x),expected)
    raw,mask,times=fixture();models=[Model(a,1) for a in c.TRAINED]
    outputs=[m(raw,mask,times) for m in models]
    assert all(torch.equal(outputs[0],q) for q in outputs[1:])
    old=pure.PureRaw(compact=True);old.load_state_dict(models[0].state_dict())
    flat=normalizer.reference_wave(raw,mask,times)[0];emb=raw.new_zeros((*mask.shape,24));emb[mask]=old.cnn(flat[:,None]).flatten(1)
    assert torch.equal(old.classify(old.aggregate(emb,mask,times)),outputs[0])


def test_temporal_and_pointwise_dependency_support():
    assert [local_receptive_field(a) for a in c.TRAINED]==[33,33,289]
    for arm in c.TRAINED[1:]:
        p=ResidualPool(arm)
        # Test-only nonzero filters demonstrate representable dependencies;
        # formal training always uses the registered zero-output initialization.
        with torch.no_grad():
            p.adapter[0].weight.fill_(.02);p.adapter[2].weight.fill_(.03)
            if p.adapter[2].bias is not None:p.adapter[2].bias.zero_()
        x=torch.ones(2,24,30,requires_grad=True);y=p.adapter(x)[...,15].sum();gradient=torch.autograd.grad(y,x)[0]
        off=gradient.clone();off[...,15]=0
        if arm=='FRESH_POINTWISE':assert torch.count_nonzero(off)==0
        else:assert off[...,7:24].abs().sum()>0 and torch.count_nonzero(off[...,:7])==torch.count_nonzero(off[...,24:])==0


def test_full_raw_order_padding_and_gradients():
    raw,mask,times=fixture();order=torch.tensor([2,0,4,1,3])
    for arm in c.TRAINED:
        m=Model(arm,1);a=m(raw,mask,times);b=m(raw[:,order],mask[:,order],times)
        assert torch.allclose(a[order],b,atol=1e-6,rtol=1e-5)
        pad=torch.cat([raw,torch.full((2,5,2,500),float('nan'))],2)
        pm=torch.cat([mask,torch.zeros(2,5,2,dtype=torch.bool)],2);pt=torch.cat([times,torch.full((2,2),float('nan'))],1)
        assert torch.allclose(a,m(pad,pm,pt),atol=1e-6,rtol=1e-5)
        (a*torch.arange(1,6)).sum().backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
        assert m.cnn[0].weight.grad.abs().sum()>0 and m.head[2].weight.grad.abs().sum()>0
        if arm!='FRESH_BASE':assert m.cnn[4].adapter[2].weight.grad.abs().sum()>0


def test_exact_macro_objective():
    x=torch.tensor([-.7,.3,.5,-.2],requires_grad=True);y=torch.tensor([0.,1.,1.,0.]);p=x.sigmoid()
    macro=(2*(p*y).sum()/(p.sum()+y.sum())+2*((1-p)*(1-y)).sum()/((1-p).sum()+(1-y).sum()))/2
    expected=.5*torch.nn.functional.binary_cross_entropy_with_logits(x,y)+.5*(1-macro)
    assert torch.equal(objective(x,y,torch.tensor(1.)),expected)
    expected.backward();assert torch.isfinite(x.grad).all()
