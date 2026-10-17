"""Five unique changed paths; synthetic checks are not classification results."""
import inspect
import numpy as np
import torch
import common as c
from model import Model,views,patient_z,aggregate,pure,objective


def fixture():
    torch.manual_seed(42);emb=torch.randn(3,5,6,24);mask=torch.ones(3,5,6,dtype=torch.bool)
    times=torch.tensor([[-4.,-2.,0.,2.,8.,10.]]*3)
    mask[2,1]=False;emb[2,1]=0
    return emb,mask,times


def test_equal_initial_states_and_shape():
    models=[Model(a,1) for a in c.TRAINED]
    assert len({c.state_hash(m.state_dict()) for m in models})==1
    assert all(sum(p.numel() for p in m.parameters())==14577 for m in models)
    assert list(inspect.signature(Model).parameters)==['arm','fold']
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42+1009);old=pure.PureRaw()
    for m in models:
        for name in ['0.weight','0.bias','2.weight','2.bias']:assert torch.equal(m.cnn.state_dict()[name],old.cnn.state_dict()[name])


def test_original_delta_exact_output_and_gradient():
    value,mask,times=fixture();a=value.clone().requires_grad_();b=value.clone().requires_grad_()
    expected=pure.PureRaw().aggregate(a,mask,times);actual=patient_z(views(b,mask,times)[0])
    assert torch.equal(expected,actual)
    w=torch.arange(expected.numel(),dtype=torch.float32).reshape_as(expected)
    (expected*w).sum().backward();(actual*w).sum().backward();assert torch.equal(a.grad,b.grad)
    z=aggregate(value,mask,times,'DELTA_DUP');assert torch.equal(z[:,:96],z[:,96:])


def test_baseline_oracle_and_retained_information():
    emb,mask,times=fixture();delta,post,baseline=views(emb,mask,times)
    oracle=[]
    for channel in range(5):
        b=np.stack([emb[s,channel,mask[s,channel]&(times[s]<0)].numpy().mean(0) for s in range(3) if mask[s,channel].any()])
        oracle.append(np.stack([b.mean(0),b.std(0,ddof=0)]).reshape(-1))
    assert np.allclose(baseline.numpy(),np.stack(oracle),atol=1e-7,rtol=1e-5)
    # Same added per-channel latent state in every seizure/window cancels in
    # delta, but not in pre/post. Constant-coordinate treatment remains exact.
    shift=torch.arange(5,dtype=emb.dtype)[None,:,None,None]/4
    changed=(emb+shift)*mask[...,None];d2,p2,b2=views(changed,mask,times)
    assert torch.allclose(delta,d2,atol=1e-6,rtol=1e-5)
    assert not torch.allclose(baseline,b2) and not torch.allclose(post,p2)
    out=aggregate(emb,mask,times,'PRE_DELTA')
    assert torch.equal(out[:,:96],patient_z(delta))
    assert torch.equal(out[:,96:144],patient_z(baseline)) and torch.equal(out[:,96:144],out[:,144:])


def test_order_masked_nan_padding_and_full_gradients():
    torch.manual_seed(43);raw=torch.randn(2,5,6,500);mask=torch.ones(2,5,6,dtype=torch.bool)
    times=torch.tensor([[-4.,-2.,0.,2.,8.,10.]]*2);order=torch.tensor([2,0,4,1,3])
    for arm in c.TRAINED:
        m=Model(arm,1);a=m(raw,mask,times);b=m(raw[:,order],mask[:,order],times)
        assert torch.allclose(a[order],b,atol=1e-6,rtol=1e-5)
        pad=torch.cat([raw,torch.full((2,5,2,500),float('nan'))],2)
        pm=torch.cat([mask,torch.zeros(2,5,2,dtype=torch.bool)],2);pt=torch.cat([times,torch.full((2,2),float('nan'))],1)
        assert torch.allclose(a,m(pad,pm,pt),atol=1e-6,rtol=1e-5)
        (a*torch.arange(1,6)).sum().backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
        assert m.cnn[0].weight.grad.abs().sum()>0 and m.head[2].weight.grad.abs().sum()>0


def test_exact_macro_loss():
    x=torch.tensor([-.7,.3,.5,-.2],requires_grad=True);y=torch.tensor([0.,1.,1.,0.]);p=x.sigmoid()
    macro=(2*(p*y).sum()/(p.sum()+y.sum())+2*((1-p)*(1-y)).sum()/((1-p).sum()+(1-y).sum()))/2
    expected=.5*torch.nn.functional.binary_cross_entropy_with_logits(x,y)+.5*(1-macro)
    assert torch.equal(objective(x,y,torch.tensor(1.)),expected)
    expected.backward();assert torch.isfinite(x.grad).all()
