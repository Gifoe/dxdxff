"""Four critical-path tests; not classification benefit evidence."""
import torch
from model import Model,ResidualPool,old,objective


def reference():
    with torch.random.fork_rng(devices=[]):torch.manual_seed(42);m=old.pure.PureRaw()
    return dict(keep32=torch.zeros(32),**m.state_dict())


def test_zero_identity_equal_capacity_and_shared_weights():
    state=reference();matrix=torch.eye(24)*.1
    a=Model('PRE_TUNE',1,state,matrix);b=Model('CHANNEL_MLP',1,state,matrix);c=Model('TEMPORAL_CONTEXT',1,state,matrix)
    x=torch.randn(8,1,500)
    assert torch.equal(a.cnn(x),b.cnn(x)) and torch.equal(a.cnn(x),c.cnn(x))
    assert sum(p.numel() for p in a.parameters())==13105
    assert sum(p.numel() for p in b.parameters())==sum(p.numel() for p in c.parameters())==14089
    for n,v in a.state_dict().items():assert torch.equal(v,b.state_dict()[n]) and torch.equal(v,c.state_dict()[n])


def test_temporal_order_vs_pointwise_capacity_control():
    torch.manual_seed(42);x=torch.randn(3,24,30);order=torch.randperm(30)
    a=ResidualPool('CHANNEL_MLP');b=ResidualPool('TEMPORAL_CONTEXT')
    with torch.no_grad():
        a.adapter[2].weight.normal_(0,.1)
        b.adapter[0].weight.fill_(1/17);b.adapter[2].weight.copy_(torch.eye(24)[:,:,None]*.2)
    assert torch.allclose(a(x),a(x[:,:,order]),atol=1e-6,rtol=1e-5)
    assert float((b(x)-b(x[:,:,order])).abs().max())>1e-5


def test_fullpatient_mask_order_and_gradients():
    torch.manual_seed(42);m=Model('TEMPORAL_CONTEXT',1,reference(),torch.eye(24)*.1)
    raw=torch.randn(2,5,6,500);mask=torch.ones(2,5,6,dtype=torch.bool);times=torch.tensor([[-4.,-2.,0.,2.,8.,10.]]*2)
    order=torch.tensor([2,0,4,1,3]);a=m(raw,mask,times);b=m(raw[:,order],mask[:,order],times)
    assert torch.allclose(a[order],b,atol=1e-6,rtol=1e-5)
    padded=torch.cat([raw,torch.full((2,5,2,500),999.)],2);pm=torch.cat([mask,torch.zeros(2,5,2,dtype=torch.bool)],2)
    pt=torch.cat([times,torch.full((2,2),100.)],1)
    assert torch.allclose(a,m(padded,pm,pt),atol=1e-6,rtol=1e-5)
    a.square().sum().backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    assert m.cnn[4].adapter[2].weight.grad.abs().sum()>0


def test_loss_both_classes_and_exact_registered_weights():
    x=torch.tensor([-.7,.3,.5,-.2],requires_grad=True);y=torch.tensor([0.,1.,1.,0.]);pos=torch.tensor(1.)
    a=objective(x,y,pos);b=objective(-x,1-y,pos)
    assert torch.allclose(a,b,atol=1e-7,rtol=1e-6)
    p=x.sigmoid();macro=((2*(p*y).sum()/(p.sum()+y.sum()))+(2*((1-p)*(1-y)).sum()/((1-p).sum()+(1-y).sum())))/2
    expected=.5*torch.nn.functional.binary_cross_entropy_with_logits(x,y)+.5*(1-macro)
    assert torch.equal(a,expected)
    a.backward();assert torch.isfinite(x.grad).all() and (x.grad.abs()>0).all()
