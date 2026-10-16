"""Four changed critical paths; none is evidence of classification improvement."""
import inspect
import torch
from common import TRAINED,state_hash
from model import Model,LogEnergy,objective


def test_exact_shared_fresh_weights_and_no_warmstate_argument():
    models=[Model(a,1) for a in TRAINED]
    assert len({state_hash(m.state_dict()) for m in models})==1
    assert all(sum(p.numel() for p in m.parameters())==12529 for m in models)
    assert list(inspect.signature(Model).parameters)==['arm','fold']
    assert all('matrix' not in m.state_dict() for m in models)


def test_logenergy_definition_and_finite_learned_response_gradients():
    h=torch.tensor([-2.,1.,0.,3.]).repeat(2,24,1).requires_grad_();pool=LogEnergy()
    assert torch.equal(pool(h),torch.log1p(h.square().mean(-1,keepdim=True)))
    assert not torch.equal(pool(h),h.mean(-1,keepdim=True))
    pool(h).sum().backward();assert torch.isfinite(h.grad).all() and h.grad.abs().sum()>0
    zero=torch.zeros(2,24,30,requires_grad=True);pool(zero).sum().backward();assert torch.isfinite(zero.grad).all()


def test_complete_patient_order_padding_and_backbone_gradients():
    torch.manual_seed(42);raw=torch.randn(2,5,6,500);mask=torch.ones(2,5,6,dtype=torch.bool)
    times=torch.tensor([[-4.,-2.,0.,2.,8.,10.]]*2);order=torch.tensor([2,0,4,1,3])
    for arm in ['WINDOW_LOGENERGY','PRE_LOGENERGY']:
        m=Model(arm,1);a=m(raw,mask,times);b=m(raw[:,order],mask[:,order],times)
        assert torch.allclose(a[order],b,atol=1e-6,rtol=1e-5)
        pad=torch.cat([raw,torch.full((2,5,2,500),999.)],2);pm=torch.cat([mask,torch.zeros(2,5,2,dtype=torch.bool)],2);pt=torch.cat([times,torch.full((2,2),100.)],1)
        assert torch.allclose(a,m(pad,pm,pt),atol=1e-6,rtol=1e-5)
        a.square().sum().backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
        assert m.cnn[0].weight.grad.abs().sum()>0 and m.head[2].weight.grad.abs().sum()>0


def test_registered_loss_is_macro_and_class_symmetric_when_balanced():
    x=torch.tensor([-.7,.3,.5,-.2],requires_grad=True);y=torch.tensor([0.,1.,1.,0.]);pos=torch.tensor(1.)
    a=objective(x,y,pos);assert torch.allclose(a,objective(-x,1-y,pos),atol=1e-7,rtol=1e-6)
    p=x.sigmoid();macro=(2*(p*y).sum()/(p.sum()+y.sum())+2*((1-p)*(1-y)).sum()/((1-p).sum()+(1-y).sum()))/2
    expected=.5*torch.nn.functional.binary_cross_entropy_with_logits(x,y)+.5*(1-macro)
    assert torch.equal(a,expected);a.backward();assert torch.isfinite(x.grad).all()
