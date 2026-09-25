import torch
from neuroez_c.task2.true_dualset.set_pooling import dual_statistics,normalized_lse
from neuroez_c.task2.true_dualset.interaction import BidirectionalCrossSet
def test_singleton_std_and_normalized_lse():
 x=torch.ones(1,48);z,p=dual_statistics(x);assert p['std'].eq(0).all();assert torch.allclose(normalized_lse(x),normalized_lse(x.repeat(20,1)))
def test_cross_set_shapes_finite_backward():
 ez=torch.randn(3,48,requires_grad=True);nez=torch.randn(4,48,requires_grad=True);a,b,_,_=BidirectionalCrossSet()(ez,nez);(a.sum()+b.sum()).backward();assert a.shape==(48,) and b.shape==(48,) and torch.isfinite(a).all() and ez.grad is not None and nez.grad is not None
