import torch
from neuroez_c.task2.tech_outcome_c1.masked_ops import masked_softmax,masked_mean,masked_max,masked_std
def test_masked_ops_exclude_padding_and_are_finite():
 x=torch.tensor([[[1.,-2.],[3.,-4.],[999.,999.]]]);m=torch.tensor([[1,1,0]],dtype=torch.bool);w=masked_softmax(x,m,1);assert w[:,2].eq(0).all() and torch.allclose(w.sum(1),torch.ones(1,2));assert torch.isfinite(masked_mean(x,m,1)).all() and torch.equal(masked_max(x,m,1),torch.tensor([[3.,-2.]])) and torch.isfinite(masked_std(x,m,1)).all()
