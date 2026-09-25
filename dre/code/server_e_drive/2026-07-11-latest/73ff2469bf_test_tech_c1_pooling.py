import torch
from neuroez_c.task2.tech_outcome_c1.masked_ops import masked_std
def test_singleton_std_is_zero():assert masked_std(torch.randn(1,96),torch.ones(1,dtype=torch.bool),0).eq(0).all()
