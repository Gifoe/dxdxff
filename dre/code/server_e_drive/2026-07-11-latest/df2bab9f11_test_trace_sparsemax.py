import torch
from neuroez_c.task2.trace_rawmil.sparsemax import sparsemax
def test_sparsemax_properties():
 x=sparsemax(torch.tensor([[1.,0.,-3.]]));assert torch.all(x>=0) and torch.allclose(x.sum(1),torch.ones(1)) and (x==0).any()
