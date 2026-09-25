import torch
from neuroez_c.task2.trace_rawmil.film import PhaseFiLM
def test_film_identity_initialization():
 x=torch.randn(4,32);assert torch.allclose(PhaseFiLM()(x,torch.tensor([0,1,2,0])),x)
