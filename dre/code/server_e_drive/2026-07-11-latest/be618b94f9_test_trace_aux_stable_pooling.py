import torch
from neuroez_c.task2.trace_aux_stable.residual_pooling import sparsemax
def test_sparsemax_variable_length():
 for n in (1,3,40):assert torch.allclose(sparsemax(torch.randn(n)).sum(),torch.tensor(1.))
