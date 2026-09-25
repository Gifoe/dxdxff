import torch
from neuroez_c.task2.trace_rawmil.model import TRACERawMIL
def test_seizure_permutation_invariant():
 torch.manual_seed(1);m=TRACERawMIL().eval();a=(torch.randn(4,1,500),torch.tensor([0,1,0,1]),torch.tensor([0,0,1,1]),torch.tensor([0,0,1,1]));b=(torch.randn(4,1,500),torch.tensor([0,1,0,1]),torch.tensor([0,0,1,1]),torch.tensor([0,0,1,1]));assert torch.allclose(m([[a,b]])[0]['logit'],m([[b,a]])[0]['logit'])
