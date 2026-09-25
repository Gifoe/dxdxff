import torch
from neuroez_c.task2.trace_rawmil.model import TRACERawMIL
def test_model_runs_with_ez_and_nez():
 m=TRACERawMIL();p=[[(torch.randn(6,1,500),torch.tensor([0,0,1,1,2,2]),torch.tensor([0,0,1,1,2,2]),torch.tensor([0,0,0,1,1,1]))]];o=m(p)[0];assert torch.isfinite(o['rho']) and torch.allclose(o['details'][0]['alpha'].sum(),torch.tensor(1.))
