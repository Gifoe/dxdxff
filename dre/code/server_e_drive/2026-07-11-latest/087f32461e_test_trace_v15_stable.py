import torch
from neuroez_c.task2.trace_rawmil.model import TRACERawMIL
from neuroez_c.task2.trace_rawmil.trainer import ModelEMA
def test_v15_ema_is_exact_copy_before_first_step():
 m=TRACERawMIL();ema=ModelEMA(m);assert all(torch.equal(x,y) for x,y in zip(m.parameters(),ema.model.parameters()))
def test_v15_uses_sparsemax_and_reliability_pooling():
 m=TRACERawMIL();assert hasattr(m,'reliability') and hasattr(m,'theta_deviation') and hasattr(m,'seizure_head')
 p=[[(torch.randn(6,1,500),torch.tensor([0,0,1,1,2,2]),torch.tensor([0,0,1,1,2,2]),torch.tensor([0,0,0,1,1,1]))]];o=m(p)[0];assert torch.allclose(o['details'][0]['alpha'].sum(),torch.tensor(1.)) and torch.isfinite(o['rho'])
