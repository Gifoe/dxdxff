import torch
from neuroez_c.task2.trace_rawmil.model import TRACERawMIL
from neuroez_c.task2.trace_rawmil.trainer import ModelEMA
def test_v15_ema_is_exact_copy_before_first_step():
 m=TRACERawMIL();ema=ModelEMA(m);assert all(torch.equal(x,y) for x,y in zip(m.parameters(),ema.model.parameters()))
def test_eval_forward_is_deterministic():
 torch.manual_seed(4);m=TRACERawMIL().eval();p=[[(torch.randn(6,1,500),torch.tensor([0,0,1,1,2,2]),torch.tensor([0,0,1,1,2,2]),torch.tensor([0,0,0,1,1,1]))]]
 with torch.inference_mode():a=m(p)[0]['logit'];b=m(p)[0]['logit']
 assert torch.equal(a,b)
def test_v15_uses_sparsemax_and_reliability_pooling_without_centering():
 m=TRACERawMIL();assert hasattr(m,'reliability') and not hasattr(m,'theta_deviation') and hasattr(m,'seizure_head')
 p=[[(torch.randn(6,1,500),torch.tensor([0,0,1,1,2,2]),torch.tensor([0,0,1,1,2,2]),torch.tensor([0,0,0,1,1,1]))]];o=m(p)[0];assert torch.allclose(o['details'][0]['alpha'].sum(),torch.tensor(1.)) and torch.isfinite(o['rho'])
