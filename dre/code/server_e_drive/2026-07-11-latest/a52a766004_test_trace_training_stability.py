import torch
from neuroez_c.task2.trace_rawmil.losses import rank_weight,trace_loss
from neuroez_c.task2.trace_rawmil.model import TRACERawMIL
def test_rank_warmup_first_ten_epochs_is_zero():
 assert all(rank_weight(e)==0 for e in range(1,11)) and rank_weight(25)==.05
def test_loss_has_only_bce_and_rank():
 o=[{'logit':torch.tensor(.2,requires_grad=True),'rho':torch.tensor(.3)},{'logit':torch.tensor(-.1,requires_grad=True),'rho':torch.tensor(.4)}];loss,parts=trace_loss(o,[1,0],1);assert set(parts)=={'bce','rank_raw','rank_weighted','rank_weight'} and torch.isfinite(loss)
def test_v15_restores_reliability_and_keeps_no_auxiliary_loss():
 names=set(dict(TRACERawMIL().named_modules()));assert 'reliability' in names and 'seizure_head' in names
