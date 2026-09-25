import torch
from neuroez_c.task2.trace_rawmil.losses import rank_weight,trace_loss
from neuroez_c.task2.trace_rawmil.model import TRACERawMIL
def test_clean_v1_rank_weight_schedule():
    expected = [0,0,0,0,0,.01,.02,.03,.04,.05,.06,.07,.08,.09,.10]
    assert all(abs(rank_weight(e)-value) < 1e-12 for e,value in enumerate(expected,1))
def test_loss_has_only_bce_and_rank():
 o=[{'logit':torch.tensor(.2,requires_grad=True),'rho':torch.tensor(.3)},{'logit':torch.tensor(-.1,requires_grad=True),'rho':torch.tensor(.4)}];loss,parts=trace_loss(o,[1,0],1);assert set(parts)=={'bce','rank_raw','rank_weighted','rank_weight'} and torch.isfinite(loss)
def test_v15_restores_reliability_and_keeps_no_auxiliary_loss():
 names=set(dict(TRACERawMIL().named_modules()));assert 'reliability' in names and 'seizure_head' in names
