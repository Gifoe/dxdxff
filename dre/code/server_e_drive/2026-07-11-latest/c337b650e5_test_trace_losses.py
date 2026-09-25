import torch
from neuroez_c.task2.trace_rawmil.losses import trace_loss
def test_losses_no_nan_with_single_class():
 o=[{'logit':torch.tensor(.2,requires_grad=True),'rho':torch.tensor(.3),'seizure_logits':torch.tensor([.1])}];v,_=trace_loss(o,[1]);assert torch.isfinite(v)
