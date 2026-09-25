import torch
from neuroez_c.task2.native_ssl_n2.losses import ssl_losses
def test_ssl_mask_loss_only_masked_and_backward():
 p=torch.randn(2,3,48,requires_grad=True);t=torch.randn(2,3,48);m=torch.tensor([[1,0,0],[0,0,0]],dtype=torch.bool);loss,x=ssl_losses(p,t,m,torch.randn(3,32),torch.randn(3,32));loss.backward();assert torch.isfinite(loss) and p.grad[~m].eq(0).all()
