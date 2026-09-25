import torch
from neuroez_c.task2.tech_outcome_c1.cotar import MaskedCoTAR
def test_cotar_single_channel_and_gradients():
 m=MaskedCoTAR();x=torch.randn(2,3,64,requires_grad=True);mask=torch.tensor([[1,0,0],[1,1,1]],dtype=torch.bool);y=m(x,mask);y.sum().backward();assert torch.isfinite(y).all() and m.last_weights[0,0].eq(1).all() and all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
def test_padding_content_does_not_change_valid_output():
 m=MaskedCoTAR().eval();x=torch.randn(1,3,64);mask=torch.tensor([[1,1,0]],dtype=torch.bool);a=m(x,mask);x[:,2]=999;b=m(x,mask);assert torch.allclose(a[:,:2],b[:,:2])
