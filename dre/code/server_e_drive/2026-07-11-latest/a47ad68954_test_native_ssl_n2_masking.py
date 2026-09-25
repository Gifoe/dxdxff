import torch
from neuroez_c.task2.native_ssl_n2.temporal_encoder import Temporal
def test_padding_values_do_not_change_temporal_output():
 t=Temporal().eval(); z=torch.randn(1,3,48); m=torch.tensor([[1,1,0]],dtype=torch.bool); p=torch.tensor([0,1,2]); r=torch.arange(3.)
 a=t(z,m,p,r)[0]; z[:,2]=99999; b=t(z,m,p,r)[0]; assert torch.allclose(a,b)
