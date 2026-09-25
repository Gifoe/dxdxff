import torch
from neuroez_c.task2.trace_aux_stable.temporal import PhaseChannelEncoder
def test_masked_out_values_invariant():
 m=PhaseChannelEncoder().eval();x=torch.randn(2,4,32);mask=torch.tensor([[1,1,1,0],[1,1,1,0]],dtype=torch.bool);p=torch.tensor([0,1,2,0]);a=m(x,mask,p);x[:,3]=1e5;assert torch.allclose(a,m(x,mask,p),atol=1e-5)
