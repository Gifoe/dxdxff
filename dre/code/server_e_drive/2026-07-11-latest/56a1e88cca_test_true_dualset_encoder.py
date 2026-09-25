import torch,pytest
from neuroez_c.task2.true_dualset.encoder import SharedWindowEncoder
from neuroez_c.task2.true_dualset.temporal import PhaseTemporalChannelEncoder
def test_window_and_temporal_encoder_gradient_and_masking():
 torch.manual_seed(1);enc=SharedWindowEncoder();temp=PhaseTemporalChannelEncoder();raw=torch.randn(6,1,500,requires_grad=True);side=torch.randn(6,12);h=enc(raw,side).reshape(2,3,32);mask=torch.tensor([[1,1,1],[1,1,1]],dtype=torch.bool);phase=torch.tensor([0,1,2]);out=temp(h,mask,phase);out.sum().backward();assert out.shape==(2,48) and raw.grad is not None and any(p.grad is not None for p in temp.parameters())
def test_masked_values_do_not_change_phase_encoding():
 temp=PhaseTemporalChannelEncoder().eval();h=torch.randn(2,4,32);mask=torch.tensor([[1,1,1,0],[1,1,1,0]],dtype=torch.bool);phase=torch.tensor([0,1,2,0]);a=temp(h,mask,phase);h[:,3]=99999;assert torch.allclose(a,temp(h,mask,phase),atol=1e-5)
def test_empty_phase_is_explicit_error():
 with pytest.raises(ValueError):PhaseTemporalChannelEncoder()(torch.randn(2,3,32),torch.ones(2,3,dtype=torch.bool),torch.tensor([0,0,2]),strict=True)
