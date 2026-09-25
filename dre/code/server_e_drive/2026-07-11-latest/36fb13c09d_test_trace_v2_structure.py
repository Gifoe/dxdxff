import torch
from neuroez_c.task2.trace_rawmil.pooling import PhaseDeltaChannelPool
from neuroez_c.task2.trace_rawmil.model import TRACERawMIL

def test_phase_delta_pool_has_explicit_99_input_and_missing_phase_is_finite():
 p=PhaseDeltaChannelPool();assert p.net[0].in_features==99
 out=p(torch.randn(2,32),torch.tensor([0,2]));assert out.shape==(32,) and torch.isfinite(out).all()

def test_topk_pooling_uses_exact_top_twenty_percent_and_zero_elsewhere():
 torch.manual_seed(2);m=TRACERawMIL().eval();n=10
 item=(torch.randn(n*3,1,500),torch.arange(3).repeat(n),torch.arange(n).repeat_interleave(3),torch.tensor([0]*5+[1]*5).repeat_interleave(3));d=m([[item]])[0]['details'][0];assert d['topk_mask'].sum().item()==1 and torch.all(d['alpha'][~d['topk_mask']]==0) and torch.allclose(d['alpha'].sum(),torch.tensor(1.))

def test_channel_centering_has_zero_mean_before_downstream_masks():
 torch.manual_seed(3);m=TRACERawMIL();x=torch.randn(4,32);common_removed=x-x.mean(0,keepdim=True);assert torch.allclose(common_removed.mean(0),torch.zeros(32),atol=1e-6) and torch.isfinite(m.center_norm(common_removed)).all()
