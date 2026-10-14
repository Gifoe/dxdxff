import os
import sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
sys.path.insert(0,os.environ.get('DGM_COMPLETED_CODE',r'D:\dgm_net_seed42_v1\code'))
sys.path.insert(0,os.environ.get('D1_PHASE1_CODE',r'D:\d1_small_module_exploration_seed42_v1\code'))
sys.path.insert(0,str(Path(__file__).parents[1]/'code'))
from direct_correction import FeatureCorrection,MorphCorrection
torch.set_num_threads(2)
class D1(nn.Module):
    def __init__(self):super().__init__();self.first=nn.Linear(14,96);self.last=nn.Linear(96,1)
    def forward(self,x):return self.last(nn.functional.gelu(self.first(x))).squeeze(-1)
def sample():
    torch.manual_seed(42);t=torch.tensor([[-4,-2,0,2,8,10,12]],dtype=torch.float32);mask=torch.ones(1,8,7,dtype=torch.bool)
    return torch.randn(8,14),torch.randn(1,8,7,500),mask,t,torch.randn(1,8,7,7),torch.ones(7)
def test_direct_initial_identity_and_budgets():
    for arm,n in [('M4_FEATURE_CORRECTION',257),('M5_STAGE_CORRECTION',721),('M6_MORPH_CORRECTION',9481)]:
        m=MorphCorrection(D1()) if arm=='M6_MORPH_CORRECTION' else FeatureCorrection(D1(),arm)
        q=m.eval()(*sample());assert torch.equal(q[0],q[2]) and torch.equal(q[1],torch.zeros_like(q[1]))
        assert sum(p.numel() for p in m.parameters() if p.requires_grad)==n
        assert not hasattr(m,'alpha')
def test_direct_loss_gradients_and_frozen_base():
    for arm in ['M4_FEATURE_CORRECTION','M5_STAGE_CORRECTION','M6_MORPH_CORRECTION']:
        m=MorphCorrection(D1()) if arm=='M6_MORPH_CORRECTION' else FeatureCorrection(D1(),arm);args=sample();opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=.001)
        for step in range(3):
            opt.zero_grad();q=m(*args);nn.functional.binary_cross_entropy_with_logits(q[0],torch.arange(8).float()%2).backward()
            assert m.head[-1].weight.grad.norm()>0 and all(p.grad is None for p in m.base.parameters())
            assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
            if step:assert m.head[0].weight.grad.norm()>0
            else:assert m.head[0].weight.grad.norm()==0
            if arm=='M6_MORPH_CORRECTION' and step==2:assert m.gate.network[0].weight.grad.norm()>0
            opt.step()
        q=m.eval()(*args,mode='residual0');assert torch.equal(q[0],q[2])
