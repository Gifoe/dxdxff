import os
import sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
sys.path.insert(0,os.environ.get('DGM_COMPLETED_CODE',r'D:\dgm_net_seed42_v1\code'))
sys.path.insert(0,os.environ.get('D1_PHASE1_CODE',r'D:\d1_small_module_exploration_seed42_v1\code'))
sys.path.insert(0,os.environ.get('D1_PHASE2_CODE',r'D:\d1_small_module_exploration_seed42_v1\phase2\code'))
sys.path.insert(0,str(Path(__file__).parents[1]/'code'))
from decision_modules import MarginGated,soft_macro_loss,objective
torch.set_num_threads(2)
class D1(nn.Module):
    def __init__(self):super().__init__();self.first=nn.Linear(14,96);self.last=nn.Linear(96,1)
    def forward(self,x):return self.last(nn.functional.gelu(self.first(x))).squeeze(-1)
def fixture():
    torch.manual_seed(42);return torch.randn(8,14),torch.randn(1,8,7,500),torch.ones(1,8,7,dtype=torch.bool),torch.tensor([[-4,-2,0,2,8,10,12.]]),torch.randn(1,8,7,7),torch.ones(7)
def test_margin_initialization_and_gradient():
    m=MarginGated(D1());a=fixture();q=m(*a);assert torch.equal(q[0],q[2]);assert sum(p.numel() for p in m.parameters() if p.requires_grad)==9483
    opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=.001)
    for step in range(3):
        opt.zero_grad();q=m(*a);nn.functional.binary_cross_entropy_with_logits(q[0],torch.arange(8).float()%2).backward()
        if step:assert m.margin_a.grad.abs()>0 and m.margin_b.grad.abs()>0
        else:assert m.margin_a.grad==0 and m.margin_b.grad==0
        opt.step()
    q=m(*a,mode='residual0');assert torch.equal(q[0],q[2])
def test_soft_macro_missing_class_and_finite_gradients():
    for y in [torch.zeros(8),torch.ones(8),torch.arange(8).float()%2]:
        z=torch.linspace(-3,3,8,requires_grad=True);v=soft_macro_loss(z,y);assert torch.isfinite(v) and 0<=v<=1;v.backward();assert torch.isfinite(z.grad).all()
    # Perfect predictions on an absent class use zero, not a spurious epsilon/epsilon1.
    assert soft_macro_loss(torch.full((8,),100.),torch.ones(8))==.5
def test_macro_objective_declared_weight():
    z=torch.tensor([-2.,0.,1.],requires_grad=True);y=torch.tensor([0.,1.,1.]);fn=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(.5))
    assert torch.equal(objective(z,y,fn,'M7_MARGIN_GATED'),fn(z,y))
    assert torch.equal(objective(z,y,fn,'M8_MACRO_AWARE'),.75*fn(z,y)+.25*soft_macro_loss(z,y))
