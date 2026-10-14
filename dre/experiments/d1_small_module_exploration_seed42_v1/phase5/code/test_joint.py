import numpy as np
import torch
from torch import nn
from joint_model import Joint,ARMS,MorphCorrection,objective

class Base(nn.Module):
    def __init__(self):super().__init__();self.frozen=nn.Parameter(torch.zeros(1537))
    def forward(self,x):return x[:,0]

def inputs():
    return torch.randn(6,14),torch.randn(1,6,3,500),torch.ones(1,6,3,dtype=torch.bool),torch.tensor([[-1.,3.,10.]]),torch.randn(1,6,3,7),torch.ones(7)

def test_warmstart_both_and_permutation():
    old=MorphCorrection(Base()).eval();nn.init.normal_(old.head[-1].weight,std=.05);state=old.state_dict();args=inputs()
    for arm in ARMS:
        m=Joint(Base(),state,arm).eval()
        with torch.no_grad():assert torch.allclose(m(*args)[0],old(*args)[0],atol=1e-6,rtol=0)
        p=torch.randperm(6);x,r,mask,t,d,s=args
        with torch.no_grad():assert torch.allclose(m(x[p],r[:,p],mask[:,p],t,d[:,p],s)[0],m(*args)[0][p],atol=1e-6,rtol=0)
        assert sum(v.numel() for v in m.parameters() if v.requires_grad)==10377+(arm==ARMS[1])

def test_joint_gradients_and_frozen_D1():
    old=MorphCorrection(Base());nn.init.normal_(old.head[-1].weight,std=.05);args=inputs();y=torch.arange(6).remainder(2).float()
    for arm in ARMS:
        m=Joint(Base(),old.state_dict(),arm);z=m(*args)[0];loss=objective(z,y,nn.BCEWithLogitsLoss());loss.backward()
        assert all(v.grad is None for v in m.base.parameters())
        assert torch.isfinite(loss) and m.head[0].weight.grad[:,96:].norm()>0 and m.cnn[0].weight.grad.norm()>0
        if arm==ARMS[1]:assert m.cap_parameter.grad.abs()>0
