"""Synthetic tests; no patient/clinical data."""
import numpy as np
import torch
from models import ARMS,Adapter,loss,soft_macro

def test_initial_identity_all_candidates():
    for arm in ARMS:
        m=Adapter(arm);x=torch.randn(9,129);b=torch.randn(9)
        assert torch.equal(m(x,b),b)

def test_shift_permutation_and_no_channel_ranking_change():
    m=Adapter(ARMS[0]);torch.nn.init.normal_(m.net[-1].weight);x=torch.randn(9,129);x[:,-16:]=x[0,-16:];b=torch.randn(9);p=torch.randperm(9);z=m(x,b)
    assert torch.allclose(m(x[p],b[p]),z[p],atol=1e-6)
    assert torch.allclose(z-b,(z-b)[0].expand_as(b),atol=1e-6)

def test_two_step_gradient_and_cap():
    for arm in ARMS:
        m=Adapter(arm);x=torch.randn(12,129);b=torch.randn(12);y=torch.arange(12).remainder(2).float();opt=torch.optim.AdamW(m.parameters(),lr=.001)
        for step in range(2):
            opt.zero_grad();z=m(x,b);v=loss(z,y,torch.nn.BCEWithLogitsLoss(),arm);v.backward();assert m.net[-1].weight.grad.norm()>0
            if step:assert m.net[0].weight.grad.norm()>0
            opt.step()
        assert (m(x,b)-b).abs().max()<=1+1e-6

def test_soft_macro_absent_class_not_fabricated_perfect():
    assert torch.isclose(soft_macro(torch.tensor([100.]),torch.tensor([1.])),torch.tensor(.5))
