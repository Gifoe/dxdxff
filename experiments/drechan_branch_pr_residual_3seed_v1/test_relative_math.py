"""Draft critical-path checks only, not accuracy evidence."""
import unittest
import torch
from torch import nn
from relative_math import patient_z,channel_z,residual_hidden

class Tests(unittest.TestCase):
    def test_initial_identity_and_gain_gradient(self):
        torch.manual_seed(1);head=nn.Sequential(nn.Linear(110,64),nn.GELU(),nn.Linear(64,1))
        a=torch.randn(7,64,requires_grad=True);b=torch.randn(7,64,requires_grad=True)
        gr=nn.Parameter(torch.zeros(64));gf=nn.Parameter(torch.zeros(64))
        expected=head[1](a+b+head[0].bias)
        h=residual_hidden(head,a,b,gr,gf)
        self.assertTrue(torch.equal(h,expected))
        head[2](h).square().mean().backward()
        for t in [a,b,gr,gf]:
            self.assertTrue(torch.isfinite(t.grad).all());self.assertGreater(float(t.grad.norm()),0)

    def test_only_patient_norm_uses_other_channels(self):
        torch.manual_seed(3);x=torch.randn(8,64);changed=x.clone();changed[1:]=changed[1:]*10+20
        self.assertTrue(torch.equal(channel_z(x)[0],channel_z(changed)[0]))
        self.assertFalse(torch.equal(patient_z(x)[0],patient_z(changed)[0]))
        self.assertTrue(torch.isfinite(patient_z(torch.ones(1,64))).all())
        self.assertTrue(torch.equal(patient_z(torch.ones(1,64)),torch.zeros(1,64)))
        self.assertTrue(torch.equal(channel_z(torch.ones(1,64)),torch.zeros(1,64)))

if __name__=='__main__':
    torch.set_num_threads(1);unittest.main()
