"""New residual/numeric-control paths only; synthetic outputs are not evidence."""
import unittest
from torch import nn
from common import *

def transform(delta,mask,times,mode,perms):return delta.clone()
def feature_stub():pass

class Gate(nn.Module):
    def __init__(self):
        super().__init__();self.network=nn.Sequential(nn.Linear(7,32),nn.GELU(),nn.Linear(32,24))
    def forward(self,u):return 1+.25*torch.tanh(self.network(u))

def source():
    torch.manual_seed(51)
    cnn=nn.Sequential(nn.Conv1d(1,12,9,4),nn.GELU(),nn.Conv1d(12,24,7,4),nn.GELU(),nn.AdaptiveAvgPool1d(1))
    return component.old.MODEL.SingleHead('F0_AFTER_GELU',cnn,Gate(),feature_stub,170054)

def batch():
    torch.manual_seed(63);raw=torch.randn(2,6,9,500);mask=torch.ones(2,6,9,dtype=torch.bool)
    times=torch.tensor([[-4,-2,0,2,4,6,8,12,18]]*2,dtype=torch.float32)
    return torch.randn(6,14),raw,mask,times,torch.randn(2,6,9,7),torch.ones(7)

class Tests(unittest.TestCase):
    def test_full_and_channel_control_initial_function_exact(self):
        s=source();args=batch();base=MODEL.BASE.PRFactorial(s,'FULL');expected=base(*args)['logit'];original=base.state_dict()
        for arm in ['FULL','CHANNEL_CONTEXT']:
            rng=torch.random.get_rng_state().clone();m=MODEL.BranchResidual(s,arm)
            self.assertTrue(torch.equal(rng,torch.random.get_rng_state()))
            self.assertTrue(all(torch.equal(v,m.state_dict()[k]) for k,v in original.items()))
            self.assertTrue(torch.equal(expected,m(*args)['logit']))
            self.assertEqual(sum(p.numel() for p in m.parameters())-sum(p.numel() for p in base.parameters()),128)

    def test_raw_no_pr_preconditioner_is_exact_independent_affine(self):
        args=batch();s=source();base=MODEL.BASE.PRFactorial(s,'WO_RAW_PR');raw,_=base.features(*args[1:])
        anchor=dict(mean=torch.linspace(-.1,.1,96,dtype=torch.float64),scale=torch.linspace(.01,.2,96,dtype=torch.float64))
        m=MODEL.BranchResidual(s,'WO_RAW_PR');m.set_anchor(anchor);out,_=m.features(*args[1:])
        self.assertTrue(torch.equal(out,((raw.double()-anchor['mean'])/anchor['scale']).float()))
        m.collect_anchor=True;collect,_=m.features(*args[1:]);self.assertTrue(torch.equal(raw,collect))

    def test_all_new_variants_gradients_and_variance_mask(self):
        args=list(batch());s=source();capacity=None
        for arm in ARMS[1:]:
            m=MODEL.BranchResidual(s,arm);count=sum(p.numel() for p in m.parameters())
            if capacity is None:capacity=count
            self.assertEqual(count,capacity);self.assertFalse(m.interaction)
            opt=torch.optim.AdamW(m.parameters(),lr=.0003)
            for step in range(2):
                opt.zero_grad(set_to_none=True);out=m(*args)['logit'];out.square().mean().backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))
                for name in ['cnn.stem.0.weight','head.0.weight','gain_raw','gain_feature']:
                    self.assertGreater(float(dict(m.named_parameters())[name].grad.norm()),0)
                opt.step()
            before=m(*args)['logit'];changed=list(args);changed[0]=args[0].clone();changed[4]=args[4].clone()
            changed[0][:,8:10]=float('nan');changed[4][...,4]=float('nan')
            self.assertTrue(torch.equal(before,m(*changed)['logit']))
            self.assertEqual(m.feature_pr,arm not in ['WO_FEATURE_PR','WO_BOTH_PR'])
            self.assertEqual(m.raw_pr,arm not in ['WO_RAW_PR','WO_BOTH_PR'])

if __name__=='__main__':
    torch.set_num_threads(2);unittest.main()
