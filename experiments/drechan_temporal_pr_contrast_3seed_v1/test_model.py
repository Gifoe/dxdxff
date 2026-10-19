"""Changed model paths only; synthetic checks are not accuracy experiments."""
import unittest
from common import *
from torch import nn

def transform(delta,mask,times,mode,perms):return delta.clone()
def stub():pass
class Gate(nn.Module):
    def __init__(self):
        super().__init__();self.network=nn.Sequential(nn.Linear(7,32),nn.GELU(),nn.Linear(32,24))
    def forward(self,u):return 1+.25*torch.tanh(self.network(u))
def source():
    torch.manual_seed(51)
    cnn=nn.Sequential(nn.Conv1d(1,12,9,4),nn.GELU(),nn.Conv1d(12,24,7,4),nn.GELU(),nn.AdaptiveAvgPool1d(1))
    return old.MODEL.SingleHead('F0_AFTER_GELU',cnn,Gate(),stub,170054)
def batch():
    torch.manual_seed(63);raw=torch.randn(2,6,9,500);mask=torch.ones(2,6,9,dtype=torch.bool)
    times=torch.tensor([[-4,-2,0,2,4,6,8,12,18]]*2,dtype=torch.float32)
    return torch.randn(6,14),raw,mask,times,torch.randn(2,6,9,7),torch.ones(7)
def ref():
    return dict(times=torch.tensor([0.,2.,4.,6.,8.,12.,18.]),median=torch.full((7,24),.02),mad=torch.full((7,24),.03),count=torch.full((7,),100))

class Tests(unittest.TestCase):
    def test_initial_identity_shared_weights_capacity_and_masks(self):
        s=source();args=list(batch());base=MODEL.BASE.Ablation(s,'FULL_F0');original=base(*args)['logit'];states=base.state_dict()
        for arm in ARMS:
            rng=torch.random.get_rng_state().clone();m=MODEL.Temporal(s,arm,1,51)
            self.assertTrue(torch.equal(rng,torch.random.get_rng_state()))
            if arm=='B2':m.set_reference(ref())
            self.assertTrue(all(torch.equal(v,m.state_dict()[k]) for k,v in states.items()))
            self.assertTrue(torch.equal(original,m(*args)['logit']),arm)
            changed=list(args);changed[0]=args[0].clone();changed[4]=args[4].clone()
            changed[0][:,8:10]=float('nan');changed[4][...,4]=float('nan')
            self.assertTrue(torch.equal(original,m(*changed)['logit']),arm)
            self.assertFalse(m.interaction);self.assertEqual(sum(isinstance(q,nn.Linear) and q.out_features==1 for q in m.modules()),1)

    def test_live_gradient_all_three_reference_modes(self):
        args=batch();s=source()
        for arm in ARMS[1:]:
            m=MODEL.Temporal(s,arm,1,51)
            if arm=='B2':m.set_reference(ref())
            opt=torch.optim.AdamW(m.parameters(),lr=.0003)
            for k in range(3):
                opt.zero_grad(set_to_none=True);out=m(*args)['logit'];loss=out.square().mean()+.1*out.mean();loss.backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))
                self.assertGreater(float(m.temporal_fusion.network[-1].weight.grad.norm()),0)
                self.assertGreater(float(m.cnn.stem[0].weight.grad.norm()),0)
                if k:self.assertGreater(float(m.temporal_fusion.network[0].weight.grad.norm()),0)
                opt.step()

    def test_prepared_encoder_exact_and_pooling_gradient(self):
        m=MODEL.Temporal(source(),'B1',1,51);args=batch()
        p=m.input_tensors(*args[1:]);emb=m.encode_prepared(p)
        self.assertTrue(emb.requires_grad)
        res=m.residuals(emb,args[2],args[3]);self.assertEqual(res.shape,emb.shape)
        self.assertTrue(res.requires_grad)
        z=m(*args)['raw_summary'];self.assertTrue(z.requires_grad)
        np.testing.assert_allclose(z.detach().mean(0),np.zeros(96),rtol=0,atol=1e-6)

if __name__=='__main__':
    torch.set_num_threads(2);unittest.main()
