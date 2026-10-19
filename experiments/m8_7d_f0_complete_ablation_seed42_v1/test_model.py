"""CPU mathematical switch tests, not surrogate performance experiments."""
import unittest
import math
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
    s=old.MODEL.SingleHead('F0_AFTER_GELU',cnn,Gate(),stub,170054)
    with torch.no_grad():s.cnn.conditioner[2].weight.normal_(0,.02);s.interaction_gain.fill_(.2)
    return s

def batch():
    torch.manual_seed(63);raw=torch.randn(2,6,9,500);mask=torch.ones(2,6,9,dtype=torch.bool)
    times=torch.tensor([[-4,-2,0,2,4,6,8,12,18]]*2,dtype=torch.float32)
    return torch.randn(6,14),raw,mask,times,torch.randn(2,6,9,7),torch.ones(7)

class ModelTests(unittest.TestCase):
    def test_counts_initializer_and_single_readout(self):
        s=source();expected=[11426,10378,10378,9330,11426,11426,11426,11425,11426,8433,1025]+[11426]*7
        for arm,n in zip(ARMS,expected):
            m=MODEL.Ablation(s,arm);self.assertEqual(sum(p.numel() for p in m.parameters()),n)
            self.assertEqual(sum(isinstance(q,nn.Linear) and q.out_features==1 for q in m.modules()),1)
            self.assertTrue(torch.equal(m.head[-1].weight,s.head[-1].weight))
            self.assertTrue(torch.equal(m.head[0].bias,s.head[0].bias))
            if m.branch=='joint':self.assertTrue(torch.equal(m.head[0].weight,s.head[0].weight))
            else:self.assertTrue(torch.equal(m.head[0].weight,s.head[0].weight[:,:96] if m.branch=='raw' else s.head[0].weight[:,96:]))

    def test_full_exact_and_padding(self):
        s=source();m=MODEL.Ablation(s,'FULL_F0');args=list(batch())
        a=s(*args);b=m(*args);self.assertTrue(torch.equal(a['logit'],b['logit']));self.assertTrue(torch.equal(a['raw_summary'],b['raw_summary']))
        args[2][0,:,0]=False;expected=m(*args)['logit'];args[1][~args[2]]=float('nan')
        self.assertTrue(torch.equal(expected,m(*args)['logit']))

    def test_all_seven_drop_paths_are_independent(self):
        s=source()
        for k,arm in enumerate(ARMS[11:]):
            m=MODEL.Ablation(s,arm);args=list(batch());before=m(*args)['logit']
            args[0][:,2*k:2*k+2]=float('nan');args[4][...,k]=float('nan')
            self.assertTrue(torch.equal(before,m(*args)['logit']),arm)

    def test_raw_z_pre_reference_and_head_switch(self):
        s=source();args=batch();m=MODEL.Ablation(s,'WO_RAW_PR');z=m(*args)['raw_summary'];f=MODEL.Ablation(s,'FULL_F0')
        q=z.double();expected=(q-q.mean(0))/torch.where(q.std(0,unbiased=False)>1e-8,q.std(0,unbiased=False),torch.ones_like(q.std(0,unbiased=False)))
        # Casting the unnormalised vector adds one FP32 rounding step.
        self.assertTrue(torch.allclose(f(*args)['raw_summary'],expected.float(),atol=1e-6,rtol=1e-5))
        m=MODEL.Ablation(s,'WO_RAW_PRE_REFERENCE');a=m(*args)['raw_summary'];changed=list(args);changed[1]=args[1].clone();changed[1][:,:,:2]*=3
        self.assertTrue(torch.equal(a,m(*changed)['raw_summary']))
        m=MODEL.Ablation(s,'WO_HEAD_INTERACTION');z=torch.randn(6,96);x=torch.randn(6,14)
        h=torch.nn.functional.linear(z*math.sqrt(110/192),m.head[0].weight[:,:96])+torch.nn.functional.linear(x*math.sqrt(110/28),m.head[0].weight[:,96:])+m.head[0].bias
        self.assertTrue(torch.equal(m.classify(z,x),m.head[2](m.head[1](h)).squeeze(-1)))

    def test_support_preserving_mismatch_and_branches(self):
        args=list(batch());args[2][0,0,0]=False
        mapping,counts=MODEL.channel_mapping(args[2],[str(i) for i in range(6)],'synthetic',1)
        again,_=MODEL.channel_mapping(args[2],[str(i) for i in range(6)],'synthetic',1)
        self.assertTrue(torch.equal(mapping,again));self.assertGreater(counts['changed_valid_windows'],0);self.assertGreater(counts['singleton_valid_windows'],0)
        self.assertTrue(torch.equal(torch.gather(args[2],1,mapping[...,None].expand(-1,-1,9)),args[2]))
        s=source();m=MODEL.Ablation(s,'EARLY_FILM_MISMATCH');self.assertTrue(torch.isfinite(m(*args,mapping)['logit']).all())
        m=MODEL.Ablation(s,'RAW_ONLY');a=m(*args)['logit'];args[0].fill_(float('nan'));args[4].fill_(float('nan'));self.assertTrue(torch.equal(a,m(*args)['logit']))
        m=MODEL.Ablation(s,'PHYSIOLOGY_ONLY');x=torch.randn(6,14);self.assertEqual(tuple(m(x)['logit'].shape),(6,))

    def test_original_loss(self):
        x=torch.tensor([-.4,.2,.8]);y=torch.tensor([0.,1.,1.]);pos=torch.tensor(.7)
        self.assertTrue(torch.equal(MODEL.objective(x,y,pos),old.MODEL.objective(x,y,pos)))

if __name__=='__main__':unittest.main()
