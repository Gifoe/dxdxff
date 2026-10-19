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
    def test_shared_initializer_counts_and_full_reference(self):
        s=source();args=batch()
        ref=MODEL.BASE.Ablation(s,'DROP_VARIANCE')
        full=MODEL.Ablation(s,'FULL_F0')
        self.assertTrue(torch.equal(ref(*args)['logit'],full(*args)['logit']))
        self.assertEqual(state_hash(ref.state_dict()),state_hash(full.state_dict()))
        for arm in DEEP_ARMS:
            rng=torch.random.get_rng_state().clone();m=MODEL.Ablation(s,arm)
            self.assertTrue(torch.equal(rng,torch.random.get_rng_state()))
            self.assertEqual(sum(isinstance(q,nn.Linear) and q.out_features==1 for q in m.modules()),1)
            self.assertTrue(torch.equal(m.head[-1].weight,s.head[-1].weight))
        oldraw=MODEL.BASE.Ablation(s,'RAW_ONLY');newraw=MODEL.Ablation(s,'RAW_ONLY')
        self.assertEqual(state_hash(oldraw.state_dict()),state_hash(newraw.state_dict()))
        self.assertTrue(torch.equal(oldraw(*args)['logit'],newraw(*args)['logit']))

    def test_variance_and_all_six_coordinate_removals(self):
        s=source()
        drops={'DROP_BETA':0,'DROP_LOW_GAMMA':1,'DROP_PARTIAL_HIGH_GAMMA':2,'DROP_RMS':3,'DROP_LINE_LENGTH':5,'DROP_SPECTRAL_ENTROPY':6}
        for arm in DEEP_ARMS:
            m=MODEL.Ablation(s,arm);args=list(batch());perm=None
            if arm=='EARLY_FILM_MISMATCH':perm,_=MODEL.channel_mapping(args[2],[str(i) for i in range(6)],'synthetic',1)
            before=m(*args,permutation=perm)['logit']
            for k in [4]+([drops[arm]] if arm in drops else []):
                args[0][:,2*k:2*k+2]=float('nan');args[4][...,k]=float('nan')
            self.assertTrue(torch.equal(before,m(*args,permutation=perm)['logit']),arm)

    def test_both_PR_switch_and_reference(self):
        s=source();args=batch();m=MODEL.Ablation(s,'WO_BOTH_PR');self.assertFalse(m.raw_pr)
        z=m(*args)['raw_summary'].double()
        expected=(z-z.mean(0))/torch.where(z.std(0,unbiased=False)>1e-8,z.std(0,unbiased=False),torch.ones_like(z.std(0,unbiased=False)))
        self.assertTrue(torch.allclose(MODEL.Ablation(s,'FULL_F0')(*args)['raw_summary'],expected.float(),atol=1e-6,rtol=1e-5))
        m=MODEL.Ablation(s,'WO_RAW_PRE_REFERENCE');a=m(*args)['raw_summary']
        changed=list(args);changed[1]=args[1].clone();changed[1][:,:,:2]*=3
        self.assertTrue(torch.equal(a,m(*changed)['raw_summary']))

    def test_mismatch_support_and_raw_independence(self):
        args=list(batch());args[2][0,0,0]=False
        perm,counts=MODEL.channel_mapping(args[2],[str(i) for i in range(6)],'synthetic',1)
        self.assertGreater(counts['singleton_valid_windows'],0)
        self.assertTrue(torch.equal(torch.gather(args[2],1,perm[...,None].expand(-1,-1,9)),args[2]))
        m=MODEL.Ablation(source(),'RAW_ONLY');a=m(*args)['logit']
        args[0].fill_(float('nan'));args[4].fill_(float('nan'))
        self.assertTrue(torch.equal(a,m(*args)['logit']))

    def test_no_interaction_and_original_loss(self):
        s=source();m=MODEL.Ablation(s,'WO_HEAD_INTERACTION');self.assertFalse(m.interaction)
        x=torch.tensor([-.4,.2,.8]);y=torch.tensor([0.,1.,1.]);pos=torch.tensor(.7)
        self.assertTrue(torch.equal(MODEL.objective(x,y,pos),old.MODEL.objective(x,y,pos)))

if __name__=='__main__':
    torch.set_num_threads(2);unittest.main()

