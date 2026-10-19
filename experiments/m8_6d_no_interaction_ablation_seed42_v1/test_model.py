"""Critical-path mathematical tests, not substitute performance experiments."""
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
    s=old.MODEL.SingleHead('F0_AFTER_GELU',cnn,Gate(),stub,170054)
    with torch.no_grad():s.cnn.conditioner[2].weight.normal_(0,.02);s.interaction_gain.fill_(.2)
    return s

def batch():
    torch.manual_seed(63);raw=torch.randn(2,6,9,500);mask=torch.ones(2,6,9,dtype=torch.bool)
    times=torch.tensor([[-4,-2,0,2,4,6,8,12,18]]*2,dtype=torch.float32)
    return torch.randn(6,14),raw,mask,times,torch.randn(2,6,9,7),torch.ones(7)

class Tests(unittest.TestCase):
    def test_references_initialization_and_all_interactions_absent(self):
        s=source();args=batch()
        for arm,name in REF_NAMES.items():
            ref=MODEL.BASE.Ablation(s,name);m=MODEL.Ablation(s,arm)
            self.assertEqual(state_hash(ref.state_dict()),state_hash(m.state_dict()))
            self.assertTrue(torch.equal(ref(*args)['logit'],m(*args)['logit']))
        for arm in ARMS:
            rng=torch.random.get_rng_state().clone();m=MODEL.Ablation(s,arm)
            self.assertTrue(torch.equal(rng,torch.random.get_rng_state()))
            self.assertFalse(m.interaction);self.assertFalse(any('interaction' in n for n,_ in m.named_parameters()))
            self.assertEqual(sum(isinstance(q,nn.Linear) and q.out_features==1 for q in m.modules()),1)
            self.assertTrue(torch.equal(m.head[-1].weight,s.head[-1].weight))
            if m.branch=='joint':self.assertTrue(torch.equal(m.head[0].weight,s.head[0].weight))
        self.assertNotIn('WO_LATE_GATE',ARMS);self.assertEqual(len(ARMS),14)

    def test_permanent_variance_and_six_deletions(self):
        drops={'DROP_BETA':0,'DROP_LOW_GAMMA':1,'DROP_PARTIAL_HIGH_GAMMA':2,'DROP_RMS':3,'DROP_LINE_LENGTH':5,'DROP_SPECTRAL_ENTROPY':6}
        s=source()
        for arm in ARMS:
            m=MODEL.Ablation(s,arm);args=list(batch());perm=None
            if arm=='EARLY_FILM_MISMATCH':perm,_=MODEL.channel_mapping(args[2],list(map(str,range(6))),'synthetic',1)
            a=m(*args,permutation=perm)['logit']
            for k in [4]+([drops[arm]] if arm in drops else []):
                args[0][:,2*k:2*k+2]=float('nan');args[4][...,k]=float('nan')
            self.assertTrue(torch.equal(a,m(*args,permutation=perm)['logit']),arm)

    def test_PR_reference_and_guidance_boundaries(self):
        s=source();args=batch();m=MODEL.Ablation(s,'WO_BOTH_PR');self.assertFalse(m.raw_pr)
        z=m(*args)['raw_summary'].double();sd=z.std(0,unbiased=False)
        expected=((z-z.mean(0))/torch.where(sd>1e-8,sd,torch.ones_like(sd))).float()
        self.assertTrue(torch.allclose(MODEL.Ablation(s,'FULL_F0')(*args)['raw_summary'],expected,atol=1e-6,rtol=1e-5))
        m=MODEL.Ablation(s,'WO_RAW_PRE_REFERENCE');a=m(*args)['raw_summary'];changed=list(args)
        changed[1]=args[1].clone();changed[1][:,:,:2]*=3
        self.assertTrue(torch.equal(a,m(*changed)['raw_summary']))
        m=MODEL.Ablation(s,'WO_EARLY_FILM');self.assertFalse(m.early);self.assertTrue(m.late)
        m=MODEL.Ablation(s,'WO_BOTH_GUIDANCE');self.assertFalse(m.early);self.assertFalse(m.late)

    def test_mismatch_scope_and_raw_independence(self):
        args=list(batch());args[2][0,0,0]=False
        perm,counts=MODEL.channel_mapping(args[2],list(map(str,range(6))),'synthetic',1)
        self.assertGreater(counts['singleton_valid_windows'],0)
        self.assertTrue(torch.equal(torch.gather(args[2],1,perm[...,None].expand(-1,-1,9)),args[2]))
        a=MODEL.Ablation(source(),'FULL_F0');b=MODEL.Ablation(source(),'EARLY_FILM_MISMATCH')
        self.assertEqual(sum(p.numel() for p in a.parameters()),sum(p.numel() for p in b.parameters()))
        m=MODEL.Ablation(source(),'RAW_ONLY');out=m(*args)['logit'];args[0].fill_(float('nan'));args[4].fill_(float('nan'))
        self.assertTrue(torch.equal(out,m(*args)['logit']))

    def test_original_loss(self):
        x=torch.tensor([-.4,.2,.8]);y=torch.tensor([0.,1.,1.]);pos=torch.tensor(.7)
        self.assertTrue(torch.equal(MODEL.objective(x,y,pos),old.MODEL.objective(x,y,pos)))

if __name__=='__main__':
    torch.set_num_threads(2);unittest.main()
