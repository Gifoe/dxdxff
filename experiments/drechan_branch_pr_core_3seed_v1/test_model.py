"""Only changed guidance/pre-reference/mismatch paths; no accuracy claim."""
import unittest
from common import *

helper=module('core_parent_test_helpers',PARENT/'test_model.py')
sys.modules['common']=OWN;sys.modules['model']=MODEL

def original():return parent.MODEL.BranchResidual(helper.source(),'FULL')

class Tests(unittest.TestCase):
    def test_rng_shared_tensors_and_capacity(self):
        source=original();expected=source.state_dict();base_count=sum(p.numel() for p in source.parameters())
        import copy
        for arm in ARMS:
            rng=torch.random.get_rng_state().clone();m=MODEL.configure(copy.deepcopy(source),arm)
            self.assertTrue(torch.equal(rng,torch.random.get_rng_state()))
            for k,v in m.state_dict().items():
                parent_key=k.replace('cnn.0.','cnn.stem.0.').replace('cnn.2.','cnn.stem.2.')
                self.assertTrue(torch.equal(v,expected[parent_key]),(arm,k))
            removed=base_count-sum(p.numel() for p in m.parameters())
            self.assertEqual(removed,2096 if arm=='WO_BOTH_GUIDANCE' else 1048 if arm in ['WO_EARLY_FILM','WO_LATE_GATE'] else 0)
            self.assertTrue(m.raw_pr and m.feature_pr and m.raw_context and m.feature_context)

    def test_removed_paths_and_all_gradients(self):
        args=list(helper.batch());source=original()
        import copy
        for arm in ARMS:
            m=MODEL.configure(copy.deepcopy(source),arm);perm=None
            if arm=='EARLY_FILM_MISMATCH':perm,_=parent.component.MODEL.BASE.channel_mapping(args[2],[str(i) for i in range(6)],'toy',1)
            out=m(*args,permutation=perm)['logit'];out.square().mean().backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))
            if arm=='WO_BOTH_GUIDANCE':
                changed=list(args);changed[4]=args[4]*17+10
                self.assertTrue(torch.equal(out,m(*changed)['logit']))
            if arm in ['WO_LATE_GATE','WO_BOTH_GUIDANCE']:self.assertFalse(hasattr(m,'gate'))
            altered=list(args);altered[0]=args[0].clone();altered[4]=args[4].clone()
            altered[0][:,8:10]=float('nan');altered[4][...,4]=float('nan')
            self.assertTrue(torch.equal(out,m(*altered,permutation=perm)['logit']))

    def test_mismatch_support_and_correct_late_gate(self):
        args=list(helper.batch());args[2][0,0,-1]=False
        mm,counts=parent.component.MODEL.BASE.channel_mapping(args[2],[str(i) for i in range(6)],'toy',1)
        self.assertTrue(torch.equal(torch.gather(args[2],1,mm[...,None].expand(-1,-1,args[2].shape[-1])),args[2]))
        self.assertGreater(counts['singleton_valid_windows'],0);self.assertGreater(counts['changed_valid_windows'],0)
        import copy
        full=original();bad=MODEL.configure(copy.deepcopy(full),'EARLY_FILM_MISMATCH')
        with torch.no_grad():
            full.cnn.conditioner[-1].weight.fill_(.01);bad.cnn.conditioner[-1].weight.fill_(.01)
        g1=full(*args)['gate_activity'];g2=bad(*args,permutation=mm)['gate_activity']
        self.assertTrue(torch.equal(g1,g2))

if __name__=='__main__':
    torch.set_num_threads(2);unittest.main()
