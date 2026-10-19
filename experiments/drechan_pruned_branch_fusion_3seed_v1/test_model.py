"""Unique changed executable paths; synthetic tensors are not outcome evidence."""
import copy
import math
import unittest
from common import *

helper=module('pbf_existing_source_fixture',Path(r'D:\drechan_branch_pr_residual_3seed_v1')/'test_model.py')
sys.modules['common']=OWN;sys.modules['model']=MODEL


class Tests(unittest.TestCase):
    def test_shared_initialization_rng_and_capacity(self):
        source=helper.source();base=MODEL.BASE.PRFactorial(source,'FULL');expected=base.state_dict()
        for arm in ARMS[1:]:
            before=torch.random.get_rng_state().clone();m=MODEL.PrunedFusion(source,arm)
            self.assertTrue(torch.equal(before,torch.random.get_rng_state()))
            for name,value in m.state_dict().items():
                if name.startswith('anchor_'):continue
                old=name.replace('cnn.0.','cnn.stem.0.').replace('cnn.2.','cnn.stem.2.')
                self.assertTrue(torch.equal(value,expected[old]),(arm,name))
            self.assertEqual(sum(p.numel() for p in base.parameters())-sum(p.numel() for p in m.parameters()),2096)
            self.assertEqual(sum(isinstance(q,torch.nn.Linear) and q.out_features==1 for q in m.modules()),1)
            self.assertFalse(m.early or m.late or m.interaction or hasattr(m,'gate'))

    def test_exact_joint_bridge_split_formula_and_gradients(self):
        source=helper.source();args=helper.batch()
        original=MODEL.remove_guidance(MODEL.BASE.PRFactorial(source,'FULL'))
        joint=MODEL.PrunedFusion(source,'JOINT_PRUNED')
        self.assertTrue(torch.equal(original(*args)['logit'],joint(*args)['logit']))
        full=MODEL.PrunedFusion(source,'FULL');z,_=full.features(*args[1:]);x=args[0].clone();x[:,8:10]=0
        first=full.head[0]
        a=torch.nn.functional.linear(z*math.sqrt(110/192),first.weight[:,:96])
        b=torch.nn.functional.linear(x*math.sqrt(110/28),first.weight[:,96:])
        manual=full.head[2]((full.head[1](a+first.bias/2)+full.head[1](b+first.bias/2))/math.sqrt(2)).squeeze(-1)
        self.assertTrue(torch.equal(manual,full(*args)['logit']))
        for arm in ARMS[1:]:
            m=MODEL.PrunedFusion(source,arm);out=m(*args)['logit'];out.square().mean().backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))
            self.assertGreater(float(m.cnn[0].weight.grad.norm()),0)
            self.assertGreater(float(m.head[0].weight.grad[:,:96].norm()),0)
            self.assertGreater(float(m.head[0].weight.grad[:,96:].norm()),0)
            altered=list(args);altered[0]=args[0].clone();altered[0][:,8:10]=float('nan');altered[4]=torch.full_like(args[4],float('nan'))
            self.assertTrue(torch.equal(out,m(*altered)['logit']))

    def test_raw_affine_and_scope_of_pr_and_pre_reference(self):
        args=helper.batch();source=helper.source();full=MODEL.PrunedFusion(source,'FULL')
        base=MODEL.remove_guidance(MODEL.BASE.PRFactorial(source,'FULL'));base.raw_pr=False
        unscaled,_=base.features(*args[1:])
        anchor=dict(mean=torch.linspace(-.1,.1,96,dtype=torch.float64),scale=torch.linspace(.01,.2,96,dtype=torch.float64))
        for arm in ['WO_RAW_PR','WO_BOTH_PR']:
            m=MODEL.PrunedFusion(source,arm);m.set_anchor(anchor)
            actual,_=m.features(*args[1:]);self.assertTrue(torch.equal(actual,((unscaled.double()-anchor['mean'])/anchor['scale']).float()))
            m.collect_anchor=True;raw,_=m.features(*args[1:]);self.assertTrue(torch.equal(raw,unscaled))
        for arm in ARMS[1:]:
            m=MODEL.PrunedFusion(source,arm)
            self.assertEqual(m.feature_pr,arm not in ['WO_FEATURE_PR','WO_BOTH_PR'])
            self.assertEqual(m.raw_pr,arm not in ['WO_RAW_PR','WO_BOTH_PR'])
            self.assertEqual(m.pre_reference,arm!='WO_PRE_REFERENCE')
        no_pre=MODEL.PrunedFusion(source,'WO_PRE_REFERENCE');same=copy.deepcopy(full);same.pre_reference=False
        self.assertTrue(torch.equal(same(*args)['logit'],no_pre(*args)['logit']))
        self.assertFalse(torch.equal(full.features(*args[1:])[0],no_pre.features(*args[1:])[0]))

    def test_nested_optimizer_state_tamper(self):
        from admit import equal_tree
        x={'state':{0:{'exp_avg':torch.ones(2)}},'param_groups':[{'params':[0],'lr':.0003}]}
        y=copy.deepcopy(x);self.assertTrue(equal_tree(x,y));y['state'][0]['exp_avg'][1]=2
        self.assertFalse(equal_tree(x,y))


if __name__=='__main__':
    torch.set_num_threads(2);unittest.main()
