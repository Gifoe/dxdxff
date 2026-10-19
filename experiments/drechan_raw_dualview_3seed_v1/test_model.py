"""Changed-path checks only; no synthetic accuracy or surrogate outcome evidence."""
import copy
import math
import unittest
from torch import nn
from common import *
from view_math import views,patient_z,channel_z


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
    def test_shared_slices_rng_and_parameter_count(self):
        s=source();old=MODEL.BASE.BASE.PRFactorial(s,'FULL');state=old.state_dict()
        for arm in NEW:
            before=torch.random.get_rng_state().clone();m=MODEL.DualView(s,arm)
            self.assertTrue(torch.equal(before,torch.random.get_rng_state()))
            for name,value in m.cnn.state_dict().items():
                original=('cnn.'+name).replace('cnn.0.','cnn.stem.0.').replace('cnn.2.','cnn.stem.2.')
                self.assertTrue(torch.equal(value,state[original]))
            self.assertTrue(torch.equal(m.head[0].weight[:,:96]*2,old.head[0].weight[:,:96]))
            self.assertTrue(torch.equal(m.head[0].weight[:,96:192]*2,old.head[0].weight[:,:96]))
            self.assertTrue(torch.equal(m.head[0].weight[:,192:],old.head[0].weight[:,96:]))
            self.assertTrue(torch.equal(m.head[0].bias,old.head[0].bias))
            self.assertTrue(torch.equal(m.head[2].weight,old.head[2].weight))
            self.assertEqual(sum(p.numel() for p in m.parameters()),15473)
            self.assertEqual(sum(isinstance(q,nn.Linear) and q.out_features==1 for q in m.modules()),1)
            self.assertFalse(m.early or m.late or m.interaction or hasattr(m,'gate'))

    def test_views_and_only_patient_context_uses_other_channels(self):
        torch.manual_seed(71);v=torch.randn(6,96);v[:,0]=2
        mu=torch.linspace(-.1,.1,96,dtype=torch.float64);scale=torch.linspace(.01,.2,96,dtype=torch.float64)
        g=((v.double()-mu)/scale).float();a=v.double();sd=a.std(0,unbiased=False)
        r=((a-a.mean(0))/torch.where(sd>1e-8,sd,torch.ones_like(sd))).float()
        self.assertTrue(torch.equal(views(v,mu,scale),torch.cat([g,r],1)))
        self.assertTrue(torch.equal(views(v,mu,scale,False),torch.cat([g,channel_z(g)],1)))
        self.assertTrue(torch.equal(views(v,mu,scale,True,False),torch.cat([channel_z(r),r],1)))
        changed=v.clone();changed[1:]=changed[1:]*3+5
        self.assertTrue(torch.equal(views(v,mu,scale,False)[0],views(changed,mu,scale,False)[0]))
        self.assertFalse(torch.equal(views(v,mu,scale)[0],views(changed,mu,scale)[0]))
        self.assertTrue(torch.equal(views(v,mu,scale,True,False),views(v,mu+50,scale*3,True,False)))
        self.assertTrue(torch.equal(patient_z(torch.ones(1,96)),torch.zeros(1,96)))
        self.assertTrue(torch.isfinite(channel_z(torch.ones(1,96))).all())

    def test_original_pooling_proxy_and_joint_head_formula(self):
        s=source();args=batch();original=MODEL.BASE.remove_guidance(MODEL.BASE.BASE.PRFactorial(s,'FULL'));original.raw_pr=False
        m=MODEL.DualView(s,'FULL');v,_=original.features(*args[1:]);new,_=m.pooled_vector(*args[1:])
        self.assertTrue(torch.equal(v,new))
        z,_=m.features(*args[1:]);x=args[0].clone();x[:,8:10]=0;first=m.head[0]
        a=torch.nn.functional.linear(z*math.sqrt(110/192),first.weight[:,:192])
        b=torch.nn.functional.linear(x*math.sqrt(110/28),first.weight[:,192:])
        expected=m.head[2](m.head[1](a+b+first.bias)).squeeze(-1)
        self.assertTrue(torch.equal(expected,m(*args)['logit']))
        original.pre_reference=False;no_pre=MODEL.DualView(s,'WO_PRE_REFERENCE')
        self.assertTrue(torch.equal(original.features(*args[1:])[0],no_pre.pooled_vector(*args[1:])[0]))

    def test_all_paths_gradients_masks_and_no_both_pr_invariance(self):
        s=source();args=list(batch())
        for arm in NEW:
            m=MODEL.DualView(s,arm);out=m(*args)['logit'];out.square().mean().backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))
            for value in [m.cnn[0].weight.grad,m.head[0].weight.grad[:,:96],m.head[0].weight.grad[:,96:192],m.head[0].weight.grad[:,192:]]:
                self.assertGreater(float(value.norm()),0)
            changed=list(args);changed[0]=args[0].clone();changed[0][:,8:10]=float('nan');changed[4]=torch.full_like(args[4],float('nan'))
            self.assertTrue(torch.equal(out,m(*changed)['logit']))
            self.assertEqual(m.raw_pr,arm not in ['WO_RAW_PR','WO_BOTH_PR'])
            self.assertEqual(m.feature_pr,arm not in ['WO_FEATURE_PR','WO_BOTH_PR'])
            self.assertEqual(m.global_view,arm!='WO_GLOBAL_VIEW');self.assertEqual(m.pre_reference,arm!='WO_PRE_REFERENCE')
        m=MODEL.DualView(s,'WO_BOTH_PR');changed=list(args);changed[0]=args[0].clone();changed[1]=args[1].clone()
        changed[0][1:]+=3;changed[1][:,1:]=changed[1][:,1:].flip(-1)
        self.assertTrue(torch.equal(m(*args)['logit'][0],m(*changed)['logit'][0]))

    def test_nested_optimizer_state_tamper(self):
        from admit import equal_tree
        x={'state':{0:{'exp_avg':torch.ones(2)}},'param_groups':[{'params':[0],'lr':.0003}]}
        y=copy.deepcopy(x);self.assertTrue(equal_tree(x,y));y['state'][0]['exp_avg'][1]=2
        self.assertFalse(equal_tree(x,y))

    def test_independent_auditor_preserves_separate_cast_gradients(self):
        from admit import independent_views
        torch.manual_seed(79);a=torch.randn(17,96,requires_grad=True);b=a.detach().clone().requires_grad_()
        mu=torch.linspace(-.2,.2,96,dtype=torch.float64);scale=torch.linspace(.01,.3,96,dtype=torch.float64)
        expected=views(a,mu,scale);actual=independent_views(b,mu,scale)
        self.assertTrue(torch.equal(expected,actual))
        weights=torch.linspace(-1,1,192)
        (expected*weights).square().mean().backward();(actual*weights).square().mean().backward()
        self.assertTrue(torch.equal(a.grad,b.grad))


if __name__=='__main__':
    torch.set_num_threads(2);unittest.main()
