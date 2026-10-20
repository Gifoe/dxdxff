"""CPU complete inherited graph/isolation/pooling/init tests. No real data access."""
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest import mock
import torch
HERE=Path(__file__).resolve().parent
parent=HERE.parent/'drechan_f0_no_lategate_ablation_seed42_v1'
if not parent.is_dir():parent=Path(r'D:\drechan_f0_no_lategate_ablation_seed42_v1')
spec=importlib.util.spec_from_file_location('nga_fixture',parent/'test_model.py')
FIX=importlib.util.module_from_spec(spec);sys.modules[spec.name]=FIX;spec.loader.exec_module(FIX)
with mock.patch.object(importlib.util,'spec_from_file_location',side_effect=FIX.source_spec):
    spec=FIX.original_spec('nga_test_math',HERE/'model.py')
    MODEL=importlib.util.module_from_spec(spec);sys.modules[spec.name]=MODEL;spec.loader.exec_module(MODEL)
pool=HERE.parent/'drechan_pooling_capacity_ablation_seed42_v1/model.py'
if not pool.is_file():pool=Path(r'D:\drechan_pooling_capacity_ablation_seed42_v1\model.py')
with mock.patch.object(importlib.util,'spec_from_file_location',side_effect=FIX.source_spec):
    spec=FIX.original_spec('nga_pool_reference_math',pool)
    POOL=importlib.util.module_from_spec(spec);sys.modules[spec.name]=POOL;spec.loader.exec_module(POOL)
sys.path.insert(0,str(HERE))
class Tests(unittest.TestCase):
    def test_main_and_restored_state_forward_loss_gradient_exact(self):
        source,args=FIX.source(),FIX.batch()
        for arm,oldarm in [('FULL','WO_EARLY_FILM'),('EARLY_FILM_RESTORED','FULL'),('RAW_ONLY','RAW_ONLY'),('PHYSIOLOGY_ONLY','PHYSIOLOGY_ONLY')]:
            old=MODEL.BASE.NoLateF0(source,oldarm);new=MODEL.NoGuidanceF0(source,arm)
            self.assertEqual(set(old.state_dict()),set(new.state_dict()))
            for k in old.state_dict():self.assertTrue(torch.equal(old.state_dict()[k],new.state_dict()[k]))
            a,b=old(*args)['logit'],new(*args)['logit'];self.assertTrue(torch.equal(a,b),arm)
            y=torch.tensor([0.,1.,0.,1.,0.,1.]);losses=[MODEL.objective(z,y,torch.tensor(1.)) for z in [a,b]]
            self.assertTrue(torch.equal(*losses));losses[0].backward();losses[1].backward()
            for n,p in old.named_parameters():self.assertTrue(torch.equal(p.grad,dict(new.named_parameters())[n].grad),arm)
    def test_all_shared_initializers_rng_head_slices_and_counts(self):
        source=FIX.source();full=MODEL.NoGuidanceF0(source,'FULL');initial=dict(full.named_parameters())
        for arm in MODEL.ARMS:
            rng=torch.random.get_rng_state().clone();m=MODEL.NoGuidanceF0(source,arm,123)
            self.assertTrue(torch.equal(rng,torch.random.get_rng_state()),arm)
            self.assertEqual(sum(p.numel() for p in m.parameters()),MODEL.PARAMETERS[arm])
            self.assertFalse(m.late or hasattr(m,'gate') or m.interaction)
            self.assertEqual(m.early,arm=='EARLY_FILM_RESTORED')
            for name,p in m.named_parameters():
                if 'conditioner' in name or name=='attention_weight':continue
                key=name.replace('cnn.stem.','cnn.') if m.early else name
                x=p.detach();expected=initial[key].detach()
                if arm=='CAPACITY_MATCHED':
                    if name=='head.0.weight':x=x[:64]
                    if name=='head.0.bias':x=x[:64]
                    if name=='head.2.weight':x=x[:,:64]
                if arm=='RAW_ONLY' and name=='head.0.weight':expected=expected[:,:96]
                if arm=='PHYSIOLOGY_ONLY' and name=='head.0.weight':expected=expected[:,96:]
                self.assertTrue(torch.equal(x,expected),(arm,name))
    def test_capacity_exact_previously_fixed_recipe(self):
        source,args=FIX.source(),FIX.batch()
        old=POOL.PoolingCapacity(source,'A2_CAPACITY_MATCHED',123);new=MODEL.NoGuidanceF0(source,'CAPACITY_MATCHED',123)
        self.assertEqual(set(old.state_dict()),set(new.state_dict()))
        for k in old.state_dict():self.assertTrue(torch.equal(old.state_dict()[k],new.state_dict()[k]))
        self.assertTrue(torch.equal(old(*args)['logit'],new(*args)['logit']))
    def test_deleted_coordinates_and_removed_condition_paths(self):
        source,args=FIX.source(),FIX.batch()
        for arm in MODEL.ARMS:
            m=MODEL.NoGuidanceF0(source,arm);a=m(*args)['logit'];change=[v.clone() for v in args]
            change[0][:,8:10]=float('nan');change[4][...,4]=float('nan')
            if arm in MODEL.DROP:
                j=MODEL.DROP[arm];change[0][:,2*j:2*j+2]=float('nan');change[4][...,j]=float('nan')
            if not m.early:change[4][:]=float('nan')
            self.assertTrue(torch.equal(a,m(*change)['logit']),arm)
    def test_PR_reference_boundaries_and_aliases(self):
        source=FIX.source()
        for arm in MODEL.ARMS:
            m=MODEL.NoGuidanceF0(source,arm)
            self.assertEqual(m.raw_pr,arm not in ['WO_RAW_PR','WO_BOTH_PR'])
            self.assertEqual(m.feature_pr,arm not in ['WO_FEATURE_PR','WO_BOTH_PR','RAW_ONLY'])
            self.assertEqual(m.pre_reference,arm!='WO_PRE_REFERENCE')
            self.assertEqual(m.guidance_pr,arm=='EARLY_FILM_RESTORED')
        self.assertEqual(set(MODEL.ALIASES.values()),{'FULL'})
        self.assertEqual(len(MODEL.ARMS),17)
    def test_pure_branch_input_isolation(self):
        source,args=FIX.source(),list(FIX.batch());m=MODEL.NoGuidanceF0(source,'RAW_ONLY')
        a=m(*args)['logit'];args[0][:]=float('nan');args[4][:]=float('nan')
        self.assertTrue(torch.equal(a,m(*args)['logit']))
        m=MODEL.NoGuidanceF0(source,'PHYSIOLOGY_ONLY');args=FIX.batch()
        self.assertTrue(torch.equal(m(*args)['logit'],m(args[0])['logit']))
    def test_masked_attention_gradient_and_uniform_original_pooling(self):
        latent=torch.randn(3,5,24);times=torch.tensor([-2.,0.,4.,8.,float('nan')])
        valid=torch.tensor([[False,True,True,False,False],[False]*5,[False,True,False,True,False]])
        w=torch.zeros(26,requires_grad=True);out,present,weights=MODEL.attention_mean(latent,times,valid,w)
        expected,_sd,available=MODEL.moments(latent,valid)
        torch.testing.assert_close(out,expected,rtol=1e-6,atol=1e-7)
        self.assertTrue(torch.equal(present,available));out.sum().backward();self.assertTrue(torch.isfinite(w.grad).all())
        source,args=FIX.source(),FIX.batch()
        full=MODEL.NoGuidanceF0(source,'FULL');attention=MODEL.NoGuidanceF0(source,'P2_TIME_ATTENTION')
        torch.testing.assert_close(full(*args)['raw_summary'],attention(*args)['raw_summary'],rtol=1e-5,atol=2e-6)
    def test_every_arm_complete_backward(self):
        source,args=FIX.source(),FIX.batch()
        for arm in MODEL.ARMS:
            m=MODEL.NoGuidanceF0(source,arm)
            MODEL.objective(m(*args)['logit'],torch.tensor([0.,1.,0.,1.,0.,1.]),torch.tensor(1.)).backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()),arm)
            self.assertGreater(float(m.head[2].weight.grad.norm()),0)
            if arm=='P2_TIME_ATTENTION':self.assertGreater(float(m.attention_weight.grad.norm()),0)
            if arm=='CAPACITY_MATCHED':self.assertGreater(float(m.head[0].weight.grad[64:].norm()),0)
if __name__=='__main__':
    torch.set_num_threads(2);unittest.main()
