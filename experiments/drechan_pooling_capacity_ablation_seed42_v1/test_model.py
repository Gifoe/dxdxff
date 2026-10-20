"""Exact inherited graphs and executable changed pooling/mask/capacity checks."""
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest import mock
import torch
HERE=Path(__file__).resolve().parent
parent=HERE.parent/'drechan_f0_no_lategate_ablation_seed42_v1'
if not parent.is_dir():parent=Path(r'D:\drechan_f0_no_lategate_ablation_seed42_v1')
spec=importlib.util.spec_from_file_location('poolcap_original_fixture',parent/'test_model.py')
FIX=importlib.util.module_from_spec(spec);sys.modules[spec.name]=FIX;spec.loader.exec_module(FIX)
with mock.patch.object(importlib.util,'spec_from_file_location',side_effect=FIX.source_spec):
    spec=FIX.original_spec('poolcap_test_math',HERE/'model.py')
    MODEL=importlib.util.module_from_spec(spec);sys.modules[spec.name]=MODEL;spec.loader.exec_module(MODEL)
sys.path.insert(0,str(HERE))

class ModelTests(unittest.TestCase):
    def test_two_reference_graphs_state_logits_loss_allgrad_exact(self):
        for arm in MODEL.REUSE:
            source,args=FIX.source(),FIX.batch()
            old=MODEL.BASE.NoLateF0(source,arm);new=MODEL.PoolingCapacity(source,arm)
            for n,v in old.state_dict().items():self.assertTrue(torch.equal(v,new.state_dict()[n]))
            a,b=old(*args)['logit'],new(*args)['logit'];self.assertTrue(torch.equal(a,b))
            y=torch.tensor([0.,1.,0.,1.,0.,1.]);losses=[MODEL.objective(x,y,torch.tensor(1.)) for x in [a,b]]
            self.assertTrue(torch.equal(*losses));losses[0].backward();losses[1].backward()
            for n,p in old.named_parameters():self.assertTrue(torch.equal(p.grad,dict(new.named_parameters())[n].grad))

    def test_shared_initializer_rng_capacity_and_finite_gradients(self):
        source,args=FIX.source(),FIX.batch()
        for arm in MODEL.ARMS:
            rng=torch.random.get_rng_state();m=MODEL.PoolingCapacity(source,arm,171001)
            self.assertTrue(torch.equal(rng,torch.random.get_rng_state()))
            self.assertEqual(sum(p.numel() for p in m.parameters()),MODEL.PARAMETERS[arm])
            self.assertFalse(m.late or hasattr(m,'gate') or m.interaction)
            self.assertTrue(torch.equal(m.head[0].weight[:64],source.head[0].weight))
            self.assertTrue(torch.equal(m.head[2].weight[:,:64],source.head[2].weight))
            self.assertEqual(m.head[0].out_features,73 if arm=='A2_CAPACITY_MATCHED' else 64)
            out=m(*args);self.assertEqual(out['raw_summary'].shape,(6,96))
            MODEL.objective(out['logit'],torch.tensor([0.,1.,0.,1.,0.,1.]),torch.tensor(1.)).backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()),arm)
            if arm=='P2_TIME_ATTENTION':self.assertGreater(float(m.attention_weight.grad.norm()),0)
            if arm=='A2_CAPACITY_MATCHED':
                self.assertGreater(float(m.head[0].weight.grad[64:].norm()),0)
                self.assertFalse(m.early or m.guidance_pr)

    def test_population_moments_mask_singleton_constant_and_gradients(self):
        value=torch.tensor([[[1.,2.],[3.,4.],[99.,99.]],[[7.,8.],[99.,99.],[99.,99.]]],requires_grad=True)
        valid=torch.tensor([[True,True,False],[True,False,False]])
        mu,sd,present=MODEL.moments(value,valid)
        self.assertTrue(torch.equal(mu,torch.tensor([[2.,3.],[7.,8.]])))
        self.assertTrue(torch.equal(sd,torch.tensor([[1.,1.],[0.,0.]])))
        self.assertTrue(present.all());(mu.sum()+sd.sum()).backward()
        self.assertTrue(torch.isfinite(value.grad).all());self.assertTrue((value.grad[~valid]==0).all())
        change=value.detach().clone();change[~valid]=float('nan')
        self.assertTrue(torch.equal(mu,MODEL.moments(change,valid)[0]))

    def test_attention_uniform_mask_empty_time_information_and_gradients(self):
        torch.manual_seed(29);value=torch.randn(3,7,24,requires_grad=True);times=torch.arange(7).float()*2
        valid=torch.tensor([[True,False,True,True,False,False,True],[False]*7,[True]+[False]*6])
        weight=torch.zeros(26,requires_grad=True)
        mu,present,w=MODEL.attention_mean(value,times,valid,weight)
        expected,_,_=MODEL.moments(value,valid)
        self.assertTrue(torch.allclose(mu,expected,atol=1e-7,rtol=1e-6))
        self.assertTrue(torch.equal(w[~valid],torch.zeros_like(w[~valid])))
        self.assertTrue(torch.allclose(w.sum(-1),torch.tensor([1.,0.,1.])))
        self.assertEqual(present.tolist(),[True,False,True]);mu.square().sum().backward()
        self.assertTrue(torch.isfinite(weight.grad).all() and torch.isfinite(value.grad).all())
        time_weight=torch.zeros(26);time_weight[24]=2.
        different=MODEL.attention_mean(value.detach(),times,valid,time_weight)[0]
        self.assertFalse(torch.allclose(different[0],expected[0]))

    def test_padding_and_variance_invariance_all_arms(self):
        source=FIX.source();args=list(FIX.batch());args[2][0,0,3]=False
        for arm in MODEL.ARMS:
            m=MODEL.PoolingCapacity(source,arm);out=m(*args)['logit']
            changed=[a.clone() for a in args];changed[0][:,8:10]=float('nan')
            changed[1][~args[2]]=float('nan');changed[4][~args[2]]=float('nan');changed[4][...,4]=float('nan')
            self.assertTrue(torch.equal(out,m(*changed)['logit']),arm)
            if arm in ['WO_EARLY_FILM','A2_CAPACITY_MATCHED']:
                changed[4][:]=float('nan');self.assertTrue(torch.equal(out,m(*changed)['logit']))

    def test_scalar_patient_intervals_and_capacity_contrast_definition(self):
        import numpy as np
        value=np.random.default_rng(42).normal(size=(80,9));draw=np.random.default_rng(42).integers(0,80,(10000,80))
        a=np.concatenate([value[draw[k:k+250]].mean(1) for k in range(0,10000,250)])
        for j in range(9):self.assertTrue(np.allclose(np.quantile(a[:,j],[.025,.975]),np.quantile(value[:,j][draw].mean(1),[.025,.975]),atol=1e-15,rtol=0))
        self.assertEqual(9329+(73-64)*112,MODEL.PARAMETERS['A2_CAPACITY_MATCHED'])
        self.assertLess(abs(10337-10377)/10377,.004)

if __name__=='__main__':
    torch.set_num_threads(2);unittest.main()
