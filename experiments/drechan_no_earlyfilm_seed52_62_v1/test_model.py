"""CPU exact inherited graph, isolation and RNG tests; no data access."""
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest import mock
import torch
HERE=Path(__file__).resolve().parent
parent=HERE.parent/'drechan_f0_no_lategate_ablation_seed42_v1'
if not parent.is_dir():parent=Path(r'D:\drechan_f0_no_lategate_ablation_seed42_v1')
spec=importlib.util.spec_from_file_location('noearly5262_fixture',parent/'test_model.py')
FIX=importlib.util.module_from_spec(spec);sys.modules[spec.name]=FIX;spec.loader.exec_module(FIX)
with mock.patch.object(importlib.util,'spec_from_file_location',side_effect=FIX.source_spec):
    spec=FIX.original_spec('noearly5262_test_math',HERE/'model.py')
    MODEL=importlib.util.module_from_spec(spec);sys.modules[spec.name]=MODEL;spec.loader.exec_module(MODEL)
class Tests(unittest.TestCase):
    def test_state_forward_loss_gradient_and_rng_exact(self):
        initial,args=FIX.source(),FIX.batch()
        a=MODEL.BASE.NoLateF0(initial,'WO_EARLY_FILM')
        rng=torch.random.get_rng_state().clone();b=MODEL.NoEarlyF0(initial,'WO_EARLY_FILM')
        self.assertTrue(torch.equal(rng,torch.random.get_rng_state()))
        self.assertEqual(set(a.state_dict()),set(b.state_dict()))
        for k in a.state_dict():self.assertTrue(torch.equal(a.state_dict()[k],b.state_dict()[k]))
        x,y=a(*args)['logit'],b(*args)['logit'];self.assertTrue(torch.equal(x,y))
        labels=torch.tensor([0.,1.,0.,1.,0.,1.])
        losses=[MODEL.objective(z,labels,torch.tensor(1.)) for z in [x,y]]
        self.assertTrue(torch.equal(*losses));losses[0].backward();losses[1].backward()
        for n,p in a.named_parameters():self.assertTrue(torch.equal(p.grad,dict(b.named_parameters())[n].grad))
    def test_guidance_removed_not_PR_reference_or_head_width(self):
        m=MODEL.NoEarlyF0(FIX.source(),'WO_EARLY_FILM')
        self.assertFalse(m.early or m.late or m.guidance_pr or m.interaction)
        self.assertTrue(m.feature_pr and m.raw_pr and m.pre_reference)
        self.assertFalse(hasattr(m,'gate') or hasattr(m.cnn,'conditioner'))
        self.assertEqual(sum(p.numel() for p in m.parameters()),9329)
        self.assertEqual(m.head[0].out_features,64)
    def test_no_window_condition_or_variance_input(self):
        m=MODEL.NoEarlyF0(FIX.source(),'WO_EARLY_FILM');args=list(FIX.batch())
        a=m(*args)['logit']
        args[4][:]=float('nan');args[0][:,8:10]=float('nan')
        self.assertTrue(torch.equal(a,m(*args)['logit']))
    def test_both_input_branches_effective_gradients(self):
        m=MODEL.NoEarlyF0(FIX.source(),'WO_EARLY_FILM');args=FIX.batch()
        MODEL.objective(m(*args)['logit'],torch.tensor([0.,1.,0.,1.,0.,1.]),torch.tensor(1.)).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))
        self.assertGreater(float(m.cnn[0].weight.grad.norm()),0)
        self.assertGreater(float(m.head[0].weight.grad[:,:96].norm()),0)
        self.assertGreater(float(m.head[0].weight.grad[:,96:].norm()),0)
if __name__=='__main__':
    torch.set_num_threads(2);unittest.main()
