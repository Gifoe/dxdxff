import unittest
import torch
from torch.optim import AdamW
from neuroez_c.p2_q10_lzu_adapter import P2Q10LZUBoundedAdapter,apply_p2_q10_lzu_adapter
from neuroez_c.p2_q10_lzu_adapter_loss import compute_p2_lzu_adapter_loss
from scripts.run_p2_q10_lzu_adapter import _optimization_status,_residual_stats
from scripts.summarize_p2_q10_lzu_adapter import _performance_status


class OptimizationTests(unittest.TestCase):
    def _loss(self,model):
        features=torch.randn(8,5); base=torch.tensor([-.4,.2,.7,-.3,-.2,.4,.8,-.7]); labels=torch.tensor([0.,1.,1.,0.,0.,1.,1.,0.]); patient=torch.tensor([0,0,0,0,1,1,1,1]); gate=torch.ones(8,dtype=torch.bool)
        delta=model(features,gate); adapted=apply_p2_q10_lzu_adapter(base,delta,gate); return compute_p2_lzu_adapter_loss(adapted,labels,patient,delta)["total_loss"]
    def test_zero_output_initialization(self):
        model=P2Q10LZUBoundedAdapter(5); self.assertTrue(torch.equal(model.output_layer.weight,torch.zeros_like(model.output_layer.weight))); self.assertTrue(torch.equal(model.output_layer.bias,torch.zeros_like(model.output_layer.bias)))
    def test_first_backward_output_gradient_nonzero(self):
        torch.manual_seed(1); model=P2Q10LZUBoundedAdapter(5); self._loss(model).backward(); self.assertGreater(float(model.output_layer.weight.grad.norm()),0)
    def test_first_backward_trunk_gradient_zero(self):
        torch.manual_seed(1); model=P2Q10LZUBoundedAdapter(5); self._loss(model).backward(); self.assertEqual(sum(float(parameter.grad.norm()) for parameter in model.trunk_parameters()),0.)
    def test_later_step_trunk_gradient_nonzero(self):
        torch.manual_seed(1); model=P2Q10LZUBoundedAdapter(5); optimizer=AdamW([{"params":list(model.trunk_parameters()),"lr":1e-3},{"params":list(model.output_parameters()),"lr":3e-3}]); loss=self._loss(model); loss.backward(); optimizer.step(); optimizer.zero_grad(); self._loss(model).backward(); self.assertGreater(sum(float(parameter.grad.norm()) for parameter in model.trunk_parameters()),0.)
    def test_optimizer_group_learning_rates(self):
        model=P2Q10LZUBoundedAdapter(5); optimizer=AdamW([{"params":list(model.trunk_parameters()),"lr":1e-3},{"params":list(model.output_parameters()),"lr":3e-3}]); self.assertEqual([group["lr"] for group in optimizer.param_groups],[1e-3,3e-3])
    def test_default_max_delta_bound(self):
        model=P2Q10LZUBoundedAdapter(5); model.output_layer.bias.data.fill_(100); delta=model(torch.randn(2,5),torch.ones(2,dtype=torch.bool)); self.assertLessEqual(float(delta.detach().abs().max()),.300001)
    def test_noop_detection(self):
        status,passed=_optimization_status(100,100,1e-5,1e-5,{"mean_abs_adapter_delta":1e-5,"fraction_abs_delta_gt_0_001":0.}); self.assertEqual(status,"OPTIMIZATION_NO_OP"); self.assertFalse(passed)
    def test_effective_detection(self):
        status,passed=_optimization_status(100,100,.01,.02,{"mean_abs_adapter_delta":.01,"fraction_abs_delta_gt_0_001":.5}); self.assertEqual(status,"OPTIMIZATION_EFFECTIVE"); self.assertTrue(passed)
    def test_insufficient_steps_detection(self):
        status,passed=_optimization_status(99,100,.01,.02,{"mean_abs_adapter_delta":.01,"fraction_abs_delta_gt_0_001":.5}); self.assertEqual(status,"OPTIMIZATION_INSUFFICIENT_STEPS"); self.assertFalse(passed)
    def test_saturation_audit_uses_95_percent_of_bound(self):
        stats=_residual_stats(torch.tensor([.284,.286,.30]).numpy(),.30); self.assertAlmostEqual(stats["saturation_rate"],2/3)
    def test_performance_status_distinguishes_noop_and_no_gain(self):
        self.assertEqual(_performance_status(optimization_status="OPTIMIZATION_NO_OP",minimum_pass=False),"OPTIMIZATION_NO_OP")
        self.assertEqual(_performance_status(optimization_status="OPTIMIZATION_EFFECTIVE",minimum_pass=False),"OPTIMIZATION_EFFECTIVE_BUT_NO_PERFORMANCE_GAIN")
