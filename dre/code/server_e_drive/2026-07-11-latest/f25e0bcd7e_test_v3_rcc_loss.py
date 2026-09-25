from types import SimpleNamespace
import unittest
import torch
from neuroez_c.v3_rcc_loss import linear_ramp_weight, patient_mean_unweighted_bce, rcc_loss_components

class RCCLossTests(unittest.TestCase):
 def test_patient_mean_is_patient_equal(self):
  logits=torch.zeros(2,100); labels=torch.zeros(2,100); mask=torch.zeros(2,100,dtype=torch.bool);mask[0,:10]=True;mask[1]=True;labels[0,:10]=1
  value=patient_mean_unweighted_bce(logits,labels,mask)
  self.assertAlmostEqual(float(value),.693147,places=5)
 def test_ramp_is_one_based(self):
  self.assertEqual(linear_ramp_weight(1),0);self.assertEqual(linear_ramp_weight(3),0);self.assertAlmostEqual(linear_ramp_weight(4),.03/7);self.assertEqual(linear_ramp_weight(10),.03);self.assertEqual(linear_ramp_weight(20),.03)
 def test_hybrid_weights(self):
  args=SimpleNamespace(v3_rcc_profile='R2_HYBRID_CALIBRATION',v3_rcc_balanced_bce_weight=.75,v3_rcc_unweighted_bce_weight=.25,v3_rcc_coverage_start_epoch=3,v3_rcc_coverage_end_epoch=10,v3_rcc_coverage_weight=.03,v3_rcc_soft_rank_tau=.1,v3_rcc_soft_topk_tau=.25)
  logits=torch.tensor([[0.,0.]],requires_grad=True);batch={'labels':torch.tensor([[1.,0.]]),'labels_ez':torch.tensor([[1.,0.]]),'channel_mask':torch.ones(1,2,dtype=torch.bool)};loss,parts=rcc_loss_components({'logits':logits,'final_nez_logit':-logits},batch,torch.tensor(2.),args,1)
  self.assertAlmostEqual(float(parts['classification_loss'].detach()),.75*2+.25*.693147,places=5);self.assertEqual(float(parts['coverage_weight_effective'].detach()),0.)
