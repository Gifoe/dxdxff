import unittest
from types import SimpleNamespace
import torch
from neuroez_c.v3_rcc_profiles import V3_RCC_PROFILES,validate_v3_rcc_args
from neuroez_c.model import NeuroEZCModel
class RCCProfileTests(unittest.TestCase):
 def test_profiles_disable_q10_boundary_by_contract(self):self.assertEqual(set(V3_RCC_PROFILES),{'R0_BASE','R1_RANK_COVERAGE','R2_HYBRID_CALIBRATION'})
 def test_pairwise_required(self):
  args=SimpleNamespace(v3_rcc_profile='R0_BASE',positive_label='ez',use_ez_ranking_loss=True,ez_ranking_loss_weight=.05)
  self.assertEqual(validate_v3_rcc_args(args).name,'R0_BASE')
 def test_rcc_model_exposes_raw_nez_outputs_without_q10(self):
  args=SimpleNamespace(use_v3_rcc=True,v3_rcc_profile='R0_BASE',use_v3_qbc=False,positive_label='ez',model_dim=8,num_heads=2,dropout=0.,use_channel_attention=True,use_patient_relative_z=True,use_physics_dynamics=False,use_diffusion_residual=False,use_negative_anchor_head=False,use_view_gated_fusion=False,use_two_expert_router=False,use_feature_separated_two_expert=False,use_a9v8_lcbo=False,use_n6_dual_view_ema=False,temporal_pooling='mean',channel_pooling_mode='mean',record_pooling='mean')
  batch={'b0_features':torch.randn(1,1,2,3,36),'labels_ez':torch.tensor([[1.,0.,0.]]),'channel_mask':torch.ones(1,3,dtype=torch.bool),'seizure_mask':torch.ones(1,1,dtype=torch.bool),'seizure_channel_mask':torch.ones(1,1,3,dtype=torch.bool),'window_mask':torch.ones(1,1,2,dtype=torch.bool)}
  out=NeuroEZCModel(args)(batch);self.assertIn('final_nez_logit',out);self.assertTrue(torch.allclose(out['score_nez'],torch.sigmoid(out['final_nez_logit'])))
