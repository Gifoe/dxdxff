from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from neuroez_c.v3_qbc_profiles import V3_QBC_PROFILES, get_v3_qbc_profile, validate_v3_qbc_args
from neuroez_c.model import NeuroEZCModel


class V3QBCProfileTests(unittest.TestCase):
    def test_final_bcr_profiles_are_q10_free(self):
        self.assertEqual(set(V3_QBC_PROFILES), {
            "BCR_BC_ONLY", "BCR_BOUNDARY_ONLY", "BCR_COVERAGE_ONLY", "BCR_BOUNDARY_COVERAGE",
        })

    def test_profile_module_matrix(self):
        self.assertFalse(get_v3_qbc_profile("BCR_BC_ONLY").use_boundary)
        self.assertTrue(get_v3_qbc_profile("BCR_BOUNDARY_COVERAGE").use_boundary)
        self.assertTrue(get_v3_qbc_profile("BCR_BOUNDARY_COVERAGE").use_coverage)
        self.assertTrue(get_v3_qbc_profile("BCR_BOUNDARY_ONLY").use_boundary)
        self.assertFalse(get_v3_qbc_profile("BCR_BOUNDARY_ONLY").use_coverage)
        self.assertFalse(get_v3_qbc_profile("BCR_COVERAGE_ONLY").use_boundary)
        self.assertTrue(get_v3_qbc_profile("BCR_COVERAGE_ONLY").use_coverage)
        self.assertFalse(hasattr(get_v3_qbc_profile("BCR_BOUNDARY_COVERAGE"), "use_q10"))

    def test_qbc_rejects_incompatible_branches(self):
        args = SimpleNamespace(
            v3_qbc_profile="BCR_BOUNDARY_COVERAGE", positive_label="ez", use_n6_dual_view_ema=True,
            use_two_expert_router=False, use_feature_separated_two_expert=False,
            use_a9v8_lcbo=False, use_broad_ez_mil_loss=False,
            use_diffusion_residual=False, use_view_gated_fusion=False,
        )
        with self.assertRaises(ValueError):
            validate_v3_qbc_args(args)

    def test_qbc_requires_v3_ez_positive_adapter(self):
        args = SimpleNamespace(v3_qbc_profile="BCR_BC_ONLY", positive_label="nez")
        with self.assertRaises(ValueError):
            validate_v3_qbc_args(args)

    def test_full_profile_model_outputs_explicit_nez_semantics(self):
        args = SimpleNamespace(
            use_v3_qbc=True, v3_qbc_profile="BCR_BOUNDARY_COVERAGE", positive_label="ez",
            model_dim=8, num_heads=2, dropout=0.0, use_channel_attention=True,
            use_patient_relative_z=True, use_physics_dynamics=False,
            use_diffusion_residual=False, use_negative_anchor_head=False,
            use_view_gated_fusion=False, use_two_expert_router=False,
            use_feature_separated_two_expert=False, use_a9v8_lcbo=False,
            use_n6_dual_view_ema=False, temporal_pooling="mean",
            channel_pooling_mode="mean", record_pooling="mean",
        )
        batch = {
            "b0_features": torch.randn(2, 2, 3, 4, 36),
            "labels_ez": torch.tensor([[1., 0., 0., 1.], [1., 0., 0., 1.]]),
            "channel_mask": torch.ones(2, 4, dtype=torch.bool),
            "seizure_mask": torch.ones(2, 2, dtype=torch.bool),
            "seizure_channel_mask": torch.ones(2, 2, 4, dtype=torch.bool),
            "window_mask": torch.ones(2, 2, 3, dtype=torch.bool),
        }
        output = NeuroEZCModel(args)(batch)
        for key in ("base_nez_logit", "final_nez_logit", "ez_semantic_logit"):
            self.assertIn(key, output)
        self.assertFalse(any("q10" in key.lower() for key in output))
        self.assertTrue(torch.allclose(output["score_nez"], torch.sigmoid(output["final_nez_logit"]), atol=1e-6))
        self.assertTrue(torch.allclose(output["score_ez"], 1.0 - output["score_nez"], atol=1e-6))


if __name__ == "__main__":
    unittest.main()
