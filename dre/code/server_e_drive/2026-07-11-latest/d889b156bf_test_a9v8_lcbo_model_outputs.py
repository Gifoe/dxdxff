from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from neuroez_c.model import NeuroEZCModel
from patient_channel_ranker import PatientChannelClassifier


def _args(**overrides):
    values = {
        "model_dim": 8,
        "num_heads": 2,
        "dropout": 0.0,
        "use_channel_attention": True,
        "use_patient_relative_z": True,
        "positive_label": "ez",
        "b0_feature_parts": "abs,delta,zdelta,ratio",
        "b0_feature_groups": "spectral_classical",
        "self_compare_eps": 1e-5,
        "temporal_pooling": "mean",
        "temporal_topk_fraction": 0.20,
        "temporal_pooling_top_p": 0.10,
        "temporal_pooling_tau": 0.10,
        "record_pooling": "mean",
        "record_pooling_top_p": 0.30,
        "record_pooling_alpha": 0.70,
        "use_physics_dynamics": False,
        "use_diffusion_residual": False,
        "use_view_gated_fusion": False,
        "use_negative_anchor_head": False,
        "use_two_expert_router": False,
        "use_feature_separated_two_expert": False,
        "use_a9v8_lcbo": False,
        "eval_score_fusion_gamma": 0.10,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _batch() -> dict:
    return {
        "b0_features": torch.randn(2, 1, 3, 4, 36),
        "labels": torch.tensor([[1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]]),
        "labels_nez": torch.tensor([[0.0, 1.0, 1.0, 0.0], [0.0, 1.0, 1.0, 0.0]]),
        "labels_ez": torch.tensor([[1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]]),
        "channel_mask": torch.ones(2, 4, dtype=torch.bool),
        "seizure_mask": torch.ones(2, 1, dtype=torch.bool),
        "seizure_channel_mask": torch.ones(2, 1, 4, dtype=torch.bool),
        "window_mask": torch.ones(2, 1, 3, dtype=torch.bool),
        "center_id": torch.tensor([1, 3], dtype=torch.long),
    }


class A9v8LCBOModelOutputTests(unittest.TestCase):
    def test_classifier_old_path_output_keys_unchanged(self):
        classifier = PatientChannelClassifier(input_dim=8, num_heads=2, dropout=0.0, positive_label="ez")
        out = classifier(torch.randn(1, 3, 8), torch.ones(1, 3, dtype=torch.bool))

        self.assertEqual(set(out), {"logits", "scores", "score_nez", "score_ez"})

    def test_classifier_lcbo_eval_score_uses_logit_fusion(self):
        gamma = 0.15
        classifier = PatientChannelClassifier(
            input_dim=8,
            num_heads=2,
            dropout=0.0,
            positive_label="ez",
            use_a9v8_lcbo=True,
            eval_score_fusion_gamma=gamma,
        )
        out = classifier(torch.randn(1, 3, 8), torch.ones(1, 3, dtype=torch.bool))

        expected_logits = out["logits_broad"] + gamma * out["logits_core"]
        self.assertTrue(torch.allclose(out["logits_eval"], expected_logits, atol=1e-6))
        self.assertTrue(torch.allclose(out["score_eval"], torch.sigmoid(expected_logits), atol=1e-6))
        self.assertTrue(torch.allclose(out["score_ez"], out["score_eval"]))

    def test_model_lcbo_forward_exposes_dual_head_scores(self):
        out = NeuroEZCModel(_args(use_a9v8_lcbo=True, eval_score_fusion_gamma=0.05))(_batch())

        for key in ("logits_core", "logits_broad", "logits_eval", "score_core", "score_broad", "score_eval"):
            self.assertIn(key, out)
        self.assertTrue(torch.allclose(out["score_eval"], torch.sigmoid(out["logits_eval"]), atol=1e-6))
        self.assertTrue(torch.allclose(out["logits"], out["logits_eval"], atol=1e-6))


if __name__ == "__main__":
    unittest.main()
