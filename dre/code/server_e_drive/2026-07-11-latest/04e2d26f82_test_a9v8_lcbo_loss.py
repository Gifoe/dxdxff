from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from exp_ez_hybrid import compute_a9v8_lcbo_loss


def _args(**overrides):
    values = {
        "lambda_core_rank": 0.20,
        "lambda_soft_mrr": 0.05,
        "lambda_subset": 0.02,
        "lambda_core_distill": 0.05,
        "core_rank_margin": 0.05,
        "soft_mrr_tau": 0.10,
        "subset_eps": 0.05,
        "lcbo_hard_neg_top_frac": 0.30,
        "lcbo_min_core_mass": 1.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _batch() -> dict:
    return {
        "labels_ez": torch.tensor([[1.0, 0.0, 0.0, 1.0]]),
        "channel_mask": torch.ones(1, 4, dtype=torch.bool),
        "subject_id": ["p1"],
        "canonical_channels": [["a", "b", "c", "d"]],
    }


class A9v8LCBOLossTests(unittest.TestCase):
    def test_loss_is_finite_and_uses_logit_eval_for_soft_mrr(self):
        outputs = {
            "logits_broad": torch.tensor([[0.0, 0.0, 0.0, 0.0]], requires_grad=True),
            "logits_core": torch.tensor([[4.0, 3.0, 2.0, 1.0]], requires_grad=True),
            "logits_eval": torch.tensor([[4.0, 3.0, 2.0, 1.0]], requires_grad=True),
            "score_broad": torch.sigmoid(torch.tensor([[0.0, 0.0, 0.0, 0.0]])),
            "score_core": torch.sigmoid(torch.tensor([[4.0, 3.0, 2.0, 1.0]])),
            "score_eval": torch.sigmoid(torch.tensor([[4.0, 3.0, 2.0, 1.0]])),
        }
        lookup = {
            ("p1", "a"): {"label_ez": 1.0, "pseudo_core_q": 1.0, "a9v3_oof_score": 0.9},
            ("p1", "b"): {"label_ez": 0.0, "pseudo_core_q": 0.0, "a9v3_oof_score": 0.2},
            ("p1", "c"): {"label_ez": 0.0, "pseudo_core_q": 0.0, "a9v3_oof_score": 0.1},
            ("p1", "d"): {"label_ez": 1.0, "pseudo_core_q": 0.0, "a9v3_oof_score": 0.3},
        }

        loss, parts = compute_a9v8_lcbo_loss(outputs, _batch(), lookup, _args())

        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(parts["mean_core_rank_loss"] >= 0.0)
        self.assertTrue(parts["mean_soft_mrr_loss"] >= 0.0)
        self.assertIn("subset_violation_rate", parts)

    def test_zero_core_target_patient_skips_core_rank_and_softmrr(self):
        outputs = {
            "logits_broad": torch.zeros(1, 4),
            "logits_core": torch.randn(1, 4),
            "logits_eval": torch.randn(1, 4),
            "score_broad": torch.full((1, 4), 0.5),
            "score_core": torch.sigmoid(torch.randn(1, 4)),
            "score_eval": torch.sigmoid(torch.randn(1, 4)),
        }
        lookup = {
            ("p1", ch): {"label_ez": 1.0 if ch in {"a", "d"} else 0.0, "pseudo_core_q": 0.0, "a9v3_oof_score": 0.1}
            for ch in ("a", "b", "c", "d")
        }

        loss, parts = compute_a9v8_lcbo_loss(outputs, _batch(), lookup, _args())

        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(parts["mean_core_rank_loss"], 0.0)
        self.assertEqual(parts["mean_soft_mrr_loss"], 0.0)


if __name__ == "__main__":
    unittest.main()
