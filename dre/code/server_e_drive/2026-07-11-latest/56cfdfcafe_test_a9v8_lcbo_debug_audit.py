from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from exp_ez_hybrid import (
    Exp_EZHybridLocalization,
    build_lcbo_target_lookup,
    compute_lcbo_target_match_stats,
)


def _batch() -> dict:
    return {
        "labels_ez": torch.tensor([[1.0, 0.0, 1.0]]),
        "labels": torch.tensor([[1.0, 0.0, 1.0]]),
        "channel_mask": torch.ones(1, 3, dtype=torch.bool),
        "subject_id": ["p1"],
        "canonical_channels": [["a", "b", "c"]],
    }


def _outputs() -> dict:
    logits_broad = torch.zeros(1, 3)
    logits_core = torch.tensor([[1.0, 0.0, -1.0]])
    logits_eval = logits_broad + 0.1 * logits_core
    return {
        "logits_broad": logits_broad,
        "logits_core": logits_core,
        "logits_eval": logits_eval,
        "score_broad": torch.sigmoid(logits_broad),
        "score_core": torch.sigmoid(logits_core),
        "score_eval": torch.sigmoid(logits_eval),
    }


class A9v8LCBODebugAuditTests(unittest.TestCase):
    def test_target_match_stats_report_missing_channels(self):
        lookup = build_lcbo_target_lookup(
            __import__("pandas").DataFrame(
                [
                    {"patient_id": "p1", "subject_id": "p1", "fold_id": 1, "channel_name": "a", "label_ez": 1, "pseudo_core_q": 0.8, "a9v3_oof_score": 0.9},
                    {"patient_id": "p1", "subject_id": "p1", "fold_id": 1, "channel_name": "b", "label_ez": 0, "pseudo_core_q": 0.0, "a9v3_oof_score": 0.1},
                ]
            )
        )

        stats = compute_lcbo_target_match_stats(_batch(), lookup)

        self.assertEqual(stats["matched_channels"], 2.0)
        self.assertEqual(stats["total_valid_channels"], 3.0)
        self.assertAlmostEqual(stats["target_match_rate"], 2.0 / 3.0)
        self.assertEqual(stats["n_core_positive_channels"], 1.0)

    def test_training_lcbo_loss_fails_closed_when_targets_missing(self):
        exp = Exp_EZHybridLocalization.__new__(Exp_EZHybridLocalization)
        exp.args = SimpleNamespace(use_a9v8_lcbo=True)
        exp.use_a9v8_lcbo = True
        exp.current_lcbo_target_lookup = {}

        with self.assertRaisesRegex(RuntimeError, "target_match_rate"):
            exp._compute_loss(_outputs(), _batch(), torch.tensor(1.0), split_name="train")


if __name__ == "__main__":
    unittest.main()
