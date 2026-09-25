from __future__ import annotations

import unittest
from types import SimpleNamespace

import pandas as pd
import torch

from exp_ez_hybrid import apply_teacher_anchor_eval_scores, build_lcbo_target_lookup


def _batch() -> dict:
    return {
        "channel_mask": torch.ones(1, 3, dtype=torch.bool),
        "labels_ez": torch.tensor([[1.0, 0.0, 1.0]]),
        "subject_id": ["p1"],
        "canonical_channels": [["a", "b", "c"]],
    }


def _outputs() -> dict:
    logits_broad = torch.tensor([[0.0, 1.0, -1.0]])
    logits_core = torch.tensor([[1.0, -1.0, 0.0]])
    logits_eval = logits_broad + 0.1 * logits_core
    return {
        "logits_broad": logits_broad.clone(),
        "logits_core": logits_core.clone(),
        "logits_eval": logits_eval.clone(),
        "score_broad": torch.sigmoid(logits_broad),
        "score_core": torch.sigmoid(logits_core),
        "score_eval": torch.sigmoid(logits_eval),
        "logits": logits_eval.clone(),
        "scores": torch.sigmoid(logits_eval),
        "score_ez": torch.sigmoid(logits_eval),
        "score_nez": 1.0 - torch.sigmoid(logits_eval),
    }


def _lookup() -> dict:
    return build_lcbo_target_lookup(pd.DataFrame([
        {"patient_id": "p1", "channel_name": "a", "label_ez": 1, "pseudo_core_q": 0.8, "a9v3_oof_score": 0.9},
        {"patient_id": "p1", "channel_name": "b", "label_ez": 0, "pseudo_core_q": 0.0, "a9v3_oof_score": 0.2},
        {"patient_id": "p1", "channel_name": "c", "label_ez": 1, "pseudo_core_q": 0.4, "a9v3_oof_score": 0.5},
    ]))


class A9v8TeacherAnchorEvalTests(unittest.TestCase):
    def test_disabled_teacher_anchor_keeps_outputs_unchanged(self):
        outputs = _outputs()
        original = outputs["score_eval"].clone()
        args = SimpleNamespace(use_teacher_anchor_eval=False)

        updated = apply_teacher_anchor_eval_scores(outputs, _batch(), _lookup(), args)

        self.assertTrue(torch.allclose(updated["score_eval"], original))
        self.assertNotIn("score_eval_original", updated)

    def test_enabled_teacher_anchor_rewrites_eval_and_preserves_original(self):
        outputs = _outputs()
        original_score = outputs["score_eval"].clone()
        original_logits = outputs["logits_eval"].clone()
        args = SimpleNamespace(
            use_teacher_anchor_eval=True,
            teacher_anchor_alpha=0.2,
            teacher_anchor_beta=-0.1,
            teacher_anchor_space="patient_zscore_probability",
            teacher_anchor_temperature=1.0,
            teacher_anchor_require_oof_score=True,
        )

        updated = apply_teacher_anchor_eval_scores(outputs, _batch(), _lookup(), args)

        self.assertTrue(torch.allclose(updated["score_eval_original"], original_score))
        self.assertTrue(torch.allclose(updated["logits_eval_original"], original_logits))
        self.assertTrue(torch.allclose(updated["score_ez"], updated["score_eval"]))
        self.assertFalse(torch.allclose(updated["score_eval"], original_score))
        self.assertAlmostEqual(float(updated["teacher_anchor_alpha"]), 0.2)
        self.assertAlmostEqual(float(updated["teacher_anchor_beta"]), -0.1)

    def test_require_oof_score_fails_closed_when_targets_missing(self):
        args = SimpleNamespace(
            use_teacher_anchor_eval=True,
            teacher_anchor_alpha=0.1,
            teacher_anchor_beta=0.0,
            teacher_anchor_space="patient_zscore_probability",
            teacher_anchor_temperature=1.0,
            teacher_anchor_require_oof_score=True,
        )

        with self.assertRaisesRegex(RuntimeError, "teacher anchor"):
            apply_teacher_anchor_eval_scores(_outputs(), _batch(), {}, args)


if __name__ == "__main__":
    unittest.main()
