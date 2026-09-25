from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.rescore_a9v8_lcbo_outputs import (
    passes_main_gate,
    patient_zscore,
    rescore_run_dir,
)


def _write_predictions(path: Path, *, include_logits: bool = True, include_teacher: bool = True) -> None:
    rows = [
        {"fold_idx": 1, "split": "test", "subject_id": "p1", "center": "lzu", "channel_name": "a", "true_ez": 1, "true_nez": 0, "score_eval": 0.90, "score_broad": 0.60, "score_core": 0.80, "a9v3_oof_score": 0.95},
        {"fold_idx": 1, "split": "test", "subject_id": "p1", "center": "lzu", "channel_name": "b", "true_ez": 0, "true_nez": 1, "score_eval": 0.10, "score_broad": 0.40, "score_core": 0.20, "a9v3_oof_score": 0.05},
        {"fold_idx": 2, "split": "test", "subject_id": "p2", "center": "hup", "channel_name": "a", "true_ez": 0, "true_nez": 1, "score_eval": 0.20, "score_broad": 0.30, "score_core": 0.90, "a9v3_oof_score": 0.10},
        {"fold_idx": 2, "split": "test", "subject_id": "p2", "center": "hup", "channel_name": "b", "true_ez": 1, "true_nez": 0, "score_eval": 0.80, "score_broad": 0.70, "score_core": 0.10, "a9v3_oof_score": 0.85},
    ]
    df = pd.DataFrame(rows)
    if include_logits:
        df["logits_broad"] = np.log(df["score_broad"] / (1.0 - df["score_broad"]))
        df["logits_core"] = np.log(df["score_core"] / (1.0 - df["score_core"]))
        df["logits_eval"] = np.log(df["score_eval"] / (1.0 - df["score_eval"]))
    if not include_teacher:
        df = df.drop(columns=["a9v3_oof_score"])
    df.to_csv(path, index=False)


class A9v8LCBOTeacherAnchorRescoreTests(unittest.TestCase):
    def test_missing_teacher_score_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            _write_predictions(run_dir / "test_channel_predictions_neuroez_v2_fold_1.csv", include_teacher=False)

            with self.assertRaisesRegex(RuntimeError, "Missing a9v3_oof_score"):
                rescore_run_dir(run_dir)

    def test_patient_zscore_does_not_cross_patients(self):
        self.assertTrue(np.allclose(patient_zscore(np.array([3.0, 3.0])), np.array([0.0, 0.0])))
        self.assertTrue(np.allclose(patient_zscore(np.array([1.0, 3.0])), np.array([-1.0, 1.0])))

    def test_rescore_writes_teacher_anchor_outputs_and_logit_space_variants(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            _write_predictions(run_dir / "test_channel_predictions_neuroez_v2_fold_1.csv", include_logits=True)

            summary, by_center, by_fold = rescore_run_dir(run_dir, expected_patient_count=2)

            self.assertIn("teacher_only", set(summary["score_variant"]))
            self.assertIn("logit_broad_plus_gamma_core_g+0.10", set(summary["score_variant"]))
            logit_rows = summary[summary["score_variant"].str.contains("logit_broad_plus_gamma_core", regex=False)]
            self.assertTrue((logit_rows["score_space"] == "logit").all())
            self.assertTrue((run_dir / "lcbo_teacher_anchor_rescore_summary.csv").exists())
            self.assertTrue((run_dir / "lcbo_teacher_anchor_rescore_by_center.csv").exists())
            self.assertTrue((run_dir / "lcbo_teacher_anchor_rescore_by_fold.csv").exists())
            self.assertTrue((run_dir / "lcbo_teacher_anchor_best_variants.json").exists())
            self.assertTrue((run_dir / "lcbo_rescore_audit.json").exists())
            self.assertGreater(len(by_center), 0)
            self.assertGreater(len(by_fold), 0)

    def test_rescore_falls_back_to_logit_from_probability(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            _write_predictions(run_dir / "test_channel_predictions_neuroez_v2_fold_1.csv", include_logits=False)

            summary, _, _ = rescore_run_dir(run_dir, expected_patient_count=2)

            fusion_rows = summary[summary["score_variant"].str.contains("logit_broad_plus_gamma_core", regex=False)]
            self.assertGreater(len(fusion_rows), 0)
            self.assertTrue((fusion_rows["score_space"] == "logit_from_probability").all())

    def test_passes_main_gate_logic(self):
        self.assertTrue(passes_main_gate({
            "patient_macro_f1": 0.7,
            "patient_macro_ez_f1": 0.5,
            "patient_macro_auprc_ez": 0.6,
            "patient_macro_ez_mrr": 0.72,
            "top1_is_ez_rate": 0.61,
        }))
        self.assertFalse(passes_main_gate({
            "patient_macro_f1": 0.7,
            "patient_macro_ez_f1": 0.5,
            "patient_macro_auprc_ez": 0.6,
            "patient_macro_ez_mrr": 0.70,
            "top1_is_ez_rate": 0.61,
        }))


if __name__ == "__main__":
    unittest.main()
