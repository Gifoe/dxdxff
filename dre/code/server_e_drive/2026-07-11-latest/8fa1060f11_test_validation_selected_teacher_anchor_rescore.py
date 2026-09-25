from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.validation_selected_teacher_anchor_rescore import (
    candidate_scores_for_channels,
    run_validation_selected_rescore,
)


def _rows(split: str, fold_idx: int, *, subject_id: str, teacher_good: bool, raw_good: bool) -> list[dict]:
    if teacher_good:
        teacher = [0.90, 0.80, 0.20, 0.10]
    else:
        teacher = [0.20, 0.10, 0.90, 0.80]
    if raw_good:
        raw = [0.90, 0.80, 0.20, 0.10]
    else:
        raw = [0.95, 0.10, 0.90, 0.20]  # improves one hit but top1/MRR can be unsafe in other folds.
    broad = [0.90, 0.80, 0.20, 0.10]
    core = [0.10, 0.20, 0.80, 0.90]
    labels = [1, 1, 0, 0]
    out = []
    for i, (ez, t, r, b, c) in enumerate(zip(labels, teacher, raw, broad, core)):
        out.append(
            {
                "fold_idx": fold_idx,
                "split": split,
                "subject_id": subject_id,
                "center": "hup" if fold_idx == 1 else "lzu",
                "channel_name": f"ch{i}",
                "true_ez": ez,
                "score_eval": r,
                "score_broad": b,
                "score_core": c,
                "a9v3_oof_score": t,
            }
        )
    return out


class ValidationSelectedTeacherAnchorRescoreTests(unittest.TestCase):
    def _make_run(self, root: Path) -> Path:
        cfg = root / "cfgA"
        cfg.mkdir()
        val_rows = []
        test_rows = []
        val_rows += _rows("val", 1, subject_id="p1", teacher_good=True, raw_good=False)
        val_rows += _rows("val", 2, subject_id="p2", teacher_good=False, raw_good=True)
        test_rows += _rows("test", 1, subject_id="p3", teacher_good=False, raw_good=True)
        test_rows += _rows("test", 2, subject_id="p4", teacher_good=True, raw_good=False)
        pd.DataFrame(val_rows).to_csv(cfg / "val_channel_predictions_neuroez_v2_fold_1.csv", index=False)
        pd.DataFrame(test_rows).to_csv(cfg / "test_channel_predictions_neuroez_v2_fold_1.csv", index=False)
        return cfg

    def test_missing_teacher_score_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = self._make_run(Path(tmpdir))
            path = cfg / "val_channel_predictions_neuroez_v2_fold_1.csv"
            df = pd.read_csv(path).drop(columns=["a9v3_oof_score"])
            df.to_csv(path, index=False)

            with self.assertRaisesRegex(RuntimeError, "a9v3_oof_score"):
                run_validation_selected_rescore(Path(tmpdir), expected_patient_count=4)

    def test_patient_zscore_does_not_cross_subjects(self):
        df = pd.DataFrame(
            [
                {"fold_idx": 1, "subject_id": "p1", "score_broad": 2.0, "score_core": 2.0, "a9v3_oof_score": 2.0, "score_eval": 0.1},
                {"fold_idx": 1, "subject_id": "p1", "score_broad": 2.0, "score_core": 2.0, "a9v3_oof_score": 2.0, "score_eval": 0.2},
                {"fold_idx": 1, "subject_id": "p2", "score_broad": 1.0, "score_core": 3.0, "a9v3_oof_score": 5.0, "score_eval": 0.3},
                {"fold_idx": 1, "subject_id": "p2", "score_broad": 3.0, "score_core": 1.0, "a9v3_oof_score": 7.0, "score_eval": 0.4},
            ]
        )
        scores = candidate_scores_for_channels(df, alpha_grid=[0.5], beta_grid=[0.0], gamma_grid=[], include_logit_broad_core=False)
        candidate = scores["teacher_plus_broad_core_a0.50_b+0.00"]

        self.assertTrue(np.allclose(candidate[:2], np.zeros(2)))
        self.assertTrue(np.allclose(candidate[2:], np.array([-1.5, 1.5])))

    def test_validation_selection_is_fold_local_and_outputs_are_written(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._make_run(Path(tmpdir))
            result = run_validation_selected_rescore(
                Path(tmpdir),
                alpha_grid=[0.02],
                beta_grid=[0.0],
                gamma_grid=[0.0],
                expected_patient_count=4,
                include_raw_score_eval=True,
                include_teacher_only=True,
                include_logit_broad_core=True,
            )
            root = Path(tmpdir)
            selected = result["selected_params"]
            fold1 = selected[selected["fold_idx"] == 1].iloc[0]
            fold2 = selected[selected["fold_idx"] == 2].iloc[0]

            self.assertEqual(fold1["selected_score_variant"], "teacher_only")
            self.assertEqual(fold2["selected_score_variant"], "raw_score_eval")
            self.assertTrue((root / "validation_selected_teacher_anchor_summary.csv").exists())
            self.assertTrue((root / "validation_selected_teacher_anchor_by_fold.csv").exists())
            self.assertTrue((root / "validation_selected_teacher_anchor_by_center.csv").exists())
            self.assertTrue((root / "validation_selected_teacher_anchor_patient_rows.csv").exists())
            self.assertTrue((root / "validation_selected_teacher_anchor_selected_params.csv").exists())
            self.assertTrue((root / "validation_selected_teacher_anchor_val_grid.csv").exists())
            self.assertTrue((root / "validation_selected_teacher_anchor_test_grid.csv").exists())
            self.assertTrue((root / "validation_selected_teacher_anchor_audit.json").exists())
            self.assertTrue((root / "validation_selected_teacher_anchor_fold_failure_audit.csv").exists())

    def test_teacher_only_and_raw_score_candidates_reproduce_input_columns(self):
        df = pd.DataFrame(
            [
                {"fold_idx": 1, "subject_id": "p1", "score_eval": 0.2, "score_broad": 0.4, "score_core": 0.6, "a9v3_oof_score": 0.8},
                {"fold_idx": 1, "subject_id": "p1", "score_eval": 0.3, "score_broad": 0.5, "score_core": 0.7, "a9v3_oof_score": 0.9},
            ]
        )
        scores = candidate_scores_for_channels(df, alpha_grid=[], beta_grid=[], gamma_grid=[], include_logit_broad_core=False)

        self.assertTrue(np.allclose(scores["teacher_only"], df["a9v3_oof_score"].to_numpy()))
        self.assertTrue(np.allclose(scores["raw_score_eval"], df["score_eval"].to_numpy()))

    def test_gate_margin_prefers_safer_candidate_over_single_metric_gain(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = Path(tmpdir) / "cfgA"
            cfg.mkdir()
            val = pd.DataFrame(
                [
                    {"fold_idx": 1, "split": "val", "subject_id": "p1", "center": "hup", "channel_name": "a", "true_ez": 1, "score_eval": 0.95, "score_broad": 0.95, "score_core": 0.10, "a9v3_oof_score": 0.90},
                    {"fold_idx": 1, "split": "val", "subject_id": "p1", "center": "hup", "channel_name": "b", "true_ez": 1, "score_eval": 0.10, "score_broad": 0.10, "score_core": 0.20, "a9v3_oof_score": 0.80},
                    {"fold_idx": 1, "split": "val", "subject_id": "p1", "center": "hup", "channel_name": "c", "true_ez": 0, "score_eval": 0.90, "score_broad": 0.90, "score_core": 0.80, "a9v3_oof_score": 0.20},
                    {"fold_idx": 1, "split": "val", "subject_id": "p1", "center": "hup", "channel_name": "d", "true_ez": 0, "score_eval": 0.20, "score_broad": 0.20, "score_core": 0.90, "a9v3_oof_score": 0.10},
                ]
            )
            test = val.copy()
            test["split"] = "test"
            test.to_csv(cfg / "val_channel_predictions_neuroez_v2_fold_1.csv", index=False)
            test.to_csv(cfg / "test_channel_predictions_neuroez_v2_fold_1.csv", index=False)

            result = run_validation_selected_rescore(Path(tmpdir), alpha_grid=[], beta_grid=[], gamma_grid=[], expected_patient_count=1)
            selected = result["selected_params"].iloc[0]

            self.assertEqual(selected["selected_score_variant"], "teacher_only")
            self.assertGreaterEqual(selected["val_pass_count"], 5)


if __name__ == "__main__":
    unittest.main()
