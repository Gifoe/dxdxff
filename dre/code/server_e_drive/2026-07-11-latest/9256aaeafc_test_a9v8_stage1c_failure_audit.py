from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.a9v8_stage1c_failure_audit import run_stage1c_failure_audit


def _write_stage1c_inputs(root: Path, *, include_optional_grids: bool = True) -> None:
    pd.DataFrame(
        [
            {
                "config": "cfg_fail",
                "patient_macro_f1": 0.640,
                "patient_macro_ez_f1": 0.480,
                "patient_macro_auprc_ez": 0.520,
                "patient_macro_ez_mrr": 0.700,
                "top1_is_ez_rate": 0.500,
            },
            {
                "config": "cfg_close",
                "patient_macro_f1": 0.644,
                "patient_macro_ez_f1": 0.472,
                "patient_macro_auprc_ez": 0.519,
                "patient_macro_ez_mrr": 0.714,
                "top1_is_ez_rate": 0.600,
            },
            {
                "config": "cfg_other",
                "patient_macro_f1": 0.700,
                "patient_macro_ez_f1": 0.600,
                "patient_macro_auprc_ez": 0.700,
                "patient_macro_ez_mrr": 0.900,
                "top1_is_ez_rate": 0.900,
            },
        ]
    ).to_csv(root / "validation_selected_teacher_anchor_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "config": "cfg_close",
                "fold_idx": 2,
                "selected_variant_family": "raw_score_eval",
                "selected_score_variant": "score_eval",
                "selected_alpha": "",
                "selected_beta": "",
                "selected_gamma": "",
                "n_patients": 2,
                "patient_macro_f1": 0.650,
                "patient_macro_ez_f1": 0.480,
                "patient_macro_auprc_ez": 0.520,
                "patient_macro_ez_mrr": 0.600,
                "top1_is_ez_rate": 0.400,
            },
            {
                "config": "cfg_close",
                "fold_idx": 4,
                "selected_variant_family": "teacher_only",
                "selected_score_variant": "teacher_only",
                "selected_alpha": "",
                "selected_beta": "",
                "selected_gamma": "",
                "n_patients": 1,
                "patient_macro_f1": 0.500,
                "patient_macro_ez_f1": 0.300,
                "patient_macro_auprc_ez": 0.400,
                "patient_macro_ez_mrr": 0.500,
                "top1_is_ez_rate": 0.000,
            },
            {
                "config": "cfg_close",
                "fold_idx": 4,
                "selected_variant_family": "raw_score_eval",
                "selected_score_variant": "score_eval",
                "selected_alpha": "",
                "selected_beta": "",
                "selected_gamma": "",
                "n_patients": 1,
                "patient_macro_f1": 0.520,
                "patient_macro_ez_f1": 0.330,
                "patient_macro_auprc_ez": 0.420,
                "patient_macro_ez_mrr": 0.520,
                "top1_is_ez_rate": 0.000,
            },
            {
                "config": "cfg_other",
                "fold_idx": 4,
                "selected_variant_family": "teacher_only",
                "selected_score_variant": "teacher_only",
                "selected_alpha": "",
                "selected_beta": "",
                "selected_gamma": "",
                "n_patients": 1,
                "patient_macro_f1": 0.510,
                "patient_macro_ez_f1": 0.310,
                "patient_macro_auprc_ez": 0.410,
                "patient_macro_ez_mrr": 0.510,
                "top1_is_ez_rate": 0.000,
            },
        ]
    ).to_csv(root / "validation_selected_teacher_anchor_by_fold.csv", index=False)
    pd.DataFrame(
        [
            {
                "config": "cfg_close",
                "fold_idx": 2,
                "selected_variant_family": "raw_score_eval",
                "selected_score_variant": "score_eval",
                "selected_alpha": "",
                "selected_beta": "",
                "selected_gamma": "",
                "subject_id": "p1",
                "center": "hup",
                "valid_channel_count": 4,
                "ez_channel_count": 1,
                "ez_fraction": 0.25,
                "patient_macro_f1": 0.6,
                "patient_macro_ez_f1": 0.4,
                "patient_macro_auprc_ez": 0.5,
                "patient_macro_ez_mrr": 0.5,
                "top1_is_ez": 0.0,
            },
            {
                "config": "cfg_close",
                "fold_idx": 2,
                "selected_variant_family": "raw_score_eval",
                "selected_score_variant": "score_eval",
                "selected_alpha": "",
                "selected_beta": "",
                "selected_gamma": "",
                "subject_id": "p2",
                "center": "lzu",
                "valid_channel_count": 6,
                "ez_channel_count": 2,
                "ez_fraction": 0.333,
                "patient_macro_f1": 0.7,
                "patient_macro_ez_f1": 0.5,
                "patient_macro_auprc_ez": 0.6,
                "patient_macro_ez_mrr": 0.7,
                "top1_is_ez": 1.0,
            },
            {
                "config": "cfg_other",
                "fold_idx": 4,
                "selected_variant_family": "teacher_only",
                "selected_score_variant": "teacher_only",
                "selected_alpha": "",
                "selected_beta": "",
                "selected_gamma": "",
                "subject_id": "p3",
                "center": "pediatric",
                "valid_channel_count": 8,
                "ez_channel_count": 2,
                "ez_fraction": 0.25,
                "patient_macro_f1": 0.2,
                "patient_macro_ez_f1": 0.1,
                "patient_macro_auprc_ez": 0.2,
                "patient_macro_ez_mrr": 0.2,
                "top1_is_ez": 0.0,
            },
        ]
    ).to_csv(root / "validation_selected_teacher_anchor_patient_rows.csv", index=False)
    if include_optional_grids:
        selected = pd.DataFrame(
            [
                {
                    "config": "cfg_close",
                    "fold_idx": 2,
                    "selected_score_variant": "score_eval",
                    "selected_alpha": "",
                    "selected_beta": "",
                    "selected_gamma": "",
                    "val_patient_macro_f1": 0.65,
                    "val_patient_macro_ez_f1": 0.48,
                    "val_patient_macro_auprc_ez": 0.52,
                    "val_patient_macro_ez_mrr": 0.60,
                    "val_top1_is_ez_rate": 0.40,
                },
                {
                    "config": "cfg_other",
                    "fold_idx": 4,
                    "selected_score_variant": "teacher_only",
                    "selected_alpha": "",
                    "selected_beta": "",
                    "selected_gamma": "",
                    "val_patient_macro_f1": 0.2,
                    "val_patient_macro_ez_f1": 0.1,
                    "val_patient_macro_auprc_ez": 0.2,
                    "val_patient_macro_ez_mrr": 0.2,
                    "val_top1_is_ez_rate": 0.0,
                },
            ]
        )
        selected.to_csv(root / "validation_selected_teacher_anchor_selected_params.csv", index=False)
        selected.rename(columns={
            "selected_score_variant": "score_variant",
            "selected_alpha": "alpha",
            "selected_beta": "beta",
            "selected_gamma": "gamma",
        }).to_csv(root / "validation_selected_teacher_anchor_val_grid.csv", index=False)
        pd.DataFrame(
            [
                {
                    "config": "cfg_close",
                    "fold_idx": 2,
                    "score_variant": "score_eval",
                    "alpha": "",
                    "beta": "",
                    "gamma": "",
                    "patient_macro_f1": 0.65,
                    "patient_macro_ez_f1": 0.48,
                    "patient_macro_auprc_ez": 0.52,
                    "patient_macro_ez_mrr": 0.60,
                    "top1_is_ez_rate": 0.40,
                },
                {
                    "config": "cfg_close",
                    "fold_idx": 2,
                    "score_variant": "teacher_only",
                    "alpha": "",
                    "beta": "",
                    "gamma": "",
                    "patient_macro_f1": 0.64,
                    "patient_macro_ez_f1": 0.47,
                    "patient_macro_auprc_ez": 0.51,
                    "patient_macro_ez_mrr": 0.80,
                    "top1_is_ez_rate": 1.0,
                },
                {
                    "config": "cfg_other",
                    "fold_idx": 4,
                    "score_variant": "teacher_only",
                    "alpha": "",
                    "beta": "",
                    "gamma": "",
                    "patient_macro_f1": 0.2,
                    "patient_macro_ez_f1": 0.1,
                    "patient_macro_auprc_ez": 0.2,
                    "patient_macro_ez_mrr": 0.2,
                    "top1_is_ez_rate": 0.0,
                },
            ]
        ).to_csv(root / "validation_selected_teacher_anchor_test_grid.csv", index=False)


class A9v8Stage1cFailureAuditTests(unittest.TestCase):
    def test_gate_deltas_fold_tags_regret_and_recommendation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_stage1c_inputs(root)

            outputs = run_stage1c_failure_audit(root, candidate_configs=["cfg_close"])

            gate = outputs["gate_report"]
            close = gate[gate["config"] == "cfg_close"].iloc[0]
            self.assertIn("low_mrr", close["failed_gate_metrics"])
            self.assertLess(close["distance_to_gate"], 0.0)
            fold = outputs["fold_failure"]
            fold2 = fold[fold["fold_idx"] == 2].iloc[0]
            self.assertIn("low_mrr", fold2["weakness_tag"])
            self.assertIn("low_top1", fold2["weakness_tag"])
            mismatch = outputs["selection_mismatch"].iloc[0]
            self.assertGreater(mismatch["test_mrr_regret"], 0.0)
            self.assertGreater(mismatch["test_top1_regret"], 0.0)
            center = outputs["center_by_fold"]
            self.assertEqual(set(center["center"]), {"hup", "lzu"})
            rec = json.loads((root / "stage1c_failure_audit_recommendation.json").read_text())
            self.assertFalse(rec["whether_to_continue_training"])
            self.assertFalse(rec["whether_any_config_passes_main_gate"])

    def test_candidate_filter_applies_to_all_outputs_and_closest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_stage1c_inputs(root)

            outputs = run_stage1c_failure_audit(root, candidate_configs=["cfg_close"])
            rec = json.loads((root / "stage1c_failure_audit_recommendation.json").read_text())

            for key in ("gate_report", "fold_failure", "selection_mismatch", "patient_failure", "center_by_fold"):
                df = outputs[key]
                if not df.empty and "config" in df.columns:
                    self.assertEqual(set(df["config"].astype(str)), {"cfg_close"})
            self.assertEqual(rec["closest_config_to_gate"], "cfg_close")

    def test_weakest_folds_are_unique_and_ranked_from_aggregate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_stage1c_inputs(root)

            run_stage1c_failure_audit(root)
            rec = json.loads((root / "stage1c_failure_audit_recommendation.json").read_text())
            ranking = pd.read_csv(root / "stage1c_weakest_fold_ranking.csv")

            self.assertEqual(len(rec["weakest_folds_unique"]), len(set(rec["weakest_folds_unique"])))
            self.assertLessEqual(len(rec["weakest_folds_unique"]), 3)
            self.assertIn("worst_min_delta", ranking.columns)
            self.assertEqual(ranking.iloc[0]["fold_idx"], 4)

    def test_weakest_center_rankings_can_differ_by_global_mean_and_worst_fold(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_stage1c_inputs(root)

            run_stage1c_failure_audit(root)
            rec = json.loads((root / "stage1c_failure_audit_recommendation.json").read_text())
            ranking = pd.read_csv(root / "stage1c_weakest_center_ranking.csv")

            self.assertIn("weakest_centers_by_global_mean", rec)
            self.assertIn("weakest_centers_by_worst_fold", rec)
            self.assertNotIn("weakest_centers", rec)
            self.assertIn("worst_fold_mrr", ranking.columns)

    def test_closest_config_is_limited_to_candidate_configs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_stage1c_inputs(root)

            run_stage1c_failure_audit(root, candidate_configs=["cfg_fail"])
            rec = json.loads((root / "stage1c_failure_audit_recommendation.json").read_text())

            self.assertEqual(rec["closest_config_to_gate"], "cfg_fail")

    def test_missing_optional_grids_warn_but_do_not_crash(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_stage1c_inputs(root, include_optional_grids=False)

            outputs = run_stage1c_failure_audit(root)

            self.assertTrue(outputs["selection_mismatch"].empty)
            rec = json.loads((root / "stage1c_failure_audit_recommendation.json").read_text())
            self.assertIn("validation_selected_teacher_anchor_selected_params.csv", rec["optional_files_missing"])

    def test_missing_required_files_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(FileNotFoundError, "validation_selected_teacher_anchor_summary"):
                run_stage1c_failure_audit(Path(tmpdir))


if __name__ == "__main__":
    unittest.main()
