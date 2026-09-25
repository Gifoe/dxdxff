from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd

from neuroez_c.clean_nez_utils import (
    build_clean_nez_distance_ledger,
    export_v3_clean_nez_ledger,
)
from neuroez_c.kcalibration import train_predict_kcal
from neuroez_c.settopo_reranker import generate_clean_nez_candidates, run_settopo_reranker
from scripts.audit_clean_nez_pipeline import audit_pipeline


def _write_v3_fold_outputs(v3_dir: Path) -> pd.DataFrame:
    rows = []
    fold_subjects = {1: ["s3", "s4"], 2: ["s1", "s2"]}
    for fold_idx, subjects in fold_subjects.items():
        fold_rows = []
        for subject_id in subjects:
            for channel_id, channel_name in enumerate(["A1", "A2", "A3", "B1", "B2"]):
                score_ez = [0.05, 0.10, 0.75, 0.55, 0.20][channel_id]
                true_ez = 1 if channel_name in {"A3", "B1"} else 0
                fold_rows.append(
                    {
                        "fold_idx": fold_idx,
                        "subject_id": subject_id,
                        "center": "hup",
                        "channel_id": channel_id,
                        "channel_name": channel_name,
                        "true_ez": true_ez,
                        "score_ez_probability": score_ez,
                        "rank_ez_desc": channel_id + 1,
                        "predicted_ez": int(score_ez >= 0.5),
                    }
                )
        pd.DataFrame(fold_rows).to_csv(v3_dir / f"test_channel_predictions_neuroez_v2_fold_{fold_idx}.csv", index=False)
        rows.extend(fold_rows)
    return pd.DataFrame(rows)


def _write_embeddings(embedding_dir: Path, fold_idx: int, subjects: list[str]) -> None:
    rows = []
    for subject_id in subjects:
        for idx, channel_name in enumerate(["A1", "A2", "A3", "B1", "B2"]):
            row = {
                "fold_idx": fold_idx,
                "subject_id": subject_id,
                "channel_name": channel_name,
                "n_records": 2,
            }
            for prefix in ("rawbb_all", "rawbb_onset", "rawbb_preictal"):
                row[f"{prefix}_0"] = float(idx)
                row[f"{prefix}_1"] = float(idx * 0.5)
            rows.append(row)
    pd.DataFrame(rows).to_csv(embedding_dir / f"rawbrainbert_patient_channel_embeddings_fold_{fold_idx}.csv", index=False)


class CleanNEZPipelineTests(unittest.TestCase):
    def test_export_v3_clean_nez_ledger_adds_clean_nez_scores_and_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            v3_dir = tmp_path / "v3"
            v3_dir.mkdir()
            input_rows = _write_v3_fold_outputs(v3_dir)
            allowed = tmp_path / "allowed.csv"
            input_rows[["subject_id"]].drop_duplicates().to_csv(allowed, index=False)

            output = tmp_path / "v3_clean_nez_ledger.csv"
            ledger, audit = export_v3_clean_nez_ledger(
                v3_dir,
                output,
                fold_start=1,
                fold_end=2,
                allowed_subjects_ledger=allowed,
                require_n_patients=4,
            )

            self.assertEqual(len(ledger), len(input_rows))
            self.assertIn("p_clean_nez", ledger.columns)
            self.assertIn("feature_non_nez_score", ledger.columns)
            self.assertTrue(np.allclose(ledger["p_clean_nez"], 1.0 - ledger["score_ez_probability"]))
            self.assertEqual(audit["n_patients"], 4)
            self.assertEqual(audit["missing_columns"], [])
            self.assertTrue((output.parent / "v3_clean_nez_ledger_audit.json").exists())

    def test_export_v3_clean_nez_ledger_ez0_preserves_clinical_labels_and_raw_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            v3_dir = tmp_path / "v3"
            v3_dir.mkdir()
            pd.DataFrame(
                {
                    "fold_idx": [1, 1],
                    "subject_id": ["s1", "s1"],
                    "center": ["hup", "hup"],
                    "channel_id": [0, 1],
                    "channel_name": ["A1", "A2"],
                    "true_ez": [1, 0],
                    "score_nez_probability": [0.10, 0.85],
                    "pred_topk": [0, 1],
                }
            ).to_csv(v3_dir / "test_channel_predictions_neuroez_v2_fold_1.csv", index=False)

            output = tmp_path / "v3_clean_nez_ledger.csv"
            ledger, audit = export_v3_clean_nez_ledger(
                v3_dir,
                output,
                fold_start=1,
                fold_end=1,
                label_encoding_mode="ez0",
            )

            self.assertEqual(set(ledger["label_encoding_mode"]), {"ez0"})
            self.assertEqual(set(ledger["ez_label_value"]), {0})
            self.assertEqual(set(ledger["nez_label_value"]), {1})
            self.assertEqual(ledger["true_ez"].tolist(), [1, 0])
            self.assertEqual(ledger["clinical_true_ez"].tolist(), [1, 0])
            self.assertEqual(ledger["clinical_true_nez"].tolist(), [0, 1])
            self.assertEqual(ledger["raw_binary_label"].tolist(), [0, 1])
            self.assertEqual(ledger["predicted_ez"].tolist(), [1, 0])
            self.assertTrue(np.allclose(ledger["p_clean_nez"], [0.10, 0.85]))
            self.assertTrue(np.allclose(ledger["feature_non_nez_score"], [0.90, 0.15]))
            self.assertEqual(audit["label_encoding_mode"], "ez0")
            self.assertFalse(audit["raw_binary_label_used_as_clinical_label"])

    def test_export_v3_clean_nez_ledger_interprets_score_eval_by_label_encoding_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            v3_dir = tmp_path / "v3"
            v3_dir.mkdir()
            pd.DataFrame(
                {
                    "fold_idx": [1, 1],
                    "subject_id": ["s1", "s1"],
                    "center": ["hup", "hup"],
                    "channel_id": [0, 1],
                    "channel_name": ["A1", "A2"],
                    "true_ez": [1, 0],
                    "score_eval": [0.80, 0.25],
                    "pred_topk": [1, 0],
                }
            ).to_csv(v3_dir / "test_channel_predictions_neuroez_v2_fold_1.csv", index=False)

            ez1, _ = export_v3_clean_nez_ledger(
                v3_dir,
                tmp_path / "ez1.csv",
                fold_start=1,
                fold_end=1,
                label_encoding_mode="ez1",
            )
            ez0, _ = export_v3_clean_nez_ledger(
                v3_dir,
                tmp_path / "ez0.csv",
                fold_start=1,
                fold_end=1,
                label_encoding_mode="ez0",
            )

            self.assertTrue(np.allclose(ez1["score_ez_probability"], [0.80, 0.25]))
            self.assertTrue(np.allclose(ez1["score_nez_probability"], [0.20, 0.75]))
            self.assertTrue(np.allclose(ez1["p_clean_nez"], [0.20, 0.75]))
            self.assertTrue(np.allclose(ez0["score_nez_probability"], [0.80, 0.25]))
            self.assertTrue(np.allclose(ez0["score_ez_probability"], [0.20, 0.75]))
            self.assertTrue(np.allclose(ez0["p_clean_nez"], [0.80, 0.25]))
            self.assertTrue(np.allclose(ez0["feature_non_nez_score"], [0.20, 0.75]))

    def test_clean_nez_distance_uses_pseudo_anchors_not_true_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            v3_dir = tmp_path / "v3"
            embedding_dir = tmp_path / "embeddings"
            v3_dir.mkdir()
            embedding_dir.mkdir()
            _write_v3_fold_outputs(v3_dir)
            ledger_path = tmp_path / "v3_clean_nez_ledger.csv"
            export_v3_clean_nez_ledger(v3_dir, ledger_path, fold_start=1, fold_end=2)
            for fold_idx in (1, 2):
                _write_embeddings(embedding_dir, fold_idx, ["s1", "s2", "s3", "s4"])

            distance, audit = build_clean_nez_distance_ledger(
                ledger_path,
                embedding_dir,
                tmp_path / "distance",
                anchor_quantile=0.70,
                min_anchors=2,
                distance_modes=("cosine", "euclidean"),
            )

            self.assertEqual(len(distance), len(pd.read_csv(ledger_path)))
            self.assertIn("raw_dist_all_cosine", distance.columns)
            self.assertIn("raw_dist_onset_z", distance.columns)
            self.assertTrue(distance.groupby("subject_id")["is_pseudo_clean_nez_anchor"].sum().ge(2).all())
            self.assertFalse(audit["whether_true_label_used_for_anchor_selection"])
            self.assertTrue((tmp_path / "distance" / "clean_nez_anchor_audit.json").exists())

    def test_candidate_generation_includes_same_shaft_neighbors_and_preserves_rows(self):
        rows = pd.DataFrame(
            {
                "fold_idx": [1, 1, 1, 1, 1],
                "subject_id": ["s1"] * 5,
                "channel_name": ["A1", "A2", "A3", "A4", "B1"],
                "feature_suspicious_z": [0.0, 0.1, 5.0, 0.2, 0.0],
                "raw_dist_onset_z": [0.0, 0.1, 4.0, 0.2, 0.0],
                "raw_dist_all_z": [0.0, 0.1, 3.0, 0.2, 0.0],
            }
        )

        out, audit = generate_clean_nez_candidates(rows)

        self.assertEqual(len(out), len(rows))
        selected = set(out.loc[out["is_candidate"].astype(bool), "channel_name"])
        self.assertTrue({"A1", "A2", "A3", "A4"}.issubset(selected))
        self.assertEqual(audit["row_count"], len(rows))

    def test_settopo_and_kcal_preserve_rows_and_avoid_forbidden_features(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            distance = pd.DataFrame(
                {
                    "fold_idx": [1, 1, 1, 1, 2, 2, 2, 2],
                    "subject_id": ["s1"] * 4 + ["s2"] * 4,
                    "center": ["hup"] * 8,
                    "channel_name": ["A1", "A2", "A3", "A4"] * 2,
                    "true_ez": [0, 0, 1, 1] * 2,
                    "true_nez": [1, 1, 0, 0] * 2,
                    "p_clean_nez": [0.95, 0.80, 0.20, 0.10] * 2,
                    "feature_non_nez_score": [0.05, 0.20, 0.80, 0.90] * 2,
                    "feature_suspicious_z": [-1.0, -0.5, 0.8, 1.2] * 2,
                    "raw_dist_all_z": [-0.8, -0.4, 0.5, 1.1] * 2,
                    "raw_dist_onset_z": [-0.9, -0.3, 0.6, 1.0] * 2,
                    "raw_dist_preictal_z": [-0.7, -0.2, 0.4, 0.8] * 2,
                    "is_pseudo_clean_nez_anchor": [1, 1, 0, 0] * 2,
                    "n_records": [2] * 8,
                }
            )
            distance_path = tmp_path / "distance.csv"
            distance.to_csv(distance_path, index=False)

            settopo, settopo_audit = run_settopo_reranker(
                distance_path,
                tmp_path / "settopo",
                alpha_list=(0.10,),
                train_alpha=0.10,
                epochs=2,
                learning_rate=1e-3,
            )
            self.assertEqual(len(settopo), len(distance))
            self.assertTrue(settopo_audit["no_center_features"])
            self.assertTrue(settopo_audit["no_outcome_features"])
            self.assertEqual(settopo_audit["train_alpha"], 0.10)
            self.assertEqual(settopo_audit["main_alpha"], 0.10)
            self.assertTrue(settopo_audit["alpha_protocol_no_test_selection"])
            self.assertTrue(settopo_audit["settopo_uses_clinical_true_ez_for_ranking"])
            self.assertTrue(settopo_audit["settopo_uses_clinical_true_nez_for_clean_loss"])

            kcal_ledger, kcal_audit = train_predict_kcal(
                tmp_path / "settopo" / "corrected_suspicious_ledger_alpha0.10.csv",
                tmp_path / "kcal",
                k_min=1,
                k_max=4,
                model="ridge_poisson",
            )
            self.assertEqual(len(kcal_ledger), len(distance))
            self.assertTrue(kcal_ledger.groupby("subject_id")["predicted_by_kcal"].sum().ge(1).all())
            self.assertNotIn("true_ez_count", kcal_audit["inference_feature_columns"])
            self.assertEqual(kcal_audit["k_true_source"], "clinical_true_ez")
            self.assertTrue(kcal_audit["raw_binary_label_not_used_for_k_true"])

    def test_settopo_marks_non_train_alpha_outputs_diagnostic_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            distance = pd.DataFrame(
                {
                    "fold_idx": [1, 1, 2, 2, 3, 3],
                    "subject_id": ["s1", "s1", "s2", "s2", "s3", "s3"],
                    "center": ["hup"] * 6,
                    "channel_name": ["A1", "A2"] * 3,
                    "true_ez": [1, 0] * 3,
                    "true_nez": [0, 1] * 3,
                    "clinical_true_ez": [1, 0] * 3,
                    "clinical_true_nez": [0, 1] * 3,
                    "raw_binary_label": [1, 0] * 3,
                    "label_encoding_mode": ["ez1"] * 6,
                    "ez_label_value": [1] * 6,
                    "nez_label_value": [0] * 6,
                    "p_clean_nez": [0.1, 0.9] * 3,
                    "feature_non_nez_score": [0.9, 0.1] * 3,
                    "feature_suspicious_z": [1.0, -1.0] * 3,
                    "raw_dist_all_z": [0.5, -0.5] * 3,
                    "raw_dist_onset_z": [0.5, -0.5] * 3,
                    "raw_dist_preictal_z": [0.2, -0.2] * 3,
                    "is_pseudo_clean_nez_anchor": [0, 1] * 3,
                    "n_records": [1] * 6,
                }
            )
            distance_path = tmp_path / "distance.csv"
            distance.to_csv(distance_path, index=False)

            _, audit = run_settopo_reranker(
                distance_path,
                tmp_path / "settopo",
                alpha_list=(0.05, 0.10),
                train_alpha=0.10,
                epochs=1,
                d_model=16,
                num_layers=1,
                num_heads=4,
            )
            diagnostic = pd.read_csv(tmp_path / "settopo" / "corrected_suspicious_ledger_alpha0.05.csv")
            main = pd.read_csv(tmp_path / "settopo" / "corrected_suspicious_ledger_alpha0.10.csv")

            self.assertTrue(audit["diagnostic_alpha_outputs"])
            self.assertEqual(audit["diagnostic_alphas"], [0.05])
            self.assertTrue(diagnostic["diagnostic_only"].all())
            self.assertFalse(main["diagnostic_only"].any())

    def test_clean_nez_audit_fails_missing_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            result = audit_pipeline(SimpleNamespace(base_dir=str(tmp_path), require_n_patients=4))
            self.assertFalse(result["ok"])
            self.assertGreater(len(result["failed_checks"]), 0)
            self.assertTrue((tmp_path / "clean_nez_pipeline_audit.json").exists())
            self.assertTrue((tmp_path / "clean_nez_pipeline_audit_errors.txt").exists())

    def test_clean_nez_audit_requires_rawbrainbert_fold_artifacts_1_to_5(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            rawbb = tmp_path / "rawbrainbert"
            embeddings = tmp_path / "embeddings"
            rawbb.mkdir()
            embeddings.mkdir()
            for fold_idx in (1, 2, 3, 4):
                (rawbb / f"rawbrainbert_pretrain_audit_fold_{fold_idx}.json").write_text(
                    json.dumps(
                        {
                            "failure_used_in_ssl": False,
                            "leakage_success_test_subjects_in_ssl": [],
                            "onset_timing_verified": False,
                        }
                    ),
                    encoding="utf-8",
                )
                (rawbb / f"rawbrainbert_encoder_fold_{fold_idx}.pt").write_text("stub", encoding="utf-8")
                (rawbb / f"rawbrainbert_preproc_fold_{fold_idx}.pkl").write_text("stub", encoding="utf-8")
                (embeddings / f"rawbrainbert_embedding_audit_fold_{fold_idx}.json").write_text(
                    json.dumps({"onset_timing_verified": False}),
                    encoding="utf-8",
                )

            result = audit_pipeline(SimpleNamespace(base_dir=str(tmp_path), require_n_patients=None))
            check_names = {entry["name"] for entry in result["failed_checks"]}

            self.assertIn("rawbrainbert_required_artifacts_fold_1_to_5_exist", check_names)

    def test_clean_nez_audit_checks_label_encoding_and_alpha_protocol(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "clean_nez_distance").mkdir()
            (tmp_path / "settopo").mkdir()
            (tmp_path / "kcal").mkdir()
            (tmp_path / "eval").mkdir()
            (tmp_path / "rawbrainbert").mkdir()
            (tmp_path / "embeddings").mkdir()
            ledger = pd.DataFrame(
                {
                    "fold_idx": [1, 1],
                    "subject_id": ["s1", "s1"],
                    "center": ["hup", "hup"],
                    "channel_name": ["A1", "A2"],
                    "true_ez": [1, 0],
                    "true_nez": [0, 1],
                    "clinical_true_ez": [1, 0],
                    "clinical_true_nez": [0, 1],
                    "raw_binary_label": [0, 1],
                    "label_encoding_mode": ["ez0", "ez0"],
                    "ez_label_value": [0, 0],
                    "nez_label_value": [1, 1],
                    "alpha": [0.20, 0.20],
                    "is_candidate": [1, 1],
                    "predicted_by_kcal": [1, 0],
                    "final_suspicious_score": [0.9, 0.1],
                }
            )
            for path in [
                tmp_path / "v3_clean_nez_ledger.csv",
                tmp_path / "clean_nez_distance" / "clean_nez_distance_ledger.csv",
                tmp_path / "settopo" / "corrected_suspicious_ledger_alpha0.10.csv",
                tmp_path / "kcal" / "settopo_ledger_with_kcal.csv",
            ]:
                ledger.to_csv(path, index=False)
            pd.DataFrame(
                {
                    "method": ["V3 baseline predicted_ez"],
                    "patient_macro_f1": [1.0],
                    "patient_ez_f1": [1.0],
                    "patient_nez_f1": [1.0],
                }
            ).to_csv(tmp_path / "eval" / "patient_level_metrics.csv", index=False)
            pd.DataFrame({"method": ["V3 baseline predicted_ez"], "patient_macro_f1": [1.0]}).to_csv(
                tmp_path / "eval" / "metrics_summary.csv",
                index=False,
            )
            (tmp_path / "v3_clean_nez_ledger_audit.json").write_text(json.dumps({"label_encoding_mode": "ez0"}), encoding="utf-8")
            (tmp_path / "clean_nez_distance" / "clean_nez_anchor_audit.json").write_text(
                json.dumps({"whether_true_label_used_for_anchor_selection": False}),
                encoding="utf-8",
            )
            (tmp_path / "settopo" / "settopo_training_audit.json").write_text(
                json.dumps(
                    {
                        "model_type": "real_settopo",
                        "real_settopo_used": True,
                        "feature_columns": [],
                        "no_center_features": True,
                        "no_outcome_features": True,
                        "true_ez_count_not_used_as_feature": True,
                        "candidate_audit": {"candidate_count": 2, "candidate_fraction": 1.0},
                        "settopo_uses_clinical_true_ez_for_ranking": True,
                        "settopo_uses_clinical_true_nez_for_clean_loss": True,
                        "raw_binary_label_not_used_as_clinical_target": True,
                        "train_alpha": 0.10,
                        "main_alpha": 0.10,
                        "alpha_list": [0.10],
                        "alpha_protocol_no_test_selection": True,
                    }
                ),
                encoding="utf-8",
            )
            (tmp_path / "kcal" / "kcal_training_audit.json").write_text(
                json.dumps(
                    {
                        "inference_feature_columns": [],
                        "forbidden_inference_feature_intersection": [],
                        "actual_model_type": "ensemble_ridge_poisson",
                        "requested_model": "ensemble_ridge_poisson",
                        "fold_audits": {"1": {"train_patients": 1, "test_patients": 1}},
                        "k_true_source": "clinical_true_ez",
                        "raw_binary_label_not_used_for_k_true": True,
                        "input_alpha": 0.20,
                    }
                ),
                encoding="utf-8",
            )
            (tmp_path / "eval" / "evaluation_audit.json").write_text(
                json.dumps(
                    {
                        "macro_f1_formula": "0.5*(ez_f1+nez_f1)",
                        "v3_subject_set_equals_main": True,
                        "evaluation_target": "clinical_true_ez",
                        "raw_binary_label_used_as_metric_target": False,
                    }
                ),
                encoding="utf-8",
            )
            for fold_idx in range(1, 6):
                (tmp_path / "rawbrainbert" / f"rawbrainbert_pretrain_audit_fold_{fold_idx}.json").write_text(
                    json.dumps({"failure_used_in_ssl": False, "leakage_success_test_subjects_in_ssl": []}),
                    encoding="utf-8",
                )
                (tmp_path / "rawbrainbert" / f"rawbrainbert_encoder_fold_{fold_idx}.pt").write_text("stub", encoding="utf-8")
                (tmp_path / "rawbrainbert" / f"rawbrainbert_preproc_fold_{fold_idx}.pkl").write_text("stub", encoding="utf-8")
                (tmp_path / "embeddings" / f"rawbrainbert_embedding_audit_fold_{fold_idx}.json").write_text("{}", encoding="utf-8")

            result = audit_pipeline(SimpleNamespace(base_dir=str(tmp_path), require_n_patients=None, label_encoding_mode="ez0"))
            failed = {entry["name"] for entry in result["failed_checks"]}

            self.assertIn("kcal_input_alpha_matches_settopo_main_alpha", failed)
            self.assertIn("alpha_protocol_no_test_selection", {entry["name"] for entry in result["checks"]})
            self.assertIn("ledgers_have_required_label_encoding_columns", {entry["name"] for entry in result["checks"]})


if __name__ == "__main__":
    unittest.main()
