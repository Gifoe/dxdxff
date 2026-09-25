from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

import pandas as pd

from neuroez_c.kcalibration import _feature_columns, build_patient_k_features, train_predict_kcal


class KCalNoTrueCountFeatureTests(unittest.TestCase):
    def test_patient_k_feature_columns_exclude_true_count_fields(self) -> None:
        rows = pd.DataFrame(
            {
                "fold_idx": [1, 1, 2, 2],
                "subject_id": ["p1", "p1", "p2", "p2"],
                "center": ["hup"] * 4,
                "true_ez": [1, 0, 0, 1],
                "true_nez": [0, 1, 1, 0],
                "clinical_true_ez": [1, 0, 0, 1],
                "clinical_true_nez": [0, 1, 1, 0],
                "raw_binary_label": [0, 1, 1, 0],
                "final_suspicious_score": [0.9, 0.2, 0.3, 0.8],
                "predicted_by_oracle_k": [1, 0, 0, 1],
            }
        )
        features = build_patient_k_features(rows)
        cols = _feature_columns(features)

        self.assertNotIn("k_true", cols)
        self.assertNotIn("true_ez_count", cols)
        self.assertNotIn("center", cols)

    def test_k_true_uses_clinical_true_ez_not_raw_binary_label_for_ez0(self) -> None:
        rows = pd.DataFrame(
            {
                "fold_idx": [1, 1, 1, 2, 2, 2],
                "subject_id": ["p1"] * 3 + ["p2"] * 3,
                "center": ["hup"] * 6,
                "true_ez": [1, 0, 0, 0, 1, 0],
                "true_nez": [0, 1, 1, 1, 0, 1],
                "clinical_true_ez": [1, 0, 0, 0, 1, 0],
                "clinical_true_nez": [0, 1, 1, 1, 0, 1],
                "raw_binary_label": [0, 1, 1, 1, 0, 1],
                "label_encoding_mode": ["ez0"] * 6,
                "final_suspicious_score": [0.9, 0.2, 0.1, 0.1, 0.8, 0.2],
            }
        )

        features = build_patient_k_features(rows)

        self.assertEqual(features.sort_values("subject_id")["k_true"].tolist(), [1, 1])

    def test_ensemble_ridge_poisson_outputs_kcal_predictions_and_audit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            rows = []
            for fold_idx, subject_id, scores, labels in [
                (1, "p1", [0.9, 0.8, 0.1, 0.0], [1, 1, 0, 0]),
                (2, "p2", [0.95, 0.2, 0.1, 0.0], [1, 0, 0, 0]),
                (3, "p3", [0.5, 0.4, 0.3, 0.2], [0, 0, 0, 1]),
            ]:
                for idx, score in enumerate(scores):
                    rows.append(
                        {
                            "fold_idx": fold_idx,
                            "subject_id": subject_id,
                            "center": "hup",
                            "channel_name": f"A{idx + 1}",
                            "true_ez": labels[idx],
                            "true_nez": 1 - labels[idx],
                            "clinical_true_ez": labels[idx],
                            "clinical_true_nez": 1 - labels[idx],
                            "raw_binary_label": labels[idx],
                            "label_encoding_mode": "ez1",
                            "alpha": 0.10,
                            "final_suspicious_score": score,
                            "final_suspicious_logit": score,
                            "predicted_by_oracle_k": 1 if idx < sum(labels) else 0,
                            "is_candidate": 1,
                        }
                    )
            ledger = tmp_path / "settopo.csv"
            pd.DataFrame(rows).to_csv(ledger, index=False)

            out, audit = train_predict_kcal(
                ledger,
                tmp_path / "kcal",
                model="ensemble_ridge_poisson",
                k_min=1,
                k_max=4,
            )

            self.assertIn("predicted_by_kcal", out.columns)
            self.assertEqual(audit["requested_model"], "ensemble_ridge_poisson")
            self.assertEqual(audit["actual_model_type"], "ensemble_ridge_poisson")
            self.assertFalse(audit["true_ez_count_used_as_inference_input"])
            self.assertEqual(audit["forbidden_inference_feature_intersection"], [])
            self.assertNotIn("k_true_train_label_only", out.columns)
            self.assertEqual(audit["input_alpha"], 0.10)
            self.assertEqual(audit["k_true_source"], "clinical_true_ez")
            self.assertTrue(audit["raw_binary_label_not_used_for_k_true"])


if __name__ == "__main__":
    unittest.main()
