from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

import pandas as pd

from scripts.evaluate_clean_nez_pipeline import _patient_rows, evaluate_clean_nez_pipeline


class CleanNEZMetricsTests(unittest.TestCase):
    def test_patient_rows_compute_ez_nez_and_macro_f1_separately(self) -> None:
        df = pd.DataFrame(
            {
                "fold_idx": [1, 1, 1, 1],
                "subject_id": ["p1"] * 4,
                "center": ["hup"] * 4,
                "true_ez": [1, 0, 0, 0],
                "score": [0.9, 0.8, 0.2, 0.1],
                "predicted_by_kcal": [1, 1, 0, 0],
            }
        )

        rows = _patient_rows(df, method="toy", score_col="score", pred_col="predicted_by_kcal")
        row = rows.iloc[0]

        self.assertEqual(int(row["tp"]), 1)
        self.assertEqual(int(row["fp"]), 1)
        self.assertEqual(int(row["fn"]), 0)
        self.assertEqual(int(row["tn"]), 2)
        self.assertAlmostEqual(float(row["patient_ez_f1"]), 2.0 / 3.0, places=6)
        self.assertAlmostEqual(float(row["patient_nez_f1"]), 0.8, places=6)
        self.assertAlmostEqual(float(row["patient_macro_f1"]), (2.0 / 3.0 + 0.8) * 0.5, places=6)
        self.assertNotEqual(float(row["patient_ez_f1"]), float(row["patient_nez_f1"]))

    def test_evaluation_filters_v3_and_requires_subject_set_equality(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            main = pd.DataFrame(
                {
                    "fold_idx": [1, 1],
                    "subject_id": ["p1", "p1"],
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
                    "final_suspicious_logit": [2.0, -1.0],
                    "predicted_by_kcal": [1, 0],
                    "predicted_by_oracle_k": [1, 0],
                }
            )
            v3 = pd.concat(
                [
                    main.assign(score_ez_probability=[0.8, 0.1], predicted_ez=[1, 0]),
                    pd.DataFrame(
                        {
                            "fold_idx": [1],
                            "subject_id": ["extra"],
                            "center": ["hup"],
                            "channel_name": ["B1"],
                            "true_ez": [1],
                            "true_nez": [0],
                            "clinical_true_ez": [1],
                            "clinical_true_nez": [0],
                            "raw_binary_label": [0],
                            "label_encoding_mode": ["ez0"],
                            "ez_label_value": [0],
                            "nez_label_value": [1],
                            "score_ez_probability": [0.9],
                            "predicted_ez": [1],
                        }
                    ),
                ],
                ignore_index=True,
            )
            main_path = tmp_path / "main.csv"
            v3_path = tmp_path / "v3.csv"
            main.to_csv(main_path, index=False)
            v3.to_csv(v3_path, index=False)

            with self.assertRaisesRegex(ValueError, "subject_id set mismatch"):
                evaluate_clean_nez_pipeline(main_path, tmp_path / "eval", v3_ledger=v3_path)

            allowed = tmp_path / "allowed.csv"
            pd.DataFrame({"subject_id": ["p1"]}).to_csv(allowed, index=False)
            _, audit = evaluate_clean_nez_pipeline(
                main_path,
                tmp_path / "eval",
                v3_ledger=v3_path,
                allowed_subjects_ledger=allowed,
                require_n_patients=1,
            )

            self.assertTrue(audit["v3_subject_set_equals_main"])
            self.assertEqual(audit["macro_f1_formula"], "0.5*(ez_f1+nez_f1)")
            self.assertEqual(audit["label_encoding_mode"], "ez0")
            self.assertEqual(audit["evaluation_target"], "clinical_true_ez")
            self.assertFalse(audit["raw_binary_label_used_as_metric_target"])
            self.assertIn("V3 baseline predicted_ez", audit["methods"])
            self.assertIn("V3 baseline oracle-K diagnostic", audit["methods"])
            patient_rows = pd.read_csv(tmp_path / "eval" / "patient_level_metrics.csv")
            self.assertTrue(
                patient_rows.loc[
                    patient_rows["method"].eq("V3 baseline predicted_ez"),
                    "diagnostic_oracle_k",
                ].eq(False).all()
            )
            self.assertTrue(
                patient_rows.loc[
                    patient_rows["method"].eq("V3 baseline oracle-K diagnostic"),
                    "diagnostic_oracle_k",
                ].eq(True).all()
            )
            self.assertTrue((tmp_path / "eval" / "evaluation_audit.json").exists())


if __name__ == "__main__":
    unittest.main()
