from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from neuroez_c.v3_qbc_decoder import (
    FORMAL_DECISION_RULE,
    apply_global_robust_z_threshold,
    fit_global_robust_z_threshold,
    robust_standardize_patient_logits,
    true_k_diagnostic_prediction,
)
from neuroez_c.v3_qbc_reporting import build_v3_qbc_reports


class V3QBCDecoderTests(unittest.TestCase):
    def test_robust_standardization_is_shift_scale_invariant(self):
        x = np.array([-2.0, 0.0, 1.0, 4.0])
        self.assertTrue(np.allclose(robust_standardize_patient_logits(x), robust_standardize_patient_logits(7 + 3 * x)))

    def test_formal_threshold_uses_no_true_count(self):
        result = apply_global_robust_z_threshold([-2, -1, 1, 2], 0.0)
        self.assertEqual(result["decision_rule"], FORMAL_DECISION_RULE)
        self.assertFalse(result["true_count_used_for_prediction"])
        self.assertTrue(result["formal_prediction"])

    def test_true_k_selects_smallest_nez_logits(self):
        result = true_k_diagnostic_prediction([2.0, -3.0, 0.0, -1.0], 2)
        self.assertEqual(result["predicted_ez"].tolist(), [False, True, False, True])
        self.assertTrue(result["true_count_used_for_prediction"])
        self.assertFalse(result["formal_prediction"])

    def test_threshold_fit_is_deterministic(self):
        rows = [
            {"final_nez_logit": np.array([-2.0, -1.0, 1.0, 2.0]), "label_nez": np.array([0, 0, 1, 1]), "center": "hup"},
            {"final_nez_logit": np.array([-3.0, 0.0, 2.0]), "label_nez": np.array([0, 1, 1]), "center": "lzu"},
        ]
        self.assertEqual(fit_global_robust_z_threshold(rows), fit_global_robust_z_threshold(rows))

    def test_threshold_fit_rejects_empty_validation(self):
        with self.assertRaises(ValueError):
            fit_global_robust_z_threshold([])

    def test_saved_fold_reporting_separates_formal_and_truek(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for subject, center, shift in (("p1", "hup", 0.0), ("p2", "lzu", 0.2)):
                for channel, label_nez, logit in (("A1", 0, -2.0), ("A2", 0, -1.0), ("A3", 1, 1.0), ("A4", 1, 2.0)):
                    rows.append({
                        "fold_idx": 1, "subject_id": subject, "center": center,
                        "channel_name": channel, "true_nez": label_nez,
                        "final_nez_logit": logit + shift, "q10_gate": 0.01,
                    })
            frame = pd.DataFrame(rows)
            frame.to_csv(root / "val_channel_predictions_neuroez_v2_fold_1.csv", index=False)
            frame.to_csv(root / "test_channel_predictions_neuroez_v2_fold_1.csv", index=False)
            result = build_v3_qbc_reports(root)
            self.assertEqual(result["n_patients"], 2)
            self.assertTrue((root / "formal_summary.csv").exists())
            self.assertTrue((root / "truek_summary.csv").exists())
            formal = pd.read_csv(root / "formal_summary.csv").iloc[0]
            diagnostic = pd.read_csv(root / "truek_summary.csv").iloc[0]
            self.assertFalse(bool(formal["true_count_used_for_prediction"]))
            self.assertTrue(bool(diagnostic["true_count_used_for_prediction"]))

    def test_a0_saved_reporting_recovers_logit_from_score_nez(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame = pd.DataFrame({
                "fold_idx": [1, 1, 1, 1],
                "subject_id": ["p1"] * 4,
                "center": ["hup"] * 4,
                "channel_name": ["A1", "A2", "A3", "A4"],
                "true_nez": [0, 0, 1, 1],
                "score_nez_probability": [0.1, 0.2, 0.8, 0.9],
            })
            frame.to_csv(root / "val_channel_predictions_neuroez_v2_fold_1.csv", index=False)
            frame.to_csv(root / "test_channel_predictions_neuroez_v2_fold_1.csv", index=False)
            result = build_v3_qbc_reports(root)
            self.assertEqual(result["n_patients"], 1)
            ledger = pd.read_csv(root / "oof_channel_ledger.csv")
            self.assertTrue(np.isfinite(ledger["final_nez_logit"]).all())
            self.assertEqual(ledger["predicted_nez"].tolist(), [0, 0, 1, 1])


if __name__ == "__main__":
    unittest.main()
