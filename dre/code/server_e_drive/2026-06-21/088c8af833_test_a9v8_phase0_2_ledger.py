"""A9v8 Phase 0 ledger tests — toy metrics from exec document spec."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.export_patient_channel_ledger import export_channel_ledger_from_run
from scripts.summarize_patient_channel_ledger import (
    _check_mixed_splits,
    summarize_ledger_dataframe,
)


class A9v8LedgerTests(unittest.TestCase):
    def test_export_split_role_test_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "test_run"
            run_dir.mkdir()
            for split_name, fold, a_l, b_l, a_s, b_s in [
                ("test", 0, 1.0, 0.0, 0.9, 0.2),
                ("val", 0, 1.0, 0.0, 0.8, 0.3),
            ]:
                pd.DataFrame([
                    {"fold_idx": fold, "split": split_name, "subject_id": "p1",
                     "center": "hup", "center_id": 0, "channel_name": "a",
                     "true_ez": a_l, "score_ez_probability": a_s},
                    {"fold_idx": fold, "split": split_name, "subject_id": "p1",
                     "center": "hup", "center_id": 0, "channel_name": "b",
                     "true_ez": b_l, "score_ez_probability": b_s},
                ]).to_csv(run_dir / f"{split_name}_channel_predictions_neuroez_v2_fold_{fold}.csv", index=False)

            ledger = export_channel_ledger_from_run(run_dir, split_role="test")
            self.assertEqual(set(ledger["split_role"].unique()), {"test"})
            self.assertEqual(len(ledger), 2)

            ledger_val = export_channel_ledger_from_run(run_dir, split_role="val")
            self.assertEqual(set(ledger_val["split_role"].unique()), {"val"})

    def test_ledger_required_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "test_run"
            run_dir.mkdir()
            pd.DataFrame([
                {"fold_idx": 0, "split": "test", "subject_id": "p1",
                 "center": "hup", "center_id": 0, "channel_name": "a",
                 "true_ez": 1.0, "score_ez_probability": 0.9},
                {"fold_idx": 0, "split": "test", "subject_id": "p1",
                 "center": "hup", "center_id": 0, "channel_name": "b",
                 "true_ez": 0.0, "score_ez_probability": 0.2},
            ]).to_csv(run_dir / "test_channel_predictions_neuroez_v2_fold_0.csv", index=False)
            ledger = export_channel_ledger_from_run(run_dir, split_role="test")
        required = {"config", "fold_id", "split_role", "subject_id", "patient_id",
                    "record_id", "center", "center_id", "channel_id", "channel_name",
                    "label_ez", "patient_ez_count", "score_raw", "score_patient",
                    "score_eval", "rank_eval", "pred_topk", "is_top1"}
        self.assertTrue(required.issubset(ledger.columns))

    def test_mixed_splits_raises_by_default(self):
        ledger = pd.DataFrame({
            "subject_id": ["p1", "p2"], "patient_id": ["p1", "p2"],
            "label_ez": [1.0, 0.0], "score_eval": [0.9, 0.1],
            "split_role": ["test", "val"],
        })
        with self.assertRaises(ValueError):
            _check_mixed_splits(ledger, allow_mixed=False)
        _check_mixed_splits(ledger, allow_mixed=True)

    # ---- Toy ledger from exec document: precise metric checks ----

    def _toy_ledger(self):
        """Patient A: k=2, labels [1,1,0,0], scores [0.9,0.1,0.8,0.2]
           Patient B: k=1, labels [0,1,0], scores [0.3,0.2,0.1]"""
        return pd.DataFrame({
            "subject_id": ["A", "A", "A", "A", "B", "B", "B"],
            "patient_id": ["A", "A", "A", "A", "B", "B", "B"],
            "label_ez": [1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            "score_eval": [0.9, 0.1, 0.8, 0.2, 0.3, 0.2, 0.1],
            "center": ["hup"]*4 + ["lzu"]*3,
        })

    def test_toy_ledger_patient_macro_f1(self):
        """From exec doc: A: macro_f1=0.5, B: macro_f1=0.25, mean=0.375"""
        summary, _ = summarize_ledger_dataframe(self._toy_ledger())
        self.assertAlmostEqual(float(summary["patient_macro_f1"]), 0.375, places=4)

    def test_toy_ledger_ez_f1(self):
        summary, _ = summarize_ledger_dataframe(self._toy_ledger())
        # Patient A: pred_ez=[1,0,1,0], y_ez=[1,1,0,0], f1_ez = f1(y_ez,pred_ez)
        # Precision(EZ=1)=1/2=0.5, Recall=1/2=0.5, f1=0.5
        # Patient B: pred_ez=[1,0,0], y_ez=[0,1,0], f1_ez=0.0
        # mean=0.25
        self.assertAlmostEqual(float(summary["patient_macro_ez_f1"]), 0.25, places=4)

    def test_toy_ledger_auprc_ez(self):
        summary, _ = summarize_ledger_dataframe(self._toy_ledger())
        # Patient A: scores [0.9,0.1,0.8,0.2], y_ez=[1,1,0,0], AP=0.75
        # Patient B: scores [0.3,0.2,0.1], y_ez=[0,1,0]
        # Precision-recall: (1.0,0.0),(0.5,0.0),(0.0,0.0)→AP=0.5
        # mean=(0.75+0.5)/2=0.625
        self.assertAlmostEqual(float(summary["patient_macro_auprc_ez"]), 0.625, places=4)

    def test_toy_ledger_mrr(self):
        summary, _ = summarize_ledger_dataframe(self._toy_ledger())
        # Patient A: first EZ rank=1 → MRR=1.0
        # Patient B: first EZ rank=2 → MRR=0.5
        # mean=0.75
        self.assertAlmostEqual(float(summary["patient_macro_ez_mrr"]), 0.75, places=6)

    def test_toy_ledger_top1(self):
        summary, _ = summarize_ledger_dataframe(self._toy_ledger())
        # Patient A: top1=ch0(score0.9,label=1) → hit=1
        # Patient B: top1=ch0(score0.3,label=0) → hit=0
        # mean=0.5
        self.assertAlmostEqual(float(summary["top1_is_ez_rate"]), 0.5, places=6)


if __name__ == "__main__":
    unittest.main()
