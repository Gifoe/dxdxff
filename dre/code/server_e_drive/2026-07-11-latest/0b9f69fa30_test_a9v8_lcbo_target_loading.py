from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from exp_ez_hybrid import Exp_EZHybridLocalization, _summarize_prediction_records, build_lcbo_target_lookup, load_lcbo_fold_targets


class A9v8LCBOTargetLoadingTests(unittest.TestCase):
    def test_fold4_loader_reads_fold4_train_and_rejects_leakage(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pd.DataFrame(
                [
                    {"patient_id": "p1", "subject_id": "p1", "fold_id": 1, "channel_name": "A-1", "label_ez": 1, "pseudo_core_q": 0.8, "a9v3_oof_score": 0.9},
                    {"patient_id": "p2", "subject_id": "p2", "fold_id": 4, "channel_name": "B-1", "label_ez": 0, "pseudo_core_q": 0.0, "a9v3_oof_score": 0.2},
                ]
            ).to_csv(root / "latent_core_targets_fold4_train.csv", index=False)

            with self.assertRaisesRegex(ValueError, "heldout fold 4"):
                load_lcbo_fold_targets(root, fold_id=4)

            pd.DataFrame(
                [
                    {"patient_id": "p1", "subject_id": "p1", "fold_id": 1, "channel_name": "A-1", "label_ez": 1, "pseudo_core_q": 0.8, "a9v3_oof_score": 0.9},
                    {"patient_id": "p2", "subject_id": "p2", "fold_id": 2, "channel_name": "B-1", "label_ez": 0, "pseudo_core_q": 0.0, "a9v3_oof_score": 0.2},
                ]
            ).to_csv(root / "latent_core_targets_fold4_train.csv", index=False)
            loaded = load_lcbo_fold_targets(root, fold_id=4)
            self.assertEqual(len(loaded), 2)

    def test_lookup_normalizes_patient_or_subject_and_channel(self):
        df = pd.DataFrame(
            [
                {"patient_id": "P1", "subject_id": "S1", "fold_id": 1, "channel_name": "A-1", "label_ez": 1, "pseudo_core_q": 0.8, "a9v3_oof_score": 0.9},
            ]
        )
        lookup = build_lcbo_target_lookup(df)

        self.assertIn(("p1", "a_1"), lookup)
        self.assertIn(("s1", "a_1"), lookup)

    def test_summary_and_saved_outputs_include_lcbo_diagnostics(self):
        record = {
            "subject_id": "p1",
            "center": "lzu",
            "center_id": 1,
            "canonical_channels": ["a", "b"],
            "labels": np.asarray([1.0, 0.0], dtype=np.float32),
            "labels_ez": np.asarray([1.0, 0.0], dtype=np.float32),
            "labels_nez": np.asarray([0.0, 1.0], dtype=np.float32),
            "scores": np.asarray([0.1, 0.9], dtype=np.float32),
            "score_nez": np.asarray([0.1, 0.9], dtype=np.float32),
            "score_ez": np.asarray([0.9, 0.1], dtype=np.float32),
            "score_core": np.asarray([0.8, 0.2], dtype=np.float32),
            "score_broad": np.asarray([0.7, 0.3], dtype=np.float32),
            "score_eval": np.asarray([0.9, 0.1], dtype=np.float32),
            "pseudo_core_q": np.asarray([1.0, 0.0], dtype=np.float32),
            "a9v3_oof_score": np.asarray([0.95, 0.05], dtype=np.float32),
            "channel_mask": np.asarray([True, True]),
            "run_ids": [],
            "sample_ids": [],
            "valid_channel_count": 2,
            "ez_channel_count": 1,
        }
        summary, enriched = _summarize_prediction_records([record])
        self.assertEqual(summary["top1_is_ez_rate"], 1.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            exp = Exp_EZHybridLocalization.__new__(Exp_EZHybridLocalization)
            exp.args = SimpleNamespace(output_dir=tmpdir)
            exp._save_outputs(enriched, fold_idx=4, split_name="test")
            channel_csv = Path(tmpdir) / "test_channel_predictions_neuroez_v2_fold_4.csv"
            saved = pd.read_csv(channel_csv)
            for col in ("score_core", "score_broad", "score_eval", "pseudo_core_q", "a9v3_oof_score"):
                self.assertIn(col, saved.columns)

    def test_classification_metrics_do_not_use_true_ez_count(self):
        record = {
            "subject_id": "p1",
            "canonical_channels": ["a", "b", "c"],
            "labels": np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
            "labels_nez": np.asarray([0.0, 1.0, 1.0], dtype=np.float32),
            "labels_ez": np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
            "score_nez": np.asarray([0.49, 0.40, 0.60], dtype=np.float32),
            "score_ez": np.asarray([0.51, 0.60, 0.40], dtype=np.float32),
            "scores": np.asarray([0.49, 0.40, 0.60], dtype=np.float32),
            "channel_mask": np.asarray([True, True, True]),
        }
        summary, enriched = _summarize_prediction_records([record], classification_threshold=0.5)

        self.assertEqual(enriched[0]["predicted_nez_channels"], ["c"])
        self.assertEqual(enriched[0]["predicted_ez_channels"], ["a", "b"])
        self.assertAlmostEqual(summary["classification_threshold"], 0.5)
        self.assertNotEqual(summary["patient_macro_f1"], summary["patient_macro_balanced_accuracy"])


if __name__ == "__main__":
    unittest.main()
