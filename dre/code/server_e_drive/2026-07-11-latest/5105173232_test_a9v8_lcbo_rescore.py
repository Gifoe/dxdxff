from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.rescore_a9v8_lcbo_outputs import rescore_run_dir


class A9v8LCBORescoreTests(unittest.TestCase):
    def test_rescore_outputs_eval_broad_core_and_gamma_grid(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            pd.DataFrame(
                [
                    {"fold_idx": 1, "split": "test", "subject_id": "p1", "center": "lzu", "channel_name": "a", "true_ez": 1, "true_nez": 0, "score_eval": 0.9, "score_broad": 0.6, "score_core": 0.8, "a9v3_oof_score": 0.9},
                    {"fold_idx": 1, "split": "test", "subject_id": "p1", "center": "lzu", "channel_name": "b", "true_ez": 0, "true_nez": 1, "score_eval": 0.1, "score_broad": 0.4, "score_core": 0.2, "a9v3_oof_score": 0.1},
                    {"fold_idx": 2, "split": "test", "subject_id": "p2", "center": "hup", "channel_name": "a", "true_ez": 0, "true_nez": 1, "score_eval": 0.2, "score_broad": 0.5, "score_core": 0.1, "a9v3_oof_score": 0.2},
                    {"fold_idx": 2, "split": "test", "subject_id": "p2", "center": "hup", "channel_name": "b", "true_ez": 1, "true_nez": 0, "score_eval": 0.8, "score_broad": 0.7, "score_core": 0.9, "a9v3_oof_score": 0.8},
                ]
            ).to_csv(run_dir / "test_channel_predictions_neuroez_v2_fold_1.csv", index=False)

            summary, by_center, by_fold = rescore_run_dir(run_dir)

            self.assertIn("score_eval", set(summary["score_name"]))
            self.assertIn("score_broad", set(summary["score_name"]))
            self.assertIn("score_core", set(summary["score_name"]))
            self.assertTrue(summary["score_name"].str.contains("logit_broad_plus_gamma_core", regex=False).any())
            self.assertTrue((run_dir / "lcbo_rescore_summary.csv").exists())
            self.assertTrue((run_dir / "lcbo_rescore_by_center.csv").exists())
            self.assertTrue((run_dir / "lcbo_rescore_by_fold.csv").exists())
            self.assertGreater(len(by_center), 0)
            self.assertGreater(len(by_fold), 0)


if __name__ == "__main__":
    unittest.main()
