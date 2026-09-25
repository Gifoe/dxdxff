from __future__ import annotations

import json
import inspect
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import scripts.a9v3_robust_pooling_ablation as pooling_script
from scripts.a9v3_robust_pooling_ablation import (
    detect_artifact_granularity,
    passes_main_gate,
    run_pooling_ablation,
)


class A9v3RobustPoolingAblationTests(unittest.TestCase):
    def test_patient_level_files_write_feasibility_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pd.DataFrame(
                [{"fold_idx": 1, "subject_id": "p1", "center": "hup", "channel_name": "a", "true_ez": 1, "score_eval": 0.9}]
            ).to_csv(root / "test_channel_predictions_neuroez_v2_fold_1.csv", index=False)
            output_dir = root / "pooling"

            result = run_pooling_ablation(root, window_cache_path=root / "cache.pkl", output_dir=output_dir)

            self.assertEqual(result["granularity"], "patient_level_only")
            report = json.loads((output_dir / "pooling_ablation_feasibility_report.json").read_text())
            self.assertEqual(report["granularity"], "patient_level_only")
            self.assertTrue((output_dir / "pooling_ablation_available_artifacts.csv").exists())
            self.assertIn("cannot be performed", (output_dir / "pooling_ablation_recommendation.txt").read_text())

    def test_granularity_detection_record_level(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "record_channel_predictions.csv").write_text("subject_id,record_id\np1,r1\n")

            artifacts = detect_artifact_granularity(root)

            self.assertEqual(artifacts["granularity"], "record_level_available")

    def test_synthetic_record_level_rows_produce_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            rows = []
            for record_id, scores in [("r1", [0.9, 0.1]), ("r2", [0.8, 0.2])]:
                rows.extend(
                    [
                        {"fold_idx": 1, "subject_id": "p1", "center": "hup", "record_id": record_id, "channel_name": "a", "true_ez": 1, "score": scores[0]},
                        {"fold_idx": 1, "subject_id": "p1", "center": "hup", "record_id": record_id, "channel_name": "b", "true_ez": 0, "score": scores[1]},
                    ]
                )
            pd.DataFrame(rows).to_csv(root / "record_channel_predictions.csv", index=False)
            output_dir = root / "pooling"

            result = run_pooling_ablation(root, window_cache_path=root / "cache.pkl", output_dir=output_dir)

            self.assertEqual(result["granularity"], "record_level_available")
            self.assertTrue((output_dir / "pooling_ablation_summary.csv").exists())
            summary = pd.read_csv(output_dir / "pooling_ablation_summary.csv")
            self.assertIn("mean", set(summary["record_pooling"]))
            self.assertFalse(summary["patient_macro_f1"].isna().any())

    def test_pass_gate_logic(self):
        self.assertTrue(passes_main_gate({
            "patient_macro_f1": 0.7,
            "patient_macro_ez_f1": 0.5,
            "patient_macro_auprc_ez": 0.6,
            "patient_macro_ez_mrr": 0.8,
            "top1_is_ez_rate": 0.6,
        }))

    def test_script_does_not_call_training_entrypoints(self):
        source = inspect.getsource(pooling_script)

        self.assertNotIn("run_neuroez_c", source)
        self.assertNotIn("subprocess", source)
        self.assertNotIn(".fit(", source)
        self.assertFalse(passes_main_gate({
            "patient_macro_f1": 0.7,
            "patient_macro_ez_f1": 0.5,
            "patient_macro_auprc_ez": 0.6,
            "patient_macro_ez_mrr": 0.7,
            "top1_is_ez_rate": 0.6,
        }))


if __name__ == "__main__":
    unittest.main()
