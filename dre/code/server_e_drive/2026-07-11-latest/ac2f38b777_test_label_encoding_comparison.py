from __future__ import annotations

import json
import tempfile
from pathlib import Path
import unittest

import pandas as pd

from scripts.compare_label_encoding_experiments import compare_label_encoding_experiments


class LabelEncodingComparisonTests(unittest.TestCase):
    def test_compare_label_encoding_experiments_writes_clinical_metric_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ez1 = tmp_path / "ez1" / "eval"
            ez0 = tmp_path / "ez0" / "eval"
            out = tmp_path / "compare"
            ez1.mkdir(parents=True)
            ez0.mkdir(parents=True)
            methods = [
                "V3 baseline predicted_ez",
                "CleanNEZ + RawDistance + SetTopo + KCal no-leak",
            ]
            for eval_dir, offset in [(ez1, 0.0), (ez0, -0.1)]:
                pd.DataFrame(
                    {
                        "method": methods,
                        "patient_macro_f1": [0.5 + offset, 0.6 + offset],
                        "patient_ez_f1": [0.4 + offset, 0.7 + offset],
                        "patient_nez_f1": [0.6 + offset, 0.5 + offset],
                    }
                ).to_csv(eval_dir / "metrics_summary.csv", index=False)
                pd.DataFrame(
                    {
                        "method": methods * 2,
                        "fold_idx": [1, 1, 2, 2],
                        "patient_macro_f1": [0.5 + offset, 0.6 + offset, 0.55 + offset, 0.65 + offset],
                    }
                ).to_csv(eval_dir / "metrics_by_fold.csv", index=False)
                pd.DataFrame(
                    {
                        "method": methods,
                        "center": ["hup", "hup"],
                        "patient_macro_f1": [0.5 + offset, 0.6 + offset],
                    }
                ).to_csv(eval_dir / "metrics_by_center.csv", index=False)
                pd.DataFrame(
                    {
                        "method": methods * 2,
                        "subject_id": ["s1", "s1", "s2", "s2"],
                        "fold_idx": [1, 1, 2, 2],
                        "center": ["hup"] * 4,
                        "n_channels": [2] * 4,
                        "patient_macro_f1": [0.5 + offset, 0.6 + offset, 0.55 + offset, 0.65 + offset],
                        "patient_ez_f1": [0.4 + offset, 0.7 + offset, 0.45 + offset, 0.75 + offset],
                        "patient_nez_f1": [0.6 + offset, 0.5 + offset, 0.65 + offset, 0.55 + offset],
                    }
                ).to_csv(eval_dir / "patient_level_metrics.csv", index=False)
                (eval_dir / "evaluation_audit.json").write_text(
                    json.dumps({"evaluation_target": "clinical_true_ez"}),
                    encoding="utf-8",
                )

            audit = compare_label_encoding_experiments(ez1, ez0, out)

            self.assertTrue((out / "label_encoding_comparison_summary.csv").exists())
            self.assertTrue((out / "label_encoding_comparison_patient_delta.csv").exists())
            self.assertTrue(audit["subject_sets_equal"])
            self.assertTrue(audit["row_counts_equal"])
            self.assertEqual(audit["clinical_metric_target"], "clinical_true_ez")


if __name__ == "__main__":
    unittest.main()
