from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from a12_vcsn.config import A12Config
from a12_vcsn.suite import ALL_VARIANTS, make_synthetic_ledger, run_a12_suite


class A12AllVariantsSyntheticTests(unittest.TestCase):
    def test_v0_through_v10_execute_with_real_capabilities(self):
        ledger = make_synthetic_ledger(n_subjects=6, n_folds=2)
        config = A12Config(variants=("all",), seeds=(3,), inner_folds=2, max_swaps=1,
                           strict=False, smoke_mode=True, max_epochs=1, patience=1,
                           batch_size=16, hidden_dim=4, catboost_max_iterations=2,
                           max_eject_candidates=2, max_add_candidates=3,
                           max_pairs_per_patient=6, anchor_min_channels=1)
        with tempfile.TemporaryDirectory() as tmp:
            result = run_a12_suite(ledger, output_dir=tmp, config=config, synthetic=True)
            statuses = {name: value["status"] for name, value in result["variants"].items()}
            self.assertEqual(set(statuses), set(ALL_VARIANTS))
            self.assertEqual(statuses["A12-D0"], "diagnostic")
            self.assertTrue(all(status == "success" for name, status in statuses.items() if name != "A12-D0"))
            self.assertTrue(Path(tmp, "comparison", "a12_action_summary.csv").exists())
            self.assertTrue(Path(tmp, "comparison", "a12_patient_delta_matrix.csv").exists())
            report = Path(tmp, "comparison", "final_report.md").read_text(encoding="utf-8")
            self.assertIn("Run classification: **synthetic**", report)


if __name__ == "__main__":
    unittest.main()
