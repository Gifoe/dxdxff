import tempfile
import unittest
import json
from pathlib import Path

from a12_vcsn.config import A12Config
from a12_vcsn.suite import make_synthetic_ledger, run_a12_suite


class A12SyntheticSuiteTests(unittest.TestCase):
    def test_runs_audit_to_comparison_with_frozen_folds(self):
        ledger = make_synthetic_ledger(n_subjects=10, n_folds=2)
        with tempfile.TemporaryDirectory() as tmp:
            result = run_a12_suite(
                ledger,
                output_dir=tmp,
                config=A12Config(variants=("A12-V0", "A12-D0", "A12-V1"), inner_folds=2, strict=False, max_swaps=1, smoke_mode=True, seeds=(42,), catboost_max_iterations=2),
                synthetic=True,
            )
            self.assertTrue((Path(tmp) / "audit" / "anchor_parity.json").exists())
            self.assertTrue((Path(tmp) / "comparison" / "a12_variant_comparison.json").exists())
            self.assertEqual(result["variants"]["A12-V0"]["status"], "success")
            self.assertEqual(result["variants"]["A12-D0"]["status"], "diagnostic")
            self.assertEqual(result["variants"]["A12-V1"]["status"], "success")
            registry = json.loads((Path(tmp) / "variants" / "A12_V1" / "feature_registry.json").read_text(encoding="utf-8"))
            self.assertEqual(registry["feature_groups"], ["patient_context", "v3"])

    def test_resume_ignores_the_resume_control_flag(self):
        ledger = make_synthetic_ledger(n_subjects=10, n_folds=2)
        with tempfile.TemporaryDirectory() as tmp:
            original = A12Config(variants=("A12-V0", "A12-V1"), inner_folds=2, strict=False, smoke_mode=True, seeds=(42,), catboost_max_iterations=2)
            run_a12_suite(ledger, output_dir=tmp, config=original, synthetic=True)
            resumed = run_a12_suite(ledger, output_dir=tmp, config=A12Config(variants=("A12-V0", "A12-V1"), inner_folds=2, strict=False, resume=True, smoke_mode=True, seeds=(42,), catboost_max_iterations=2), synthetic=True)
            self.assertTrue(resumed["variants"]["A12-V0"]["resumed"])
            self.assertTrue(resumed["variants"]["A12-V1"]["resumed"])
