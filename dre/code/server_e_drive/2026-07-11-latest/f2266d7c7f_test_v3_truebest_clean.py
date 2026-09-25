import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.audit_v3_truebest_clean import audit_run_dir


REPO_ROOT = Path(__file__).resolve().parents[1]
TRUEBEST_RUN = (
    REPO_ROOT
    / "b0_m1_A9v3_search_20260624"
    / "A9v3_s5_anchor_w002_rank005_m005_all90_posEZ"
)


class V3TrueBestCleanAuditTests(unittest.TestCase):
    def test_existing_truebest_metrics_match_with_legacy_disabled_flags_allowed(self):
        self.assertTrue(TRUEBEST_RUN.exists(), TRUEBEST_RUN)

        result = audit_run_dir(
            TRUEBEST_RUN,
            tolerance=1e-6,
            allow_missing_disabled_flags=True,
            write_outputs=True,
        )

        self.assertEqual(result["status"], "pass", result["failures"])
        metrics = result["metrics"]
        self.assertAlmostEqual(metrics["patient_macro_f1"], 0.6427135781, places=9)
        self.assertAlmostEqual(metrics["patient_macro_ez_f1"], 0.4705912943, places=9)
        self.assertAlmostEqual(metrics["patient_macro_auprc_ez"], 0.5184277652, places=9)
        self.assertAlmostEqual(metrics["patient_macro_ez_mrr"], 0.7099510182, places=9)
        self.assertEqual(metrics["n_patient_rows"], 90)
        self.assertEqual(metrics["n_unique_subjects"], 90)
        self.assertTrue((TRUEBEST_RUN / "heldout_center_summary_neuroez_v3.csv").exists())

    def test_clean_audit_requires_explicit_legacy_disabled_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            pd.DataFrame(
                [
                    {
                        "patient_macro_f1": 0.6427135780776422,
                        "patient_macro_ez_f1": 0.47059129425797075,
                        "patient_macro_auprc_ez": 0.5184277651748298,
                        "patient_macro_ez_mrr": 0.7099510181940522,
                        "n_patient_rows": 90,
                        "n_unique_subjects": 90,
                    }
                ]
            ).to_csv(run_dir / "heldout_summary_neuroez_v3.csv", index=False)
            (run_dir / "heldout_summary_neuroez_v3.json").write_text("{}", encoding="utf-8")
            pd.DataFrame([{"fold_idx": 1, "patient_macro_f1": 0.1}]).to_csv(
                run_dir / "heldout_fold_summary_neuroez_v3.csv", index=False
            )
            for fold_idx in range(1, 6):
                pd.DataFrame(
                    [{"subject_id": f"s{fold_idx}", "center": "hup", "patient_macro_f1": 1.0}]
                ).to_csv(run_dir / f"test_patient_predictions_neuroez_v2_fold_{fold_idx}.csv", index=False)
                pd.DataFrame([{"subject_id": f"s{fold_idx}", "channel": "A1"}]).to_csv(
                    run_dir / f"test_channel_predictions_neuroez_v2_fold_{fold_idx}.csv", index=False
                )
            args = {
                "use_negative_anchor_head": True,
                "negative_anchor_loss_weight": 0.02,
                "use_ez_ranking_loss": True,
                "ez_ranking_loss_weight": 0.05,
                "ez_ranking_margin": 0.05,
                "use_physics_dynamics": True,
                "loss_mode": "patient_balanced_bce",
                "patient_loss_weighting": "uniform",
                "positive_label": "ez",
                "drop_high_ez_fraction_lzu": False,
                "split_strategy": "5fold",
                "n_splits": 5,
                "random_seed": 42,
                "epochs": 40,
                "patience": 8,
            }
            (run_dir / "run_args_b0_pruned.json").write_text(json.dumps(args), encoding="utf-8")

            result = audit_run_dir(run_dir, allow_missing_disabled_flags=False, write_outputs=False)

            self.assertEqual(result["status"], "fail")
            self.assertTrue(
                any("missing_required_arg:use_patient_context_reranker" in item for item in result["failures"]),
                result["failures"],
            )

    def test_runner_script_is_single_clean_v3_path(self):
        script_path = REPO_ROOT / "scripts" / "run_v3_truebest_clean.ps1"
        text = script_path.read_text(encoding="utf-8")

        self.assertIn("IsNullOrWhiteSpace", text)
        self.assertIn("use_patient_context_reranker", text)
        self.assertIn("audit_v3_truebest_clean.py", text)
        self.assertNotIn("run_a9v14_candidate_grid", text)
        self.assertNotIn("summarize_a9v14_candidate_grid", text)


if __name__ == "__main__":
    unittest.main()
