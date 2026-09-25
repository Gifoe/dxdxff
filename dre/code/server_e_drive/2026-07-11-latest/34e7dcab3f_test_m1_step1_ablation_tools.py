from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from scripts.report_step3_rank_decision import summarize_step3_rank_decision
from scripts.m1_step1_ablation_common import (
    build_candidate_multiseed_experiments,
    build_stage1a_experiments,
    build_stage1b_experiments,
    build_stage1c_experiments,
    build_step1_continue_feature_experiments,
    choose_best_result,
    collect_seed_summary_frame,
    collect_results_rows,
    collect_multiseed_run_rows,
    parse_multiseed_candidate_name,
    summarize_multiseed_decision,
    summarize_step2_diffusion_decision,
    summarize_multiseed_runs,
    summarize_decision,
)


class M1Step1AblationToolsTests(unittest.TestCase):
    def test_stage1b_uses_best_gate_from_stage1a_when_available(self):
        stage1a_results = [
            {
                "experiment_name": "m1_gate_m4",
                "physics_gate_init": -4.0,
                "pooled_macro_f1": 0.670,
                "patient_macro_f1": 0.640,
                "pooled_auprc_ez": 0.350,
                "pooled_auroc_ez": 0.690,
            },
            {
                "experiment_name": "m1_gate_m2",
                "physics_gate_init": -2.0,
                "pooled_macro_f1": 0.679,
                "patient_macro_f1": 0.650,
                "pooled_auprc_ez": 0.362,
                "pooled_auroc_ez": 0.701,
            },
        ]

        experiments = build_stage1b_experiments(stage1a_results)

        self.assertEqual(len(experiments), 4)
        self.assertTrue(all(experiment["physics_gate_init"] == -2.0 for experiment in experiments))

    def test_stage1c_uses_best_feature_setting_from_stage1b_when_available(self):
        stage1b_results = [
            {
                "experiment_name": "m1_feat_HL",
                "physics_state_features": "log_bp_high_gamma,line_length_per_sec",
                "pooled_macro_f1": 0.677,
                "patient_macro_f1": 0.649,
                "pooled_auprc_ez": 0.361,
                "pooled_auroc_ez": 0.700,
            },
            {
                "experiment_name": "m1_feat_HLV",
                "physics_state_features": "log_bp_high_gamma,line_length_per_sec,variance",
                "pooled_macro_f1": 0.681,
                "patient_macro_f1": 0.654,
                "pooled_auprc_ez": 0.365,
                "pooled_auroc_ez": 0.706,
            },
        ]

        experiments = build_stage1c_experiments(stage1b_results, gate_init=-3.0)

        self.assertEqual(len(experiments), 3)
        self.assertTrue(
            all(
                experiment["physics_state_features"] == "log_bp_high_gamma,line_length_per_sec,variance"
                for experiment in experiments
            )
        )

    def test_collect_results_prefers_richest_summary_and_sorts_by_required_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir)
            run_dir = output_root / "m1_gate_m3"
            run_dir.mkdir(parents=True)
            with open(run_dir / "run_args_b0_pruned.json", "w", encoding="utf-8") as fout:
                json.dump(
                    {
                        "output_dir": str(run_dir),
                        "physics_gate_init": -3.0,
                        "physics_state_features": "log_bp_high_gamma,line_length_per_sec,rms,variance",
                        "physics_feature_parts": "zdelta,delta",
                        "physics_loss_weight": 0.0,
                        "physics_source_sparse_weight": 0.0,
                        "physics_velocity_l2_weight": 0.0,
                        "epochs": 30,
                        "patience": 6,
                        "random_seed": 42,
                    },
                    fout,
                    indent=2,
                )
            with open(run_dir / "weak_summary.json", "w", encoding="utf-8") as fout:
                json.dump({"pooled_macro_f1": 0.6701}, fout)
            with open(run_dir / "heldout_summary_neuroez_v3.json", "w", encoding="utf-8") as fout:
                json.dump(
                    {
                        "patient_macro_accuracy": 0.61,
                        "patient_macro_balanced_accuracy": 0.62,
                        "patient_macro_f1": 0.6532,
                        "patient_macro_ez_f1": 0.4201,
                        "patient_macro_ez_recall_at_true_count": 0.4555,
                        "patient_macro_ez_mrr": 0.5123,
                        "patient_macro_auroc_ez": 0.6888,
                        "patient_macro_auprc_ez": 0.3555,
                        "pooled_accuracy": 0.66,
                        "pooled_balanced_accuracy": 0.67,
                        "pooled_macro_f1": 0.6761,
                        "pooled_auroc_nez": 0.6937,
                        "pooled_auroc_ez": 0.6937,
                        "pooled_auprc_nez": 0.8123,
                        "pooled_auprc_ez": 0.3603,
                    },
                    fout,
                    indent=2,
                )

            other_run_dir = output_root / "m1_gate_m2"
            other_run_dir.mkdir(parents=True)
            pd.DataFrame(
                [
                    {
                        "patient_macro_accuracy": 0.60,
                        "patient_macro_balanced_accuracy": 0.61,
                        "patient_macro_f1": 0.6500,
                        "patient_macro_ez_f1": 0.4100,
                        "patient_macro_ez_recall_at_true_count": 0.4500,
                        "patient_macro_ez_mrr": 0.5000,
                        "patient_macro_auroc_ez": 0.6800,
                        "patient_macro_auprc_ez": 0.3500,
                        "pooled_accuracy": 0.65,
                        "pooled_balanced_accuracy": 0.66,
                        "pooled_macro_f1": 0.6740,
                        "pooled_auroc_nez": 0.6900,
                        "pooled_auroc_ez": 0.6900,
                        "pooled_auprc_nez": 0.8100,
                        "pooled_auprc_ez": 0.3580,
                    }
                ]
            ).to_csv(other_run_dir / "heldout_summary_neuroez_v3.csv", index=False)
            with open(other_run_dir / "run_args_b0_pruned.json", "w", encoding="utf-8") as fout:
                json.dump(
                    {
                        "output_dir": str(other_run_dir),
                        "physics_gate_init": -2.0,
                        "physics_state_features": "log_bp_high_gamma,line_length_per_sec,rms,variance",
                        "physics_feature_parts": "zdelta,delta",
                        "physics_loss_weight": 0.0,
                        "physics_source_sparse_weight": 0.0,
                        "physics_velocity_l2_weight": 0.0,
                        "epochs": 30,
                        "patience": 6,
                        "random_seed": 42,
                    },
                    fout,
                    indent=2,
                )

            rows = collect_results_rows(output_root)

            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["experiment_name"], "m1_gate_m3")
            self.assertAlmostEqual(rows[0]["pooled_macro_f1"], 0.6761, places=6)
            self.assertAlmostEqual(rows[0]["physics_gate_init"], -3.0, places=6)

    def test_summarize_decision_uses_expected_threshold_language(self):
        summary = summarize_decision(
            best_row={
                "experiment_name": "m1_gate_m2",
                "pooled_macro_f1": 0.6822,
                "patient_macro_f1": 0.6590,
                "pooled_auprc_ez": 0.3620,
                "pooled_auroc_ez": 0.7010,
            },
            b0_pooled_macro_f1=0.6665047098784886,
            current_m1_pooled_macro_f1=0.6761773349,
            current_m1_pooled_auprc_ez=0.3603683291,
        )

        self.assertEqual(summary["best_experiment"], "m1_gate_m2")
        self.assertIn("freeze this as M1-best and proceed to Step 2", summary["conclusion"])

    def test_choose_best_result_uses_primary_secondary_and_tertiary_order(self):
        best = choose_best_result(
            [
                {
                    "experiment_name": "a",
                    "pooled_macro_f1": 0.680,
                    "patient_macro_f1": 0.650,
                    "pooled_auprc_ez": 0.360,
                    "pooled_auroc_ez": 0.700,
                },
                {
                    "experiment_name": "b",
                    "pooled_macro_f1": 0.680,
                    "patient_macro_f1": 0.651,
                    "pooled_auprc_ez": 0.359,
                    "pooled_auroc_ez": 0.705,
                },
            ]
        )

        self.assertEqual(best["experiment_name"], "b")

    def test_build_step1_continue_feature_experiments_uses_fixed_gate_m4(self):
        experiments = build_step1_continue_feature_experiments()

        self.assertEqual([experiment["experiment_name"] for experiment in experiments], ["m1_feat_HL", "m1_feat_HLR", "m1_feat_HLV", "m1_feat_all"])
        self.assertTrue(all(experiment["physics_gate_init"] == -4.0 for experiment in experiments))
        self.assertTrue(all(experiment["physics_loss_weight"] == 0.0 for experiment in experiments))

    def test_build_candidate_multiseed_experiments_creates_a_and_b_per_seed(self):
        experiments = build_candidate_multiseed_experiments([43, 44])

        self.assertEqual(len(experiments), 4)
        self.assertEqual(
            [experiment["experiment_name"] for experiment in experiments],
            [
                "m1_candidate_A_gate_m4_loss0_seed43",
                "m1_candidate_B_gate_m4_loss3e4_seed43",
                "m1_candidate_A_gate_m4_loss0_seed44",
                "m1_candidate_B_gate_m4_loss3e4_seed44",
            ],
        )

    def test_parse_multiseed_candidate_name_extracts_candidate_and_seed(self):
        candidate_name, seed = parse_multiseed_candidate_name("m1_candidate_B_gate_m4_loss3e4_seed44")

        self.assertEqual(candidate_name, "B")
        self.assertEqual(seed, 44)

    def test_collect_multiseed_run_rows_reuses_seed42_from_ablation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            multiseed_root = root / "multiseed"
            ablation_root = root / "ablation"
            run_dir = multiseed_root / "m1_candidate_A_gate_m4_loss0_seed43"
            run_dir.mkdir(parents=True)
            with open(run_dir / "run_args_b0_pruned.json", "w", encoding="utf-8") as fout:
                json.dump(
                    {
                        "output_dir": str(run_dir),
                        "experiment_name": "m1_candidate_A_gate_m4_loss0_seed43",
                        "physics_gate_init": -4.0,
                        "physics_state_features": "log_bp_high_gamma,line_length_per_sec,rms,variance",
                        "physics_feature_parts": "zdelta,delta",
                        "physics_loss_weight": 0.0,
                        "physics_source_sparse_weight": 0.0,
                        "physics_velocity_l2_weight": 0.0,
                        "epochs": 30,
                        "patience": 6,
                        "random_seed": 43,
                    },
                    fout,
                    indent=2,
                )
            pd.DataFrame(
                [
                    {
                        "patient_macro_accuracy": 0.60,
                        "patient_macro_balanced_accuracy": 0.61,
                        "patient_macro_f1": 0.6510,
                        "patient_macro_ez_f1": 0.4120,
                        "patient_macro_ez_recall_at_true_count": 0.4510,
                        "patient_macro_ez_mrr": 0.5010,
                        "patient_macro_auroc_ez": 0.6810,
                        "patient_macro_auprc_ez": 0.3510,
                        "pooled_accuracy": 0.66,
                        "pooled_balanced_accuracy": 0.67,
                        "pooled_macro_f1": 0.6780,
                        "pooled_auroc_nez": 0.6920,
                        "pooled_auroc_ez": 0.6920,
                        "pooled_auprc_nez": 0.8100,
                        "pooled_auprc_ez": 0.3610,
                    }
                ]
            ).to_csv(run_dir / "heldout_summary_neuroez_v3.csv", index=False)

            ablation_run = ablation_root / "m1_loss_3e4"
            ablation_run.mkdir(parents=True)
            with open(ablation_run / "run_args_b0_pruned.json", "w", encoding="utf-8") as fout:
                json.dump(
                    {
                        "output_dir": str(ablation_run),
                        "experiment_name": "m1_loss_3e4",
                        "physics_gate_init": -4.0,
                        "physics_state_features": "log_bp_high_gamma,line_length_per_sec,rms,variance",
                        "physics_feature_parts": "zdelta,delta",
                        "physics_loss_weight": 0.0003,
                        "physics_source_sparse_weight": 0.0,
                        "physics_velocity_l2_weight": 0.0,
                        "epochs": 30,
                        "patience": 6,
                        "random_seed": 42,
                    },
                    fout,
                    indent=2,
                )
            with open(ablation_run / "heldout_summary_neuroez_v3.json", "w", encoding="utf-8") as fout:
                json.dump(
                    {
                        "patient_macro_accuracy": 0.61,
                        "patient_macro_balanced_accuracy": 0.62,
                        "patient_macro_f1": 0.6559,
                        "patient_macro_ez_f1": 0.4200,
                        "patient_macro_ez_recall_at_true_count": 0.4550,
                        "patient_macro_ez_mrr": 0.5100,
                        "patient_macro_auroc_ez": 0.6901,
                        "patient_macro_auprc_ez": 0.3600,
                        "pooled_accuracy": 0.67,
                        "pooled_balanced_accuracy": 0.68,
                        "pooled_macro_f1": 0.6795,
                        "pooled_auroc_nez": 0.6901,
                        "pooled_auroc_ez": 0.6901,
                        "pooled_auprc_nez": 0.8100,
                        "pooled_auprc_ez": 0.3647,
                    },
                    fout,
                    indent=2,
                )

            rows = collect_multiseed_run_rows(multiseed_root, ablation_root=ablation_root)

            self.assertEqual(len(rows), 2)
            self.assertEqual({row["candidate_name"] for row in rows}, {"A", "B"})
            self.assertIn(42, {row["seed"] for row in rows})

    def test_summarize_multiseed_runs_and_decision_prefers_simpler_candidate(self):
        rows = [
            {
                "candidate_name": "A",
                "seed": 42,
                "pooled_macro_f1": 0.6800,
                "patient_macro_f1": 0.6550,
                "pooled_auprc_ez": 0.3610,
                "pooled_auroc_ez": 0.6920,
                "patient_macro_ez_f1": 0.4200,
                "patient_macro_ez_mrr": 0.5100,
            },
            {
                "candidate_name": "A",
                "seed": 43,
                "pooled_macro_f1": 0.6810,
                "patient_macro_f1": 0.6560,
                "pooled_auprc_ez": 0.3620,
                "pooled_auroc_ez": 0.6930,
                "patient_macro_ez_f1": 0.4210,
                "patient_macro_ez_mrr": 0.5110,
            },
            {
                "candidate_name": "B",
                "seed": 42,
                "pooled_macro_f1": 0.6815,
                "patient_macro_f1": 0.6565,
                "pooled_auprc_ez": 0.3625,
                "pooled_auroc_ez": 0.6940,
                "patient_macro_ez_f1": 0.4220,
                "patient_macro_ez_mrr": 0.5120,
            },
            {
                "candidate_name": "B",
                "seed": 43,
                "pooled_macro_f1": 0.6818,
                "patient_macro_f1": 0.6568,
                "pooled_auprc_ez": 0.3628,
                "pooled_auroc_ez": 0.6945,
                "patient_macro_ez_f1": 0.4225,
                "patient_macro_ez_mrr": 0.5125,
            },
        ]

        summary_frame = summarize_multiseed_runs(rows)
        decision = summarize_multiseed_decision(summary_frame.to_dict(orient="records"))

        self.assertEqual(set(summary_frame["candidate_name"]), {"A", "B"})
        self.assertEqual(decision["selected_candidate"], "A")
        self.assertEqual(decision["step"], "Stay in Step 1")

    def test_collect_seed_summary_frame_computes_mean_std_min_max(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for seed, pooled_macro_f1, pooled_auprc_ez in ((42, 0.680, 0.361), (43, 0.684, 0.363)):
                run_dir = root / f"m1_clean_gate_m4_loss0_seed{seed}"
                run_dir.mkdir(parents=True)
                with open(run_dir / "run_args_b0_pruned.json", "w", encoding="utf-8") as fout:
                    json.dump({"experiment_name": run_dir.name, "random_seed": seed}, fout)
                with open(run_dir / "heldout_summary_neuroez_v3.json", "w", encoding="utf-8") as fout:
                    json.dump(
                        {
                            "patient_macro_accuracy": 0.61,
                            "patient_macro_balanced_accuracy": 0.62,
                            "patient_macro_f1": 0.653 + 0.001 * (seed - 42),
                            "patient_macro_ez_f1": 0.420 + 0.001 * (seed - 42),
                            "patient_macro_ez_recall_at_true_count": 0.45,
                            "patient_macro_ez_mrr": 0.51,
                            "patient_macro_auroc_ez": 0.69,
                            "patient_macro_auprc_ez": 0.36,
                            "pooled_accuracy": 0.66,
                            "pooled_balanced_accuracy": 0.67,
                            "pooled_macro_f1": pooled_macro_f1,
                            "pooled_auroc_nez": 0.69,
                            "pooled_auroc_ez": 0.69,
                            "pooled_auprc_nez": 0.81,
                            "pooled_auprc_ez": pooled_auprc_ez,
                        },
                        fout,
                    )
            summary = collect_seed_summary_frame(root)

            self.assertEqual(len(summary), 1)
            self.assertAlmostEqual(float(summary.iloc[0]["pooled_macro_f1_mean"]), 0.682, places=6)
            self.assertAlmostEqual(float(summary.iloc[0]["pooled_macro_f1_min"]), 0.680, places=6)
            self.assertAlmostEqual(float(summary.iloc[0]["pooled_macro_f1_max"]), 0.684, places=6)

    def test_step2_decision_does_not_keep_absolute_score_without_mean_gain(self):
        m1_rows = [
            {"seed": 42, "pooled_macro_f1": 0.6800, "pooled_auprc_ez": 0.3620, "pooled_auroc_ez": 0.7010},
            {"seed": 43, "pooled_macro_f1": 0.6800, "pooled_auprc_ez": 0.3620, "pooled_auroc_ez": 0.7010},
        ]
        step2_rows = [
            {"seed": 42, "pooled_macro_f1": 0.6820, "pooled_auprc_ez": 0.3620, "pooled_auroc_ez": 0.7010},
            {"seed": 43, "pooled_macro_f1": 0.6820, "pooled_auprc_ez": 0.3620, "pooled_auroc_ez": 0.7010},
        ]

        decision = summarize_step2_diffusion_decision(m1_rows, step2_rows)

        self.assertEqual(decision["decision"], "DO_NOT_KEEP_STEP2")
        self.assertIn("below 0.005", decision["reason"])

    def test_step3_rank_decision_keeps_seed42_gain_without_metric_regression(self):
        m1_rows = [
            {
                "seed": 42,
                "pooled_macro_f1": 0.6760,
                "pooled_auprc_ez": 0.3600,
                "pooled_auroc_ez": 0.7000,
                "patient_macro_f1": 0.6500,
            }
        ]
        step3_rows = [
            {
                "seed": 42,
                "pooled_macro_f1": 0.6815,
                "pooled_auprc_ez": 0.3560,
                "pooled_auroc_ez": 0.6960,
                "patient_macro_f1": 0.6460,
            }
        ]

        decision = summarize_step3_rank_decision(m1_rows, step3_rows)

        self.assertEqual(decision["decision"], "KEEP_STEP3_CANDIDATE")
        self.assertAlmostEqual(decision["deltas"]["pooled_macro_f1"], 0.0055, places=6)

    def test_step3_rank_decision_rejects_when_seed42_missing(self):
        decision = summarize_step3_rank_decision(
            m1_rows=[
                {
                    "seed": 43,
                    "pooled_macro_f1": 0.6760,
                    "pooled_auprc_ez": 0.3600,
                    "pooled_auroc_ez": 0.7000,
                    "patient_macro_f1": 0.6500,
                }
            ],
            step3_rows=[],
        )

        self.assertEqual(decision["decision"], "DO_NOT_KEEP_STEP3")
        self.assertIn("M1 seed42 result not found", decision["reason"])


if __name__ == "__main__":
    unittest.main()
