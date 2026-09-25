from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch

from exp_ez_hybrid import Exp_EZHybridLocalization, _add_center_robustness_fields, _summary_score
from run_neuroez_c import assert_fixed_all90_protocol
from scripts.summarize_a10_rank_simple_grid import summarize


class A10RankSimpleTests(unittest.TestCase):
    def test_rank_robust_composite_formula(self):
        summary = {
            "patient_macro_f1": 0.60,
            "patient_macro_ez_f1": 0.40,
            "patient_macro_auprc_ez": 0.50,
            "patient_macro_ez_mrr": 0.70,
            "center_gap_f1": 0.20,
        }
        args = SimpleNamespace(early_stop_metric="rank_robust_composite")

        score = _summary_score(summary, args)

        self.assertAlmostEqual(score, 0.60 + 0.5 * 0.40 + 0.25 * 0.50 + 0.10 * 0.70 - 0.15 * 0.20)

    def test_center_gap_two_centers(self):
        summary: dict[str, float] = {}
        enriched = [
            {"center": "hup", "patient_macro_f1": 0.8},
            {"center": "hup", "patient_macro_f1": 0.9},
            {"center": "lzu", "patient_macro_f1": 0.6},
            {"center": "lzu", "patient_macro_f1": 0.7},
        ]

        _add_center_robustness_fields(summary, enriched)

        self.assertAlmostEqual(summary["center_hup_patient_macro_f1"], 0.85)
        self.assertAlmostEqual(summary["center_lzu_patient_macro_f1"], 0.65)
        self.assertAlmostEqual(summary["center_gap_f1"], 0.2)
        self.assertAlmostEqual(summary["worst_center_f1"], 0.65)
        self.assertAlmostEqual(summary["best_center_f1"], 0.85)
        self.assertAlmostEqual(summary["center_multicenter_patient_macro_f1"], 0.0)
        self.assertAlmostEqual(summary["center_pediatric_patient_macro_f1"], 0.0)

    def test_center_gap_is_zero_for_single_center(self):
        summary: dict[str, float] = {}
        enriched = [
            {"center": "hup", "patient_macro_f1": 0.7},
            {"center": "hup", "patient_macro_f1": 0.9},
        ]

        _add_center_robustness_fields(summary, enriched)

        self.assertAlmostEqual(summary["center_hup_patient_macro_f1"], 0.8)
        self.assertAlmostEqual(summary["center_gap_f1"], 0.0)
        self.assertAlmostEqual(summary["worst_center_f1"], 0.8)
        self.assertAlmostEqual(summary["best_center_f1"], 0.8)

    def test_protocol_assert_fails_positive_label_nez(self):
        args = SimpleNamespace(
            positive_label="nez",
            split_strategy="5fold",
            n_splits=5,
            random_seed=42,
            drop_high_ez_fraction_lzu=False,
        )

        with self.assertRaisesRegex(ValueError, "positive_label"):
            assert_fixed_all90_protocol(args)

    def test_protocol_assert_fails_lzu_drop_true(self):
        args = SimpleNamespace(
            positive_label="ez",
            split_strategy="5fold",
            n_splits=5,
            random_seed=42,
            drop_high_ez_fraction_lzu=True,
        )

        with self.assertRaisesRegex(ValueError, "drop_high_ez_fraction_lzu"):
            assert_fixed_all90_protocol(args)

    def test_protocol_assert_passes_fixed_all90_minimal(self):
        args = SimpleNamespace(
            positive_label="ez",
            split_strategy="5fold",
            n_splits=5,
            random_seed=42,
            drop_high_ez_fraction_lzu=False,
        )

        audit = assert_fixed_all90_protocol(
            args,
            patient_index={f"p{i}": {} for i in range(90)},
            outer_splits=[object() for _ in range(5)],
        )

        self.assertEqual(audit["protocol_name"], "fixed_all90_patient_topk_ez")
        self.assertEqual(audit["n_patients"], 90)
        self.assertEqual(audit["n_outer_splits"], 5)
        self.assertFalse(audit["center_as_input_allowed"])

    def test_summarizer_minimal_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_dir = root / "A10_00_pbce"
            run_dir.mkdir()
            (run_dir / "a10_config.json").write_text(
                json.dumps(
                    {
                        "use_rank": False,
                        "ez_ranking_loss_weight": 0.0,
                        "ez_ranking_margin": 0.10,
                        "use_hard_topk": False,
                        "hard_topk_loss_weight": 0.0,
                        "hard_topk_margin": 0.05,
                        "positive_label": "ez",
                        "drop_high_ez_fraction_lzu": False,
                        "loss_mode": "patient_balanced_bce",
                        "patient_loss_weighting": "uniform",
                    }
                ),
                encoding="utf-8",
            )
            pd.DataFrame(
                [
                    {
                        "patient_macro_f1": 0.71,
                        "patient_macro_ez_f1": 0.53,
                        "patient_macro_auprc_ez": 0.57,
                        "patient_macro_ez_mrr": 0.75,
                        "top1_is_ez_rate": 0.66,
                    }
                ]
            ).to_csv(run_dir / "heldout_summary_neuroez_v3.csv", index=False)
            pd.DataFrame(
                [
                    {"subject_id": "p1", "center": "hup", "patient_macro_f1": 0.8},
                    {"subject_id": "p2", "center": "lzu", "patient_macro_f1": 0.6},
                ]
            ).to_csv(run_dir / "test_patient_predictions_neuroez_v2_fold_1.csv", index=False)

            output_csv = root / "a10_rank_simple_grid_summary.csv"

            summarize(root, output_csv)

            out = pd.read_csv(output_csv)
            self.assertEqual(out.loc[0, "config_name"], "A10_00_pbce")
            self.assertAlmostEqual(float(out.loc[0, "center_gap_f1"]), 0.2)
            self.assertIn("rank_robust_composite", out.columns)
            self.assertTrue(bool(out.loc[0, "passes_aaai_target_070"]))

    def test_evaluate_uses_pred_outputs_for_optional_diagnostics(self):
        class TinyModel(torch.nn.Module):
            def forward(self, batch):
                logits = torch.tensor([[2.0, -2.0]], dtype=torch.float32)
                score_ez = torch.sigmoid(logits)
                return {
                    "logits": logits,
                    "score_ez": score_ez,
                    "score_nez": 1.0 - score_ez,
                    "two_expert_lambda_patient": torch.tensor([0.25], dtype=torch.float32),
                }

        exp = Exp_EZHybridLocalization.__new__(Exp_EZHybridLocalization)
        exp.device = torch.device("cpu")
        exp.args = SimpleNamespace(
            loss_mode="patient_balanced_bce",
            positive_label="ez",
            class_weight_mode="none",
            early_stop_metric="patient_macro_f1",
            use_physics_dynamics=False,
            use_diffusion_residual=False,
            use_negative_anchor_head=False,
            use_view_gated_fusion=False,
            use_ez_ranking_loss=False,
            use_hard_topk_loss=False,
        )
        exp.use_a9v8_lcbo = False
        exp.use_broad_ez_mil_loss = False
        exp.broad_ez_mil_loss_weight = 0.0
        exp.feature_sep = False
        exp.aux_loss_weight = 0.0
        exp.anchor_rank_weight = 0.0
        exp.final_rank_weight = 0.0
        exp.pediatric_preserve = False
        exp.gate_l2 = 0.0
        exp.entropy_reg = 0.0
        batch = {
            "labels": torch.tensor([[1.0, 0.0]], dtype=torch.float32),
            "labels_ez": torch.tensor([[1.0, 0.0]], dtype=torch.float32),
            "labels_nez": torch.tensor([[0.0, 1.0]], dtype=torch.float32),
            "channel_mask": torch.tensor([[True, True]]),
            "subject_id": ["p1"],
            "center": ["hup"],
            "center_id": torch.tensor([0], dtype=torch.long),
            "ez_fraction": torch.tensor([0.5], dtype=torch.float32),
            "valid_channel_count": torch.tensor([2], dtype=torch.long),
            "ez_channel_count": torch.tensor([1], dtype=torch.long),
            "canonical_channels": [["a", "b"]],
            "channel_meta": [[]],
            "run_ids": [[]],
            "sample_ids": [[]],
        }

        _, _, records = exp._evaluate(TinyModel(), [batch], torch.tensor(1.0), split_name="val")

        self.assertEqual(len(records), 1)
        self.assertAlmostEqual(records[0]["two_expert_lambda_patient"], 0.25)


if __name__ == "__main__":
    unittest.main()
