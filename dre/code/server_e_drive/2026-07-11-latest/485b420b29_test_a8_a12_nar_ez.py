from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from exp_ez_hybrid import (
    Exp_EZHybridLocalization,
    _broad_ez_mil_ranking_loss,
    _compute_broad_aware_supervised_loss,
    _compute_supervised_loss,
    _parse_broad_ez_centers,
)
from neuroez_c.config import apply_pruned_defaults
from neuroez_c.dataset import build_patient_examples, collate_patient_ez_batch, fit_window_tensor_normalizer
from neuroez_c.model import NeuroEZCModel
from patient_channel_ranker import PatientChannelClassifier
from run_neuroez_c import build_parser


def _args(**overrides):
    values = {
        "b0_feature_parts": "abs,delta,zdelta,ratio",
        "b0_feature_groups": "spectral_classical",
        "self_compare_eps": 1e-5,
        "physics_state_features": "log_bp_high_gamma,line_length_per_sec,rms,variance",
        "physics_feature_parts": "abs",
        "positive_label": "ez",
        "class_weight_mode": "none",
        "loss_mode": "masked_bce",
        "patient_loss_weighting": "uniform",
        "use_diffusion_residual": False,
        "diffusion_graph_source": "cache_or_functional",
        "diffusion_gate_init": -4.0,
        "diffusion_beta_init": 0.10,
        "diffusion_source_sparse_weight": 0.0,
        "diffusion_residual_l2_weight": 0.0,
        "diffusion_score_residual": False,
        "diffusion_center_mode": "all",
        "diffusion_center_gate_init": -4.0,
        "use_physics_dynamics": False,
        "physics_model_dim": None,
        "physics_num_heads": None,
        "physics_gate_init": -3.0,
        "physics_loss_weight": 0.0,
        "physics_source_sparse_weight": 0.0,
        "physics_velocity_l2_weight": 0.0,
        "physics_detach_next_target": True,
        "use_view_gated_fusion": False,
        "physics_view_slices": "",
        "view_gate_init": -3.0,
        "view_dropout": 0.0,
        "static_fallback": True,
        "use_negative_anchor_head": False,
        "negative_anchor_dim": 4,
        "negative_anchor_loss_weight": 0.03,
        "negative_anchor_margin": 1.0,
        "negative_anchor_gamma": 0.5,
        "negative_anchor_gate_init": -2.0,
        "negative_anchor_center_prototypes": False,
        "use_broad_ez_mil_loss": False,
        "broad_ez_mil_loss_weight": 0.05,
        "broad_ez_core_frac": 0.30,
        "broad_ez_min_fraction": 0.35,
        "broad_ez_centers": "lzu,pediatric",
        "broad_ez_positive_bce_scale": 0.5,
        "broad_ez_margin": 0.05,
        "broad_ez_hard_neg_multiplier": 1.0,
        "use_channel_attention": True,
        "use_patient_relative_z": True,
        "temporal_pooling": "mean",
        "temporal_topk_fraction": 0.20,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _sample(subject_id: str = "lzu:p1") -> dict:
    return {
        "subject_id": subject_id,
        "run_id": f"{subject_id}_r1",
        "sample_id": f"{subject_id}_s1",
        "channel_names_norm": ["a", "b", "c", "d"],
        "labels": np.asarray([1.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "window_features": np.ones((3, 4, 20), dtype=np.float32),
        "window_adjacency": np.ones((3, 4, 4), dtype=np.float32),
        "window_relative_centers_sec": np.asarray([-2.0, 0.0, 2.0], dtype=np.float32),
    }


def _model_batch(physics_dim: int = 8) -> dict:
    return {
        "b0_features": torch.randn(2, 1, 3, 4, 36),
        "physics_features": torch.randn(2, 1, 3, 4, physics_dim),
        "diffusion_adjacency": torch.ones(2, 1, 3, 4, 4),
        "labels": torch.tensor([[1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]]),
        "labels_nez": torch.tensor([[0.0, 1.0, 1.0, 0.0], [0.0, 1.0, 1.0, 0.0]]),
        "labels_ez": torch.tensor([[1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]]),
        "channel_mask": torch.ones(2, 4, dtype=torch.bool),
        "seizure_mask": torch.ones(2, 1, dtype=torch.bool),
        "seizure_channel_mask": torch.ones(2, 1, 4, dtype=torch.bool),
        "window_mask": torch.ones(2, 1, 3, dtype=torch.bool),
        "window_centers": torch.tensor([[[-2.0, 0.0, 2.0]], [[-2.0, 0.0, 2.0]]]),
        "center_id": torch.tensor([0, 1], dtype=torch.long),
    }


class NAREZInfrastructureTests(unittest.TestCase):
    def test_apply_defaults_preserves_explicit_positive_ez_semantics(self):
        args = SimpleNamespace(
            positive_label="ez",
            window_cache_path="cache.pkl",
            b0_feature_parts="abs",
            b0_feature_groups="spectral_classical",
            drop_high_ez_fraction_lzu=False,
        )

        apply_pruned_defaults(args)

        self.assertEqual(args.positive_label, "ez")
        self.assertEqual(args.score_semantics, "ez_probability")
        self.assertFalse(args.drop_high_ez_fraction_lzu)

    def test_a8_a12_cli_flags_default_disabled_and_parse(self):
        args = build_parser().parse_args([])

        self.assertEqual(args.loss_mode, "masked_bce")
        self.assertFalse(args.use_negative_anchor_head)
        self.assertFalse(args.use_view_gated_fusion)
        self.assertFalse(args.diffusion_score_residual)
        self.assertEqual(args.diffusion_center_mode, "all")
        self.assertFalse(args.pretrain_masked_windows)

        enabled = build_parser().parse_args(
            [
                "--loss_mode",
                "patient_balanced_bce",
                "--use_negative_anchor_head",
                "--use_view_gated_fusion",
                "--physics_view_slices",
                "static:0:6,burst:6:8,hfo:8:12",
                "--diffusion_score_residual",
                "--diffusion_center_mode",
                "lzu_only",
                "--pretrain_masked_windows",
                "--pretrain_epochs",
                "1",
            ]
        )

        self.assertEqual(enabled.loss_mode, "patient_balanced_bce")
        self.assertTrue(enabled.use_negative_anchor_head)
        self.assertTrue(enabled.use_view_gated_fusion)
        self.assertTrue(enabled.diffusion_score_residual)
        self.assertEqual(enabled.diffusion_center_mode, "lzu_only")
        self.assertTrue(enabled.pretrain_masked_windows)
        self.assertEqual(enabled.pretrain_epochs, 1)

    def test_patient_channel_classifier_respects_positive_ez_semantics(self):
        classifier = PatientChannelClassifier(input_dim=4, num_heads=2, dropout=0.0, positive_label="ez")
        embeddings = torch.randn(1, 3, 4)
        mask = torch.ones(1, 3, dtype=torch.bool)

        out = classifier(embeddings, mask)

        self.assertTrue(torch.allclose(out["score_ez"], torch.sigmoid(out["logits"])))
        self.assertTrue(torch.allclose(out["score_nez"], 1.0 - out["score_ez"]))

    def test_patient_balanced_bce_averages_ez_and_nez_per_patient(self):
        logits = torch.tensor([[0.0, 2.0, -2.0, 0.5]])
        batch = {
            "labels": torch.tensor([[1.0, 0.0, 0.0, 1.0]]),
            "labels_ez": torch.tensor([[1.0, 0.0, 0.0, 1.0]]),
            "channel_mask": torch.tensor([[True, True, True, True]]),
        }

        loss, parts = _compute_supervised_loss(
            logits,
            batch,
            _args(loss_mode="patient_balanced_bce"),
            torch.tensor(1.0),
        )

        raw = torch.nn.functional.binary_cross_entropy_with_logits(logits, batch["labels"], reduction="none")
        expected = 0.5 * raw[0, [0, 3]].mean() + 0.5 * raw[0, [1, 2]].mean()
        self.assertTrue(torch.allclose(loss, expected))
        self.assertEqual(parts["loss_mode"], "patient_balanced_bce")
        self.assertEqual(parts["patient_balanced_single_class_count"], 0.0)

    def test_dataset_collate_exposes_center_and_prevalence_metadata(self):
        sample = _sample("lzu:p1")
        patient_index = {
            "lzu:p1": {
                "canonical_channels": ["a", "b", "c", "d"],
                "labels": np.asarray([1.0, 0.0, 0.0, 1.0], dtype=np.float32),
                "label_mask": np.asarray([True, True, True, True]),
                "source_center": "lzu",
            }
        }
        args = _args()
        normalizer = fit_window_tensor_normalizer([sample], args=args)

        examples = build_patient_examples([sample], patient_index, normalizer=normalizer, args=args)
        batch = collate_patient_ez_batch(examples)

        self.assertEqual(batch["center"], ["lzu"])
        self.assertEqual(batch["center_id"].tolist(), [1])
        self.assertEqual(batch["valid_channel_count"].tolist(), [4])
        self.assertEqual(batch["ez_channel_count"].tolist(), [2])
        self.assertTrue(torch.allclose(batch["ez_fraction"], torch.tensor([0.5])))

    def test_negative_anchor_head_reports_loss_and_updates_ez_scores(self):
        from neuroez_c.negative_anchor import NegativeAnchorHead

        head = NegativeAnchorHead(input_dim=6, anchor_dim=3, gate_init=0.0)
        embeddings = torch.randn(1, 4, 6)
        labels_ez = torch.tensor([[1.0, 0.0, 0.0, 1.0]])
        mask = torch.ones(1, 4, dtype=torch.bool)
        base_logits = torch.zeros(1, 4)

        out = head(embeddings, labels_ez, mask, base_logits, positive_label="ez")

        self.assertEqual(out["logits"].shape, (1, 4))
        self.assertIn("negative_anchor_loss", out)
        self.assertTrue(torch.isfinite(out["negative_anchor_loss"]))
        self.assertFalse(torch.allclose(out["score_ez_final"], torch.sigmoid(base_logits)))

    def test_view_slice_parser_validates_bounds(self):
        from neuroez_c.view_gated_fusion import parse_view_slices

        views = parse_view_slices("static:0:6,burst:6:8,hfo:8:12", feature_dim=12)
        self.assertEqual([view.name for view in views], ["static", "burst", "hfo"])

        with self.assertRaisesRegex(ValueError, "outside available physics feature dim"):
            parse_view_slices("static:0:6,hfo:8:13", feature_dim=12)

    def test_model_forward_uses_negative_anchor_when_enabled(self):
        out = NeuroEZCModel(_args(model_dim=8, use_negative_anchor_head=True))(_model_batch())

        self.assertIn("negative_anchor_loss", out)
        self.assertIn("negative_anchor_distance", out)
        self.assertEqual(out["negative_anchor_distance"].shape, (2, 4))
        self.assertTrue(torch.allclose(out["score_ez_final"], out["score_ez"]))

    def test_model_forward_uses_view_gated_fusion_when_enabled(self):
        out = NeuroEZCModel(
            _args(
                model_dim=8,
                use_physics_dynamics=True,
                use_view_gated_fusion=True,
                physics_view_slices="static:0:6,burst:6:8,hfo:8:12",
            )
        )(_model_batch(physics_dim=12))

        self.assertIn("view_gate_burst", out)
        self.assertIn("view_gate_hfo", out)
        self.assertIn("view_gate_l1_loss", out)

    def test_model_forward_supports_mean_max_topk_temporal_pooling(self):
        out = NeuroEZCModel(_args(model_dim=8, temporal_pooling="mean_max_topk"))(_model_batch())

        self.assertEqual(out["logits"].shape, (2, 4))
        self.assertEqual(out["temporal_attention"].shape, (2, 1, 4, 3))

    def test_lzu_only_diffusion_score_residual_masks_non_lzu_patients(self):
        out = NeuroEZCModel(
            _args(
                model_dim=8,
                use_diffusion_residual=True,
                diffusion_score_residual=True,
                diffusion_center_mode="lzu_only",
            )
        )(_model_batch())

        self.assertIn("graph_delta_logits", out)
        self.assertEqual(float(out["diffusion_applied_patient_count"].detach().cpu()), 1.0)
        self.assertAlmostEqual(float(out["mean_abs_graph_delta_non_lzu"].detach().cpu()), 0.0, places=6)

    def test_dry_run_config_only_writes_args_without_loading_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            result = subprocess.run(
                [
                    sys.executable,
                    "run_neuroez_c.py",
                    "--dry_run_config_only",
                    "--output_dir",
                    str(output_dir),
                    "--window_cache_path",
                    str(output_dir / "missing_remote.pkl"),
                    "--positive_label",
                    "ez",
                    "--drop_high_ez_fraction_lzu",
                    "false",
                    "--loss_mode",
                    "patient_balanced_bce",
                    "--use_negative_anchor_head",
                ],
                cwd=Path(__file__).resolve().parents[1],
                text=True,
                capture_output=True,
                check=True,
            )

            self.assertIn("Dry-run config only", result.stdout)
            with open(output_dir / "run_args_b0_pruned.json", "r", encoding="utf-8") as fin:
                args = json.load(fin)
            self.assertEqual(args["positive_label"], "ez")
            self.assertEqual(args["score_semantics"], "ez_probability")
            self.assertFalse(args["drop_high_ez_fraction_lzu"])
            self.assertTrue(args["use_negative_anchor_head"])


    def test_negative_anchor_center_gate_returns_per_patient_tensor(self):
        from neuroez_c.negative_anchor import NegativeAnchorHead

        head = NegativeAnchorHead(input_dim=8, anchor_dim=4, center_gate=True, gate_init=0.0, pediatric_gate_delta_init=-2.0)
        batch = 3
        embeddings = torch.randn(batch, 6, 8)
        labels_ez = torch.tensor([[1.0, 0.0, 0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]])
        mask = torch.ones(batch, 6, dtype=torch.bool)
        base_logits = torch.zeros(batch, 6)
        center_id = torch.tensor([0, 1, 3], dtype=torch.long)

        out = head(embeddings, labels_ez, mask, base_logits, positive_label="ez", center_id=center_id)

        self.assertIn("negative_anchor_gate_per_patient", out)
        gate_pp = out["negative_anchor_gate_per_patient"]
        self.assertEqual(gate_pp.shape, (batch,))
        # pediatric (center_id=3, delta=-2.0) gate should be lower than HUP (center_id=0, delta=0.0)
        self.assertLess(float(gate_pp[2].detach().cpu()), float(gate_pp[0].detach().cpu()))

    def test_negative_anchor_pediatric_gate_below_hup_with_neg_delta(self):
        from neuroez_c.negative_anchor import NegativeAnchorHead

        head = NegativeAnchorHead(input_dim=8, anchor_dim=4, center_gate=True, gate_init=0.0, pediatric_gate_delta_init=-2.0)
        embeddings = torch.randn(2, 6, 8)
        labels_ez = torch.ones(2, 6)
        mask = torch.ones(2, 6, dtype=torch.bool)
        base_logits = torch.zeros(2, 6)
        center_id = torch.tensor([0, 3], dtype=torch.long)  # HUP=0, pediatric=3

        out = head(embeddings, labels_ez, mask, base_logits, positive_label="ez", center_id=center_id)

        gate_pp = out["negative_anchor_gate_per_patient"]
        self.assertLess(float(gate_pp[1].detach().cpu()), float(gate_pp[0].detach().cpu()),
                        "Pediatric gate (delta=-2.0) should be strictly less than HUP gate (delta=0.0)")

    def test_hard_topk_cli_flags_parse(self):
        parser = build_parser()
        args = parser.parse_args([
            "--use_hard_topk_loss",
            "--hard_topk_loss_weight", "0.05",
            "--hard_topk_margin", "0.10",
            "--hard_topk_multiplier", "2.0",
        ])
        self.assertTrue(args.use_hard_topk_loss)
        self.assertEqual(args.hard_topk_loss_weight, 0.05)
        self.assertEqual(args.hard_topk_margin, 0.10)
        self.assertEqual(args.hard_topk_multiplier, 2.0)

    def test_broad_ez_cli_flags_parse(self):
        parser = build_parser()
        args = parser.parse_args([
            "--use_broad_ez_mil_loss",
            "--broad_ez_mil_loss_weight", "0.05",
            "--broad_ez_core_frac", "0.30",
            "--broad_ez_min_fraction", "0.35",
            "--broad_ez_centers", "lzu,pediatric",
            "--broad_ez_positive_bce_scale", "0.5",
            "--broad_ez_margin", "0.05",
            "--broad_ez_hard_neg_multiplier", "1.0",
        ])

        self.assertTrue(args.use_broad_ez_mil_loss)
        self.assertEqual(args.broad_ez_mil_loss_weight, 0.05)
        self.assertEqual(args.broad_ez_core_frac, 0.30)
        self.assertEqual(args.broad_ez_min_fraction, 0.35)
        self.assertEqual(args.broad_ez_centers, "lzu,pediatric")
        self.assertEqual(args.broad_ez_positive_bce_scale, 0.5)
        self.assertEqual(args.broad_ez_margin, 0.05)
        self.assertEqual(args.broad_ez_hard_neg_multiplier, 1.0)

    def test_parse_broad_ez_centers_accepts_names_and_ids(self):
        self.assertEqual(_parse_broad_ez_centers("lzu,pediatric"), {1, 3})
        self.assertEqual(_parse_broad_ez_centers("1,3"), {1, 3})
        with self.assertRaisesRegex(ValueError, "Unsupported broad EZ center token"):
            _parse_broad_ez_centers("lzu,badcenter")

    def test_broad_ez_mil_loss_only_targets_broad_or_high_fraction_patients(self):
        logits = torch.tensor([
            [0.9, 0.8, 0.7, 0.6],
            [0.1, 0.2, 1.2, 1.1],
            [0.1, 0.2, 1.3, 1.1],
        ])
        labels_ez = torch.tensor([
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
        ])
        mask = torch.ones_like(labels_ez, dtype=torch.bool)
        center_id = torch.tensor([0, 1, 3], dtype=torch.long)
        ez_fraction = torch.tensor([0.25, 0.25, 0.25])

        loss = _broad_ez_mil_ranking_loss(
            logits,
            labels_ez,
            mask,
            center_id,
            ez_fraction,
            positive_label="ez",
            broad_centers={1, 3},
            min_ez_fraction=0.35,
            core_frac=0.30,
            margin=0.05,
            hard_neg_multiplier=1.0,
        )
        self.assertGreater(float(loss.detach().cpu()), 0.0)

        hup_only = _broad_ez_mil_ranking_loss(
            logits[:1],
            labels_ez[:1],
            mask[:1],
            torch.tensor([0], dtype=torch.long),
            torch.tensor([0.25]),
            positive_label="ez",
            broad_centers={1, 3},
            min_ez_fraction=0.35,
            core_frac=0.30,
            margin=0.05,
            hard_neg_multiplier=1.0,
        )
        self.assertAlmostEqual(float(hup_only.detach().cpu()), 0.0, places=7)

    def test_broad_aware_supervised_loss_scales_only_broad_center_ez_positives(self):
        logits = torch.zeros(1, 4)
        labels = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
        batch_lzu = {
            "labels": labels,
            "labels_ez": labels,
            "channel_mask": torch.ones(1, 4, dtype=torch.bool),
            "center_id": torch.tensor([1], dtype=torch.long),
        }
        batch_hup = dict(batch_lzu)
        batch_hup["center_id"] = torch.tensor([0], dtype=torch.long)
        base_args = _args(
            use_broad_ez_mil_loss=True,
            loss_mode="patient_balanced_bce",
            broad_ez_centers="lzu,pediatric",
            broad_ez_positive_bce_scale=1.0,
        )
        scaled_args = _args(
            use_broad_ez_mil_loss=True,
            loss_mode="patient_balanced_bce",
            broad_ez_centers="lzu,pediatric",
            broad_ez_positive_bce_scale=0.5,
        )

        lzu_base, _ = _compute_broad_aware_supervised_loss(logits, batch_lzu, base_args, torch.tensor(1.0))
        lzu_scaled, _ = _compute_broad_aware_supervised_loss(logits, batch_lzu, scaled_args, torch.tensor(1.0))
        hup_base, _ = _compute_broad_aware_supervised_loss(logits, batch_hup, base_args, torch.tensor(1.0))
        hup_scaled, _ = _compute_broad_aware_supervised_loss(logits, batch_hup, scaled_args, torch.tensor(1.0))

        self.assertLess(float(lzu_scaled.detach().cpu()), float(lzu_base.detach().cpu()))
        self.assertTrue(torch.allclose(hup_scaled, hup_base))

    def test_compute_loss_adds_broad_ez_mil_diagnostics_when_enabled(self):
        args = _args(
            use_broad_ez_mil_loss=True,
            broad_ez_mil_loss_weight=0.10,
            broad_ez_core_frac=0.30,
            broad_ez_min_fraction=0.35,
            broad_ez_centers="lzu,pediatric",
            broad_ez_positive_bce_scale=0.5,
            broad_ez_margin=0.05,
            loss_mode="patient_balanced_bce",
        )
        exp = Exp_EZHybridLocalization.__new__(Exp_EZHybridLocalization)
        exp.args = args
        exp.use_broad_ez_mil_loss = True
        exp.broad_ez_mil_loss_weight = 0.10
        exp.broad_ez_core_frac = 0.30
        exp.broad_ez_min_fraction = 0.35
        exp.broad_ez_centers = {1, 3}
        exp.broad_ez_positive_bce_scale = 0.5
        exp.broad_ez_margin = 0.05
        exp.broad_ez_hard_neg_multiplier = 1.0
        exp.pediatric_preserve = False
        exp.gate_l2 = 0.0
        exp.entropy_reg = 0.0
        exp.feature_sep = False
        exp.aux_loss_weight = 0.0
        exp.anchor_rank_weight = 0.0
        exp.final_rank_weight = 0.0
        logits = torch.tensor([[0.1, 0.2, 1.2, 1.1]])
        batch = {
            "labels": torch.tensor([[1.0, 1.0, 0.0, 0.0]]),
            "labels_ez": torch.tensor([[1.0, 1.0, 0.0, 0.0]]),
            "channel_mask": torch.ones(1, 4, dtype=torch.bool),
            "center_id": torch.tensor([1], dtype=torch.long),
            "ez_fraction": torch.tensor([0.5]),
        }
        outputs = {"logits": logits}

        loss, parts = exp._compute_loss(outputs, batch, torch.tensor(1.0))
        bce_only, _ = _compute_broad_aware_supervised_loss(logits, batch, args, torch.tensor(1.0))

        self.assertIn("broad_ez_mil_loss", parts)
        self.assertGreater(parts["broad_ez_mil_loss"], 0.0)
        self.assertGreater(float(loss.detach().cpu()), float(bce_only.detach().cpu()))

    def test_two_expert_cli_flags_parse(self):
        parser = build_parser()
        args = parser.parse_args([
            "--use_two_expert_router",
            "--two_expert_router_mode", "center_fixed",
            "--two_expert_lambda_hup", "1.0",
            "--two_expert_lambda_lzu", "0.75",
            "--two_expert_lambda_multicenter", "1.0",
            "--two_expert_lambda_pediatric", "0.0",
        ])

        self.assertTrue(args.use_two_expert_router)
        self.assertEqual(args.two_expert_router_mode, "center_fixed")
        self.assertEqual(args.two_expert_lambda_hup, 1.0)
        self.assertEqual(args.two_expert_lambda_lzu, 0.75)
        self.assertEqual(args.two_expert_lambda_multicenter, 1.0)
        self.assertEqual(args.two_expert_lambda_pediatric, 0.0)

    def test_neuroez_model_two_expert_forward_reports_requested_outputs(self):
        batch = _model_batch(physics_dim=12)
        out = NeuroEZCModel(
            _args(
                model_dim=8,
                use_two_expert_router=True,
                two_expert_router_mode="center_fixed",
                two_expert_lambda_hup=1.0,
                two_expert_lambda_lzu=0.75,
                two_expert_lambda_multicenter=1.0,
                two_expert_lambda_pediatric=0.0,
                use_physics_dynamics=True,
                use_negative_anchor_head=True,
                physics_state_features="early_high_gamma_slope,early_line_length_slope,onset_latency_high_gamma,onset_latency_line_length,onset_rank_high_gamma,onset_rank_line_length,high_gamma_top20pct_mean,line_length_top20pct_mean,hfo80_150_event_rate,hfo80_150_duration_fraction,hfo80_150_mean_envelope_z,hfo80_150_max_envelope_z",
            )
        )(batch)

        self.assertIn("score_ez_static", out)
        self.assertIn("score_ez_anchor", out)
        self.assertIn("score_ez_final", out)
        self.assertIn("two_expert_lambda_patient", out)
        self.assertEqual(out["score_ez_static"].shape, out["score_ez_final"].shape)
        self.assertEqual(out["score_ez_anchor"].shape, out["score_ez_final"].shape)
        self.assertEqual(out["two_expert_lambda_patient"].shape, (2,))

    def test_two_expert_center_fixed_uses_pediatric_lambda(self):
        batch = _model_batch(physics_dim=12)
        batch["center_id"] = torch.tensor([3, 3], dtype=torch.long)
        out = NeuroEZCModel(
            _args(
                model_dim=8,
                use_two_expert_router=True,
                two_expert_router_mode="center_fixed",
                two_expert_lambda_hup=1.0,
                two_expert_lambda_lzu=0.75,
                two_expert_lambda_multicenter=1.0,
                two_expert_lambda_pediatric=0.25,
                use_physics_dynamics=True,
                use_negative_anchor_head=True,
                physics_state_features="early_high_gamma_slope,early_line_length_slope,onset_latency_high_gamma,onset_latency_line_length,onset_rank_high_gamma,onset_rank_line_length,high_gamma_top20pct_mean,line_length_top20pct_mean,hfo80_150_event_rate,hfo80_150_duration_fraction,hfo80_150_mean_envelope_z,hfo80_150_max_envelope_z",
            )
        )(batch)

        self.assertTrue(torch.allclose(out["two_expert_lambda_patient"], torch.full((2,), 0.25)))

    def test_two_expert_center_learned_pediatric_gate_below_hup(self):
        batch = _model_batch(physics_dim=12)
        batch["center_id"] = torch.tensor([0, 3], dtype=torch.long)
        out = NeuroEZCModel(
            _args(
                model_dim=8,
                use_two_expert_router=True,
                two_expert_router_mode="center_learned",
                two_expert_gate_init_hup=1.0,
                two_expert_gate_init_lzu=0.5,
                two_expert_gate_init_multicenter=1.0,
                two_expert_gate_init_pediatric=-3.0,
                use_physics_dynamics=True,
                use_negative_anchor_head=True,
                physics_state_features="early_high_gamma_slope,early_line_length_slope,onset_latency_high_gamma,onset_latency_line_length,onset_rank_high_gamma,onset_rank_line_length,high_gamma_top20pct_mean,line_length_top20pct_mean,hfo80_150_event_rate,hfo80_150_duration_fraction,hfo80_150_mean_envelope_z,hfo80_150_max_envelope_z",
            )
        )(batch)

        self.assertLess(
            float(out["two_expert_lambda_patient"][1].detach().cpu()),
            float(out["two_expert_lambda_patient"][0].detach().cpu()),
        )

    def test_two_expert_fused_score_respects_lambda_extremes(self):
        batch = _model_batch(physics_dim=12)
        batch["center_id"] = torch.tensor([0, 3], dtype=torch.long)
        args = _args(
                model_dim=8,
                use_two_expert_router=True,
                two_expert_router_mode="center_fixed",
                two_expert_lambda_hup=0.0,
                two_expert_lambda_lzu=0.75,
                two_expert_lambda_multicenter=1.0,
                two_expert_lambda_pediatric=1.0,
                use_physics_dynamics=True,
                use_negative_anchor_head=True,
                physics_state_features="early_high_gamma_slope,early_line_length_slope,onset_latency_high_gamma,onset_latency_line_length,onset_rank_high_gamma,onset_rank_line_length,high_gamma_top20pct_mean,line_length_top20pct_mean,hfo80_150_event_rate,hfo80_150_duration_fraction,hfo80_150_mean_envelope_z,hfo80_150_max_envelope_z",
            )
        out = NeuroEZCModel(args)(batch)

        self.assertTrue(torch.allclose(out["score_ez_final"][0], out["score_ez_static"][0]))
        self.assertTrue(torch.allclose(out["score_ez_final"][1], out["score_ez_anchor"][1]))

    # ---- A9v6 feature-separated two-expert tests ----

    def test_feature_sep_two_expert_forward_reports_all_outputs(self):
        batch = _model_batch(physics_dim=12)
        out = NeuroEZCModel(
            _args(
                model_dim=8,
                use_feature_separated_two_expert=True,
                two_expert_router_mode="center_fixed",
                two_expert_lambda_hup=1.0, two_expert_lambda_lzu=0.75,
                two_expert_lambda_multicenter=1.0, two_expert_lambda_pediatric=0.0,
                use_physics_dynamics=True,
                use_negative_anchor_head=True,
                physics_state_features="early_high_gamma_slope,early_line_length_slope,onset_latency_high_gamma,onset_latency_line_length,onset_rank_high_gamma,onset_rank_line_length,high_gamma_top20pct_mean,line_length_top20pct_mean,hfo80_150_event_rate,hfo80_150_duration_fraction,hfo80_150_mean_envelope_z,hfo80_150_max_envelope_z",
            )
        )(batch)

        for required_key in ("logits_static", "logits_anchor", "score_ez_static",
                             "score_ez_anchor", "score_ez_final", "two_expert_lambda_patient"):
            self.assertIn(required_key, out, f"Missing {required_key}")
        self.assertEqual(out["logits_static"].shape, (2, 4))
        self.assertEqual(out["logits_anchor"].shape, (2, 4))
        self.assertEqual(out["two_expert_lambda_patient"].shape, (2,))

    def test_feature_sep_static_slice_uses_six_features(self):
        batch = _model_batch(physics_dim=12)
        out = NeuroEZCModel(
            _args(
                model_dim=8,
                use_feature_separated_two_expert=True,
                two_expert_router_mode="center_fixed",
                two_expert_lambda_hup=1.0, two_expert_lambda_lzu=0.75,
                two_expert_lambda_multicenter=1.0, two_expert_lambda_pediatric=0.0,
                use_physics_dynamics=True, use_negative_anchor_head=True,
                two_expert_static_feature_slice="0:6",
                two_expert_anchor_feature_slice="all",
                physics_state_features="early_high_gamma_slope,early_line_length_slope,onset_latency_high_gamma,onset_latency_line_length,onset_rank_high_gamma,onset_rank_line_length,high_gamma_top20pct_mean,line_length_top20pct_mean,hfo80_150_event_rate,hfo80_150_duration_fraction,hfo80_150_mean_envelope_z,hfo80_150_max_envelope_z",
            )
        )(batch)

        self.assertIn("two_expert_static_feature_dim", out)
        self.assertIn("two_expert_anchor_feature_dim", out)
        self.assertEqual(int(out["two_expert_static_feature_dim"].detach().cpu()), 6)
        self.assertEqual(int(out["two_expert_anchor_feature_dim"].detach().cpu()), 12)

    def test_feature_sep_slice_out_of_range_raises(self):
        from neuroez_c.model import _parse_feature_slice

        with self.assertRaises(ValueError):
            _parse_feature_slice("0:20", total_dim=12)

        with self.assertRaises(ValueError):
            _parse_feature_slice("bad", total_dim=12)

        # valid slice must not raise
        sl = _parse_feature_slice("all", total_dim=12)
        self.assertEqual(sl, slice(0, 12))
        sl = _parse_feature_slice("2:8", total_dim=12)
        self.assertEqual(sl, slice(2, 8))

    def test_feature_sep_lambda_extremes_match_scores(self):
        batch = _model_batch(physics_dim=12)
        batch["center_id"] = torch.tensor([0, 3], dtype=torch.long)
        args = _args(
            model_dim=8,
            use_feature_separated_two_expert=True,
            two_expert_router_mode="center_fixed",
            two_expert_lambda_hup=0.0,
            two_expert_lambda_lzu=0.75,
            two_expert_lambda_multicenter=1.0,
            two_expert_lambda_pediatric=1.0,
            use_physics_dynamics=True,
            use_negative_anchor_head=True,
            physics_state_features="early_high_gamma_slope,early_line_length_slope,onset_latency_high_gamma,onset_latency_line_length,onset_rank_high_gamma,onset_rank_line_length,high_gamma_top20pct_mean,line_length_top20pct_mean,hfo80_150_event_rate,hfo80_150_duration_fraction,hfo80_150_mean_envelope_z,hfo80_150_max_envelope_z",
        )
        out = NeuroEZCModel(args)(batch)

        # HUP (idx 0) lambda=0 → final equals static
        self.assertTrue(torch.allclose(out["score_ez_final"][0], out["score_ez_static"][0], atol=1e-6))
        # pediatric (idx 1) lambda=1 → final equals anchor
        self.assertTrue(torch.allclose(out["score_ez_final"][1], out["score_ez_anchor"][1], atol=1e-6))

    def test_feature_sep_center_learned_pediatric_lambda_below_hup(self):
        batch = _model_batch(physics_dim=12)
        batch["center_id"] = torch.tensor([0, 3], dtype=torch.long)
        out = NeuroEZCModel(
            _args(
                model_dim=8,
                use_feature_separated_two_expert=True,
                two_expert_router_mode="center_learned",
                two_expert_gate_init_hup=1.0, two_expert_gate_init_lzu=0.5,
                two_expert_gate_init_multicenter=1.0, two_expert_gate_init_pediatric=-3.0,
                use_physics_dynamics=True, use_negative_anchor_head=True,
                physics_state_features="early_high_gamma_slope,early_line_length_slope,onset_latency_high_gamma,onset_latency_line_length,onset_rank_high_gamma,onset_rank_line_length,high_gamma_top20pct_mean,line_length_top20pct_mean,hfo80_150_event_rate,hfo80_150_duration_fraction,hfo80_150_mean_envelope_z,hfo80_150_max_envelope_z",
            )
        )(batch)

        self.assertLess(
            float(out["two_expert_lambda_patient"][1].detach().cpu()),
            float(out["two_expert_lambda_patient"][0].detach().cpu()),
        )

    def test_feature_sep_aux_loss_in_compute_loss_parts(self):
        """Verify that _compute_loss emits all A9v6 aux/ranking diagnostics."""
        from exp_ez_hybrid import _compute_supervised_loss, _ez_pairwise_ranking_loss

        batch = _model_batch(physics_dim=12)
        batch_t = {k: v.to(torch.device("cpu")) if torch.is_tensor(v) else v for k, v in batch.items()}
        args = _args(
            model_dim=8,
            use_feature_separated_two_expert=True,
            two_expert_router_mode="center_fixed",
            two_expert_lambda_hup=1.0, two_expert_lambda_lzu=0.75,
            two_expert_lambda_multicenter=1.0, two_expert_lambda_pediatric=0.0,
            use_physics_dynamics=True, use_negative_anchor_head=True,
            two_expert_aux_loss_weight=0.20,
            two_expert_anchor_ranking_loss_weight=0.05,
            two_expert_final_ranking_loss_weight=0.05,
            physics_state_features="early_high_gamma_slope,early_line_length_slope,onset_latency_high_gamma,onset_latency_line_length,onset_rank_high_gamma,onset_rank_line_length,high_gamma_top20pct_mean,line_length_top20pct_mean,hfo80_150_event_rate,hfo80_150_duration_fraction,hfo80_150_mean_envelope_z,hfo80_150_max_envelope_z",
        )
        model = NeuroEZCModel(args)

        with torch.no_grad():
            outputs = model(batch_t)

        # Manually call _compute_loss' aux/ranking logic inline
        ez_weight = torch.tensor(2.0)
        parts: dict[str, float] = {}

        # aux loss
        bce_s, _ = _compute_supervised_loss(outputs["logits_static"], batch_t, args, ez_weight)
        bce_a, _ = _compute_supervised_loss(outputs["logits_anchor"], batch_t, args, ez_weight)
        self.assertGreater(float(bce_s.detach()), 0.0)
        self.assertGreater(float(bce_a.detach()), 0.0)
        parts["two_expert_static_bce"] = float(bce_s.detach().cpu())
        parts["two_expert_anchor_bce"] = float(bce_a.detach().cpu())

        # anchor ranking
        rank_a = _ez_pairwise_ranking_loss(
            outputs["logits_anchor"], batch_t["labels_ez"], batch_t["channel_mask"],
            margin=0.05, positive_label="ez",
        )
        parts["two_expert_anchor_ranking_loss"] = float(rank_a.detach().cpu())

        # final ranking
        rank_f = _ez_pairwise_ranking_loss(
            outputs["logits"], batch_t["labels_ez"], batch_t["channel_mask"],
            margin=0.05, positive_label="ez",
        )
        parts["two_expert_final_ranking_loss"] = float(rank_f.detach().cpu())

        for required_key in ("two_expert_static_bce", "two_expert_anchor_bce",
                             "two_expert_anchor_ranking_loss", "two_expert_final_ranking_loss"):
            self.assertIn(required_key, parts, f"Missing {required_key}")
            self.assertTrue(np.isfinite(parts[required_key]))


if __name__ == "__main__":
    unittest.main()
