from __future__ import annotations

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd
import torch

import exp_ez_hybrid
from exp_ez_hybrid import Exp_EZHybridLocalization
from neuroez_c.a9v14_candidate_modules import A9V14_LEDGER_REQUIRED_COLUMNS
from neuroez_c.model import NeuroEZCModel
from run_neuroez_c import build_parser
from scripts.run_a9v14_candidate import build_command


def _args(**overrides):
    values = {
        "config_name": "A9v14_Test",
        "model_dim": 8,
        "num_heads": 2,
        "dropout": 0.0,
        "use_channel_attention": True,
        "use_patient_relative_z": True,
        "positive_label": "ez",
        "b0_feature_parts": "abs,delta,zdelta,ratio",
        "b0_feature_groups": "spectral_classical",
        "self_compare_eps": 1e-5,
        "temporal_pooling": "mean",
        "channel_pooling_mode": "mean",
        "temporal_topk_fraction": 0.20,
        "temporal_pooling_top_p": 0.10,
        "temporal_pooling_tau": 0.10,
        "record_pooling": "mean",
        "record_pooling_top_p": 0.30,
        "record_pooling_alpha": 0.70,
        "use_physics_dynamics": False,
        "use_diffusion_residual": False,
        "use_view_gated_fusion": False,
        "use_negative_anchor_head": False,
        "use_two_expert_router": False,
        "use_feature_separated_two_expert": False,
        "use_a9v8_lcbo": False,
        "eval_score_fusion_gamma": 0.10,
        "use_patient_context_reranker": True,
        "reranker_type": "set_transformer",
        "reranker_hidden_dim": 16,
        "reranker_num_layers": 1,
        "reranker_dropout": 0.0,
        "reranker_residual_scale": 0.2,
        "reranker_delta_l2_weight": 0.001,
        "reranker_use_rank_features": True,
        "reranker_use_patient_gate": True,
        "reranker_gate_l2_weight": 0.001,
        "use_multi_seizure_consistency": True,
        "consistency_hidden_dim": 16,
        "consistency_dropout": 0.0,
        "consistency_residual_scale": 0.15,
        "consistency_delta_l2_weight": 0.001,
        "use_shaft_local_residual": True,
        "local_window": 1,
        "local_hidden_dim": 16,
        "local_dropout": 0.0,
        "local_residual_scale": 0.10,
        "local_delta_l2_weight": 0.001,
        "use_clinical_mixture_head": True,
        "mixture_core_loss_weight": 0.2,
        "mixture_broad_loss_weight": 0.5,
        "mixture_final_loss_weight": 1.0,
        "freeze_a9v3_backbone": True,
        "base_aux_loss_weight": 0.2,
        "final_loss_weight": 1.0,
        "module_dropout": 0.0,
        "candidate_module_seed": 42,
        "save_record_level_outputs": True,
        "save_prediction_ledger": True,
        "save_module_diagnostics": True,
        "class_weight_mode": "none",
        "device": "cpu",
        "output_dir": ".",
        "num_workers": 0,
        "epochs": 1,
        "patience": 0,
        "batch_size": 2,
        "patient_batch_size": 2,
        "learning_rate": 1e-4,
        "weight_decay": 1e-3,
        "grad_clip": 1.0,
        "early_stop_metric": "patient_macro_f1",
        "drop_high_ez_fraction_lzu": False,
        "log_interval": 1,
        "loss_mode": "masked_bce",
        "patient_loss_weighting": "uniform",
        "rank_loss_weight": 0.0,
        "hard_pairwise_weight": 0.0,
        "soft_topk_weight": 0.0,
        "first_rank_weight": 0.0,
        "diversity_weight": 0.0,
        "group_robust_mode": "none",
        "use_ez_ranking_loss": False,
        "use_hard_topk_loss": False,
        "use_broad_ez_mil_loss": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _batch() -> dict:
    return {
        "b0_features": torch.randn(2, 2, 3, 4, 36),
        "labels": torch.tensor([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]]),
        "labels_nez": torch.tensor([[0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0]]),
        "labels_ez": torch.tensor([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]]),
        "channel_mask": torch.ones(2, 4, dtype=torch.bool),
        "seizure_mask": torch.ones(2, 2, dtype=torch.bool),
        "seizure_channel_mask": torch.ones(2, 2, 4, dtype=torch.bool),
        "window_mask": torch.ones(2, 2, 3, dtype=torch.bool),
        "center_id": torch.tensor([0, 3], dtype=torch.long),
        "canonical_channels": [["A1", "A2", "B1", "B2"], ["LA1", "LA2", "HH1-2", "X"]],
    }


def _features(offset: float = 0.0) -> np.ndarray:
    return np.arange(4 * 4 * 20, dtype=np.float32).reshape(4, 4, 20) + 1.0 + offset


def _sample(subject_id: str, offset: float = 0.0) -> dict:
    return {
        "subject_id": subject_id,
        "run_id": f"{subject_id}_r1",
        "sample_id": f"{subject_id}_s1",
        "channel_names_norm": ["A1", "A2", "B1", "B2"],
        "labels": np.asarray([1.0, 0.0, 1.0, 0.0], dtype=np.float32),
        "window_features": _features(offset),
        "window_adjacency": np.ones((4, 4, 4), dtype=np.float32),
        "window_relative_centers_sec": np.asarray([-2.0, -1.0, 0.0, 1.0], dtype=np.float32),
    }


def _patient_index(*subject_ids: str) -> dict:
    return {
        subject_id: {
            "canonical_channels": ["A1", "A2", "B1", "B2"],
            "labels": np.asarray([1.0, 0.0, 1.0, 0.0], dtype=np.float32),
            "label_mask": np.asarray([True, True, True, True]),
            "center": "hup" if int(subject_id[1:]) % 2 == 0 else "pediatric",
        }
        for subject_id in subject_ids
    }


class A9v14ContextRerankerIntegrationTests(unittest.TestCase):
    def test_parser_accepts_candidate_module_flags(self):
        args = build_parser().parse_args(
            [
                "--config_name",
                "A9v14_M1M2",
                "--use_patient_context_reranker",
                "--reranker_type",
                "set_transformer",
                "--reranker_use_rank_features",
                "--use_multi_seizure_consistency",
                "--use_shaft_local_residual",
                "--use_clinical_mixture_head",
                "--save_prediction_ledger",
            ]
        )

        self.assertEqual(args.config_name, "A9v14_M1M2")
        self.assertEqual(args.reranker_type, "set_transformer")
        self.assertTrue(args.use_patient_context_reranker)
        self.assertTrue(args.reranker_use_rank_features)
        self.assertTrue(args.use_multi_seizure_consistency)
        self.assertTrue(args.use_shaft_local_residual)
        self.assertTrue(args.use_clinical_mixture_head)
        self.assertTrue(args.save_prediction_ledger)

    def test_candidate_runner_skips_empty_string_cli_values(self):
        command = build_command(
            "python",
            {
                "physics_view_slices": "",
                "latent_core_target_dir": "   ",
                "positive_label": "ez",
                "drop_high_ez_fraction_lzu": False,
            },
        )

        self.assertNotIn("--physics_view_slices", command)
        self.assertNotIn("--latent_core_target_dir", command)
        self.assertIn("--positive_label", command)
        self.assertIn("ez", command)
        self.assertIn("--drop_high_ez_fraction_lzu", command)
        self.assertIn("false", command)

    def test_model_forward_outputs_a9v14_fields(self):
        torch.manual_seed(7)
        out = NeuroEZCModel(_args())(_batch())

        for key in (
            "logits_base",
            "base_logit_a9v3",
            "base_score_a9v3",
            "final_logit",
            "final_score",
            "reranker_delta",
            "reranker_gate",
            "consistency_delta",
            "consistency_gate",
            "local_delta",
            "local_gate",
            "score_core",
            "score_broad",
            "score_clinical",
            "record_level_base_logit_a9v3",
            "record_level_base_score_a9v3",
            "record_level_aux_logit_head",
            "record_level_aux_score_head",
            "patient_channel_embedding",
            "seizure_channel_embedding",
        ):
            self.assertIn(key, out)
        self.assertEqual(out["final_score"].shape, (2, 4))
        self.assertEqual(out["record_level_base_score_a9v3"].shape, (2, 2, 4))
        self.assertTrue(torch.allclose(out["record_level_base_score"], out["record_level_base_score_a9v3"]))
        self.assertFalse(torch.allclose(out["record_level_aux_score_head"], out["record_level_base_score_a9v3"]))
        self.assertTrue(torch.isfinite(out["final_score"]).all())
        self.assertTrue(torch.all((out["reranker_gate"] >= 0.0) & (out["reranker_gate"] <= 1.0)))

    def test_freeze_policy_freezes_backbone_and_optimizer_filters_params(self):
        torch.manual_seed(13)
        model = NeuroEZCModel(_args())

        counts = exp_ez_hybrid._apply_a9v14_freeze_policy(model, freeze=True)
        trainable_names = {name for name, param in model.named_parameters() if param.requires_grad}

        self.assertGreater(counts["frozen_parameter_count"], 0)
        self.assertGreater(counts["trainable_parameter_count"], 0)
        self.assertTrue(all(name.startswith("a9v14_") for name in trainable_names))
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4)
        optimized_ids = {id(param) for group in optimizer.param_groups for param in group["params"]}
        self.assertEqual(optimized_ids, {id(param) for param in model.parameters() if param.requires_grad})

    def test_loss_includes_a9v14_auxiliary_parts(self):
        torch.manual_seed(11)
        args = _args()
        batch = _batch()
        outputs = NeuroEZCModel(args)(batch)
        exp = object.__new__(Exp_EZHybridLocalization)
        exp.args = args
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

        loss, parts = exp._compute_loss(outputs, batch, torch.tensor(1.0))

        self.assertTrue(torch.isfinite(loss))
        for key in (
            "final_loss",
            "a9v14_base_aux_loss",
            "reranker_delta_l2_loss",
            "consistency_delta_l2_loss",
            "local_delta_l2_loss",
            "mixture_loss",
            "total_loss",
            "positive_core_target_min",
        ):
            self.assertIn(key, parts)
        expected_min_total = (
            parts["final_loss"] * args.final_loss_weight
            + parts["a9v14_base_aux_loss"] * args.base_aux_loss_weight
            + parts["reranker_delta_l2_loss"] * args.reranker_delta_l2_weight
            + parts["consistency_delta_l2_loss"] * args.consistency_delta_l2_weight
            + parts["local_delta_l2_loss"] * args.local_delta_l2_weight
            + parts["mixture_loss"]
        )
        self.assertAlmostEqual(parts["total_loss"], float(loss.detach().cpu()), places=6)
        self.assertGreaterEqual(parts["total_loss"], expected_min_total - 1e-6)
        self.assertGreaterEqual(parts["positive_core_target_min"], 0.5)

    def test_synthetic_run_writes_ledger_and_record_outputs(self):
        samples = [_sample(f"p{i}", offset=float(i)) for i in range(4)]
        patient_index = _patient_index(*(f"p{i}" for i in range(4)))
        run_records = [
            {
                "subject_id": sample["subject_id"],
                "run_id": sample["run_id"],
                "center": patient_index[sample["subject_id"]]["center"],
                "channel_names_norm": sample["channel_names_norm"],
                "labels": sample["labels"],
                "sample": {
                    "sample_id": sample["sample_id"],
                    "window_features": sample["window_features"],
                    "window_adjacency": sample["window_adjacency"],
                    "window_relative_centers_sec": sample["window_relative_centers_sec"],
                },
            }
            for sample in samples
        ]
        splits = [
            {"fold_idx": 1, "train_subjects": ["p0", "p1"], "test_subjects": ["p2", "p3"]},
        ]
        original_data_provider = exp_ez_hybrid.data_provider
        try:
            exp_ez_hybrid.data_provider = lambda args: (run_records, patient_index, splits)
            with tempfile.TemporaryDirectory() as tmpdir:
                output_dir = Path(tmpdir)
                args = _args(output_dir=str(output_dir), reranker_type="deepset", reranker_use_rank_features=False)
                records = Exp_EZHybridLocalization(args).run()

                self.assertEqual(len(records), 2)
                ledger_path = output_dir / "a9v14_prediction_ledger.csv"
                record_path = output_dir / "test_record_channel_predictions_fold_1.csv"
                embedding_path = output_dir / "test_a9v14_hidden_embeddings_fold_1.npz"
                self.assertTrue(ledger_path.exists())
                self.assertTrue(record_path.exists())
                self.assertTrue(embedding_path.exists())
                ledger = pd.read_csv(ledger_path)
                for key in A9V14_LEDGER_REQUIRED_COLUMNS:
                    self.assertIn(key, ledger.columns)
                for key in (
                    "base_score_a9v3",
                    "final_score",
                    "reranker_delta",
                    "reranker_gate",
                    "consistency_delta",
                    "consistency_gate",
                    "local_delta",
                    "local_gate",
                    "score_core",
                    "score_broad",
                    "score_clinical",
                    "neighbor_count",
                    "mean_score_across_seizures",
                    "top1_frequency",
                    "failure_flags",
                ):
                    self.assertIn(key, ledger.columns)
                record_rows = pd.read_csv(record_path)
                for key in (
                    "record_level_base_logit_a9v3",
                    "record_level_base_score_a9v3",
                    "record_level_aux_logit_head",
                    "record_level_aux_score_head",
                ):
                    self.assertIn(key, record_rows.columns)
                self.assertTrue(np.isfinite(ledger["final_score"]).all())
        finally:
            exp_ez_hybrid.data_provider = original_data_provider


if __name__ == "__main__":
    unittest.main()
