from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch

from exp_ez_hybrid import (
    build_lcbo_target_lookup,
    compute_a9v8_lcbo_loss,
    load_lcbo_fold_targets,
)
from neuroez_c.model import NeuroEZCModel
from run_neuroez_c import build_parser


def _args(**overrides):
    values = {
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
        "use_a9v8_lcbo": True,
        "eval_score_fusion_gamma": 0.10,
        "lambda_core_rank": 0.2,
        "lambda_soft_mrr": 0.1,
        "lambda_subset": 0.05,
        "lambda_core_distill": 0.1,
        "core_rank_margin": 0.1,
        "soft_mrr_tau": 0.1,
        "subset_eps": 0.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _batch() -> dict:
    return {
        "b0_features": torch.randn(2, 1, 3, 4, 36),
        "labels": torch.tensor([[1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]]),
        "labels_nez": torch.tensor([[0.0, 1.0, 1.0, 0.0], [0.0, 1.0, 1.0, 0.0]]),
        "labels_ez": torch.tensor([[1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]]),
        "channel_mask": torch.ones(2, 4, dtype=torch.bool),
        "seizure_mask": torch.ones(2, 1, dtype=torch.bool),
        "seizure_channel_mask": torch.ones(2, 1, 4, dtype=torch.bool),
        "window_mask": torch.ones(2, 1, 3, dtype=torch.bool),
        "center_id": torch.tensor([1, 3], dtype=torch.long),
        "subject_id": ["p1", "p2"],
        "canonical_channels": [["a", "b", "c", "d"], ["a", "b", "c", "d"]],
    }


class A9v8LCBOMeanStage1Tests(unittest.TestCase):
    def test_cli_exposes_lcbo_flags(self):
        args = build_parser().parse_args(["--use_a9v8_lcbo", "--latent_core_target_dir", "targets"])

        self.assertTrue(args.use_a9v8_lcbo)
        self.assertEqual(args.latent_core_target_dir, "targets")
        self.assertAlmostEqual(args.eval_score_fusion_gamma, 0.10)

    def test_model_forward_outputs_core_broad_eval_scores(self):
        out = NeuroEZCModel(_args())(_batch())

        self.assertIn("score_core", out)
        self.assertIn("score_broad", out)
        self.assertIn("score_eval", out)
        self.assertEqual(out["score_core"].shape, (2, 4))
        self.assertTrue(torch.allclose(out["score_ez"], out["score_eval"]))

    def test_fold_target_loader_rejects_heldout_fold_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            target_dir = Path(tmpdir)
            pd.DataFrame(
                [
                    {"patient_id": "p1", "subject_id": "p1", "fold_id": 0, "channel_name": "a", "label_ez": 1, "pseudo_core_q": 0.8, "a9v3_oof_score": 0.7},
                    {"patient_id": "p2", "subject_id": "p2", "fold_id": 1, "channel_name": "a", "label_ez": 1, "pseudo_core_q": 0.8, "a9v3_oof_score": 0.7},
                ]
            ).to_csv(target_dir / "latent_core_targets_fold0_train.csv", index=False)

            with self.assertRaisesRegex(ValueError, "heldout fold"):
                load_lcbo_fold_targets(target_dir, fold_id=0)

    def test_lcbo_loss_returns_required_parts(self):
        batch = _batch()
        out = NeuroEZCModel(_args())(batch)
        target_df = pd.DataFrame(
            [
                {"patient_id": "p1", "subject_id": "p1", "fold_id": 1, "channel_name": "a", "label_ez": 1, "pseudo_core_q": 0.8, "a9v3_oof_score": 0.7},
                {"patient_id": "p1", "subject_id": "p1", "fold_id": 1, "channel_name": "b", "label_ez": 0, "pseudo_core_q": 0.0, "a9v3_oof_score": 0.1},
                {"patient_id": "p2", "subject_id": "p2", "fold_id": 2, "channel_name": "a", "label_ez": 1, "pseudo_core_q": 0.6, "a9v3_oof_score": 0.8},
                {"patient_id": "p2", "subject_id": "p2", "fold_id": 2, "channel_name": "b", "label_ez": 0, "pseudo_core_q": 0.0, "a9v3_oof_score": 0.2},
            ]
        )

        loss, parts = compute_a9v8_lcbo_loss(out, batch, build_lcbo_target_lookup(target_df), _args())

        self.assertTrue(torch.isfinite(loss))
        for key in (
            "mean_broad_bce_loss",
            "mean_core_rank_loss",
            "mean_soft_mrr_loss",
            "mean_subset_loss",
            "mean_core_distill_loss",
            "score_core_broad_corr",
            "subset_violation_rate",
        ):
            self.assertIn(key, parts)


if __name__ == "__main__":
    unittest.main()
