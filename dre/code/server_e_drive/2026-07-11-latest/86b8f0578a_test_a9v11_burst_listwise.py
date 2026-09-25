from __future__ import annotations

import argparse
import inspect
import unittest
from pathlib import Path

import torch

from exp_ez_hybrid import (
    _a9v11_first_positive_rank_loss,
    _a9v11_group_reduce,
    _a9v11_hard_pairwise_loss,
    _a9v11_soft_topk_coverage_loss,
    _compute_supervised_loss,
    Exp_EZHybridLocalization,
)
from neuroez_c.model import NeuroEZCModel
from run_neuroez_c import build_parser
from temporal_encoder import ChannelTemporalEncoder


class A9v11BurstListwiseTests(unittest.TestCase):
    def test_soft_topk_coverage_loss_near_zero_for_perfect_topk(self):
        scores = torch.tensor([[10.0, 9.0, 0.0, -1.0]], requires_grad=True)
        labels_ez = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
        mask = torch.tensor([[True, True, True, True]])

        loss = _a9v11_soft_topk_coverage_loss(
            scores,
            labels_ez,
            mask,
            soft_rank_tau=0.05,
            soft_topk_tau=0.10,
        )

        self.assertTrue(torch.isfinite(loss))
        self.assertLess(float(loss.detach()), 0.10)
        loss.backward()
        self.assertIsNotNone(scores.grad)

    def test_soft_topk_coverage_loss_prefers_covering_all_positives(self):
        labels_ez = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
        mask = torch.tensor([[True, True, True, True]])
        scores_good = torch.tensor([[4.0, 3.0, 2.0, 1.0]], requires_grad=True)
        scores_bad = torch.tensor([[4.0, 0.0, 3.0, 2.0]], requires_grad=True)

        loss_good = _a9v11_soft_topk_coverage_loss(scores_good, labels_ez, mask)
        loss_bad = _a9v11_soft_topk_coverage_loss(scores_bad, labels_ez, mask)

        self.assertLess(float(loss_good.detach()), float(loss_bad.detach()))

    def test_soft_topk_coverage_loss_handles_single_positive(self):
        scores = torch.tensor([[2.0, 1.0, 0.0]], requires_grad=True)
        labels_ez = torch.tensor([[1.0, 0.0, 0.0]])
        mask = torch.tensor([[True, True, True]])

        loss = _a9v11_soft_topk_coverage_loss(scores, labels_ez, mask)

        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(scores.grad)

    def test_soft_topk_coverage_loss_handles_multi_positive(self):
        scores = torch.tensor([[1.0, 0.8, 0.6, 0.4]], requires_grad=True)
        labels_ez = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
        mask = torch.tensor([[True, True, True, True]])

        loss = _a9v11_soft_topk_coverage_loss(scores, labels_ez, mask)

        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.any(scores.grad != 0.0))

    def test_hard_pairwise_loss_prefers_positive_above_negative(self):
        labels_ez = torch.tensor([[1.0, 0.0, 0.0]])
        mask = torch.tensor([[True, True, True]])
        good = torch.tensor([[3.0, 1.0, 0.0]])
        bad = torch.tensor([[0.0, 3.0, 1.0]])

        self.assertLess(
            float(_a9v11_hard_pairwise_loss(good, labels_ez, mask).detach()),
            float(_a9v11_hard_pairwise_loss(bad, labels_ez, mask).detach()),
        )

    def test_dynamic_pooling_shapes(self):
        x = torch.randn(2, 3, 5, 4, 8)
        seizure_channel_mask = torch.ones(2, 3, 4, dtype=torch.bool)
        window_mask = torch.ones(2, 3, 5, dtype=torch.bool)
        for mode in ("mean", "mean_max_top20", "mean_logsumexp_early", "gated_dynamic"):
            encoder = ChannelTemporalEncoder(
                model_dim=8,
                channel_pooling_mode=mode,
                lse_pool_tau=0.7,
                early_pool_frac=0.25,
            )
            pooled, weights = encoder(x, seizure_channel_mask, window_mask=window_mask)
            self.assertEqual(tuple(pooled.shape), (2, 3, 4, 8))
            self.assertEqual(tuple(weights.shape), (2, 3, 4, 5))

    def test_dynamic_pooling_no_label_or_center_input(self):
        sig = inspect.signature(ChannelTemporalEncoder.forward)

        for forbidden in ("center", "center_id", "label", "labels_ez", "n_ez", "ez_fraction", "true_ez_count"):
            self.assertNotIn(forbidden, sig.parameters)

    def test_group_robust_center_not_feature(self):
        source = inspect.getsource(NeuroEZCModel.forward) + inspect.getsource(NeuroEZCModel._forward_single_expert)

        self.assertNotIn("group_robust_mode", source)
        self.assertNotIn("group_dro", source)

    def test_center_group_dro_updates_persistent_weights(self):
        args = argparse.Namespace(group_robust_mode="center_group_dro", group_dro_eta=1.0, group_dro_min_count=1)
        patient_losses = torch.tensor([0.1, 2.0])
        patient_valid = torch.tensor([True, True])
        batch = {"center_id": torch.tensor([0, 1])}
        group_weights = torch.full((5,), 0.2)

        _, parts = _a9v11_group_reduce(
            patient_losses,
            patient_valid,
            batch,
            args,
            group_weights=group_weights,
            update_group_dro=True,
        )

        self.assertGreater(float(group_weights[1]), float(group_weights[0]))
        self.assertAlmostEqual(float(group_weights.sum()), 1.0, places=6)
        self.assertEqual(parts["group_robust_mode_effective"], "persistent_center_group_dro")

    def test_a9v11_group_weights_reset_per_fold(self):
        source = inspect.getsource(Exp_EZHybridLocalization.run)
        seed_idx = source.index("_set_random_seed(base_seed + fold_idx)")
        reset_idx = source.index("self.a9v11_group_weights = torch.full")
        split_idx = source.index("split_train_val_subjects")

        self.assertLess(seed_idx, reset_idx)
        self.assertLess(reset_idx, split_idx)
        self.assertIn("reset A9v11 group-DRO weights to uniform", source)

    def test_a9v11_protocol_audit_fields(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "--loss_mode",
                "patient_dynamic_listwise",
                "--channel_pooling_mode",
                "gated_dynamic",
                "--soft_topk_weight",
                "0.2",
                "--hard_pairwise_weight",
                "0.05",
                "--first_rank_weight",
                "0.05",
                "--group_robust_mode",
                "center_balanced",
            ]
        )
        self.assertEqual(args.channel_pooling_mode, "gated_dynamic")
        self.assertEqual(args.loss_mode, "patient_dynamic_listwise")

        batch = {
            "labels": torch.tensor([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]),
            "labels_ez": torch.tensor([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]),
            "channel_mask": torch.tensor([[True, True, True], [True, True, True]]),
            "center_id": torch.tensor([0, 1]),
        }
        logits = torch.tensor([[2.0, 0.0, -1.0], [2.0, 1.0, 0.0]], requires_grad=True)
        loss, parts = _compute_supervised_loss(logits, batch, args, torch.tensor(1.0))

        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(parts["loss_mode"], "patient_dynamic_listwise")
        self.assertEqual(parts["soft_topk_weight"], 0.2)
        self.assertEqual(parts["hard_pairwise_weight"], 0.05)
        self.assertEqual(parts["first_rank_weight"], 0.05)
        self.assertEqual(parts["group_robust_mode"], "center_balanced")
        self.assertTrue(parts["no_center_features"])

    def test_runner_supports_base_args_and_config_names(self):
        script = Path("scripts/run_a9v11_burst_listwise_grid.ps1").read_text(encoding="utf-8")

        self.assertIn("BaseRunArgsPath", script)
        self.assertIn("ConfigNames", script)
        self.assertIn("a9v11_base_args_diff.json", script)

    def test_runner_skips_empty_string_base_args(self):
        script = Path("scripts/run_a9v11_burst_listwise_grid.ps1").read_text(encoding="utf-8")

        self.assertIn("IsNullOrWhiteSpace", script)


if __name__ == "__main__":
    unittest.main()
