from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from neuroez_c.v3_qbc_loss import boundary_hard_ranking_loss, compute_v3_qbc_loss, soft_topk_coverage_loss


class V3QBCLossTests(unittest.TestCase):
    def test_boundary_prefers_correct_ez_ranking(self):
        labels = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
        mask = torch.ones_like(labels, dtype=torch.bool)
        correct, _ = boundary_hard_ranking_loss(torch.tensor([[-2.0, -1.0, 1.0, 2.0]]), labels, mask)
        wrong, _ = boundary_hard_ranking_loss(torch.tensor([[2.0, 1.0, -1.0, -2.0]]), labels, mask)
        self.assertLess(float(correct), float(wrong))

    def test_coverage_prefers_correct_topk(self):
        labels = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
        mask = torch.ones_like(labels, dtype=torch.bool)
        correct, _ = soft_topk_coverage_loss(torch.tensor([[-3.0, 2.0, -2.0, 3.0]]), labels, mask)
        wrong, _ = soft_topk_coverage_loss(torch.tensor([[3.0, -2.0, 2.0, -3.0]]), labels, mask)
        self.assertLess(float(correct), float(wrong))

    def test_losses_are_differentiable(self):
        logits = torch.tensor([[-1.0, 0.0, 1.0]], requires_grad=True)
        labels = torch.tensor([[1.0, 0.0, 0.0]])
        mask = torch.ones_like(labels, dtype=torch.bool)
        boundary, _ = boundary_hard_ranking_loss(logits, labels, mask)
        coverage, _ = soft_topk_coverage_loss(logits, labels, mask)
        (boundary + coverage).backward()
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_single_class_patient_is_skipped_for_boundary(self):
        logits = torch.zeros((1, 3), requires_grad=True)
        loss, diagnostics = boundary_hard_ranking_loss(
            logits, torch.ones((1, 3)), torch.ones((1, 3), dtype=torch.bool)
        )
        self.assertEqual(float(loss.detach()), 0.0)
        self.assertEqual(float(diagnostics["n_boundary_valid_patients"]), 0.0)

    def test_profile_gates_auxiliary_terms_without_dropping_bce(self):
        logits = torch.tensor([[-2.0, 1.0, 2.0]], requires_grad=True)
        batch = {
            "labels_ez": torch.tensor([[1.0, 0.0, 0.0]]),
            "channel_mask": torch.ones((1, 3), dtype=torch.bool),
        }
        outputs = {"final_nez_logit": logits}
        for profile, boundary_enabled, coverage_enabled in (
            ("BCR_BC_ONLY", False, False),
            ("BCR_BOUNDARY_ONLY", True, False),
            ("BCR_COVERAGE_ONLY", False, True),
            ("BCR_BOUNDARY_COVERAGE", True, True),
        ):
            loss, diagnostics = compute_v3_qbc_loss(
                outputs, batch, SimpleNamespace(v3_qbc_profile=profile)
            )
            self.assertEqual(float(diagnostics["boundary_loss_enabled"]), float(boundary_enabled))
            self.assertEqual(float(diagnostics["coverage_loss_enabled"]), float(coverage_enabled))
            self.assertEqual(float(loss.detach()) == 0.0, not (boundary_enabled or coverage_enabled))


if __name__ == "__main__":
    unittest.main()
