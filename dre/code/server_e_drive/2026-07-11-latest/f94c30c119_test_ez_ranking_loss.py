from __future__ import annotations

import unittest

import torch

from exp_ez_hybrid import _ez_pairwise_ranking_loss


class EZPairwiseRankingLossTests(unittest.TestCase):
    def test_penalizes_ez_channels_ranked_below_nez_channels(self):
        logits = torch.tensor([[3.0, -3.0]], requires_grad=True)
        labels_ez = torch.tensor([[1.0, 0.0]])
        channel_mask = torch.tensor([[True, True]])

        loss = _ez_pairwise_ranking_loss(logits, labels_ez, channel_mask, margin=0.10)

        expected_score_ez_ez = 1.0 - torch.sigmoid(logits[0, 0])
        expected_score_ez_nez = 1.0 - torch.sigmoid(logits[0, 1])
        expected = torch.relu(torch.tensor(0.10) - expected_score_ez_ez + expected_score_ez_nez)
        self.assertTrue(torch.allclose(loss, expected))
        self.assertGreater(float(loss.detach()), 0.0)
        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.any(logits.grad != 0.0))

    def test_averages_only_patients_with_both_ez_and_nez_channels(self):
        logits = torch.tensor(
            [
                [-4.0, 4.0, 0.0],
                [-1.0, -2.0, -3.0],
            ],
            requires_grad=True,
        )
        labels_ez = torch.tensor(
            [
                [1.0, 0.0, -1.0],
                [1.0, 1.0, 1.0],
            ]
        )
        channel_mask = torch.tensor(
            [
                [True, True, True],
                [True, True, True],
            ]
        )

        loss = _ez_pairwise_ranking_loss(logits, labels_ez, channel_mask, margin=0.10)

        score_ez = 1.0 - torch.sigmoid(logits)
        expected = torch.relu(torch.tensor(0.10) - score_ez[0, 0] + score_ez[0, 1])
        self.assertTrue(torch.allclose(loss, expected))

    def test_returns_connected_zero_when_no_valid_pairs_exist(self):
        logits = torch.tensor([[0.0, 1.0]], requires_grad=True)
        labels_ez = torch.tensor([[1.0, 1.0]])
        channel_mask = torch.tensor([[True, True]])

        loss = _ez_pairwise_ranking_loss(logits, labels_ez, channel_mask, margin=0.10)

        self.assertEqual(float(loss.detach()), 0.0)
        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.all(logits.grad == 0.0))


if __name__ == "__main__":
    unittest.main()
