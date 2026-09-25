from __future__ import annotations

import unittest

import torch

from neuroez_c.v3_simple_q10 import V3SimpleQ10Evidence, robust_patient_zscore


class V3SimpleQ10Tests(unittest.TestCase):
    def test_robust_z_is_patient_relative_and_masked(self):
        values = torch.tensor([[1.0, 2.0, 3.0, 99.0], [5.0, 5.0, 5.0, 5.0]])
        mask = torch.tensor([[True, True, True, False], [True, True, True, True]])
        result = robust_patient_zscore(values, mask)
        self.assertAlmostEqual(float(result[0, 1]), 0.0, places=6)
        self.assertEqual(float(result[0, 3]), 0.0)
        self.assertTrue(torch.equal(result[1], torch.zeros(4)))

    def test_q10_uses_true_quantile_and_valid_seizures(self):
        module = V3SimpleQ10Evidence(2)
        module.scorer[0] = torch.nn.Identity()
        with torch.no_grad():
            module.scorer[-1].weight.copy_(torch.tensor([[1.0, 0.0]]))
            module.scorer[-1].bias.zero_()
        embedding = torch.tensor([[[[-2.0, 0.0]], [[0.0, 0.0]], [[2.0, 0.0]]]])
        result = module(
            embedding,
            torch.ones((1, 3), dtype=torch.bool),
            torch.ones((1, 3, 1), dtype=torch.bool),
            torch.ones((1, 1), dtype=torch.bool),
            torch.zeros((1, 1)),
        )
        expected = torch.quantile(torch.sigmoid(torch.tensor([-2.0, 0.0, 2.0])), 0.10)
        self.assertTrue(torch.allclose(result["q10_nez_probability"][0, 0], expected))
        self.assertEqual(int(result["valid_seizure_count"][0, 0]), 3)

    def test_no_valid_seizure_is_neutral(self):
        module = V3SimpleQ10Evidence(3)
        base = torch.tensor([[0.7, -0.2]])
        result = module(
            torch.zeros((1, 2, 2, 3)),
            torch.zeros((1, 2), dtype=torch.bool),
            torch.zeros((1, 2, 2), dtype=torch.bool),
            torch.ones((1, 2), dtype=torch.bool),
            base,
        )
        self.assertTrue(torch.equal(result["final_nez_logit"], base))
        self.assertTrue(torch.equal(result["q10_nez_probability"], torch.full_like(base, 0.5)))

    def test_nez_logit_orientation(self):
        module = V3SimpleQ10Evidence(2)
        base = torch.tensor([[-1.0, 1.0]])
        result = module(
            torch.zeros((1, 1, 2, 2)),
            torch.ones((1, 1), dtype=torch.bool),
            torch.ones((1, 1, 2), dtype=torch.bool),
            torch.ones((1, 2), dtype=torch.bool),
            base,
        )
        self.assertGreater(float(result["score_nez"][0, 1].detach()), float(result["score_nez"][0, 0].detach()))
        self.assertLess(float(result["score_ez"][0, 1].detach()), float(result["score_ez"][0, 0].detach()))


if __name__ == "__main__":
    unittest.main()
