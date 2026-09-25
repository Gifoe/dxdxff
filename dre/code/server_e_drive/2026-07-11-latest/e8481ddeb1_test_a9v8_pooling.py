"""A9v8 pooling tests — default values and numeric semantics."""

import unittest

import numpy as np
import torch

from run_neuroez_c import build_parser
from seizure_aggregator import CrossSeizureMILAggregator
from temporal_encoder import ChannelTemporalEncoder


class A9v8PoolingTests(unittest.TestCase):
    # ---- CLI defaults ----

    def test_cli_default_temporal_pooling_is_mean(self):
        p = build_parser()
        a = p.parse_args([])
        self.assertEqual(a.temporal_pooling, "mean")

    def test_cli_default_record_pooling_is_mean(self):
        p = build_parser()
        a = p.parse_args([])
        self.assertEqual(a.record_pooling, "mean")

    def test_cli_default_record_pooling_top_p_is_030(self):
        p = build_parser()
        a = p.parse_args([])
        self.assertEqual(a.record_pooling_top_p, 0.30)

    def test_cli_default_record_pooling_alpha_is_070(self):
        p = build_parser()
        a = p.parse_args([])
        self.assertEqual(a.record_pooling_alpha, 0.70)

    # ---- seizure_aggregator defaults ----

    def test_aggregator_default_top_p(self):
        agg = CrossSeizureMILAggregator(model_dim=8)
        self.assertEqual(agg.top_p, 0.30)

    def test_aggregator_default_alpha(self):
        agg = CrossSeizureMILAggregator(model_dim=8)
        self.assertEqual(agg.alpha, 0.70)

    # ---- numeric semantics ----

    def _forward(self, agg, x, B=2, S=3, C=5, D=4):
        """Helper: forward with correct args."""
        sz_mask = torch.ones(B, S, dtype=torch.bool)
        sz_ch_mask = torch.ones(B, S, C, dtype=torch.bool)
        out, _weights = agg(x, sz_mask, sz_ch_mask)
        return out

    def test_mean_pooling_finite(self):
        agg = CrossSeizureMILAggregator(model_dim=4, pooling="mean")
        x = torch.randn(2, 3, 5, 4)
        out = self._forward(agg, x)
        self.assertEqual(out.shape, (2, 5, 8))
        self.assertTrue(torch.isfinite(out).all())

    def test_top_p_mean_finite(self):
        agg = CrossSeizureMILAggregator(model_dim=4, pooling="top_p_mean", top_p=0.3)
        x = torch.ones(2, 10, 3, 4)
        out = self._forward(agg, x, S=10, C=3)
        self.assertTrue(torch.isfinite(out).all())

    def test_logsumexp_no_nan(self):
        x = torch.randn(2, 3, 4, 6) + 10.0
        tau = 0.10
        se = tau * torch.logsumexp(x / tau, dim=1)
        self.assertFalse(torch.isnan(se).any())
        self.assertFalse(torch.isinf(se).any())

    def test_noisy_or_finite(self):
        agg = CrossSeizureMILAggregator(model_dim=4, pooling="noisy_or")
        x = torch.randn(2, 3, 5, 4)
        sz_mask = torch.ones(2, 3, dtype=torch.bool)
        sz_ch_mask = torch.ones(2, 3, 5, dtype=torch.bool)
        sz_ch_mask[0, 2, :] = False
        out, _ = agg(x, sz_mask, sz_ch_mask)
        self.assertTrue(torch.isfinite(out).all())

    def test_top_p_mean_median_hybrid_finite(self):
        agg = CrossSeizureMILAggregator(model_dim=4, pooling="top_p_mean_median_hybrid",
                                         top_p=0.30, alpha=0.70)
        x = torch.randn(2, 5, 3, 4)
        out = self._forward(agg, x, S=5, C=3)
        self.assertTrue(torch.isfinite(out).all())
        self.assertEqual(out.shape, (2, 3, 8))

    # ---- temporal encoder numeric semantics ----

    def test_temporal_mean_equals_masked_mean(self):
        enc = ChannelTemporalEncoder(model_dim=1, pooling="mean")
        # shape [B=1, S=1, T=3, C=1, D=1] — explicit
        x = torch.zeros(1, 1, 3, 1, 1)
        x[0, 0, 0, 0, 0] = 1.0
        x[0, 0, 1, 0, 0] = 3.0
        x[0, 0, 2, 0, 0] = 100.0
        wm = torch.tensor([[[True, True, False]]])
        sz_ch_mask = torch.ones(1, 1, 1, dtype=torch.bool)
        out, _ = enc(x, sz_ch_mask, window_mask=wm)
        # mean of unmasked windows [1, 3] = 2
        self.assertAlmostEqual(float(out[0, 0, 0, 0]), 2.0, places=5)

    def test_temporal_top_p_mean_selects_top_fraction(self):
        enc = ChannelTemporalEncoder(model_dim=1, pooling="top_p_mean", top_p=0.5)
        x = torch.zeros(1, 1, 4, 1, 1)
        x[0, 0, 0, 0, 0] = 1.0
        x[0, 0, 1, 0, 0] = 2.0
        x[0, 0, 2, 0, 0] = 10.0
        x[0, 0, 3, 0, 0] = 20.0
        wm = torch.ones(1, 1, 4, dtype=torch.bool)
        sz_ch_mask = torch.ones(1, 1, 1, dtype=torch.bool)
        out, _ = enc(x, sz_ch_mask, window_mask=wm)
        # top 2 (50% of 4) = mean(10, 20) = 15
        self.assertAlmostEqual(float(out[0, 0, 0, 0]), 15.0, places=5)


if __name__ == "__main__":
    unittest.main()
