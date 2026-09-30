import numpy as np
import torch

from common import relative_features, score_from_segment_logits, sigmoid
from official_embedding import ResidualHead


def test_head_identity_and_size():
    torch.manual_seed(42)
    head = ResidualHead().eval()
    value = head(torch.randn(7, 96))
    assert torch.equal(value, torch.zeros_like(value))
    assert sum(p.numel() for p in head.parameters()) == 1761


def test_leave_one_out_and_ranks():
    value = np.stack([np.full(32, 1), np.full(32, 3), np.full(32, 9)]).astype(np.float32)
    difference, ranks, fallback = relative_features(value)
    assert not fallback
    np.testing.assert_allclose(difference[:, 0], [-5, -2, 7])
    np.testing.assert_allclose(ranks[:, 0], [-1, 0, 1])


def test_small_set_fallback():
    value = np.stack([np.zeros(32), np.ones(32)]).astype(np.float32)
    difference, ranks, fallback = relative_features(value)
    assert fallback
    np.testing.assert_allclose(difference[:, 0], [-0.5, 0.5])
    np.testing.assert_allclose(ranks[:, 0], [-1, 1])


def test_official_aggregation_identity():
    logits = [np.asarray([-2.0, 0.0, 3.0]), np.asarray([1.0])]
    observed = score_from_segment_logits(logits, np.zeros(2))
    expected = np.asarray([sigmoid(logits[0]).mean(), sigmoid(logits[1]).mean()])
    np.testing.assert_allclose(observed, expected, rtol=0, atol=0)
