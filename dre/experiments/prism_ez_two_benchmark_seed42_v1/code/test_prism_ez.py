"""Small executable Phase-A audits for PRiSM-EZ's changed critical paths."""
from __future__ import annotations

import numpy as np
import torch

from prism_ez import (FEATURE_DIM, PRiSMEZ, empirical_rank, parameter_audit,
                      spectral_availability, spectral_sketch)


def test_parameter_gate():
    audit = parameter_audit()
    assert audit["budget_pass"] and audit["no_batchnorm"]
    assert 50000 <= audit["trainable_parameters"] <= 90000


def test_rank_single_ties_and_permutation_equivariance():
    values = torch.tensor([[[1.0, 2.0], [3.0, 0.0]], [[1.0, 8.0], [5.0, 0.0]],
                           [[9.0, 4.0], [7.0, 0.0]]])
    mask = torch.tensor([True, True, True])
    windows = torch.ones((3, 2), dtype=torch.bool)
    rank = empirical_rank(values, mask, windows)
    # Tied first feature gets exactly the average rank: (1+2)/2 -> -0.5.
    assert torch.allclose(rank[:2, 0, 0], torch.tensor([-0.5, -0.5]))
    single = empirical_rank(values[:1], mask[:1], windows[:1])
    assert torch.equal(single, torch.zeros_like(single))
    permutation = torch.tensor([2, 0, 1])
    assert torch.allclose(empirical_rank(values[permutation], mask[permutation], windows[permutation]),
                          rank[permutation])


def test_spectral_native_frequency_mask_and_finiteness():
    waveform = np.random.default_rng(42).normal(size=(2, 15000)).astype(np.float32)
    valid = np.ones((2, 59), dtype=bool)
    sketch, availability = spectral_sketch(waveform, 250.0, valid)
    physical = spectral_availability(250.0)
    assert np.all(sketch[:, :, ~physical] == 0) and not availability[:, :, ~physical].any()
    assert np.isfinite(sketch).all()


def test_variable_record_pooling_and_valid_window_mask():
    torch.manual_seed(42)
    model = PRiSMEZ().eval()
    record = {"features": torch.randn(3, 59, FEATURE_DIM),
              "channel_mask": torch.tensor([True, True, True]),
              "window_mask": torch.ones(3, 59, dtype=torch.bool)}
    record["window_mask"][:, -1] = False
    with torch.no_grad():
        logits = model.forward_group([record, record])
    assert logits.shape == (3,) and torch.isfinite(logits).all()
