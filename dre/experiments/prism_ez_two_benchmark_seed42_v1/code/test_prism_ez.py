"""Small executable Phase-A audits for PRiSM-EZ's changed critical paths."""
from __future__ import annotations

import numpy as np
import torch

from prism_ez import (FEATURE_DIM, PRiSMEZ, _quantile_statistics, empirical_rank,
                      parameter_audit, spectral_availability, spectral_sketch)
from finalize_ictal_vloo import choose_excluding, fixed_query


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
    # Missing canonical channels must not be assigned fabricated time quantiles.
    record["channel_mask"][-1] = False
    with torch.no_grad():
        logits = model.forward_group([record, record])
    assert torch.isfinite(logits).all() and logits[-1].item() == 0.0


def test_vloo_epoch_choice_excludes_the_target_labels():
    ids = [f"patient_{index:02d}" for index in range(13)]
    labels = [0, 1, 0, 1]
    good, bad = [0.1, 0.9, 0.2, 0.8], [0.9, 0.1, 0.8, 0.2]
    first = {patient: {"labels": labels, "scores": good} for patient in ids}
    second = {patient: {"labels": labels, "scores": bad} for patient in ids}
    # This target favors epoch 2, but it must not influence its own selection.
    first[ids[0]] = {"labels": labels, "scores": bad}
    second[ids[0]] = {"labels": labels, "scores": good}
    epoch, threshold = choose_excluding([first, second], ids, ids[0])
    assert epoch == 0 and 0.05 <= threshold <= 0.95
    assert np.array_equal(fixed_query(8, 1, ids[0], 0), fixed_query(8, 1, ids[0], 0))


def _loop_quantiles(value, valid, dimension):
    if dimension == 1:
        return torch.stack([torch.cat((chosen.mean(0), torch.quantile(chosen, .25, dim=0),
                                       torch.quantile(chosen, .50, dim=0),
                                       torch.quantile(chosen, .75, dim=0), chosen.max(0).values))
                            for chosen in (value[channel, valid[channel]] for channel in range(len(value)))])
    chosen = value[valid]
    return torch.cat((chosen.mean(0), torch.quantile(chosen, .25, dim=0),
                      torch.quantile(chosen, .50, dim=0), torch.quantile(chosen, .75, dim=0),
                      chosen.max(0).values))


def test_vectorized_masked_quantiles_match_the_original_rule_and_gradients():
    torch.manual_seed(7)
    time_value = torch.randn(4, 7, 3, requires_grad=True)
    time_mask = torch.tensor([[True, True, False, True, False, False, False],
                              [False, True, True, True, True, True, True],
                              [True, False, True, False, True, False, True],
                              [True, False, False, False, False, False, False]])
    expected_time = _loop_quantiles(time_value, time_mask, 1)
    actual_time = _quantile_statistics(time_value, time_mask, dimension=1)
    assert torch.allclose(actual_time, expected_time, atol=1e-6, rtol=0.0)
    actual_time.sum().backward()
    assert torch.isfinite(time_value.grad).all()
    record_value = torch.randn(5, 3)
    record_mask = torch.tensor([True, False, True, True, False])
    assert torch.allclose(_quantile_statistics(record_value, record_mask, dimension=0),
                          _loop_quantiles(record_value, record_mask, 0), atol=1e-6, rtol=0.0)


def test_native_supervisor_allows_only_known_host_crashes():
    from run_native_supervisor import is_known_native_failure
    assert is_known_native_failure("returned non-zero exit status 2147483651")
    assert is_known_native_failure("faulting module nvcuda64.dll")
    assert not is_known_native_failure("ValueError: invalid split manifest")
