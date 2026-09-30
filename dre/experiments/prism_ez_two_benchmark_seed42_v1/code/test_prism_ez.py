"""Small executable Phase-A audits for PRiSM-EZ's changed critical paths."""
from __future__ import annotations

import numpy as np
import torch

from prism_ez import (FEATURE_DIM, PRiSMEZ, _quantile_statistics, empirical_rank,
                      parameter_audit, spectral_availability, spectral_sketch)
from finalize_ictal_vloo import canonical_patient, choose_excluding, fixed_query


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


def _scalar_empirical_rank(values, channel_mask, window_mask):
    channels, steps, features = values.shape
    out = torch.zeros_like(values)
    for step in range(steps):
        active = torch.nonzero(channel_mask & window_mask[:, step], as_tuple=False).flatten()
        count = int(active.numel())
        if count <= 1:
            continue
        table = values[active, step]
        ordered, order = torch.sort(table, dim=0, stable=True)
        ranks = torch.empty_like(table)
        for feature in range(features):
            cursor = 0
            while cursor < count:
                end = cursor + 1
                while end < count and bool(ordered[end, feature] == ordered[cursor, feature]):
                    end += 1
                average = (float(cursor + 1) + float(end)) / 2.0
                ranks[order[cursor:end, feature], feature] = average
                cursor = end
        out[active, step] = 2.0 * (ranks - 1.0) / float(count - 1) - 1.0
    return out


def test_batched_rank_matches_scalar_rule_with_masks_and_ties():
    torch.manual_seed(19)
    values = torch.randint(-2, 3, (11, 9, 7), dtype=torch.int64).float()
    channel_mask = torch.tensor([True, False, True, True, True, False, True, True, True, True, True])
    window_mask = torch.rand(11, 9) > .25
    window_mask[:, 0] = False
    window_mask[0, 0] = True  # singleton slice must remain zero
    window_mask[:, 1] = True
    assert torch.equal(empirical_rank(values, channel_mask, window_mask),
                       _scalar_empirical_rank(values, channel_mask, window_mask))


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


def test_vloo_record_rows_use_historical_patient_channel_mean():
    rows = [{"channel": ["B", "A"], "label": [0, 1], "score": [.2, .7]},
            {"channel": ["A", "B"], "label": [1, 0], "score": [.9, .4]}]
    assert canonical_patient(rows, ["B", "A"]) == {
        "labels": [0, 1], "scores": [.30000000000000004, .8]}


def test_vloo_one_class_query_matches_historical_nan_semantics():
    from finalize_ictal_vloo import query_metrics
    metrics = query_metrics(np.zeros(4, dtype=np.int8), np.asarray([-.4, -.1, .2, .3]))
    assert np.isnan(metrics["auroc"]) and np.isnan(metrics["ap"])
    assert np.isnan(metrics["mrr"]) and np.isnan(metrics["top1"])
    assert np.isnan(metrics["balanced_accuracy"])
    assert np.isfinite(metrics["macro_f1"]) and metrics["ez_f1"] == 0.0


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
    expected_input = time_value.detach().clone().requires_grad_(True)
    expected_time = _loop_quantiles(expected_input, time_mask, 1)
    actual_time = _quantile_statistics(time_value, time_mask, dimension=1)
    assert torch.allclose(actual_time, expected_time, atol=1e-6, rtol=0.0)
    actual_time.sum().backward(); expected_time.sum().backward()
    assert torch.isfinite(time_value.grad).all()
    assert torch.allclose(time_value.grad, expected_input.grad, atol=1e-6, rtol=0.0)
    record_value = torch.randn(5, 3)
    record_mask = torch.tensor([True, False, True, True, False])
    assert torch.allclose(_quantile_statistics(record_value, record_mask, dimension=0),
                          _loop_quantiles(record_value, record_mask, 0), atol=1e-6, rtol=0.0)
    # The production record path pools all channels in one call.
    record_batch = torch.randn(5, 4, 3)
    batch_mask = torch.tensor([[True, True, False, True], [True, False, True, True],
                               [False, True, True, False], [True, False, False, True],
                               [False, True, True, False]])
    expected_batch = torch.stack([
        _loop_quantiles(record_batch[:, channel], batch_mask[:, channel], 0)
        for channel in range(record_batch.shape[1])
    ])
    assert torch.allclose(_quantile_statistics(record_batch, batch_mask, dimension=0),
                          expected_batch, atol=1e-6, rtol=0.0)


def test_native_supervisor_allows_only_known_host_crashes():
    from run_native_supervisor import is_known_native_failure
    assert is_known_native_failure("returned non-zero exit status 2147483651")
    assert is_known_native_failure("faulting module nvcuda64.dll")
    assert not is_known_native_failure("ValueError: invalid split manifest")


def test_host_watchdog_terminal_status_rules():
    from run_host_watchdog import terminal_state
    assert terminal_state({"status": "COMPLETE"}) == "complete"
    assert terminal_state({"status": "STOPPED_NON_NATIVE_FAILURE"}) == "non_native_failure"
    assert terminal_state({"status": "RUNNING"}) is None
    assert terminal_state(None) is None
