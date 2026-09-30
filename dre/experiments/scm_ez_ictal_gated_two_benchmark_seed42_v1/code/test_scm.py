from __future__ import annotations

import numpy as np
import torch

from scm_core import (BIN_EDGES_HZ, SCMEZ, aggregate_states, parameter_count,
                      patient_references, scm_matrices, state_membership,
                      visible_bins, window_spectra)


def test_state_boundaries_and_real_pre_split():
    centers = np.arange(-29, 30, dtype=float)
    membership = state_membership(centers)
    assert [int(np.sum(membership == state)) for state in range(6)] == [15, 14, 5, 5, 20, 0]
    assert membership[centers == 0][0] == 2
    assert membership[centers == 5][0] == 3
    assert membership[centers == 10][0] == 4


def test_physical_frequency_availability():
    assert len(BIN_EDGES_HZ) == 17
    mask = visible_bins(250)
    assert mask.sum() == 13
    assert mask[:13].all() and not mask[13:].any()


def test_explicit_center_raw_alignment_and_spectrum():
    fs = 250
    time = np.arange(15000) / fs
    raw = np.stack((np.sin(2 * np.pi * 10 * time), np.sin(2 * np.pi * 80 * time))).astype(np.float32)
    centers = np.arange(-29, 30, dtype=float)
    value, valid = window_spectra(raw, fs, centers, onset_sample=7500,
                                  valid_start=0, valid_samples=15000)
    assert value.shape == (2, 59, 16) and valid.all()
    assert np.all(value[..., ~visible_bins(fs)] == 0)


def test_absent_state_and_matrix_masks():
    rng = np.random.default_rng(42)
    spectra = rng.normal(size=(4, 59, 16)).astype(np.float32)
    centers = np.arange(-29, 30, dtype=float)
    states, valid = aggregate_states(spectra, centers, np.ones((4, 59), bool), visible_bins(250))
    assert not valid[:, 5].any()
    matrix, fallback = scm_matrices(states, valid)
    assert matrix.shape == (4, 34, 6, 6)
    assert np.all(matrix[:, :32, 5, :] == 0)
    assert fallback == 0  # an entirely absent state is invalid, not an all-valid fallback


def test_leave_current_channel_out_reference():
    states = np.zeros((3, 6, 16), dtype=np.float32)
    states[0] = 1; states[1] = 3; states[2] = 9
    valid = np.ones((3, 6), dtype=bool)
    reference, reference_valid, fallback = patient_references(states, valid)
    assert fallback == 0 and reference_valid.all()
    assert np.all(reference[0] == 6)  # median of channels with values 3 and 9
    assert np.all(reference[1] == 5)  # median of 1 and 9
    assert np.all(reference[2] == 2)  # median of 1 and 3


def test_model_exact_parameter_budget_and_output_shape():
    assert parameter_count() == 6353
    model = SCMEZ()
    x = torch.randn(7, 34, 6, 6)
    index = torch.tensor([0, 1, 2, 0, 1, 2, 2])
    y = model(x, index, 3)
    assert y.shape == (3,)
    assert parameter_count() < 15000
