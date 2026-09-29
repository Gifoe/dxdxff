"""Metric unit tests without EEG or outcome data."""

from __future__ import annotations

from raw_metrics import ictal_validation, omni_validation


def test_ictal_mean_by_patient_channel():
    rows = {"synthetic": [
        {"channel": ["a", "b"], "label": [1, 0], "score": [0.8, 0.3], "edf": ""},
        {"channel": ["a", "b"], "label": [1, 0], "score": [0.6, 0.1], "edf": ""},
    ]}
    result, private = ictal_validation(rows)
    assert result["patients"] == 1
    assert private["synthetic"]["scores"] == [0.7, 0.2]
    assert result["auroc"] == 1


def test_omni_mean_within_edf_not_across_edfs():
    rows = {"synthetic": [
        {"channel": ["a", "b"], "label": [1, 0], "score": [0.9, 0.1], "edf": "one"},
        {"channel": ["a", "b"], "label": [1, 0], "score": [0.7, 0.3], "edf": "one"},
        {"channel": ["a", "b"], "label": [1, 0], "score": [0.6, 0.4], "edf": "two"},
    ]}
    result, private = omni_validation(rows)
    assert result["edf_channel_units"] == 4
    assert private["synthetic"]["scores"] == [0.8, 0.2, 0.6, 0.4]
    assert result["auroc"] == 1
