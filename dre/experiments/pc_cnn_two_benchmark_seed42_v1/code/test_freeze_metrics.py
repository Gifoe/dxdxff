"""Synthetic frozen-threshold and official metric-unit tests."""

from __future__ import annotations

import json

from evaluate_omni_frozen import scored
from freeze_before_test import validation_threshold


def test_validation_threshold_does_not_need_test_data(tmp_path):
    path = tmp_path / "validation_private.json"
    path.write_text(json.dumps({"private_patient_scores": {
        "synthetic": {"labels": [0, 0, 1, 1],
                      "scores": [0.1, 0.2, 0.7, 0.8]}}}))
    selected = validation_threshold(path)
    assert selected["validation_macro_f1"] == 1
    assert 0.2 < selected["threshold"] <= 0.7
    assert selected["pooled_edf_channel_units"] == 4


def test_omni_pooled_unit_and_patient_ranking():
    private = {"synthetic": {"labels": [1, 0, 1, 0],
                             "scores": [0.9, 0.1, 0.8, 0.2],
                             "edf": ["a", "a", "b", "b"],
                             "channel": ["X", "Y", "X", "Y"]}}
    result = scored(private, ["synthetic"], 0.5)
    assert result["edf_channel_units"] == 4
    assert result["patients"] == 1
    assert result["auroc"] == 1
    assert result["mrr"] == 1
    assert result["tp"] == 2 and result["tn"] == 2
