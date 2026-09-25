from __future__ import annotations

import numpy as np

from outcome_hifos.cache_schema import load_cache_contract
from task1_baselines.cache_io import task1_feature_records


def test_task1_cache_adapter_maps_patient_ez_labels_to_record_nez_labels() -> None:
    payload = {
        "run_records": [
            {
                "subject_id": "p1",
                "run_id": "r1",
                "channel_names_norm": ["A2", "A1"],
                "sample": {
                    "window_features": np.ones((2, 2, 1), dtype=np.float32),
                    "window_feature_names": ["f"],
                },
            }
        ],
        "patient_index": {
            "p1": {"canonical_channels": ["A1", "A2"], "labels": [1, 0], "source_center": "c1"}
        },
    }
    records = task1_feature_records(load_cache_contract(payload), {"p1"})
    assert len(records) == 1
    assert records[0]["channel_names"] == ["A2", "A1"]
    assert records[0]["labels_nez"].tolist() == [1, 0]


def test_task1_cache_adapter_rejects_missing_old90_patient() -> None:
    payload = {"run_records": [], "patient_index": {}}
    try:
        task1_feature_records(load_cache_contract(payload), {"missing"})
    except ValueError as exc:
        assert "missing" in str(exc).lower()
    else:
        raise AssertionError("missing old-90 patient must fail")


def test_task1_cache_adapter_accepts_explicit_record_aligned_ez_labels() -> None:
    payload = {
        "run_records": [
            {
                "subject_id": "p1",
                "run_id": "r1",
                "channel_names_norm": ["A2", "A1"],
                "labels": [0, 1],
                "sample": {"window_features": np.ones((2, 2, 1)), "window_feature_names": ["f"]},
            }
        ],
        "patient_index": {"p1": {"source_center": "c1"}},
    }
    records = task1_feature_records(load_cache_contract(payload), {"p1"})
    assert records[0]["labels_nez"].tolist() == [1, 0]
