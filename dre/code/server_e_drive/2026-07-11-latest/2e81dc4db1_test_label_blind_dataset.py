from __future__ import annotations

import numpy as np
import pytest

from outcome_hifos.cache_schema import load_cache_contract
from outcome_hifos.dataset import DatasetContractError, build_outcome_patient_examples
from outcome_hifos.leakage_guard import assert_label_blind_tree


def _cache(local_channels: list[str]) -> dict:
    return {
        "run_records": [
            {
                "subject_id": "p1",
                "run_id": "r1",
                "channel_names_norm": local_channels,
                "labels": np.asarray([1, 0], dtype=np.float32),
                "sample": {
                    "sample_id": "s1",
                    "window_features": np.arange(2 * len(local_channels) * 3, dtype=np.float32).reshape(2, len(local_channels), 3),
                    "window_relative_centers_sec": np.asarray([2.0, -1.0], dtype=np.float32),
                    "clinical_ez": ["A1"],
                },
                "source_center": "c1",
            }
        ],
        "patient_index": {
            "p1": {
                "canonical_channels": ["A1", "A2", "A3"],
                "labels": np.asarray([1, 0, 0], dtype=np.float32),
                "label_mask": np.ones(3, dtype=bool),
                "outcome_group": "success",
                "source_center": "c1",
            }
        },
    }


def test_example_has_target_but_model_input_is_label_blind() -> None:
    examples = build_outcome_patient_examples(load_cache_contract(_cache(["A1", "A3"])))
    assert len(examples) == 1
    example = examples[0]
    assert example.target == 1.0
    assert_label_blind_tree(example.model_input, stage="example")
    assert "outcome" not in example.model_input
    assert "labels" not in example.model_input


def test_missing_run_channel_sets_mask_and_windows_are_time_sorted() -> None:
    example = build_outcome_patient_examples(load_cache_contract(_cache(["A1", "A3"])))[0]
    assert example.model_input["seizure_channel_mask"][0].tolist() == [True, False, True]
    assert example.model_input["window_centers"][0].tolist() == [-1.0, 2.0]
    aligned = example.model_input["feature_runs"][0]
    assert np.all(aligned[:, 1, :] == 0.0)


def test_duplicate_local_channel_name_fails_fast() -> None:
    with pytest.raises(DatasetContractError, match="duplicate local channel"):
        build_outcome_patient_examples(load_cache_contract(_cache(["A1", "A1"])))


def test_conflicting_duplicate_run_is_excluded_with_audit_reason() -> None:
    payload = _cache(["A1", "A3"])
    duplicate = dict(payload["run_records"][0])
    duplicate["sample"] = dict(duplicate["sample"])
    duplicate["sample"]["window_features"] = duplicate["sample"]["window_features"] + 100.0
    payload["run_records"].append(duplicate)
    exclusions: list[dict] = []
    examples = build_outcome_patient_examples(load_cache_contract(payload), exclusion_audit=exclusions)
    assert examples == []
    assert exclusions == [
        {
            "subject_id": "p1",
            "run_id": "r1",
            "sample_id": "s1",
            "reason": "duplicate_run_conflict_excluded",
            "duplicate_count": 2,
        }
    ]
