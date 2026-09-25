from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from task1_baselines.feature_aggregation import build_channel_feature_table
from task1_baselines.fold_protocol import Task1ProtocolError, freeze_v3_fold_manifest


def test_v3_manifest_freezes_unique_patient_fold_and_nez_direction() -> None:
    rows = []
    for fold in range(1, 6):
        for patient in range(2):
            subject = f"c{fold}:p{patient}"
            rows.extend(
                [
                    {"subject_id": subject, "center": f"c{fold}", "fold_idx": fold, "channel_name": "A1", "true_ez": 1},
                    {"subject_id": subject, "center": f"c{fold}", "fold_idx": fold, "channel_name": "A2", "true_ez": 0},
                ]
            )
    manifest = freeze_v3_fold_manifest(pd.DataFrame(rows), expected_subjects=10)
    assert manifest["subject_id"].is_unique
    assert sorted(manifest["outer_fold"].unique()) == [1, 2, 3, 4, 5]
    assert manifest.attrs["label_semantics"] == {"NEZ": 1, "EZ": 0}


def test_v3_manifest_rejects_patient_in_multiple_folds() -> None:
    ledger = pd.DataFrame(
        [
            {"subject_id": "p1", "center": "c", "fold_idx": 1},
            {"subject_id": "p1", "center": "c", "fold_idx": 2},
        ]
    )
    with pytest.raises(Task1ProtocolError, match="multiple folds"):
        freeze_v3_fold_manifest(ledger, expected_subjects=1, expected_folds=None)


def test_channel_features_aggregate_windows_then_seizures() -> None:
    records = [
        {
            "subject_id": "p1",
            "run_id": "s1",
            "center": "c1",
            "channel_names": ["A1", "A2"],
            "features": np.asarray([[[1.0], [10.0]], [[3.0], [14.0]]]),
            "labels_nez": np.asarray([1, 0]),
            "feature_names": ["f"],
        },
        {
            "subject_id": "p1",
            "run_id": "s2",
            "center": "c1",
            "channel_names": ["A1", "A2"],
            "features": np.asarray([[[5.0], [18.0]], [[7.0], [22.0]]]),
            "labels_nez": np.asarray([1, 0]),
            "feature_names": ["f"],
        },
    ]
    table, manifest = build_channel_feature_table(records)
    a1 = table.loc[table["channel_name"] == "A1"].iloc[0]
    assert a1["label_nez"] == 1
    assert a1["f__seizure_median__median"] == pytest.approx(4.0)
    assert a1["f__seizure_mean__mean"] == pytest.approx(4.0)
    assert a1["valid_seizure_count"] == 2
    assert manifest["label_semantics"] == {"NEZ": 1, "EZ": 0}

