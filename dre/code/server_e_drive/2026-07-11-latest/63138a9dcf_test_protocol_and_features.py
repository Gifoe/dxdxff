from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from task1_baselines.feature_aggregation import build_channel_feature_table
from task1_baselines.fold_protocol import Task1ProtocolError, freeze_v3_fold_manifest, load_task1_sensitivity_protocol


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


def test_sensitivity_protocol_requires_existing_fit_validation_test_manifest(tmp_path) -> None:
    subjects = []
    folds = []
    splits = []
    for index in range(10):
        subject = f"p{index}"
        fold = index % 5 + 1
        subjects.append({"patient_key": subject, "center": "c"})
        folds.append({"patient_key": subject, "fold_idx": fold})
    for fold in range(1, 6):
        for index in range(10):
            partition = "test" if index % 5 + 1 == fold else "validation" if index == fold else "train"
            splits.append({"patient_key": f"p{index}", "fold_idx": fold, "role": partition})
    for name, rows in (("subjects.csv", subjects), ("folds.csv", folds), ("splits.csv", splits)):
        pd.DataFrame(rows).to_csv(tmp_path / name, index=False)
    frozen, parsed = load_task1_sensitivity_protocol(tmp_path / "subjects.csv", tmp_path / "folds.csv", tmp_path / "splits.csv", expected_subjects=10)
    assert len(frozen) == 10
    assert set(parsed["partition"]) == {"fit", "validation", "test"}
