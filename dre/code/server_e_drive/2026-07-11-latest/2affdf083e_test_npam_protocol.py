import pandas as pd
import pytest

from neuroez_c.task2.protocol import ProtocolError, load_fold_manifest, manifest_hash, manifest_patient_keys


def _outcomes():
    return pd.DataFrame([
        {"patient_key": f"P{index}", "center": "x", "outcome_group": "success" if index % 2 else "failure", "outcome_label": index % 2}
        for index in range(10)
    ])


def test_strict_fixed_outer_manifest_is_exact_unique_and_five_fold(tmp_path):
    path = tmp_path / "folds.csv"
    pd.DataFrame({"patient_key": [f"P{i}" for i in range(10)], "outer_fold": [1, 2, 3, 4, 5] * 2}).to_csv(path, index=False)
    left = load_fold_manifest(path, _outcomes(), strict=True)
    right = load_fold_manifest(path, _outcomes(), strict=True)
    assert not left["patient_key"].duplicated().any()
    assert set(left["outer_fold"]) == {1, 2, 3, 4, 5}
    assert manifest_hash(left) == manifest_hash(right)
    assert manifest_patient_keys(path) == {f"P{i}" for i in range(10)}


def test_strict_manifest_rejects_missing_patient_or_not_five_folds(tmp_path):
    path = tmp_path / "folds.csv"
    pd.DataFrame({"patient_key": [f"P{i}" for i in range(9)], "outer_fold": [1, 2, 3] * 3}).to_csv(path, index=False)
    with pytest.raises(ProtocolError):
        load_fold_manifest(path, _outcomes(), strict=True)
