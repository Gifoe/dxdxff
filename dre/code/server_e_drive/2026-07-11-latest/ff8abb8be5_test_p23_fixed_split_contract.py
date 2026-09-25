from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
P23 = ROOT / "P23_TRN_NEZ_80"


def _factory():
    sys.path.insert(0, str(P23))
    try:
        spec = importlib.util.spec_from_file_location("p23_fixed_factory_test", P23 / "data_factory.py")
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


def test_fixed_split_manifest_preserves_fit_validation_test(tmp_path: Path) -> None:
    factory = _factory()
    subjects = ["hup:A", "hup:B", "lzu:C", "lzu:D"]
    patient_index = {subject: {} for subject in subjects}
    rows = []
    for role, members in {
        "fit": ["hup:A", "hup:B"], "validation": ["lzu:C"], "test": ["lzu:D"],
    }.items():
        rows.extend({"subject_id": subject, "outer_fold": 1, "split_role": role} for subject in members)
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    selected, splits, audit = factory._fixed_split_manifest(manifest, patient_index, allow_partial_test_union=True)
    assert audit["status"] == "passed"
    assert set(selected) == set(subjects)
    assert splits[0]["fit_subjects"] == ["hup:A", "hup:B"]
    assert splits[0]["validation_subjects"] == ["lzu:C"]
    assert splits[0]["test_subjects"] == ["lzu:D"]


def test_fixed_split_manifest_rejects_validation_test_overlap(tmp_path: Path) -> None:
    factory = _factory()
    patient_index = {"hup:A": {}, "lzu:B": {}}
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame([
        {"subject_id": "hup:A", "outer_fold": 1, "split_role": "fit"},
        {"subject_id": "lzu:B", "outer_fold": 1, "split_role": "validation"},
        {"subject_id": "lzu:B", "outer_fold": 1, "split_role": "test"},
    ]).to_csv(manifest, index=False)
    with pytest.raises(ValueError, match="assigns a patient"):
        factory._fixed_split_manifest(manifest, patient_index, allow_partial_test_union=True)


def test_data_provider_writes_fixed_manifest_audit_without_path_shadowing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory = _factory()
    patient_index = {"hup:A": {}, "lzu:B": {}, "pediatric:C": {}}
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame([
        {"subject_id": "hup:A", "outer_fold": 1, "split_role": "fit"},
        {"subject_id": "lzu:B", "outer_fold": 1, "split_role": "validation"},
        {"subject_id": "pediatric:C", "outer_fold": 1, "split_role": "test"},
        {"subject_id": "hup:A", "outer_fold": 2, "split_role": "validation"},
        {"subject_id": "lzu:B", "outer_fold": 2, "split_role": "test"},
        {"subject_id": "pediatric:C", "outer_fold": 2, "split_role": "fit"},
        {"subject_id": "hup:A", "outer_fold": 3, "split_role": "test"},
        {"subject_id": "lzu:B", "outer_fold": 3, "split_role": "fit"},
        {"subject_id": "pediatric:C", "outer_fold": 3, "split_role": "validation"},
    ]).to_csv(manifest, index=False)
    monkeypatch.setattr(
        factory,
        "build_or_load_run_records",
        lambda _: ([{"subject_id": subject} for subject in patient_index], patient_index),
    )
    args = SimpleNamespace(
        cohort_mode="sensitivity80", require_n_patients=3,
        fixed_split_manifest=str(manifest), allow_partial_fixed_test_manifest=False,
        output_dir=str(tmp_path / "output"), split_strategy="5fold", n_splits=5,
        random_seed=42, use_p2_rtc_shift=False, use_p2_atc=False, use_p2_scope_v2=False,
    )
    records, selected, splits = factory.data_provider(args)
    assert len(records) == len(selected) == 3
    assert len(splits) == 3
    assert (tmp_path / "output" / "audit" / "fixed_split_manifest_audit.json").is_file()


def test_p23_parser_exposes_fixed_manifest_contract() -> None:
    sys.path.insert(0, str(P23))
    try:
        spec = importlib.util.spec_from_file_location("p23_runner_parser_test", P23 / "run_neuroez_c.py")
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
        actions = {action.dest for action in module.build_parser()._actions}
        assert {"fixed_split_manifest", "selected_outer_fold", "allow_partial_fixed_test_manifest"} <= actions
    finally:
        sys.path.pop(0)
