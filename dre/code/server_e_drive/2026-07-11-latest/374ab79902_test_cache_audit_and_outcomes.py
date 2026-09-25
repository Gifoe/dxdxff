from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from outcome_hifos.cache_audit import REQUIRED_AUDIT_FILES, audit_caches, build_alignment_rows
from outcome_hifos.cache_schema import CacheContractError, load_cache_contract
from outcome_hifos.outcome_resolver import (
    OutcomeConflictError,
    OutcomePolicy,
    resolve_patient_outcome,
)


def _record(subject: str, run: str, *, channels: list[str], outcome: object | None = None, raw: bool = False) -> dict:
    sample: dict = {
        "sample_id": f"{subject}-{run}",
        "window_relative_centers_sec": np.asarray([-1.0, 1.0], dtype=np.float32),
        "channel_names_norm": list(channels),
    }
    if raw:
        sample.update(
            {
                "raw_waveform": np.arange(len(channels) * 20, dtype=np.float32).reshape(len(channels), 20),
                "raw_temporal_sfreq": 10.0,
                "raw_window_waveforms": np.ones((2, len(channels), 10), dtype=np.float32),
            }
        )
    else:
        sample["window_features"] = np.ones((2, len(channels), 3), dtype=np.float32)
        sample["window_feature_names"] = ["f1", "f2", "f3"]
    if outcome is not None:
        sample["engel_score"] = outcome
    return {
        "subject_id": subject,
        "run_id": run,
        "channel_names_norm": list(channels),
        "sample": sample,
        "source_center": "c1" if subject.endswith("1") else "c2",
    }


def _payload(*, raw: bool, reverse: bool = False) -> dict:
    records = [
        _record("p1", "r1", channels=["A1", "A2"], outcome=1, raw=raw),
        _record("p2", "r2", channels=["B1", "B2"], outcome=3, raw=raw),
    ]
    if reverse:
        records.reverse()
    return {
        "cache_version": "synthetic-v1",
        "run_records": records,
        "patient_index": {
            "p1": {"canonical_channels": ["A1", "A2"], "outcome_group": "success", "source_center": "c1"},
            "p2": {"canonical_channels": ["B1", "B2"], "engel_score": 3, "source_center": "c2"},
        },
    }


def _dump(path: Path, payload: object) -> Path:
    with path.open("wb") as handle:
        pickle.dump(payload, handle)
    return path


def test_cache_contract_rejects_missing_patient_index(tmp_path: Path) -> None:
    path = _dump(tmp_path / "bad.pkl", {"run_records": []})
    with pytest.raises(CacheContractError, match="patient_index"):
        load_cache_contract(path)


def test_conflicting_high_confidence_outcomes_fail_closed() -> None:
    records = [_record("p1", "r1", channels=["A1"], outcome=3)]
    with pytest.raises(OutcomeConflictError, match="p1"):
        resolve_patient_outcome("p1", {"outcome_group": "success"}, records, OutcomePolicy())


def test_success_used_false_is_unknown_not_failure() -> None:
    result = resolve_patient_outcome("p1", {"success_used": False}, [], OutcomePolicy())
    assert result.group == "unknown"
    assert result.normalized_label is None


def test_raw_feature_alignment_uses_explicit_keys_not_record_order() -> None:
    feature = load_cache_contract(_payload(raw=False))
    raw = load_cache_contract(_payload(raw=True, reverse=True))
    rows = build_alignment_rows(feature, raw)
    assert len(rows) == 2
    assert all(row["alignment_status"] == "matched" for row in rows)
    assert all(row["channel_alignment_ratio"] == 1.0 for row in rows)
    assert all(row["window_center_alignment_ratio"] == 1.0 for row in rows)


def test_audit_writes_required_files_and_trainable_counts(tmp_path: Path) -> None:
    feature_path = _dump(tmp_path / "feature.pkl", _payload(raw=False))
    raw_path = _dump(tmp_path / "raw.pkl", _payload(raw=True, reverse=True))
    output_dir = tmp_path / "audit-output"

    summary = audit_caches(feature_path, raw_path, output_dir)

    assert summary.patient_count == 2
    assert summary.outcome_counts == {"success": 1, "failure": 1, "unknown": 0, "conflict": 0}
    assert summary.aligned_run_count == 2
    for name in REQUIRED_AUDIT_FILES:
        assert (output_dir / name).exists(), name
    schema = json.loads((output_dir / "outcome_cache_schema_feature.json").read_text(encoding="utf-8"))
    assert schema["top_level_keys"] == ["cache_version", "patient_index", "run_records"]
    leakage = json.loads((output_dir / "outcome_leakage_audit.json").read_text(encoding="utf-8"))
    assert leakage["passed"] is True
