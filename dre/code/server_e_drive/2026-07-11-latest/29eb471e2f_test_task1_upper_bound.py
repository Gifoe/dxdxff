from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) in sys.path:
    sys.path.remove(str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT))

from task1_baselines.upper_bound import (
    UpperBoundAuditError,
    _clean_label_metrics,
    evaluate_candidate,
    patient_bootstrap,
    patient_oracle,
    standardize_ledger,
    true_k_result,
    validate_candidate,
)


def _raw(
    *,
    labels: list[int] | None = None,
    scores_ez: list[float] | None = None,
    predictions_nez: list[int] | None = None,
    subject: str = "p1",
) -> pd.DataFrame:
    labels = labels or [0, 0, 1, 1]
    scores_ez = scores_ez or [0.9, 0.8, 0.2, 0.1]
    predictions_nez = predictions_nez if predictions_nez is not None else labels
    n = len(labels)
    return pd.DataFrame(
        {
            "model": "m",
            "seed": 42,
            "subject_id": [subject] * n,
            "center": ["c"] * n,
            "outer_fold": [1] * n,
            "channel_name": [f"A{i}" for i in range(n)],
            "label_nez": labels,
            "score_ez_probability": scores_ez,
            "predicted_nez": predictions_nez,
        }
    )


def _standard(raw: pd.DataFrame) -> pd.DataFrame:
    return standardize_ledger(raw, source_path="m_seed_42.csv")[0]


def _manifest_and_canonical(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    subjects = table[["subject_id", "center", "outer_fold"]].drop_duplicates()
    canonical = table[["subject_id", "center", "outer_fold", "channel_name", "label_nez"]].copy()
    return subjects, canonical


def test_perfect_ranking_and_strict_predictions_reach_one() -> None:
    table = _standard(_raw())
    summary, _ = evaluate_candidate(table)
    assert summary["strict_count_free_patient_macro_f1"] == 1.0
    assert summary["true_k_patient_macro_f1"] == 1.0
    assert summary["patient_oracle_threshold_macro_f1"] == 1.0


def test_wrong_threshold_but_correct_ranking_has_oracle_gain() -> None:
    table = _standard(_raw(predictions_nez=[1, 1, 1, 1]))
    summary, _ = evaluate_candidate(table)
    assert summary["patient_oracle_threshold_macro_f1"] > summary["strict_count_free_patient_macro_f1"]
    assert summary["true_k_patient_macro_f1"] == 1.0


def test_reversed_ranking_limits_oracle() -> None:
    table = _standard(_raw(scores_ez=[0.1, 0.2, 0.8, 0.9]))
    summary, _ = evaluate_candidate(table)
    assert summary["patient_oracle_threshold_macro_f1"] < 0.75


def test_nez_one_ez_zero_direction_is_preserved() -> None:
    raw = _raw(labels=[0, 1], scores_ez=[0.8, 0.2], predictions_nez=[0, 1])
    raw["true_ez"] = [1, 0]
    table = _standard(raw)
    assert table["label_nez"].tolist() == [0, 1]
    assert np.allclose(table["score_nez"], [0.2, 0.8])


def test_duplicate_patient_channel_key_raises() -> None:
    table = _standard(pd.concat([_raw(), _raw().iloc[[0]]], ignore_index=True))
    manifest, canonical = _manifest_and_canonical(table.drop_duplicates(["subject_id", "channel_name"]))
    with pytest.raises(UpperBoundAuditError, match="duplicate"):
        validate_candidate(table, manifest, canonical, strict=True)


def test_fold_assignment_conflict_raises() -> None:
    table = _standard(_raw())
    manifest, canonical = _manifest_and_canonical(table)
    manifest["outer_fold"] = 2
    with pytest.raises(UpperBoundAuditError, match="outer_fold"):
        validate_candidate(table, manifest, canonical, strict=True)


def test_multiple_label_columns_conflict_raises() -> None:
    raw = _raw()
    raw["true_nez"] = 1 - raw["label_nez"]
    with pytest.raises(UpperBoundAuditError, match="Conflicting NEZ label"):
        _standard(raw)


def test_probability_outside_unit_interval_raises() -> None:
    raw = _raw()
    raw.loc[0, "score_ez_probability"] = 1.1
    with pytest.raises(UpperBoundAuditError, match=r"\[0,1\]"):
        _standard(raw)


def test_stored_prediction_threshold_mismatch_raises_in_strict_mode() -> None:
    raw = _raw()
    raw["selected_threshold"] = 0.5
    raw["threshold_source"] = "inner_oof_patient_macro_f1"
    raw.loc[0, "predicted_nez"] = 1
    table = _standard(raw)
    manifest, canonical = _manifest_and_canonical(table)
    with pytest.raises(UpperBoundAuditError, match="mismatches"):
        validate_candidate(table, manifest, canonical, strict=True)


def test_primary_oracle_does_not_split_ties_but_prefix_can() -> None:
    table = _standard(_raw(labels=[0, 1, 1], scores_ez=[0.9, 0.9, 0.1], predictions_nez=[0, 1, 1]))
    threshold = patient_oracle(table, allow_tie_split=False)
    prefix = patient_oracle(table, allow_tie_split=True)
    assert threshold["ez_count"] in {0, 2, 3}
    assert prefix["ez_count"] == 1
    assert prefix["macro_f1"] > threshold["macro_f1"]


def test_true_k_tie_break_uses_channel_name_not_label() -> None:
    raw = _raw(labels=[1, 0], scores_ez=[0.5, 0.5], predictions_nez=[1, 0])
    raw["channel_name"] = ["B", "A"]
    table = _standard(raw)
    result = true_k_result(table)
    assert result["selected_channels"] == ["A"]


def test_hard_prediction_only_ledger_keeps_strict_and_disables_ranking() -> None:
    raw = _raw().drop(columns=["score_ez_probability"])
    table = _standard(raw)
    summary, _ = evaluate_candidate(table)
    assert summary["strict_count_free_patient_macro_f1"] == 1.0
    assert summary["ranking_status"] == "UNAVAILABLE_HARD_PREDICTIONS_ONLY"
    assert np.isnan(summary["true_k_patient_macro_f1"])


def test_misaligned_candidate_is_not_library_eligible() -> None:
    table = _standard(_raw())
    manifest, canonical = _manifest_and_canonical(table)
    candidate = table.iloc[:-1].copy()
    result = validate_candidate(candidate, manifest, canonical, strict=False)
    assert not result["aligned_for_library"]
    assert result["missing_keys"] == 1


def test_patient_bootstrap_resamples_patient_vectors() -> None:
    vectors = {
        "a": pd.Series([0.0, 1.0], index=["p1", "p2"]),
        "b": pd.Series([0.5, 0.5], index=["p1", "p2"]),
    }
    result = patient_bootstrap(vectors, samples=200, seed=42, gap_pairs=(("gap", "a", "b"),))
    assert set(result["metric"]) == {"a", "b", "gap"}
    assert result.set_index("metric").loc["a", "estimate"] == 0.5
    assert result.set_index("metric").loc["gap", "estimate"] == 0.0


def test_clean_label_subset_does_not_fill_missing_adjudications(tmp_path) -> None:
    table = _standard(_raw())
    clean = pd.DataFrame(
        {
            "subject_id": ["p1", "p1", "p1"],
            "channel_name": ["A0", "A1", "A2"],
            "adjudicated_label_nez": [0, np.nan, 1],
            "include_clean_subset": [True, True, False],
        }
    )
    path = tmp_path / "clean.csv"
    clean.to_csv(path, index=False)
    status, metrics = _clean_label_metrics(path, {"m::seed_42": table})
    assert status == "AVAILABLE"
    assert metrics.loc[0, "n_clean_channels"] == 1
