from __future__ import annotations

import pandas as pd
import pytest

from outcome_hifos.calibration import ProtocolLeakageError
from outcome_hifos.dataset import OutcomePatientExample
from outcome_hifos.training.fusion_protocol import build_fusion_cohort
from outcome_hifos.training.late_fusion import fit_cross_fitted_stacker


def _inner(branch_shift: float = 0.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subject_id": [f"p{i}" for i in range(8)],
            "outcome": [0, 0, 0, 0, 1, 1, 1, 1],
            "probability": [0.1 + branch_shift, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9 - branch_shift],
            "raw_logit": [-2.0 + branch_shift, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0 - branch_shift],
            "input_representation": ["raw_logit"] * 8,
            "role": ["inner_oof"] * 8,
            "outer_fold_idx": [1] * 8,
            "inner_fold_idx": [0, 1, 0, 1, 0, 1, 0, 1],
            "seed": [42] * 8,
        }
    )


def _outer() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subject_id": ["z0", "z1"],
            "feature_raw_logit": [-1.0, 1.0],
            "fm_raw_logit": [-0.8, 0.8],
            "feature_input_representation": ["raw_logit", "raw_logit"],
            "fm_input_representation": ["raw_logit", "raw_logit"],
            "role": ["outer_test", "outer_test"],
            "outer_fold_idx": [1, 1],
            "seed": [42, 42],
        }
    )


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [("outer_fold_idx", 2, "outer fold"), ("seed", 99, "seed"), ("role", "outer_test", "non-inner")],
)
def test_fusion_rejects_mixed_fold_seed_or_role(column: str, value, message: str) -> None:
    fm = _inner(0.02)
    fm.loc[0, column] = value
    with pytest.raises((ValueError, ProtocolLeakageError), match=message):
        fit_cross_fitted_stacker(_inner(), fm, _outer())


def test_fusion_rejects_cohort_and_outcome_mismatch() -> None:
    fm = _inner(0.02).iloc[:-1].copy()
    with pytest.raises(ValueError, match="cohorts"):
        fit_cross_fitted_stacker(_inner(), fm, _outer())
    fm = _inner(0.02)
    fm.loc[0, "outcome"] = 1
    with pytest.raises(ValueError, match="outcomes"):
        fit_cross_fitted_stacker(_inner(), fm, _outer())


def test_fusion_accepts_aligned_shared_fold_protocol() -> None:
    result = fit_cross_fitted_stacker(_inner(), _inner(0.02), _outer())
    assert set(result.inner_oof["role"]) == {"inner_oof"}
    assert set(result.outer_predictions["role"]) == {"outer_test"}


def _example(subject: str, center: str, target: float) -> OutcomePatientExample:
    return OutcomePatientExample(
        subject,
        center,
        target,
        ("A1",),
        {"feature_runs": []},
        {"run_ids": ("r1",)},
    )


def test_fusion_cohort_is_explicit_intersection_and_rejects_metadata_mismatch() -> None:
    feature = [_example("a", "c1", 0.0), _example("b", "c2", 1.0)]
    fm = [_example("b", "c2", 1.0), _example("c", "c1", 0.0)]
    cohort = build_fusion_cohort(feature, fm)
    assert cohort.manifest["subject_id"].tolist() == ["b"]
    assert [example.subject_id for example in cohort.feature_examples] == ["b"]
    assert [example.subject_id for example in cohort.fm_examples] == ["b"]

    with pytest.raises(ValueError, match="center"):
        build_fusion_cohort(feature, [_example("b", "wrong", 1.0)])
    with pytest.raises(ValueError, match="outcome"):
        build_fusion_cohort(feature, [_example("b", "c2", 0.0)])


def test_fusion_run_alignment_policy_excludes_incomplete_patient() -> None:
    feature = OutcomePatientExample("a", "c1", 0.0, ("A1",), {"feature_runs": []}, {"run_ids": ("r1", "r2", "r3", "r4")})
    fm = OutcomePatientExample("a", "c1", 0.0, ("A1",), {"feature_runs": []}, {"run_ids": ("r1", "r2")})
    with pytest.raises(ValueError, match="alignment policy"):
        build_fusion_cohort([feature], [fm], minimum_run_alignment_ratio=1.0)
