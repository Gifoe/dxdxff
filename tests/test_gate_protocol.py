from __future__ import annotations

import pandas as pd

from reva_dlm.pipeline import _decide_g0, _decide_g1_gate


def _g0_summary(
    *,
    rcr_50: float = 0.011,
    corruption_50: int = 11,
    rcr_60: float = 0.016,
    corruption_60: int = 17,
) -> pd.DataFrame:
    rows = []
    for checkpoint in (0.25, 0.40, 0.50, 0.60, 0.75):
        rcr = rcr_50 if checkpoint == 0.50 else rcr_60 if checkpoint == 0.60 else 0.0
        corruption = (
            corruption_50
            if checkpoint == 0.50
            else corruption_60
            if checkpoint == 0.60
            else 0
        )
        rows.append(
            {
                "checkpoint": checkpoint,
                "rcr_joint": rcr,
                "n_corruption": corruption,
                "n_rescue": 40,
            }
        )
    return pd.DataFrame(rows)


def test_g0_strong_gate_is_fixed_at_50_percent():
    summary = _g0_summary(rcr_50=0.03, corruption_50=30)
    decision = _decide_g0(summary)
    assert decision.status == "G0_STRONG_PASS"
    assert decision.selected_checkpoint == 0.50


def test_g0_exploratory_checkpoint_is_frozen_not_reselected():
    summary = _g0_summary()
    summary.loc[summary["checkpoint"].eq(0.75), ["rcr_joint", "n_corruption"]] = [
        0.20,
        200,
    ]
    decision = _decide_g0(summary)
    assert decision.status == "G0_EXPLORATORY_PASS"
    assert decision.selected_checkpoint == 0.60


def test_g0_exploratory_requires_frozen_adjacent_support():
    decision = _decide_g0(_g0_summary(rcr_50=0.009, corruption_50=9))
    assert decision.status == "BORDERLINE_REVIEW_REQUIRED"
    assert decision.selected_checkpoint is None


def _matched(auroc_a: float, lower_a: float, auroc_c: float, lower_c: float):
    return {
        "A_same_current_wrong": {
            "auroc": auroc_a,
            "auroc_ci_lower": lower_a,
            "n_positive": 100,
        },
        "C_same_final_correct": {
            "auroc": auroc_c,
            "auroc_ci_lower": lower_c,
            "n_positive": 100,
        },
    }


def test_g1_fail_requires_stable_primary_and_both_frozen_key_tasks():
    status, _ = _decide_g1_gate(0.81, 0.71, 0.84, _matched(0.82, 0.71, 0.83, 0.72))
    assert status == "G1_FAIL"
    status, _ = _decide_g1_gate(0.81, 0.71, 0.84, _matched(0.79, 0.71, 0.83, 0.72))
    assert status == "G1_PASS"


def test_g1_strong_pass_uses_primary_ci_upper():
    status, _ = _decide_g1_gate(0.65, 0.60, 0.69, _matched(0.75, 0.60, 0.70, 0.55))
    assert status == "G1_STRONG_PASS"
    status, _ = _decide_g1_gate(0.65, 0.60, 0.71, _matched(0.75, 0.60, 0.70, 0.55))
    assert status == "G1_PASS"
