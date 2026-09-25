from __future__ import annotations

import numpy as np
import pandas as pd

from outcome_hifos.baselines.training import run_simple_baseline_loco, run_simple_baseline_oof


def test_task2_simple_oof_is_one_row_per_patient() -> None:
    rows = []
    folds = []
    for index in range(20):
        subject = f"p{index:02d}"
        target = index % 2
        rows.append({"subject_id": subject, "center": f"c{index % 2}", "target": target, "f1": target + index / 100, "f2": index % 3})
        folds.append({"subject_id": subject, "center": f"c{index % 2}", "outcome_label": target, "fold_idx": index % 5 + 1})
    result = run_simple_baseline_oof(pd.DataFrame(rows), pd.DataFrame(folds), model_name="elasticnet", seed=42, inner_folds=2, compact=True)
    assert len(result.oof) == 20
    assert result.oof["subject_id"].is_unique
    assert set(result.oof["outcome"]) == {0, 1}
    assert set(result.oof["threshold_source"]) == {"inner_oof_patient_macro_f1"}
    assert set(result.oof["calibration_method"]) == {"platt_inner_oof"}
    assert "f1" not in result.oof
    assert "f2" not in result.oof


def test_task2_simple_loco_holds_out_each_center() -> None:
    rows = []
    for index in range(30):
        target = index % 2
        rows.append({
            "subject_id": f"p{index:02d}", "center": f"c{index % 3}", "target": target,
            "f1": target + index / 100, "f2": index % 4,
        })
    result = run_simple_baseline_loco(
        pd.DataFrame(rows), model_name="elasticnet", seed=42, inner_folds=2, compact=True
    )
    assert result.oof["subject_id"].is_unique
    assert (result.oof["center"] == result.oof["held_out_center"]).all()
    assert set(result.oof["held_out_center"]) == {"c0", "c1", "c2"}


def test_task2_fixed_baseline_has_no_inner_calibration() -> None:
    rows = []
    folds = []
    for index in range(20):
        subject = f"p{index:02d}"
        target = index % 2
        rows.append({"subject_id": subject, "center": f"c{index % 2}", "target": target, "f1": target + index / 100})
        folds.append({"subject_id": subject, "center": f"c{index % 2}", "outcome_label": target, "fold_idx": index % 5 + 1})
    result = run_simple_baseline_oof(
        pd.DataFrame(rows), pd.DataFrame(folds), model_name="elasticnet", seed=42, fixed_baseline=True
    )
    assert set(result.oof["selected_threshold"]) == {0.5}
    assert set(result.oof["calibration_method"]) == {"none_fixed_baseline"}
    assert set(result.oof["threshold_source"]) == {"fixed_0.5"}
