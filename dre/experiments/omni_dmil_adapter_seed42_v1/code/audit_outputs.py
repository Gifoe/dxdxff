#!/usr/bin/env python3
"""Validate the public D-MIL artifacts without reading private identifiers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    args = parser.parse_args()
    root = args.experiment

    predictions = pd.read_csv(root / "TRAIN_OOF_PREDICTIONS.csv")
    summary = pd.read_csv(root / "TRAIN_OOF_SUMMARY.csv").set_index("model")
    split = pd.read_csv(root / "TRAIN_OOF_SPLIT.csv")
    folds = pd.read_csv(root / "TRAIN_OOF_FOLD_METRICS.csv")
    parameters = pd.read_csv(root / "TRAIN_OOF_PARAMETERS.csv")
    gate = json.loads((root / "TRAIN_GATE.json").read_text(encoding="utf-8"))
    status = json.loads((root / "RUN_STATUS.json").read_text(encoding="utf-8"))

    forbidden = sorted({"patient", "patient_id", "edf", "edf_name", "channel", "channel_name"} & set(predictions.columns))
    assert not forbidden, forbidden
    assert split.patient_hash.nunique() == 141
    assert sorted(split.fold.unique().tolist()) == [1, 2, 3, 4, 5]
    assert not split.patient_hash.duplicated().any()
    assert status["official_test_accessed"] is False
    assert gate["official_test_accessed"] is False
    assert gate["terminal"] == "STOP_DMIL_TRAIN_GATE_FAILED"
    assert not (root / "TEST_METRICS.csv").exists()
    assert not (root / "MODEL_FREEZE_BEFORE_TEST.json").exists()

    models = summary.index.tolist()
    recomputed: dict[str, dict[str, float]] = {}
    for model in models:
        rows = predictions[predictions.model == model]
        assert len(rows) == 13350
        assert rows.anonymous_unit_index.nunique() == 13350
        auroc = float(roc_auc_score(rows.y, rows.score))
        ap = float(average_precision_score(rows.y, rows.score))
        np.testing.assert_allclose([auroc, ap], summary.loc[model, ["auroc", "ap"]], atol=1e-12, rtol=0)
        recomputed[model] = {"auroc": auroc, "ap": ap}

    baseline = predictions[predictions.model == "V0_MEAN"]
    np.testing.assert_allclose(baseline.score, baseline.mean_score, atol=1e-12, rtol=0)
    full_parameters = parameters[parameters.model == "V4_FULL_DMIL"]
    assert len(full_parameters) == 5
    assert np.all(np.isfinite(full_parameters[["beta_T", "beta_Q", "beta_S"]]))
    full_folds = folds[folds.model == "V4_FULL_DMIL"]
    nonnegative = int((full_folds.delta_auroc_vs_mean >= -1e-15).sum())
    assert nonnegative == gate["nonnegative_folds"] == 1

    audit = {
        "status": "PASS",
        "public_predictions_deidentified": True,
        "forbidden_identifier_columns": forbidden,
        "patients": 141,
        "folds": 5,
        "channel_units_per_model": 13350,
        "models": models,
        "metrics_recomputed": recomputed,
        "zero_adapter_identity_max_abs_error": float(np.max(np.abs(baseline.score - baseline.mean_score))),
        "full_trainable_parameters": 3,
        "nonnegative_full_folds": nonnegative,
        "official_test_accessed": False,
        "terminal": gate["terminal"],
    }
    (root / "IMPLEMENTATION_AUDIT.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
