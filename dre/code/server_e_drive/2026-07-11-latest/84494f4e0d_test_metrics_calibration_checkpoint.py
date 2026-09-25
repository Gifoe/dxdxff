from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from outcome_hifos.calibration import PlattCalibrator, ProtocolLeakageError, select_macro_f1_threshold
from outcome_hifos.checkpoint import load_checkpoint, save_checkpoint_atomic
from outcome_hifos.metrics import compute_patient_metrics


def test_single_class_auroc_is_nan_with_reason() -> None:
    metrics = compute_patient_metrics(np.ones(4), np.asarray([0.2, 0.4, 0.6, 0.8]), 0.5)
    assert np.isnan(metrics.values["auroc"])
    assert metrics.undefined_reasons["auroc"] == "single_class_target"


def test_complete_binary_metrics_and_ece_are_finite() -> None:
    metrics = compute_patient_metrics(np.asarray([0, 0, 1, 1]), np.asarray([0.1, 0.4, 0.7, 0.9]), 0.5)
    for key in ("macro_f1", "auroc", "auprc", "balanced_accuracy", "brier", "ece", "sensitivity", "specificity", "ppv", "npv"):
        assert np.isfinite(metrics.values[key]), key
    assert metrics.confusion_matrix == ((2, 0), (0, 2))


def test_threshold_rejects_non_inner_oof_rows() -> None:
    table = pd.DataFrame(
        {
            "subject_id": ["p1", "p2", "p3", "p4"],
            "role": ["inner_oof", "inner_oof", "inner_oof", "outer_test"],
            "outcome": [0, 0, 1, 1],
            "probability": [0.1, 0.4, 0.6, 0.9],
        }
    )
    with pytest.raises(ProtocolLeakageError, match="outer_test"):
        select_macro_f1_threshold(table)


def test_threshold_and_platt_are_fit_from_inner_oof() -> None:
    table = pd.DataFrame(
        {
            "subject_id": [f"p{i}" for i in range(6)],
            "role": ["inner_oof"] * 6,
            "outcome": [0, 0, 0, 1, 1, 1],
            "probability": [0.05, 0.2, 0.45, 0.55, 0.8, 0.95],
            "logit": [-3.0, -1.4, -0.2, 0.2, 1.4, 3.0],
        }
    )
    selection = select_macro_f1_threshold(table)
    assert 0.45 <= selection.threshold <= 0.55
    calibrator = PlattCalibrator().fit(table["logit"].to_numpy(), table["outcome"].to_numpy(), roles=table["role"].tolist())
    probabilities = calibrator.predict(table["logit"].to_numpy())
    assert np.all(np.diff(probabilities) >= 0.0)
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))


def test_checkpoint_roundtrip_restores_epoch_model_and_optimizer(tmp_path: Path) -> None:
    model = torch.nn.Linear(3, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    expected = {key: value.detach().clone() for key, value in model.state_dict().items()}
    path = tmp_path / "checkpoint.pt"
    save_checkpoint_atomic(path, model=model, optimizer=optimizer, epoch=4, best_metric=0.72, extra={"fold": 2})
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(10.0)
    state = load_checkpoint(path, model=model, optimizer=optimizer, map_location="cpu")
    assert state.epoch == 4
    assert state.best_metric == 0.72
    assert state.extra["fold"] == 2
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, expected[key])
