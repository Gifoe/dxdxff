from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from outcome_hifos.dataset import OutcomePatientExample
from outcome_hifos.models.model_registry import build_model
from outcome_hifos.reports.shortcut import replay_model_perturbations
from outcome_hifos.reports.shortcut import compute_shortcut_risk


class IdentityCalibrator:
    def predict(self, logits: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.asarray(logits)))


class LowCalibrator:
    def predict(self, logits: np.ndarray) -> np.ndarray:
        return np.full(len(logits), 0.3, dtype=np.float64)


def _example(subject: str, target: float) -> OutcomePatientExample:
    rng = np.random.default_rng(int(target) + 3)
    runs = [rng.normal(size=(3, 3, 4)).astype(np.float32), rng.normal(size=(2, 3, 4)).astype(np.float32)]
    return OutcomePatientExample(
        subject,
        "c1" if target == 0 else "c2",
        target,
        ("A1", "A2", "A3"),
        {
            "feature_runs": runs,
            "window_centers": [np.arange(3, dtype=np.float32), np.arange(2, dtype=np.float32)],
            "seizure_channel_mask": [np.ones(3, dtype=bool), np.ones(3, dtype=bool)],
            "canonical_index": np.arange(3, dtype=np.int64),
        },
        {},
    )


def test_shortcut_replay_runs_real_frozen_model_perturbations(tmp_path: Path) -> None:
    examples = [_example("p0", 0.0), _example("p1", 1.0)]
    model = build_model("H3_ATTENTION_MIL", {"model_dim": 8, "dropout": 0.0}, input_dim=4).eval()
    predictions = replay_model_perturbations(model, examples, tmp_path, torch.device("cpu"), calibrator=IdentityCalibrator(), threshold=0.6, seed=42, outer_fold_idx=1, checkpoint_path="checkpoint.pt")
    expected = {"original", "shuffled_signal", "channel_permutation", "window_order_shuffle", "channel_dropout_10", "channel_dropout_20", "channel_dropout_30", "seizure_dropout"}
    assert set(predictions["perturbation"]) == expected
    assert predictions["probability"].between(0.0, 1.0).all()
    assert (predictions["predicted"] == (predictions["probability"] >= 0.6).astype(int)).all()
    metrics = pd.read_csv(tmp_path / "shortcut_replay_metrics.csv")
    assert set(metrics["status"]) == {"evaluated"}
    assert not metrics["status"].str.contains("implemented").any()


def test_shortcut_replay_uses_fold_calibrator_before_threshold(tmp_path: Path) -> None:
    model = build_model("H2_HIER_POOL", {"model_dim": 8, "dropout": 0.0}, input_dim=4).eval()
    predictions = replay_model_perturbations(
        model,
        [_example("p0", 0.0)],
        tmp_path,
        torch.device("cpu"),
        calibrator=LowCalibrator(),
        threshold=0.5,
        seed=42,
        outer_fold_idx=2,
        checkpoint_path="checkpoint.pt",
    )
    assert (predictions["calibrated_probability"] == 0.3).all()
    assert (predictions["predicted"] == 0).all()
    assert set(predictions["calibration_method"]) == {"platt_inner_oof"}


def test_shortcut_risk_excludes_original_and_robustness_rows() -> None:
    metrics = pd.DataFrame(
        [
            {"shortcut": "original", "status": "evaluated", "macro_f1": 0.80},
            {"shortcut": "channel_permutation", "status": "evaluated", "macro_f1": 0.80},
            {"shortcut": "center_only", "status": "evaluated", "macro_f1": 0.52},
            {"shortcut": "channel_count_only", "status": "evaluated", "macro_f1": 0.50},
        ]
    )
    risk, considered = compute_shortcut_risk(0.80, metrics)
    assert risk is False
    assert set(considered) == {"center_only", "channel_count_only"}
    metrics.loc[len(metrics)] = {"shortcut": "metadata_combined", "status": "evaluated", "macro_f1": 0.79}
    assert compute_shortcut_risk(0.80, metrics)[0] is True
