from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

from outcome_hifos.calibration import PlattCalibrator
from outcome_hifos.dataset import OutcomePatientExample
from outcome_hifos.folds import iter_outer_partitions
from outcome_hifos.training.experiment_runner import _run_neural_fold
from .registry import build_estimator, parameter_grid


@dataclass
class SimpleBaselineResult:
    oof: pd.DataFrame
    selected_parameters: pd.DataFrame


def _features(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame if column not in {"subject_id", "center", "target", "outcome_label", "fold_idx"} and pd.api.types.is_numeric_dtype(frame[column])]


def _threshold(y: np.ndarray, probability: np.ndarray) -> float:
    candidates = []
    for threshold in np.round(np.arange(0.05, 0.951, 0.01), 2):
        predicted = (probability >= threshold).astype(int)
        macro = f1_score(y, predicted, average="macro", labels=[0, 1], zero_division=0)
        failure_recall = np.mean(predicted[y == 0] == 0) if np.any(y == 0) else 0.0
        balanced = balanced_accuracy_score(y, predicted) if np.unique(y).size == 2 else -1.0
        candidates.append((macro, failure_recall, balanced, -abs(threshold - 0.5), -threshold, threshold))
    return float(max(candidates)[-1])


def _crossfit(train: pd.DataFrame, columns: list[str], name: str, params: dict, seed: int, folds: int, compact: bool) -> np.ndarray:
    output = np.zeros(len(train), dtype=float)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for fit_index, valid_index in splitter.split(train, train["target"]):
        if name == "majority":
            output[valid_index] = float(train.iloc[fit_index]["target"].mean())
            continue
        estimator = build_estimator(name, dict(params), seed, compact).fit(train.iloc[fit_index][columns], train.iloc[fit_index]["target"])
        output[valid_index] = estimator.predict_proba(train.iloc[valid_index][columns])[:, 1]
    return output


def run_simple_baseline_oof(
    summary: pd.DataFrame,
    fold_ledger: pd.DataFrame,
    *,
    model_name: str,
    seed: int,
    inner_folds: int = 4,
    compact: bool = False,
    fixed_baseline: bool = False,
) -> SimpleBaselineResult:
    frame = summary.merge(fold_ledger[["subject_id", "fold_idx"]], on="subject_id", validate="one_to_one")
    columns = _features(frame)
    oof_rows = []
    selected = []
    for fold in sorted(frame["fold_idx"].unique()):
        train = frame[frame["fold_idx"] != fold].reset_index(drop=True)
        test = frame[frame["fold_idx"] == fold].copy()
        if fixed_baseline:
            # Fixed, non-tuned baseline: exactly one fit per outer fold.
            params = parameter_grid(model_name, compact=True)[0]
            threshold = 0.5
            if model_name == "majority":
                raw_probability = np.full(len(test), float(train["target"].mean()), dtype=float)
            else:
                estimator = build_estimator(model_name, dict(params), seed, compact=True).fit(
                    train[columns], train["target"]
                )
                raw_probability = estimator.predict_proba(test[columns])[:, 1]
            calibrated = raw_probability
            calibration_method = "none_fixed_baseline"
            threshold_source = "fixed_0.5"
        else:
            candidate_rows = []
            for params in parameter_grid(model_name, compact):
                raw = _crossfit(train, columns, model_name, params, seed, inner_folds, compact)
                score = f1_score(train["target"], raw >= _threshold(train["target"].to_numpy(), raw), average="macro", labels=[0, 1], zero_division=0)
                candidate_rows.append((score, str(sorted(params.items())), params, raw))
            _, _, params, raw_inner = max(candidate_rows)
            eps = np.finfo(float).eps
            inner_logits = np.log(np.clip(raw_inner, eps, 1 - eps) / np.clip(1 - raw_inner, eps, 1 - eps))
            calibrator = PlattCalibrator().fit(inner_logits, train["target"].to_numpy(), roles=["inner_oof"] * len(train))
            inner_calibrated = calibrator.predict(inner_logits)
            threshold = _threshold(train["target"].to_numpy(), inner_calibrated)
            if model_name == "majority":
                raw_probability = np.full(len(test), float(train["target"].mean()), dtype=float)
            else:
                estimator = build_estimator(model_name, dict(params), seed, compact).fit(train[columns], train["target"])
                raw_probability = estimator.predict_proba(test[columns])[:, 1]
            calibrated = calibrator.predict(np.log(np.clip(raw_probability, eps, 1 - eps) / np.clip(1 - raw_probability, eps, 1 - eps)))
            calibration_method = "platt_inner_oof"
            threshold_source = "inner_oof_patient_macro_f1"
        eps = np.finfo(float).eps
        raw_logit = np.log(np.clip(raw_probability, eps, 1 - eps) / np.clip(1 - raw_probability, eps, 1 - eps))
        test["model"] = model_name
        test["seed"] = seed
        test["outcome"] = test["target"].astype(int)
        test["outer_fold"] = fold
        test["raw_logit"] = raw_logit
        test["raw_probability"] = raw_probability
        test["calibrated_probability"] = calibrated
        test["selected_threshold"] = threshold
        test["predicted"] = (calibrated >= threshold).astype(int)
        test["calibration_method"] = calibration_method
        test["threshold_source"] = threshold_source
        oof_rows.append(test)
        selected.append({"outer_fold": fold, "model": model_name, "seed": seed, "parameters": params, "selected_threshold": threshold})
    oof = pd.concat(oof_rows).sort_values("subject_id").reset_index(drop=True)
    ledger_columns = [
        "model", "seed", "subject_id", "center", "outcome", "outer_fold",
        "raw_logit", "raw_probability", "calibrated_probability",
        "selected_threshold", "predicted", "calibration_method", "threshold_source",
    ]
    return SimpleBaselineResult(oof[ledger_columns], pd.DataFrame(selected))


def run_simple_baseline_loco(
    summary: pd.DataFrame,
    *,
    model_name: str,
    seed: int,
    inner_folds: int = 4,
    compact: bool = False,
    fixed_baseline: bool = False,
) -> SimpleBaselineResult:
    centers = sorted(summary["center"].astype(str).unique())
    if len(centers) < 2:
        raise ValueError("Task 2 LOCO requires at least two centers.")
    center_fold = {center: index for index, center in enumerate(centers, start=1)}
    ledger = summary[["subject_id", "center", "target"]].copy()
    ledger["outcome_label"] = ledger["target"].astype(int)
    ledger["fold_idx"] = ledger["center"].astype(str).map(center_fold)
    result = run_simple_baseline_oof(
        summary, ledger, model_name=model_name, seed=seed, inner_folds=inner_folds,
        compact=compact, fixed_baseline=fixed_baseline,
    )
    result.oof["held_out_center"] = result.oof["center"].astype(str)
    result.selected_parameters["held_out_center"] = result.selected_parameters["outer_fold"].map(
        {fold: center for center, fold in center_fold.items()}
    )
    return result


__all__ = ["SimpleBaselineResult", "run_simple_baseline_loco", "run_simple_baseline_oof"]


def run_hierarchical_token_oof(
    examples: Sequence[OutcomePatientExample],
    fold_ledger: pd.DataFrame,
    *,
    variant: str,
    seed: int,
    output_dir: str | Path,
    device: str,
    inner_folds: int,
    epochs: int,
    batch_size: int = 1,
    max_outer_folds: int = 0,
    model_config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    manifest = pd.DataFrame([{"subject_id": item.subject_id, "center": item.center, "outcome_label": int(item.target)} for item in examples])
    partitions = list(iter_outer_partitions(manifest, fold_ledger))
    if max_outer_folds > 0:
        partitions = partitions[: int(max_outer_folds)]
    root = Path(output_dir)
    rows = []
    active_device = torch.device(device)
    base_model = {"model_dim": 128, "dropout": 0.2, "input_pre_normalized": True, **dict(model_config or {})}
    training = {
        "epochs": int(epochs), "patience": min(8, max(1, int(epochs))), "batch_size": int(batch_size),
        "gradient_accumulation": 1, "num_workers": 0, "amp": active_device.type == "cuda",
        "learning_rate": 1e-3, "weight_decay": 1e-4, "grad_clip": 1.0, "save_diagnostics": False,
    }
    protocol = {"inner_folds": int(inner_folds), "calibration": "platt", "full_inner_oof": True}
    for partition in partitions:
        run_dir = root / variant / f"outer_fold_{partition.fold_idx}" / f"seed_{int(seed)}"
        frame = _run_neural_fold(
            list(examples), partition.train_manifest, partition.test_subjects, variant, "screening",
            base_model, training, protocol, {}, int(partition.fold_idx), int(seed), run_dir, active_device,
        )
        frame["model"] = variant
        frame["seed"] = int(seed)
        frame["outer_fold"] = int(partition.fold_idx)
        frame = frame.rename(columns={"probability": "calibrated_probability", "threshold": "selected_threshold"})
        frame["raw_logit"] = frame["logit"]
        frame["raw_probability"] = 1.0 / (1.0 + np.exp(-frame["logit"].to_numpy(dtype=float)))
        frame["calibration_method"] = "platt_inner_oof"
        frame["threshold_source"] = "inner_oof_patient_macro_f1"
        rows.append(frame)
    output = pd.concat(rows, ignore_index=True)
    if output.duplicated(["subject_id"]).any():
        raise ValueError(f"{variant} produced duplicate patient OOF rows.")
    return output


__all__.extend(["run_hierarchical_token_oof"])
