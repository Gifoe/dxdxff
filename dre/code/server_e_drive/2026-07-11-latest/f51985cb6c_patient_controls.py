"""Patient-relative control models evaluated under the frozen Task 1 protocol.

These controls intentionally use the same feature table, fixed validation split,
and patient-equal threshold selection as the main Task 1 evaluator.  They do not
use center, resection, stimulation, or true-count information as model inputs.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from task1_baselines.thresholds import select_patient_macro_threshold


CONTROL_MODELS = {
    "deepsets_bce",
    "patient_z_mlp",
    "patient_rank_mlp",
    "patient_z_logistic",
    "patient_z_rbf_svm",
}

_METADATA = {
    "subject_id", "center", "channel_name", "label_nez", "clinical_true_nez",
    "clinical_true_ez", "valid_seizure_count", "valid_window_count", "outer_fold",
}


def feature_columns(table: pd.DataFrame) -> list[str]:
    columns = [
        column for column in table.columns
        if column not in _METADATA and pd.api.types.is_numeric_dtype(table[column])
    ]
    if not columns:
        raise ValueError("Patient-relative controls require at least one numeric feature.")
    return columns


def patientwise_zscore(values: np.ndarray, subjects: np.ndarray) -> np.ndarray:
    """Z-score each feature within a patient without consulting labels."""
    output = np.asarray(values, dtype=np.float64).copy()
    for subject in np.unique(subjects.astype(str)):
        mask = subjects.astype(str) == subject
        current = output[mask]
        mean = np.nanmean(current, axis=0)
        std = np.nanstd(current, axis=0)
        output[mask] = (current - mean) / np.where(std > 1e-8, std, 1.0)
    return output


def patientwise_score_rank(scores: np.ndarray, subjects: np.ndarray) -> np.ndarray:
    """Map a NEZ score to an increasing within-patient percentile rank."""
    values = np.asarray(scores, dtype=np.float64)
    output = np.empty_like(values)
    for subject in np.unique(subjects.astype(str)):
        mask = subjects.astype(str) == subject
        count = int(mask.sum())
        if count < 2:
            raise ValueError(f"Patient {subject!r} has fewer than two channels.")
        output[mask] = rankdata(values[mask], method="average") / float(count + 1)
    return output


@dataclass
class _Preprocessor:
    imputer: SimpleImputer
    scaler: StandardScaler
    patient_z: bool

    def fit_transform(self, table: pd.DataFrame, columns: list[str]) -> np.ndarray:
        values = table[columns].to_numpy(dtype=float)
        if self.patient_z:
            values = patientwise_zscore(values, table["subject_id"].to_numpy())
        return self.scaler.fit_transform(self.imputer.fit_transform(values)).astype(np.float32)

    def transform(self, table: pd.DataFrame, columns: list[str]) -> np.ndarray:
        values = table[columns].to_numpy(dtype=float)
        if self.patient_z:
            values = patientwise_zscore(values, table["subject_id"].to_numpy())
        return self.scaler.transform(self.imputer.transform(values)).astype(np.float32)


class _ChannelMLP(torch.nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.LayerNorm(input_dim), torch.nn.Linear(input_dim, 96), torch.nn.GELU(),
            torch.nn.Dropout(0.15), torch.nn.Linear(96, 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values).squeeze(-1)


class _DeepSetsBCE(torch.nn.Module):
    """Set encoder with permutation-invariant patient context and BCE supervision."""
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.phi = torch.nn.Sequential(
            torch.nn.LayerNorm(input_dim), torch.nn.Linear(input_dim, 96), torch.nn.GELU(),
            torch.nn.Dropout(0.15), torch.nn.Linear(96, 64), torch.nn.GELU(),
        )
        self.rho = torch.nn.Sequential(
            torch.nn.LayerNorm(192), torch.nn.Linear(192, 96), torch.nn.GELU(),
            torch.nn.Dropout(0.15), torch.nn.Linear(96, 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        embedding = self.phi(values)
        mean = embedding.mean(dim=0, keepdim=True).expand_as(embedding)
        maximum = embedding.max(dim=0, keepdim=True).values.expand_as(embedding)
        return self.rho(torch.cat([embedding, mean, maximum], dim=1)).squeeze(-1)


def _seed(value: int) -> None:
    np.random.seed(int(value))
    torch.manual_seed(int(value))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(value))


def _torch_scores(model: torch.nn.Module, values: np.ndarray, *, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.as_tensor(values, dtype=torch.float32, device=device))).cpu().numpy()


def _fit_torch(
    model: torch.nn.Module,
    fit_x: np.ndarray,
    fit_y: np.ndarray,
    fit_subjects: np.ndarray,
    validation: pd.DataFrame,
    validation_x: np.ndarray,
    *,
    device: torch.device,
    seed: int,
    max_epochs: int,
    patience: int,
    checkpoint_path: Path | None,
    rank_scores: bool,
) -> tuple[torch.nn.Module, Any, int, list[dict[str, Any]]]:
    _seed(seed)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    positive = max(int(fit_y.sum()), 1)
    negative = max(int((1 - fit_y).sum()), 1)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negative / positive, device=device))
    subject_rows = [np.flatnonzero(fit_subjects == subject) for subject in np.unique(fit_subjects)]
    best_key: tuple[float, float, int] | None = None
    best_state: dict[str, torch.Tensor] | None = None
    best_threshold = None
    stale = 0
    best_epoch = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, int(max_epochs) + 1):
        model.train()
        losses: list[float] = []
        order = np.random.default_rng(seed + epoch).permutation(len(subject_rows))
        for position in order:
            indices = subject_rows[int(position)]
            batch_x = torch.as_tensor(fit_x[indices], dtype=torch.float32, device=device)
            batch_y = torch.as_tensor(fit_y[indices], dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        validation_scores = _torch_scores(model, validation_x, device=device)
        if rank_scores:
            validation_scores = patientwise_score_rank(validation_scores, validation["subject_id"].to_numpy())
        selected = validation[["subject_id", "label_nez"]].copy()
        selected["score_nez_probability"] = validation_scores
        threshold = select_patient_macro_threshold(selected, source="outer_validation_patient_macro_f1")
        key = (threshold.patient_macro_f1, threshold.patient_ez_f1, -epoch)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_patient_macro_f1": key[0], "validation_patient_ez_f1": key[1], "validation_threshold": threshold.threshold})
        if best_key is None or key > best_key:
            best_key, best_threshold, best_epoch, stale = key, threshold, epoch, 0
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        else:
            stale += 1
        if epoch >= 6 and stale >= int(patience):
            break
    if best_state is None or best_threshold is None:
        raise RuntimeError("Control model did not select a validation checkpoint.")
    model.load_state_dict(best_state)
    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": best_state, "best_epoch": best_epoch, "selected_threshold": best_threshold.threshold}, checkpoint_path)
    return model, best_threshold, best_epoch, history


def _classical_scores(name: str, fit_x: np.ndarray, fit_y: np.ndarray, target_x: np.ndarray, seed: int) -> np.ndarray:
    if name == "patient_z_logistic":
        estimator = LogisticRegression(C=1.0, class_weight="balanced", solver="liblinear", max_iter=2000, random_state=seed)
    elif name == "patient_z_rbf_svm":
        estimator = SVC(C=1.0, gamma="scale", kernel="rbf", probability=True, class_weight="balanced", random_state=seed)
    else:
        raise ValueError(f"Unknown classical control {name}")
    estimator.fit(fit_x, fit_y)
    return estimator.predict_proba(target_x)[:, 1]


@dataclass
class PatientControlResult:
    oof: pd.DataFrame
    training_audit: pd.DataFrame


def run_patient_relative_control_oof(
    table: pd.DataFrame,
    fold_manifest: pd.DataFrame,
    split_manifest: pd.DataFrame,
    *,
    model_name: str,
    seed: int,
    device: str = "cpu",
    max_epochs: int = 30,
    patience: int = 6,
    checkpoint_root: str | Path | None = None,
) -> PatientControlResult:
    if model_name not in CONTROL_MODELS:
        raise ValueError(f"Unknown patient-relative control {model_name!r}.")
    merged = table.merge(fold_manifest[["subject_id", "outer_fold"]], on="subject_id", how="inner", validate="many_to_one")
    if len(merged) != len(table):
        raise ValueError("Control feature table does not exactly match the frozen cohort.")
    columns = feature_columns(merged)
    active = torch.device(device)
    oof: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    for fold in sorted(merged["outer_fold"].unique()):
        split = split_manifest[split_manifest["outer_fold"] == fold]
        fit_ids = set(split.loc[split["partition"] == "fit", "subject_id"])
        val_ids = set(split.loc[split["partition"] == "validation", "subject_id"])
        test_ids = set(split.loc[split["partition"] == "test", "subject_id"])
        if test_ids != set(merged.loc[merged["outer_fold"] == fold, "subject_id"]):
            raise ValueError(f"Fixed test partition mismatch for fold {fold}.")
        fit = merged[merged["subject_id"].isin(fit_ids)].reset_index(drop=True)
        validation = merged[merged["subject_id"].isin(val_ids)].reset_index(drop=True)
        test = merged[merged["subject_id"].isin(test_ids)].copy().reset_index(drop=True)
        use_patient_z = model_name in {"patient_z_mlp", "patient_z_logistic", "patient_z_rbf_svm"}
        preprocessor = _Preprocessor(SimpleImputer(), StandardScaler(), use_patient_z)
        fit_x = preprocessor.fit_transform(fit, columns)
        val_x = preprocessor.transform(validation, columns)
        test_x = preprocessor.transform(test, columns)
        rank_scores = model_name == "patient_rank_mlp"
        if model_name in {"patient_z_logistic", "patient_z_rbf_svm"}:
            validation_scores = _classical_scores(model_name, fit_x, fit["label_nez"].to_numpy(int), val_x, seed + int(fold))
            threshold_input = validation[["subject_id", "label_nez"]].copy()
            threshold_input["score_nez_probability"] = validation_scores
            threshold = select_patient_macro_threshold(threshold_input, source="outer_validation_patient_macro_f1")
            test_scores = _classical_scores(model_name, fit_x, fit["label_nez"].to_numpy(int), test_x, seed + int(fold))
            best_epoch = np.nan
            history: list[dict[str, Any]] = []
        else:
            model = _DeepSetsBCE(fit_x.shape[1]) if model_name == "deepsets_bce" else _ChannelMLP(fit_x.shape[1])
            destination = None if checkpoint_root is None else Path(checkpoint_root) / f"fold_{fold}" / "best_model.pt"
            model, threshold, best_epoch, history = _fit_torch(
                model, fit_x, fit["label_nez"].to_numpy(int), fit["subject_id"].to_numpy(), validation, val_x,
                device=active, seed=seed + 1009 * int(fold), max_epochs=max_epochs, patience=patience,
                checkpoint_path=destination, rank_scores=rank_scores,
            )
            test_scores = _torch_scores(model, test_x, device=active)
            if rank_scores:
                test_scores = patientwise_score_rank(test_scores, test["subject_id"].to_numpy())
        test["score_nez_probability"] = test_scores
        test["score_ez_probability"] = 1.0 - test_scores
        test["selected_threshold"] = threshold.threshold
        test["threshold_source"] = threshold.source
        test["predicted_nez"] = (test_scores >= threshold.threshold).astype(int)
        test["predicted_ez"] = 1 - test["predicted_nez"]
        test["model"] = model_name
        test["seed"] = int(seed)
        test["clinical_true_nez"] = test["label_nez"]
        test["clinical_true_ez"] = 1 - test["label_nez"]
        oof.append(test)
        audits.extend({"outer_fold": fold, "seed": seed, "model": model_name, "record_type": "epoch", **row} for row in history)
        audits.append({"outer_fold": fold, "seed": seed, "model": model_name, "record_type": "fold", "best_epoch": best_epoch, "selected_threshold": threshold.threshold, "patient_zscore": use_patient_z, "patient_rank_normalization": rank_scores, "fit_patients": len(fit_ids), "validation_patients": len(val_ids), "test_patients": len(test_ids)})
    ledger = pd.concat(oof, ignore_index=True).sort_values(["subject_id", "channel_name"], kind="stable")
    keep = ["model", "seed", "subject_id", "center", "outer_fold", "channel_name", "label_nez", "clinical_true_nez", "clinical_true_ez", "score_nez_probability", "score_ez_probability", "selected_threshold", "threshold_source", "predicted_nez", "predicted_ez", "valid_seizure_count", "valid_window_count"]
    return PatientControlResult(ledger[[column for column in keep if column in ledger]], pd.DataFrame(audits))


__all__ = ["CONTROL_MODELS", "PatientControlResult", "feature_columns", "patientwise_score_rank", "patientwise_zscore", "run_patient_relative_control_oof"]
