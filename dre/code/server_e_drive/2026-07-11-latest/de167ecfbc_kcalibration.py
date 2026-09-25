from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from neuroez_c.clean_nez_utils import add_label_encoding_columns, apply_allowed_subject_filter, json_safe


FORBIDDEN_INFERENCE_FEATURES = {
    "subject_id",
    "fold_idx",
    "center",
    "center_id",
    "outcome_group",
    "surgery_success",
    "true_ez",
    "true_nez",
    "true_ez_count",
    "k_true",
    "predicted_by_oracle_k",
    "raw_binary_label",
    "clinical_true_ez",
    "clinical_true_nez",
}


def _safe_skew(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    std = float(np.std(values))
    if not np.isfinite(std) or std < 1e-8:
        return 0.0
    z = (values - float(np.mean(values))) / std
    return float(np.mean(z ** 3))


def build_patient_k_features(rows: pd.DataFrame) -> pd.DataFrame:
    score_col = "final_suspicious_score" if "final_suspicious_score" in rows.columns else "final_suspicious_logit"
    feature_rows: list[dict[str, Any]] = []
    for subject_id, group in rows.groupby("subject_id", sort=False):
        scores = pd.to_numeric(group[score_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        probs = np.exp(scores - np.max(scores)) if scores.size else np.asarray([1.0])
        probs = probs / max(float(probs.sum()), 1e-12)
        top = np.sort(scores)[::-1]
        raw_onset = pd.to_numeric(group.get("raw_dist_onset_z", pd.Series([0.0] * len(group))), errors="coerce").fillna(0.0)
        feature_non_nez = pd.to_numeric(group.get("feature_non_nez_score", pd.Series([0.0] * len(group))), errors="coerce").fillna(0.0)
        feature_rows.append(
            {
                "subject_id": str(subject_id),
                "fold_idx": int(pd.to_numeric(group["fold_idx"], errors="coerce").dropna().iloc[0]),
                "num_channels": int(len(group)),
                "candidate_count": int(group.get("is_candidate", pd.Series([1] * len(group))).astype(bool).sum()),
                "score_entropy": float(-(probs * np.log(probs + 1e-12)).sum()),
                "top1_top2_gap": float(top[0] - top[1]) if top.size >= 2 else 0.0,
                "top3_score_mass": float(top[: min(3, top.size)].sum()) if top.size else 0.0,
                "top5_score_mass": float(top[: min(5, top.size)].sum()) if top.size else 0.0,
                "final_score_mean": float(np.mean(scores)) if scores.size else 0.0,
                "final_score_std": float(np.std(scores)) if scores.size else 0.0,
                "final_score_skew": _safe_skew(scores),
                "raw_dist_onset_top_mass": float(raw_onset.sort_values(ascending=False).head(min(5, len(raw_onset))).sum()),
                "feature_non_nez_top_mass": float(feature_non_nez.sort_values(ascending=False).head(min(5, len(feature_non_nez))).sum()),
                "num_shafts_in_candidates": int(
                    group.loc[group.get("is_candidate", pd.Series([True] * len(group))).astype(bool), "shaft_code"].nunique()
                )
                if "shaft_code" in group.columns
                else 0,
                "largest_local_cluster_size": float(
                    pd.to_numeric(group.get("local_cluster_size", pd.Series([0.0] * len(group))), errors="coerce")
                    .fillna(0.0)
                    .max()
                ),
                "mean_n_records": float(
                    pd.to_numeric(group.get("n_records", pd.Series([0.0] * len(group))), errors="coerce").fillna(0.0).mean()
                ),
                "raw_consistency_summary": float(
                    pd.to_numeric(group.get("raw_dist_preictal_z", pd.Series([0.0] * len(group))), errors="coerce")
                    .fillna(0.0)
                    .std(ddof=0)
                ),
                "k_true": int(
                    pd.to_numeric(
                        group["clinical_true_ez"] if "clinical_true_ez" in group.columns else group["true_ez"],
                        errors="coerce",
                    ).fillna(0).sum()
                )
                if "true_ez" in group.columns or "clinical_true_ez" in group.columns
                else 0,
            }
        )
    return pd.DataFrame(feature_rows)


def _feature_columns(patient_features: pd.DataFrame) -> list[str]:
    excluded = {"subject_id", "fold_idx", "k_true"}
    cols = [col for col in patient_features.columns if col not in excluded]
    forbidden = sorted(FORBIDDEN_INFERENCE_FEATURES.intersection(cols))
    if forbidden:
        raise ValueError(f"Forbidden KCal inference feature columns: {forbidden}")
    return cols


def _model(name: str) -> Any:
    if name == "poisson":
        return Pipeline([("scaler", StandardScaler()), ("model", PoissonRegressor(alpha=0.1, max_iter=1000))])
    if name == "ridge":
        return Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))])
    raise ValueError(f"Unsupported KCal model component: {name}")


def train_predict_kcal(
    settopo_ledger: str | Path,
    output_dir: str | Path,
    *,
    k_min: int = 1,
    k_max: int = 40,
    model: str = "ensemble_ridge_poisson",
    allowed_subjects_ledger: str | Path | None = None,
    allowed_subjects_file: str | Path | None = None,
    require_n_patients: int | None = None,
    label_encoding_mode: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    ledger = pd.read_csv(settopo_ledger)
    ledger["subject_id"] = ledger["subject_id"].astype(str)
    ledger = add_label_encoding_columns(ledger, label_encoding_mode=label_encoding_mode)
    ledger, filter_audit = apply_allowed_subject_filter(
        ledger,
        allowed_subjects_ledger=allowed_subjects_ledger,
        allowed_subjects_file=allowed_subjects_file,
        require_n_patients=require_n_patients,
    )
    patient_features = build_patient_k_features(ledger)
    feature_cols = _feature_columns(patient_features)
    requested_model = str(model)
    if requested_model == "ridge_poisson":
        requested_model = "ensemble_ridge_poisson"
    if requested_model not in {"ridge", "poisson", "ensemble_ridge_poisson"}:
        raise ValueError("model must be ridge, poisson, or ensemble_ridge_poisson")
    folds = sorted(int(value) for value in patient_features["fold_idx"].dropna().unique())
    pred_rows: list[dict[str, Any]] = []
    fold_audits: dict[str, Any] = {}
    models_fit_by_fold: dict[str, list[str]] = {}
    for fold_idx in folds:
        train = patient_features[~patient_features["fold_idx"].astype(int).eq(int(fold_idx))].copy()
        test = patient_features[patient_features["fold_idx"].astype(int).eq(int(fold_idx))].copy()
        fallback = True
        models_fit: list[str] = []
        if len(train) >= 2 and test.shape[0] > 0:
            x_train = train[feature_cols].to_numpy(dtype=np.float64)
            y_train = train["k_true"].to_numpy(dtype=np.float64)
            x_test = test[feature_cols].to_numpy(dtype=np.float64)
            if requested_model == "ensemble_ridge_poisson":
                ridge = _model("ridge")
                poisson = _model("poisson")
                ridge.fit(x_train, y_train)
                poisson.fit(x_train, np.clip(y_train, 0.0, None))
                pred = 0.5 * ridge.predict(x_test) + 0.5 * poisson.predict(x_test)
                models_fit = ["ridge", "poisson"]
            else:
                estimator = _model(requested_model)
                estimator.fit(x_train, np.clip(y_train, 0.0, None) if requested_model == "poisson" else y_train)
                pred = estimator.predict(x_test)
                models_fit = [requested_model]
            fallback = False
        else:
            default = float(np.median(train["k_true"].to_numpy(dtype=np.float64))) if len(train) else float(k_min)
            pred = np.full(len(test), default, dtype=np.float64)
        for (_, row), value in zip(test.iterrows(), pred):
            max_for_patient = min(int(k_max), int(row["num_channels"]))
            k_hat = int(round(float(value)))
            k_hat = max(int(k_min), min(max_for_patient, k_hat))
            pred_rows.append(
                {
                    "fold_idx": int(fold_idx),
                    "subject_id": str(row["subject_id"]),
                    "k_hat": int(k_hat),
                    "k_true_train_label_only": int(row["k_true"]),
                }
            )
        fold_audits[str(fold_idx)] = {
            "train_patients": int(len(train)),
            "test_patients": int(len(test)),
            "fallback_median_train_k": bool(fallback),
        }
        models_fit_by_fold[str(fold_idx)] = models_fit

    predictions = pd.DataFrame(pred_rows)
    out = ledger.merge(predictions[["subject_id", "k_hat"]], on="subject_id", how="left")
    out["k_hat"] = pd.to_numeric(out["k_hat"], errors="coerce").fillna(k_min).astype(int)
    out["predicted_by_kcal"] = 0
    out["predicted_by_oracle_k"] = 0
    score_col = "final_suspicious_score" if "final_suspicious_score" in out.columns else "final_suspicious_logit"
    for subject_id, group in out.groupby("subject_id", sort=False):
        k_hat = int(group["k_hat"].iloc[0])
        kcal_idx = group.sort_values(score_col, ascending=False, kind="mergesort").head(k_hat).index
        out.loc[kcal_idx, "predicted_by_kcal"] = 1
        k_true = int(pd.to_numeric(group["clinical_true_ez"], errors="coerce").fillna(0).sum()) if "clinical_true_ez" in group.columns else 0
        if k_true > 0:
            oracle_idx = group.sort_values(score_col, ascending=False, kind="mergesort").head(k_true).index
            out.loc[oracle_idx, "predicted_by_oracle_k"] = 1

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / "kcal_patient_predictions.csv", index=False)
    out.to_csv(output_dir / "settopo_ledger_with_kcal.csv", index=False)
    audit = {
        **filter_audit,
        "requested_model": requested_model,
        "actual_model_type": requested_model,
        "models_fit_by_fold": models_fit_by_fold,
        "k_min": int(k_min),
        "k_max": int(k_max),
        "inference_feature_columns": feature_cols,
        "forbidden_inference_feature_intersection": sorted(FORBIDDEN_INFERENCE_FEATURES.intersection(feature_cols)),
        "true_ez_count_used_as_inference_input": False,
        "k_true_source": "clinical_true_ez",
        "raw_binary_label_not_used_for_k_true": True,
        "input_alpha": float(pd.to_numeric(ledger["alpha"], errors="coerce").dropna().iloc[0]) if "alpha" in ledger.columns and not pd.to_numeric(ledger["alpha"], errors="coerce").dropna().empty else None,
        "label_encoding_mode": str(ledger["label_encoding_mode"].iloc[0]) if "label_encoding_mode" in ledger.columns and not ledger.empty else None,
        "fold_audits": fold_audits,
    }
    with (output_dir / "kcal_training_audit.json").open("w", encoding="utf-8") as fout:
        json.dump(json_safe(audit), fout, indent=2, ensure_ascii=False, sort_keys=True)
    return out, audit


__all__ = ["FORBIDDEN_INFERENCE_FEATURES", "build_patient_k_features", "train_predict_kcal"]
