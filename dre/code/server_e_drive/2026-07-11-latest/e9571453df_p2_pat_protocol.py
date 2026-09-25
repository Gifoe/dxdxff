"""Leak-free protocol, features, metrics, and LOPO selection for P2-Q10-PAT."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score, f1_score,
    precision_recall_fscore_support, roc_auc_score,
)

from .p2_pat_decoder import (
    MAX_PREDICTION_RESIDUAL_GRID, RIDGE_ALPHA_GRID, SHRINKAGE_GRID,
    P2PATDecoder, logit_to_probability, probability_to_logit,
)


PAT1_FEATURES = (
    "median_logit_offset", "q25_logit_offset", "q75_logit_offset", "logit_iqr",
    "base_predicted_nez_fraction", "mean_binary_entropy", "log1p_n_channels",
)
PAT2_EXTRA_FEATURES = (
    "mean_q10_nez_probability", "std_q10_nez_probability", "mean_anchor_evidence",
    "std_anchor_evidence", "mean_temporal_delta_norm", "std_temporal_delta_norm",
    "mean_normalized_valid_seizure_count",
)
PAT_PROFILES = ("PAT0_GLOBAL", "PAT1_MINIMAL", "PAT2_EXTENDED")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonicalize_pat_channel_frame(frame: pd.DataFrame, *, profile: str = "PAT1_MINIMAL") -> pd.DataFrame:
    aliases = {
        "true_nez": "label_nez", "true_ez": "label_ez", "final_nez_logit": "base_nez_logit",
        "direct_nez_logit": "base_nez_logit", "seizure_nez_probability_q10": "q10_nez_probability",
        "negative_anchor_distance": "anchor_evidence", "anchor_distance_z": "anchor_evidence",
        "anchor_residual": "anchor_evidence", "fold_idx": "outer_fold",
    }
    result = frame.rename(columns={source: target for source, target in aliases.items() if source in frame and target not in frame}).copy()
    if "label_nez" not in result and "label_ez" in result:
        result["label_nez"] = 1 - pd.to_numeric(result.label_ez).astype(int)
    if "label_ez" not in result and "label_nez" in result:
        result["label_ez"] = 1 - pd.to_numeric(result.label_nez).astype(int)
    required = {"subject_id", "center", "channel_name", "outer_fold", "label_nez", "label_ez", "base_nez_logit"}
    missing = required - set(result)
    if missing:
        raise ValueError(f"P2-Q10 PAT ledger missing required fields: {sorted(missing)}")
    if profile == "PAT2_EXTENDED":
        pat2_required = {"q10_nez_probability", "anchor_evidence", "temporal_delta_norm", "valid_seizure_count"}
        missing_pat2 = pat2_required - set(result)
        if missing_pat2:
            raise ValueError(f"PAT2_EXTENDED missing diagnostic fields: {sorted(missing_pat2)}")
    result["subject_id"] = result.subject_id.astype(str)
    result["center"] = result.center.astype(str).str.lower()
    result["channel_name"] = result.channel_name.astype(str)
    result["outer_fold"] = pd.to_numeric(result.outer_fold, errors="raise").astype(int)
    for field in ("label_nez", "label_ez"):
        result[field] = pd.to_numeric(result[field], errors="raise").astype(int)
        if not set(result[field].unique()).issubset({0, 1}):
            raise ValueError(f"{field} must be binary")
    if not np.array_equal(result.label_ez.to_numpy(), 1 - result.label_nez.to_numpy()):
        raise ValueError("Label semantics must be NEZ=1, EZ=0 exactly once")
    numeric = ["base_nez_logit"]
    if profile == "PAT2_EXTENDED":
        numeric.extend(["q10_nez_probability", "anchor_evidence", "temporal_delta_norm", "valid_seizure_count"])
    for field in numeric:
        result[field] = pd.to_numeric(result[field], errors="coerce")
        if not np.isfinite(result[field]).all():
            raise ValueError(f"PAT field {field} contains missing or non-finite values")
    if result.duplicated(["subject_id", "channel_name"]).any():
        raise ValueError("PAT ledger has duplicate subject/channel rows")
    return result


def threshold_probability(frame: pd.DataFrame, outer_fold: int) -> float:
    for field in ("selected_threshold", "predicted_threshold", "classification_threshold", "frozen_threshold"):
        if field in frame:
            values = pd.to_numeric(frame[field], errors="coerce").dropna().unique()
            if len(values) == 1 and 0.0 < float(values[0]) < 1.0:
                return float(values[0])
    raise RuntimeError(f"outer fold {outer_fold}: missing one validation-only global probability threshold")


def build_patient_features(frame: pd.DataFrame, global_threshold_logit: float, profile: str) -> pd.DataFrame:
    if profile not in PAT_PROFILES:
        raise ValueError(f"Unknown PAT profile: {profile}")
    canonical_profile = "PAT2_EXTENDED" if profile == "PAT2_EXTENDED" else "PAT1_MINIMAL"
    data = canonicalize_pat_channel_frame(frame, profile=canonical_profile)
    rows: list[dict] = []
    for subject, group in data.groupby("subject_id", sort=True):
        logits = group.base_nez_logit.to_numpy(float)
        if len(logits) < 2:
            raise ValueError(f"PAT patient {subject} has fewer than two valid channels")
        q25, median, q75 = np.quantile(logits, [0.25, 0.50, 0.75])
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        clipped = np.clip(probabilities, 1e-12, 1.0 - 1e-12)
        row = {
            "subject_id": str(subject), "center": str(group.center.iloc[0]),
            "outer_fold": int(group.outer_fold.iloc[0]),
            "median_logit_offset": float(median - global_threshold_logit),
            "q25_logit_offset": float(q25 - global_threshold_logit),
            "q75_logit_offset": float(q75 - global_threshold_logit),
            "logit_iqr": float(q75 - q25),
            "base_predicted_nez_fraction": float((logits >= global_threshold_logit).mean()),
            "mean_binary_entropy": float(np.mean(-clipped*np.log(clipped) - (1-clipped)*np.log(1-clipped))),
            "log1p_n_channels": float(np.log1p(len(group))),
            "n_channels": int(len(group)),
        }
        if profile == "PAT2_EXTENDED":
            row.update({
                "mean_q10_nez_probability": float(group.q10_nez_probability.mean()),
                "std_q10_nez_probability": float(group.q10_nez_probability.std(ddof=0)),
                "mean_anchor_evidence": float(group.anchor_evidence.mean()),
                "std_anchor_evidence": float(group.anchor_evidence.std(ddof=0)),
                "mean_temporal_delta_norm": float(group.temporal_delta_norm.mean()),
                "std_temporal_delta_norm": float(group.temporal_delta_norm.std(ddof=0)),
                "mean_normalized_valid_seizure_count": float(np.clip(group.valid_seizure_count.to_numpy(float)/3.0, 0, 1).mean()),
            })
        rows.append(row)
    return pd.DataFrame(rows)


def feature_columns(profile: str) -> list[str]:
    if profile == "PAT1_MINIMAL":
        return list(PAT1_FEATURES)
    if profile == "PAT2_EXTENDED":
        return list(PAT1_FEATURES + PAT2_EXTRA_FEATURES)
    if profile == "PAT0_GLOBAL":
        return []
    raise ValueError(f"Unknown PAT profile: {profile}")


def _patient_metrics(labels: np.ndarray, logits: np.ndarray, threshold_logit: float, *, truek: bool = False) -> dict[str, float]:
    y = np.asarray(labels, dtype=int); score_nez = 1/(1+np.exp(-np.asarray(logits, dtype=float))); score_ez = 1-score_nez
    if truek:
        pred_ez = np.zeros(len(y), dtype=int); pred_ez[np.argsort(-score_ez, kind="mergesort")[:int((y == 0).sum())]] = 1; pred = 1-pred_ez
    else:
        pred = (np.asarray(logits) >= threshold_logit).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y, pred, labels=[1, 0], zero_division=0)
    ez_positions = np.flatnonzero((y == 0)[np.argsort(-score_ez, kind="mergesort")])
    return {
        "patient_macro_accuracy": float(accuracy_score(y, pred)),
        "patient_macro_balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "patient_macro_f1": float(f1_score(y, pred, labels=[0,1], average="macro", zero_division=0)),
        "patient_weighted_f1": float(f1_score(y, pred, labels=[0,1], average="weighted", zero_division=0)),
        "patient_macro_nez_precision": float(precision[0]), "patient_macro_nez_recall": float(recall[0]), "patient_macro_nez_f1": float(f1[0]),
        "patient_macro_ez_precision": float(precision[1]), "patient_macro_ez_recall": float(recall[1]), "patient_macro_ez_f1": float(f1[1]),
        "patient_macro_auroc_nez": float(roc_auc_score(y, score_nez)) if np.unique(y).size > 1 else 0.0,
        "patient_macro_auprc_nez": float(average_precision_score(y, score_nez)) if np.unique(y).size > 1 else float(y[0]),
        "patient_macro_auroc_ez": float(roc_auc_score(1-y, score_ez)) if np.unique(y).size > 1 else 0.0,
        "patient_macro_auprc_ez": float(average_precision_score(1-y, score_ez)) if np.unique(y).size > 1 else float(1-y[0]),
        "patient_macro_ez_mrr": float(1/(ez_positions[0]+1)) if len(ez_positions) else 0.0,
        "top1_is_ez_rate": float(y[np.argmax(score_ez)] == 0),
        "predicted_nez_count_mae": float(abs(pred.sum()-y.sum())), "predicted_nez_fraction_mae": float(abs(pred.mean()-y.mean())),
        "predicted_ez_count_mae": float(abs((1-pred).sum()-(1-y).sum())), "predicted_ez_fraction_mae": float(abs((1-pred).mean()-(1-y).mean())),
    }


def evaluate_pat_ledger(frame: pd.DataFrame, *, threshold_column: str = "patient_threshold_logit", truek: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = canonicalize_pat_channel_frame(frame)
    rows = []
    for subject, group in data.groupby("subject_id", sort=True):
        threshold = float(group[threshold_column].iloc[0]) if not truek else float("nan")
        row = {"subject_id": subject, "center": group.center.iloc[0], "outer_fold": int(group.outer_fold.iloc[0]), "n_channels": len(group), **_patient_metrics(group.label_nez.to_numpy(), group.base_nez_logit.to_numpy(), threshold, truek=truek)}
        if not truek:
            residual = float(group.predicted_threshold_residual.iloc[0])
            shrinkage = float(group.shrinkage.iloc[0]) if "shrinkage" in group else 0.0
            maximum = float(group.max_prediction_residual.iloc[0]) if "max_prediction_residual" in group else 0.0
            clip_limit = shrinkage * maximum
            row.update({"global_threshold_probability": float(group.global_threshold_probability.iloc[0]), "patient_threshold_probability": float(group.patient_threshold_probability.iloc[0]), "predicted_threshold_residual": residual, "residual_clip_rate": float(clip_limit > 0 and abs(residual) >= clip_limit - 1e-12)})
        rows.append(row)
    patients = pd.DataFrame(rows)
    metrics = [column for column in patients if column.startswith("patient_") or column in {"top1_is_ez_rate", "predicted_nez_count_mae", "predicted_nez_fraction_mae", "predicted_ez_count_mae", "predicted_ez_fraction_mae"}]
    def aggregate(source: pd.DataFrame, key: str | None = None) -> pd.DataFrame:
        extra = {}
        if not truek:
            extra = {"mean_global_threshold_probability": ("global_threshold_probability", "mean"), "mean_patient_threshold_probability": ("patient_threshold_probability", "mean"), "mean_threshold_residual": ("predicted_threshold_residual", "mean"), "mean_abs_threshold_residual": ("predicted_threshold_residual", lambda values: float(np.abs(values).mean())), "p90_abs_threshold_residual": ("predicted_threshold_residual", lambda values: float(np.quantile(np.abs(values), .90))), "p95_abs_threshold_residual": ("predicted_threshold_residual", lambda values: float(np.quantile(np.abs(values), .95))), "max_abs_threshold_residual": ("predicted_threshold_residual", lambda values: float(np.abs(values).max())), "positive_residual_fraction": ("predicted_threshold_residual", lambda values: float((values > 0).mean())), "negative_residual_fraction": ("predicted_threshold_residual", lambda values: float((values < 0).mean())), "residual_clip_rate": ("residual_clip_rate", "mean")}
        if key is None:
            values = {column: float(source[column].mean()) for column in metrics}
            for output, (column, operation) in extra.items():
                values[output] = float(source[column].mean()) if operation == "mean" else float(operation(source[column]))
            return pd.DataFrame([{**values, "n_patients": int(source.subject_id.nunique())}])
        return source.groupby(key, as_index=False).agg(n_patients=("subject_id", "nunique"), **{column:(column,"mean") for column in metrics}, **extra)
    return patients, aggregate(patients), aggregate(patients, "outer_fold"), aggregate(patients, "center")


def run_lopo_selection(
    validation_channels: pd.DataFrame,
    patient_features: pd.DataFrame,
    oracle_targets: pd.DataFrame,
    *, profile: str,
) -> tuple[dict, pd.DataFrame, P2PATDecoder]:
    columns = feature_columns(profile)
    if not columns:
        raise ValueError("PAT0_GLOBAL does not fit a decoder")
    features = patient_features.set_index("subject_id").sort_index()
    targets = oracle_targets.set_index("subject_id").sort_index()
    if list(features.index) != list(targets.index) or len(features) < 3:
        raise ValueError("LOPO requires >=3 aligned validation patients")
    channels = canonicalize_pat_channel_frame(validation_channels, profile=profile)
    rows: list[dict] = []
    configs: list[tuple[tuple, dict]] = []
    subjects = list(features.index)
    for alpha in RIDGE_ALPHA_GRID:
        for shrinkage in SHRINKAGE_GRID:
            for maximum in MAX_PREDICTION_RESIDUAL_GRID:
                config_rows = []
                for heldout in subjects:
                    train = [subject for subject in subjects if subject != heldout]
                    decoder = P2PATDecoder(alpha, shrinkage, maximum).fit(features.loc[train, columns].to_numpy(), targets.loc[train, "oracle_target_residual"].to_numpy())
                    residual = float(decoder.predict_residual(features.loc[[heldout], columns].to_numpy())[0])
                    global_logit = float(targets.loc[heldout, "global_threshold_logit"]); threshold = global_logit + residual
                    group = channels[channels.subject_id.eq(heldout)]
                    metrics = _patient_metrics(group.label_nez.to_numpy(), group.base_nez_logit.to_numpy(), threshold)
                    true_ez = int((group.label_nez == 0).sum()); predicted_ez = int((group.base_nez_logit < threshold).sum())
                    config_rows.append({"outer_fold": int(group.outer_fold.iloc[0]), "profile": profile, "ridge_alpha": alpha, "shrinkage": shrinkage, "max_prediction_residual": maximum, "heldout_subject_id": heldout, "heldout_macro_f1": metrics["patient_macro_f1"], "heldout_ez_f1": metrics["patient_macro_ez_f1"], "heldout_nez_f1": metrics["patient_macro_nez_f1"], "heldout_predicted_ez_count": predicted_ez, "heldout_true_ez_count": true_ez, "heldout_ez_count_error": abs(predicted_ez-true_ez), "predicted_threshold_logit": threshold, "predicted_threshold_probability": logit_to_probability(threshold), "predicted_threshold_residual": residual})
                summary = {"mean_lopo_macro_f1_for_config": float(np.mean([row["heldout_macro_f1"] for row in config_rows])), "mean_lopo_ez_f1_for_config": float(np.mean([row["heldout_ez_f1"] for row in config_rows])), "mean_lopo_count_mae_for_config": float(np.mean([row["heldout_ez_count_error"] for row in config_rows])), "worst_lopo_macro_f1_for_config": float(np.min([row["heldout_macro_f1"] for row in config_rows])), "mean_abs_residual_for_config": float(np.mean(np.abs([row["predicted_threshold_residual"] for row in config_rows])))}
                for row in config_rows: row.update(summary, selected_config=False)
                rows.extend(config_rows)
                key = (-summary["mean_lopo_macro_f1_for_config"], -summary["mean_lopo_ez_f1_for_config"], summary["mean_lopo_count_mae_for_config"], -summary["worst_lopo_macro_f1_for_config"], summary["mean_abs_residual_for_config"], shrinkage, maximum, -alpha)
                configs.append((key, {"profile": profile, "ridge_alpha": alpha, "shrinkage": shrinkage, "max_prediction_residual": maximum, **summary}))
    selected = min(configs, key=lambda item: item[0])[1]
    for row in rows:
        row["selected_config"] = bool(row["ridge_alpha"] == selected["ridge_alpha"] and row["shrinkage"] == selected["shrinkage"] and row["max_prediction_residual"] == selected["max_prediction_residual"])
    decoder = P2PATDecoder(selected["ridge_alpha"], selected["shrinkage"], selected["max_prediction_residual"]).fit(features[columns].to_numpy(), targets["oracle_target_residual"].to_numpy())
    return selected, pd.DataFrame(rows), decoder


def score_invariance_audit(base: pd.DataFrame, pat: pd.DataFrame) -> dict:
    keys = ["subject_id", "outer_fold", "channel_name"]
    left = canonicalize_pat_channel_frame(base).sort_values(keys).reset_index(drop=True)
    right = canonicalize_pat_channel_frame(pat).sort_values(keys).reset_index(drop=True)
    if left[keys].to_dict("records") != right[keys].to_dict("records"):
        raise RuntimeError("PAT score invariance audit channel keys do not align")
    base_logit = left.base_nez_logit.to_numpy(float); pat_logit = right.base_nez_logit.to_numpy(float)
    base_nez = 1/(1+np.exp(-base_logit)); pat_nez = 1/(1+np.exp(-pat_logit))
    # Ranking metrics are recomputed from the unchanged scores under true-K.
    base_truek = evaluate_pat_ledger(left.assign(patient_threshold_logit=0.0), truek=True)[1].iloc[0]
    pat_truek = evaluate_pat_ledger(right.assign(patient_threshold_logit=0.0), truek=True)[1].iloc[0]
    fields = {
        "max_abs_logit_delta": float(np.max(np.abs(base_logit-pat_logit))),
        "max_abs_score_nez_delta": float(np.max(np.abs(base_nez-pat_nez))),
        "max_abs_score_ez_delta": float(np.max(np.abs((1-base_nez)-(1-pat_nez)))),
        "ez_auprc_delta": float(pat_truek.patient_macro_auprc_ez-base_truek.patient_macro_auprc_ez),
        "ez_auroc_delta": float(pat_truek.patient_macro_auroc_ez-base_truek.patient_macro_auroc_ez),
        "ez_mrr_delta": float(pat_truek.patient_macro_ez_mrr-base_truek.patient_macro_ez_mrr),
        "top1_delta": float(pat_truek.top1_is_ez_rate-base_truek.top1_is_ez_rate),
        "truek_macro_f1_delta": float(pat_truek.patient_macro_f1-base_truek.patient_macro_f1),
    }
    fields["passed"] = bool(max(abs(value) for value in fields.values()) <= 1e-12)
    return fields


__all__ = [
    "PAT1_FEATURES", "PAT2_EXTRA_FEATURES", "PAT_PROFILES", "sha256_file",
    "canonicalize_pat_channel_frame", "threshold_probability", "build_patient_features",
    "feature_columns", "evaluate_pat_ledger", "run_lopo_selection", "score_invariance_audit",
]
