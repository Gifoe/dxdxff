from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from ..metrics import bootstrap_metrics, compute_metrics


def metric_table(predictions: pd.DataFrame, group: str | None = None) -> pd.DataFrame:
    groups = [("pooled", predictions)] if group is None else list(predictions.groupby(group, sort=True))
    rows = []
    for key, frame in groups:
        rows.append({(group or "scope"): key, "n_patients": len(frame), **compute_metrics(frame.outcome_true, frame.probability_success, prediction_success=frame.prediction)})
    return pd.DataFrame(rows)


def bootstrap_ci(predictions: pd.DataFrame, repeats: int, seed: int) -> pd.DataFrame:
    samples, skipped = bootstrap_metrics(predictions.outcome_true, predictions.probability_success, prediction_success=predictions.prediction, repeats=repeats, seed=seed)
    metrics = [x for x in ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "auroc", "success_auprc", "failure_auprc", "success_precision", "success_recall", "failure_precision", "failure_recall", "brier", "ece") if x in samples]
    rows = [{"metric": name, "estimate": float(metric_table(predictions).iloc[0][name]), "ci_lower": float(samples[name].quantile(.025)), "ci_upper": float(samples[name].quantile(.975)), "bootstrap_repeats": repeats, "bootstrap_valid_repeats": len(samples), "bootstrap_skipped": skipped} for name in metrics]
    return pd.DataFrame(rows)


def univariate_audit(frame: pd.DataFrame, features: list[str], *, repeats: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed); y = frame.outcome_true.to_numpy(dtype=int); rows = []
    for name in features:
        x = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float); valid = np.isfinite(x); auroc = roc_auc_score(y[valid], x[valid]) if valid.sum() and np.unique(y[valid]).size == 2 else np.nan; rho = spearmanr(x[valid], y[valid]).statistic if valid.sum() >= 3 else np.nan
        boot = []
        for _ in range(repeats):
            indices = rng.integers(0, len(frame), len(frame)); keep = valid[indices]
            if keep.sum() and np.unique(y[indices][keep]).size == 2: boot.append(roc_auc_score(y[indices][keep], x[indices][keep]))
        success = x[(y == 1) & np.isfinite(x)]; failure = x[(y == 0) & np.isfinite(x)]
        rows.append({"feature": name, "n_valid": int(valid.sum()), "univariate_auroc": auroc, "auroc_ci_lower": float(np.quantile(boot, .025)) if boot else np.nan, "auroc_ci_upper": float(np.quantile(boot, .975)) if boot else np.nan, "spearman_outcome": rho, "success_median": float(np.median(success)) if success.size else np.nan, "failure_median": float(np.median(failure)) if failure.size else np.nan, "used_for_selection": False})
    return pd.DataFrame(rows)


def biomarker_nez_association(joint: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for patient, frame in joint.groupby("patient_key", sort=True):
        for name in ("fragility", "ei", "propagation", "low_entropy", "hfo", "biomarker_consensus"):
            local = frame[[name, "q_nez", "clinical_target", "channel"]].drop_duplicates("channel"); valid = np.isfinite(local[name]) & np.isfinite(local.q_nez)
            rho = spearmanr(local.loc[valid, name], local.loc[valid, "q_nez"]).statistic if valid.sum() >= 3 else np.nan
            count = int(valid.sum()); k = max(1, int(np.ceil(count * .10))) if count else 0
            bio_top = set(local.loc[valid].nlargest(k, name).channel) if k else set(); nez_top = set(local.loc[valid].nlargest(k, "q_nez").channel) if k else set(); union = bio_top | nez_top
            denominator = float(local.loc[valid, name].sum()); alignment = float(local.loc[valid & local.clinical_target.astype(bool), name].sum() / denominator) if denominator > 0 else np.nan
            rows.append({"patient_key": patient, "biomarker": name, "n_valid_channels": count, "spearman_with_q_nez": rho, "top10_jaccard_with_q_nez": len(bio_top & nez_top) / len(union) if union else np.nan, "target_alignment": alignment})
    return pd.DataFrame(rows)


__all__ = ["biomarker_nez_association", "bootstrap_ci", "metric_table", "univariate_audit"]
