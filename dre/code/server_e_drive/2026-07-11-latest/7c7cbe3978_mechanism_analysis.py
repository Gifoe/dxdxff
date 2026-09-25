"""Complementarity and near-boundary mechanism analyses."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def score_correlations(ledger: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for subject, patient in ledger.groupby("subject_id", sort=True):
        p2 = patient.p2_score_nez.to_numpy(float); v3 = patient.v3_score_nez.to_numpy(float)
        rows.append({
            "subject_id": subject, "center": patient.center.iloc[0], "outer_fold": int(patient.outer_fold.iloc[0]), "n_channels": len(patient),
            "pearson": float(pearsonr(p2, v3).statistic) if np.std(p2) and np.std(v3) else np.nan,
            "spearman": float(spearmanr(p2, v3).statistic) if np.std(p2) and np.std(v3) else np.nan,
        })
    return pd.DataFrame(rows)


def disagreement_analysis(ledger: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = ledger.copy()
    frame["p2_pred_nez"] = (frame.p2_score_nez >= frame.p2_threshold).astype(int)
    frame["v3_pred_nez"] = (frame.v3_score_nez >= frame.v3_threshold).astype(int)
    frame["fusion_pred_nez"] = (frame.fused_score_nez >= frame.fusion_threshold).astype(int)
    frame["p2_v3_disagree"] = frame.p2_pred_nez != frame.v3_pred_nez
    frame["fusion_changed_p2"] = frame.p2_pred_nez != frame.fusion_pred_nez
    frame["change_correct"] = frame.fusion_changed_p2 & (frame.fusion_pred_nez == frame.label_nez) & (frame.p2_pred_nez != frame.label_nez)
    frame["change_wrong"] = frame.fusion_changed_p2 & (frame.fusion_pred_nez != frame.label_nez) & (frame.p2_pred_nez == frame.label_nez)
    frame["near_p2_boundary"] = (frame.p2_score_nez - frame.p2_threshold).abs() <= 0.05
    summary = frame.groupby(["outer_fold", "center"], as_index=False).agg(
        n_channels=("channel_name", "size"), p2_v3_disagreement_rate=("p2_v3_disagree", "mean"),
        p2_fusion_change_rate=("fusion_changed_p2", "mean"), corrected_channels=("change_correct", "sum"), introduced_errors=("change_wrong", "sum"),
    )
    near = frame[frame.near_p2_boundary].groupby(["outer_fold", "center"], as_index=False).agg(
        near_boundary_channels=("channel_name", "size"), fusion_changed_channels=("fusion_changed_p2", "sum"),
        corrected_channels=("change_correct", "sum"), introduced_errors=("change_wrong", "sum"),
    )
    near["net_corrected_channels"] = near.corrected_channels - near.introduced_errors
    changed = frame[frame.fusion_changed_p2].copy()
    return summary, changed, near

