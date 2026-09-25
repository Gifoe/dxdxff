"""RCC formal/raw-threshold and true-K diagnostic reporting."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score

from .v3_rcc_decoder import apply_raw_probability_threshold, fit_raw_probability_threshold
from .v3_qbc_decoder import true_k_diagnostic_prediction


def _m(y: np.ndarray, pred: np.ndarray, score_ez: np.ndarray) -> dict[str, float]:
    return {
        "patient_macro_f1": float(f1_score(y, pred, labels=[0, 1], average="macro", zero_division=0)),
        "patient_ez_f1": float(f1_score(y, pred, labels=[0], average="macro", zero_division=0)),
        "patient_nez_f1": float(f1_score(y, pred, labels=[1], average="macro", zero_division=0)),
        "patient_balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "patient_ez_auprc": float(average_precision_score(1-y, score_ez)) if np.unique(y).size > 1 else float((1-y)[0]),
        "patient_ez_mrr": float(1.0 / (np.flatnonzero((1-y)[np.argsort(score_ez)[::-1]])[0] + 1)) if np.any(1-y) else 0.0,
    }


def _patients(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if "score_nez" not in frame.columns and "score_nez_probability" in frame.columns:
        frame = frame.rename(columns={"score_nez_probability": "score_nez"})
    required = {"subject_id", "center", "channel_name", "true_nez", "score_nez", "final_nez_logit"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"RCC channel output missing fields: {sorted(missing)}")
    rows = []
    for subject, group in frame.groupby("subject_id", sort=True):
        rows.append({"subject_id": str(subject), "center": str(group.center.iloc[0]).lower(), "outer_fold": int(group.fold_idx.iloc[0]), "labels_nez": group.true_nez.to_numpy(int), "score_nez": group.score_nez.to_numpy(float), "channel_mask": np.ones(len(group), dtype=bool), "frame": group.reset_index(drop=True)})
    return rows


def _aggregate(frame: pd.DataFrame, by: str | None = None) -> pd.DataFrame:
    metrics = ["patient_macro_f1", "patient_ez_f1", "patient_nez_f1", "patient_balanced_accuracy", "patient_ez_auprc", "patient_ez_mrr"]
    if by is None:
        return pd.DataFrame([{**{m: float(frame[m].mean()) for m in metrics}, "n_patients": int(frame.subject_id.nunique())}])
    return frame.groupby(by, as_index=False).agg(n_patients=("subject_id", "nunique"), **{m: (m, "mean") for m in metrics})


def build_v3_rcc_reports(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir)
    formal_rows, truek_rows, ledger, thresholds = [], [], [], []
    for test_path in sorted(root.glob("test_channel_predictions_neuroez_v2_fold_*.csv")):
        fold = int(test_path.stem.rsplit("_", 1)[-1])
        validation = _patients(pd.read_csv(root / f"val_channel_predictions_neuroez_v2_fold_{fold}.csv"))
        fit = fit_raw_probability_threshold(validation)
        thresholds.append({"outer_fold": fold, **fit})
        for row in _patients(pd.read_csv(test_path)):
            y, score = row["labels_nez"], row["score_nez"]
            formal = apply_raw_probability_threshold(score, fit["threshold"])
            truek = true_k_diagnostic_prediction(-np.log(np.clip(score, 1e-7, 1-1e-7) / np.clip(1-score, 1e-7, 1.0)), int((1-y).sum()))
            base = {"subject_id": row["subject_id"], "center": row["center"], "outer_fold": row["outer_fold"], "n_channels": len(y), "true_ez_count": int((1-y).sum())}
            formal_rows.append({**base, **_m(y, formal["predicted_nez"].astype(int), 1-score), **formal})
            truek_rows.append({**base, **_m(y, truek["predicted_nez"].astype(int), 1-score), **truek})
            source = row["frame"]
            for i in range(len(source)):
                item = source.iloc[i].to_dict()
                item.update({"outer_fold": row["outer_fold"], "label_nez": int(y[i]), "label_ez": int(1-y[i]), "predicted_nez": int(formal["predicted_nez"][i]), "predicted_ez": int(formal["predicted_ez"][i]), "selected_threshold": float(fit["threshold"]), "decision_rule": formal["decision_rule"], "threshold_source": formal["threshold_source"], "true_count_used_for_prediction": False, "analysis_status": "PRIMARY_LEGAL"})
                ledger.append(item)
    formal, truek = pd.DataFrame(formal_rows), pd.DataFrame(truek_rows)
    pd.DataFrame(thresholds).to_csv(root / "fold_thresholds.csv", index=False)
    pd.DataFrame(ledger).to_csv(root / "oof_channel_ledger.csv", index=False)
    for prefix, frame in (("formal", formal), ("truek", truek)):
        frame.to_csv(root / f"{prefix}_by_patient.csv", index=False)
        _aggregate(frame, "outer_fold").to_csv(root / f"{prefix}_by_fold.csv", index=False)
        _aggregate(frame, "center").to_csv(root / f"{prefix}_by_center.csv", index=False)
        summary = _aggregate(frame)
        summary["analysis_status"] = "PRIMARY_LEGAL" if prefix == "formal" else "DIAGNOSTIC_ONLY_NOT_DEPLOYABLE"
        summary["true_count_used_for_prediction"] = prefix == "truek"
        summary.to_csv(root / f"{prefix}_summary.csv", index=False)
    report = "# V3-RCC Report\n\nFormal metrics use only fold validation fitted raw NEZ-probability thresholds. True-K metrics are diagnostic only.\n"
    (root / "V3_RCC_REPORT.md").write_text(report, encoding="utf-8")
    return {"n_patients": int(formal.subject_id.nunique()), "formal_patient_macro_f1": float(formal.patient_macro_f1.mean()), "truek_patient_macro_f1": float(truek.patient_macro_f1.mean())}


__all__ = ["build_v3_rcc_reports"]
