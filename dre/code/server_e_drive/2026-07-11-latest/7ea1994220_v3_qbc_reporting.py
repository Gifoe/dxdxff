"""Build formal and diagnostic BCR-Net ledgers from saved fold predictions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score

from .v3_qbc_decoder import true_k_diagnostic_prediction
from P2_V3_AAAI_ABLATIONS.threshold_protocol import select_fold_threshold


def _reciprocal_rank(labels_ez: np.ndarray, score_ez: np.ndarray) -> float:
    order = np.argsort(score_ez, kind="stable")[::-1]
    hits = np.flatnonzero(labels_ez[order] == 1)
    return 0.0 if hits.size == 0 else 1.0 / float(hits[0] + 1)


def _patient_metrics(y_nez: np.ndarray, pred_nez: np.ndarray, score_ez: np.ndarray) -> dict[str, float]:
    y_ez = 1 - y_nez
    return {
        "patient_macro_f1": float(f1_score(y_nez, pred_nez, labels=[0, 1], average="macro", zero_division=0)),
        "patient_ez_f1": float(f1_score(y_nez, pred_nez, labels=[0], average="macro", zero_division=0)),
        "patient_nez_f1": float(f1_score(y_nez, pred_nez, labels=[1], average="macro", zero_division=0)),
        "patient_balanced_accuracy": float(balanced_accuracy_score(y_nez, pred_nez)),
        "patient_ez_auprc": (
            float(average_precision_score(y_ez, score_ez)) if np.unique(y_ez).size > 1 else float(y_ez[0])
        ),
        "patient_ez_mrr": _reciprocal_rank(y_ez, score_ez),
    }


def _records_from_channel_frame(frame: pd.DataFrame) -> list[dict[str, Any]]:
    required = {"subject_id", "center", "channel_name", "true_nez", "fold_idx"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Saved channel predictions are missing V3-QBC fields: {missing}")
    if "final_nez_logit" not in frame.columns:
        if "base_nez_logit" in frame.columns:
            frame = frame.copy()
            frame["final_nez_logit"] = pd.to_numeric(frame["base_nez_logit"], errors="coerce")
        elif "score_nez_probability" in frame.columns or "score_nez" in frame.columns:
            # score_nez is exactly sigmoid(final_nez_logit). Recovering the
            # logit preserves both ordering and the robust-z decoder scale.
            score_column = "score_nez_probability" if "score_nez_probability" in frame.columns else "score_nez"
            probability = pd.to_numeric(frame[score_column], errors="coerce").clip(1e-7, 1.0 - 1e-7)
            frame = frame.copy()
            frame["final_nez_logit"] = np.log(probability / (1.0 - probability))
        else:
            raise ValueError(
                "Saved channel predictions require final_nez_logit, "
                "base_nez_logit, score_nez_probability, or score_nez for V3-QBC reporting"
            )
    if not np.isfinite(frame["final_nez_logit"].to_numpy(dtype=np.float64)).all():
        raise ValueError("Saved V3-QBC NEZ scores contain non-finite values")
    records = []
    for subject_id, group in frame.groupby("subject_id", sort=True):
        records.append(
            {
                "subject_id": str(subject_id),
                "center": str(group["center"].iloc[0]).lower(),
                "outer_fold": int(group["fold_idx"].iloc[0]),
                "channel_name": group["channel_name"].astype(str).tolist(),
                "label_nez": group["true_nez"].to_numpy(dtype=np.int64),
                "final_nez_logit": group["final_nez_logit"].to_numpy(dtype=np.float64),
                "frame": group.copy(),
            }
        )
    return records


def _validation_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for record in records:
        for index, name in enumerate(record["channel_name"]):
            rows.append({
                "subject_id": record["subject_id"], "center": record["center"],
                "outer_fold": record["outer_fold"], "channel_name": name,
                "label_nez": int(record["label_nez"][index]),
                "score_nez": float(1.0 / (1.0 + np.exp(-record["final_nez_logit"][index]))),
            })
    return pd.DataFrame(rows)


def _aggregate(frame: pd.DataFrame, group_column: str | None = None) -> pd.DataFrame:
    metrics = [
        "patient_macro_f1", "patient_ez_f1", "patient_nez_f1",
        "patient_balanced_accuracy", "patient_ez_auprc", "patient_ez_mrr",
    ]
    if group_column is None:
        row = {key: float(frame[key].mean()) for key in metrics}
        row["n_patients"] = int(frame["subject_id"].nunique())
        return pd.DataFrame([row])
    rows = []
    for group, data in frame.groupby(group_column, sort=True):
        rows.append({group_column: group, "n_patients": int(data["subject_id"].nunique()), **{
            key: float(data[key].mean()) for key in metrics
        }})
    return pd.DataFrame(rows)


def build_v3_qbc_reports(run_dir: str | Path) -> dict[str, Any]:
    output = Path(run_dir)
    fold_thresholds = []
    formal_patients = []
    truek_patients = []
    oof_channels = []
    test_files = sorted(output.glob("test_channel_predictions_neuroez_v2_fold_*.csv"))
    if not test_files:
        raise FileNotFoundError(f"No held-out fold channel predictions found under {output}")
    for test_path in test_files:
        fold = int(test_path.stem.rsplit("_", 1)[-1])
        val_path = output / f"val_channel_predictions_neuroez_v2_fold_{fold}.csv"
        if not val_path.exists():
            raise FileNotFoundError(f"Missing fixed-validation predictions: {val_path}")
        validation = _records_from_channel_frame(pd.read_csv(val_path))
        threshold, threshold_search = select_fold_threshold(
            _validation_frame(validation), score_column="score_nez", step=0.005,
        )
        fold_thresholds.append({
            "outer_fold": fold, "threshold": float(threshold), "threshold_source": "validation_only",
            "grid_step": 0.005, "n_validation_rows": int(len(threshold_search)),
        })
        for record in _records_from_channel_frame(pd.read_csv(test_path)):
            y_nez = record["label_nez"]
            score_ez = -record["final_nez_logit"]
            score_nez = 1.0 / (1.0 + np.exp(-record["final_nez_logit"]))
            formal = {
                "predicted_nez": (score_nez >= threshold).astype(int),
                "predicted_ez": (score_nez < threshold).astype(int),
                "selected_threshold": float(threshold),
                "decision_rule": "fold_validation_global_nez_probability_threshold",
                "formal_prediction_source": "validation_only_threshold",
                "threshold_source": "validation_only",
                "true_count_used_for_prediction": False,
                "analysis_status": "PRIMARY_LEGAL",
                "formal_prediction": True,
            }
            truek = true_k_diagnostic_prediction(record["final_nez_logit"], int((1 - y_nez).sum()))
            base = {
                "subject_id": record["subject_id"],
                "center": record["center"],
                "outer_fold": record["outer_fold"],
                "n_channels": int(y_nez.size),
                "true_ez_count": int((1 - y_nez).sum()),
            }
            formal_patients.append({**base, **_patient_metrics(y_nez, formal["predicted_nez"].astype(int), score_ez), **{
                key: formal[key] for key in (
                    "selected_threshold", "decision_rule", "formal_prediction_source",
                    "threshold_source", "true_count_used_for_prediction", "analysis_status", "formal_prediction",
                )
            }})
            truek_patients.append({**base, **_patient_metrics(y_nez, truek["predicted_nez"].astype(int), score_ez), **{
                key: truek[key] for key in (
                    "decision_rule", "formal_prediction_source", "threshold_source",
                    "true_count_used_for_prediction", "analysis_status", "formal_prediction",
                )
            }})
            source = record["frame"].reset_index(drop=True)
            for channel_idx, channel_name in enumerate(record["channel_name"]):
                row = source.iloc[channel_idx].to_dict()
                row.update({
                    "outer_fold": record["outer_fold"],
                    "channel_name": channel_name,
                    "label_nez": int(y_nez[channel_idx]),
                    "label_ez": int(1 - y_nez[channel_idx]),
                    "score_nez_probability": float(score_nez[channel_idx]),
                    "predicted_nez": int(formal["predicted_nez"][channel_idx]),
                    "predicted_ez": int(formal["predicted_ez"][channel_idx]),
                    "selected_threshold": threshold,
                    "decision_rule": formal["decision_rule"],
                    "formal_prediction_source": formal["formal_prediction_source"],
                    "threshold_source": formal["threshold_source"],
                    "true_count_used_for_prediction": False,
                    "analysis_status": "PRIMARY_LEGAL",
                })
                oof_channels.append(row)
    formal = pd.DataFrame(formal_patients)
    truek = pd.DataFrame(truek_patients)
    pd.DataFrame(fold_thresholds).to_csv(output / "fold_thresholds.csv", index=False)
    pd.DataFrame(oof_channels).to_csv(output / "oof_channel_ledger.csv", index=False)
    formal.to_csv(output / "formal_by_patient.csv", index=False)
    _aggregate(formal, "outer_fold").to_csv(output / "formal_by_fold.csv", index=False)
    _aggregate(formal, "center").to_csv(output / "formal_by_center.csv", index=False)
    formal_summary = _aggregate(formal)
    formal_summary["analysis_status"] = "PRIMARY_LEGAL"
    formal_summary["true_count_used_for_prediction"] = False
    formal_summary.to_csv(output / "formal_summary.csv", index=False)
    truek.to_csv(output / "truek_by_patient.csv", index=False)
    _aggregate(truek, "outer_fold").to_csv(output / "truek_by_fold.csv", index=False)
    _aggregate(truek, "center").to_csv(output / "truek_by_center.csv", index=False)
    truek_summary = _aggregate(truek)
    truek_summary["analysis_status"] = "DIAGNOSTIC_ONLY_NOT_DEPLOYABLE"
    truek_summary["true_count_used_for_prediction"] = True
    truek_summary.to_csv(output / "truek_summary.csv", index=False)
    oof = pd.DataFrame(oof_channels)
    report = [
        "# BCR-Net Report",
        "",
        "Formal predictions use one outer-fold validation-only NEZ-probability threshold (grid step 0.005).",
        "Test labels and true EZ counts are not used by the formal decoder.",
        "",
        f"- Held-out patients: {int(formal['subject_id'].nunique())}",
        f"- Formal patient Macro-F1: {float(formal_summary['patient_macro_f1'].iloc[0]):.6f}",
        f"- True-K diagnostic Macro-F1: {float(truek_summary['patient_macro_f1'].iloc[0]):.6f}",
        "- True-K status: DIAGNOSTIC_ONLY_NOT_DEPLOYABLE",
    ]
    (output / "BCR_NET_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return {
        "n_patients": int(formal["subject_id"].nunique()),
        "formal_patient_macro_f1": float(formal_summary["patient_macro_f1"].iloc[0]),
        "truek_patient_macro_f1": float(truek_summary["patient_macro_f1"].iloc[0]),
    }


__all__ = ["build_v3_qbc_reports"]
