from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, f1_score, precision_recall_fscore_support, roc_auc_score


A9V3_GATE = {
    "patient_macro_f1": 0.642714,
    "patient_macro_ez_f1": 0.470591,
    "patient_macro_auprc_ez": 0.518635,
    "patient_macro_ez_mrr": 0.715136,
    "top1_is_ez_rate": 0.600000,
}

AAAI_TARGET_070 = {
    "patient_macro_f1": 0.700,
    "patient_macro_ez_f1": 0.520,
    "patient_macro_auprc_ez": 0.560,
    "patient_macro_ez_mrr": 0.740,
    "top1_is_ez_rate": 0.650,
}

REQUIRED_SUMMARY_COLUMNS = [
    "patient_macro_accuracy",
    "patient_macro_balanced_accuracy",
    "patient_macro_f1",
    "patient_weighted_f1",
    "patient_macro_nez_precision",
    "patient_macro_nez_recall",
    "patient_macro_nez_f1",
    "patient_macro_ez_precision",
    "patient_macro_ez_recall",
    "patient_macro_ez_f1",
    "patient_macro_auroc_nez",
    "patient_macro_auprc_nez",
    "patient_macro_auroc_ez",
    "patient_macro_auprc_ez",
    "patient_macro_ez_recall_at_true_count",
    "patient_macro_ez_mrr",
    "top1_is_ez_rate",
    "macro_topk_recall",
    "ez_recall_at_true_count",
    "rank_robust_composite",
    "center_gap_f1",
    "worst_center_f1",
    "best_center_f1",
    "center_hup_patient_macro_f1",
    "center_lzu_patient_macro_f1",
    "center_multicenter_patient_macro_f1",
    "center_pediatric_patient_macro_f1",
    "n_patient_rows",
    "n_unique_subjects",
    "passes_a9v3_gate",
    "passes_aaai_target_070",
]


def read_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: str | Path, data: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def normalize_center(value: Any) -> str:
    text = str(value or "").strip().lower()
    if "pediatric" in text or text == "ped":
        return "pediatric"
    if "multicenter" in text or text.startswith("multi"):
        return "multicenter"
    if "hup" in text:
        return "hup"
    if "lzu" in text:
        return "lzu"
    return text.split(":", 1)[0] if ":" in text else (text or "unknown")


def patientwise_zscore(rows: pd.DataFrame, col: str) -> pd.Series:
    values = rows[col].astype(float)
    mean = values.groupby(rows["subject_id"]).transform("mean")
    std = values.groupby(rows["subject_id"]).transform(lambda x: x.std(ddof=0))
    std = std.replace(0.0, np.nan)
    return ((values - mean) / std).replace([np.inf, -np.inf], 0.0).fillna(0.0)


def patientwise_rankpct(rows: pd.DataFrame, col: str) -> pd.Series:
    out = pd.Series(np.zeros(len(rows), dtype=float), index=rows.index)
    for _, group in rows.groupby("subject_id", sort=False):
        n = len(group)
        if n <= 1:
            out.loc[group.index] = 1.0
            continue
        ranks = group[col].astype(float).rank(method="first", ascending=False)
        out.loc[group.index] = 1.0 - (ranks - 1.0) / max(n - 1, 1)
    return out.astype(float)


def _select_topk(scores: np.ndarray, k: int) -> np.ndarray:
    pred = np.zeros(scores.shape[0], dtype=bool)
    if scores.size == 0 or int(k) <= 0:
        return pred
    order = np.argsort(scores)[::-1]
    pred[order[: min(int(k), scores.size)]] = True
    return pred


def _mrr(y_ez: np.ndarray, scores: np.ndarray) -> float:
    if y_ez.size == 0 or int(y_ez.sum()) <= 0:
        return 0.0
    order = np.argsort(scores)[::-1]
    hits = np.where(y_ez[order] == 1)[0]
    return float(1.0 / float(hits[0] + 1)) if hits.size else 0.0


def patient_topk_evaluate(
    rows: pd.DataFrame,
    *,
    score_col: str,
    method: str,
    fold_idx: int,
    selected_params: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    selected_json = json.dumps(dict(selected_params or {}), sort_keys=True)
    channel_rows: list[dict[str, Any]] = []
    patient_rows: list[dict[str, Any]] = []
    for subject_id, group in rows.groupby("subject_id", sort=False):
        group = group.reset_index(drop=True)
        y_ez = group["label_ez"].astype(int).to_numpy()
        y_nez = 1 - y_ez
        scores = group[score_col].astype(float).fillna(0.0).to_numpy()
        pred_ez = _select_topk(scores, int(y_ez.sum()))
        pred_nez = (~pred_ez).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(y_nez, pred_nez, labels=[1, 0], zero_division=0)
        patient_rows.append(
            {
                "method": method,
                "fold_idx": int(fold_idx),
                "subject_id": str(subject_id),
                "center": normalize_center(group.get("center", pd.Series(["unknown"])).iloc[0]),
                "valid_channel_count": int(len(group)),
                "ez_channel_count": int(y_ez.sum()),
                "ez_fraction": float(y_ez.mean()) if y_ez.size else 0.0,
                "patient_macro_accuracy": float(accuracy_score(y_nez, pred_nez)) if y_nez.size else 0.0,
                "patient_macro_balanced_accuracy": float(balanced_accuracy_score(y_nez, pred_nez)) if np.unique(y_nez).size > 1 else 0.0,
                "patient_macro_f1": float(f1_score(y_nez, pred_nez, average="macro", zero_division=0)),
                "patient_weighted_f1": float(f1_score(y_nez, pred_nez, average="weighted", zero_division=0)),
                "patient_macro_nez_precision": float(precision[0]),
                "patient_macro_nez_recall": float(recall[0]),
                "patient_macro_nez_f1": float(f1[0]),
                "patient_macro_ez_precision": float(precision[1]),
                "patient_macro_ez_recall": float(recall[1]),
                "patient_macro_ez_f1": float(f1[1]),
                "patient_macro_auroc_nez": float(roc_auc_score(y_nez, 1.0 - scores)) if np.unique(y_nez).size > 1 else 0.0,
                "patient_macro_auprc_nez": float(average_precision_score(y_nez, 1.0 - scores)) if np.unique(y_nez).size > 1 else 0.0,
                "patient_macro_auroc_ez": float(roc_auc_score(y_ez, scores)) if np.unique(y_ez).size > 1 else 0.0,
                "patient_macro_auprc_ez": float(average_precision_score(y_ez, scores)) if np.unique(y_ez).size > 1 else 0.0,
                "patient_macro_ez_recall_at_true_count": float(((y_ez == 1) & pred_ez).sum() / max(int(y_ez.sum()), 1)),
                "patient_macro_ez_mrr": _mrr(y_ez, scores),
                "top1_is_ez": float(y_ez[int(np.argmax(scores))] == 1) if scores.size else 0.0,
                "selected_hyperparams_json": selected_json,
            }
        )
        order = np.argsort(scores)[::-1]
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, len(order) + 1)
        for idx, row in group.iterrows():
            channel_rows.append(
                {
                    "method": method,
                    "fold_idx": int(fold_idx),
                    "subject_id": str(subject_id),
                    "center": normalize_center(row.get("center", "unknown")),
                    "channel_name": str(row["channel_name"]),
                    "label_ez": int(row["label_ez"]),
                    "score_ez_meta": float(scores[idx]),
                    "pred_ez_topk": int(pred_ez[idx]),
                    "rank_ez": int(ranks[idx]),
                    "selected_hyperparams_json": selected_json,
                }
            )
    summary = summarize_patient_rows(pd.DataFrame(patient_rows))
    return channel_rows, patient_rows, summary


def compute_rank_robust_composite(row: Mapping[str, Any]) -> float:
    return float(
        finite_float(row.get("patient_macro_f1"))
        + 0.5 * finite_float(row.get("patient_macro_ez_f1"))
        + 0.25 * finite_float(row.get("patient_macro_auprc_ez"))
        + 0.10 * finite_float(row.get("patient_macro_ez_mrr"))
        - 0.15 * finite_float(row.get("center_gap_f1"))
    )


def passes_a9v3_gate(row: Mapping[str, Any]) -> bool:
    return (
        finite_float(row.get("patient_macro_f1")) > A9V3_GATE["patient_macro_f1"]
        and finite_float(row.get("patient_macro_ez_f1")) > A9V3_GATE["patient_macro_ez_f1"]
        and finite_float(row.get("patient_macro_auprc_ez")) > A9V3_GATE["patient_macro_auprc_ez"]
        and finite_float(row.get("patient_macro_ez_mrr")) >= A9V3_GATE["patient_macro_ez_mrr"]
        and finite_float(row.get("top1_is_ez_rate")) >= A9V3_GATE["top1_is_ez_rate"]
    )


def passes_aaai_target(row: Mapping[str, Any]) -> bool:
    return (
        finite_float(row.get("patient_macro_f1")) >= AAAI_TARGET_070["patient_macro_f1"]
        and finite_float(row.get("patient_macro_ez_f1")) >= AAAI_TARGET_070["patient_macro_ez_f1"]
        and finite_float(row.get("patient_macro_auprc_ez")) >= AAAI_TARGET_070["patient_macro_auprc_ez"]
        and finite_float(row.get("patient_macro_ez_mrr")) >= AAAI_TARGET_070["patient_macro_ez_mrr"]
        and finite_float(row.get("top1_is_ez_rate")) >= AAAI_TARGET_070["top1_is_ez_rate"]
    )


def summarize_patient_rows(patient_rows: pd.DataFrame) -> dict[str, Any]:
    if patient_rows.empty:
        return {key: 0.0 for key in REQUIRED_SUMMARY_COLUMNS}
    summary: dict[str, Any] = {}
    metric_cols = [col for col in patient_rows.columns if col.startswith("patient_macro_") or col == "patient_weighted_f1"]
    for col in metric_cols:
        summary[col] = float(patient_rows[col].astype(float).mean())
    summary["top1_is_ez_rate"] = float(patient_rows["top1_is_ez"].astype(float).mean()) if "top1_is_ez" in patient_rows else 0.0
    summary["macro_topk_recall"] = summary.get("patient_macro_ez_recall_at_true_count", 0.0)
    summary["ez_recall_at_true_count"] = summary.get("patient_macro_ez_recall_at_true_count", 0.0)
    center_means = patient_rows.groupby("center", dropna=False)["patient_macro_f1"].mean().to_dict() if "center" in patient_rows else {}
    center_means = {normalize_center(k): float(v) for k, v in center_means.items()}
    if center_means:
        worst, best = min(center_means.values()), max(center_means.values())
    else:
        worst, best = 0.0, 0.0
    summary["worst_center_f1"] = float(worst)
    summary["best_center_f1"] = float(best)
    summary["center_gap_f1"] = float(best - worst) if len(center_means) >= 2 else 0.0
    for center in ("hup", "lzu", "multicenter", "pediatric"):
        summary[f"center_{center}_patient_macro_f1"] = float(center_means.get(center, 0.0))
    summary["n_patient_rows"] = float(len(patient_rows))
    summary["n_unique_subjects"] = float(patient_rows["subject_id"].nunique()) if "subject_id" in patient_rows else 0.0
    summary["rank_robust_composite"] = compute_rank_robust_composite(summary)
    summary["passes_a9v3_gate"] = bool(passes_a9v3_gate(summary))
    summary["passes_aaai_target_070"] = bool(passes_aaai_target(summary))
    for key in REQUIRED_SUMMARY_COLUMNS:
        summary.setdefault(key, 0.0)
    return summary


def forbidden_feature_columns(columns: Sequence[str]) -> set[str]:
    forbidden = {
        "subject_id",
        "channel_name",
        "center",
        "center_id",
        "fold_idx",
        "label_ez",
        "label_nez",
        "ez_fraction",
        "n_ez",
        "true_ez_count",
        "ez_channel_count",
        "valid_channel_count",
        "n_records",
        "n_valid_records",
        "n_valid_windows",
    }
    out = set()
    for col in columns:
        lower = str(col).lower()
        if (
            lower in forbidden
            or "label" in lower
            or "true" in lower
            or "center" in lower
            or lower.endswith("_count")
        ):
            out.add(str(col))
    return out
