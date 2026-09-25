from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_recall_fscore_support


METRIC_KEYS = (
    "patient_macro_f1",
    "patient_macro_ez_f1",
    "patient_macro_auprc_ez",
    "patient_macro_ez_mrr",
    "top1_is_ez_rate",
)


def _reciprocal_rank(y_ez: np.ndarray, score_ez: np.ndarray) -> float:
    positives = np.where(y_ez > 0.5)[0]
    if positives.size == 0:
        return 0.0
    order = np.argsort(score_ez)[::-1]
    ranks = {idx: rank for rank, idx in enumerate(order, start=1)}
    return float(max(1.0 / ranks[int(idx)] for idx in positives))


def _ensure_patient_predictions(group: pd.DataFrame) -> pd.DataFrame:
    group = group.copy()
    if "rank_eval" not in group.columns or (group["rank_eval"].fillna(-1).astype(int) <= 0).any():
        order = group["score_eval"].astype(float).rank(method="first", ascending=False)
        group["rank_eval"] = order.astype(int)
    k = int(max(round(float(group["label_ez"].sum())), 0))
    if "patient_ez_count" in group.columns:
        k = int(max(round(float(group["patient_ez_count"].iloc[0])), k))
    if k <= 0:
        group["pred_topk"] = 0
    else:
        group["pred_topk"] = (group["rank_eval"].astype(int) <= k).astype(int)
    group["is_top1"] = (group["rank_eval"].astype(int) == 1).astype(int)
    group["patient_ez_count"] = k
    return group


def normalize_ledger_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    required = {"subject_id", "label_ez", "score_eval"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Ledger missing required columns: {sorted(missing)}")
    if "patient_id" not in df.columns:
        df = df.copy()
        df["patient_id"] = df["subject_id"].astype(str)
    groups = []
    for _, group in df.groupby(["patient_id"], sort=False):
        groups.append(_ensure_patient_predictions(group))
    return pd.concat(groups, ignore_index=True) if groups else df.copy()


def summarize_ledger_dataframe(df: pd.DataFrame) -> tuple[dict[str, float], pd.DataFrame]:
    ledger = normalize_ledger_dataframe(df)
    patient_metrics: dict[str, list[float]] = {
        "macro_f1": [],
        "ez_f1": [],
        "auprc_ez": [],
        "ez_mrr": [],
        "top1_is_ez": [],
    }

    def add_patient_metrics(group: pd.DataFrame, bucket: dict[str, list[float]]) -> None:
        y_ez = group["label_ez"].astype(float).to_numpy()
        y_nez = 1 - y_ez
        pred_ez = group["pred_topk"].astype(int).to_numpy()
        pred_nez = 1 - pred_ez
        score_ez = group["score_eval"].astype(float).to_numpy()
        _, _, f1, _ = precision_recall_fscore_support(y_nez, pred_nez, labels=[1, 0], zero_division=0)
        bucket["macro_f1"].append(float(f1_score(y_nez, pred_nez, average="macro", zero_division=0)))
        bucket["ez_f1"].append(float(f1[1]))
        bucket["auprc_ez"].append(float(average_precision_score(y_ez, score_ez)) if np.unique(y_ez).size > 1 else 0.0)
        bucket["ez_mrr"].append(_reciprocal_rank(y_ez, score_ez))
        top1_rows = group[group["is_top1"].astype(int) == 1]
        bucket["top1_is_ez"].append(float(top1_rows["label_ez"].max()) if not top1_rows.empty else 0.0)

    for _, group in ledger.groupby("patient_id", sort=False):
        add_patient_metrics(group, patient_metrics)

    summary = {
        "n_patients": float(ledger["patient_id"].nunique()),
        "n_records": float(ledger[["patient_id", "fold_id"]].drop_duplicates().shape[0]) if "fold_id" in ledger.columns else float(ledger["patient_id"].nunique()),
        "n_channels_total": float(len(ledger)),
        "n_centers": float(ledger["center"].nunique()) if "center" in ledger.columns else 0.0,
        "patient_macro_f1": float(np.mean(patient_metrics["macro_f1"])) if patient_metrics["macro_f1"] else 0.0,
        "patient_macro_ez_f1": float(np.mean(patient_metrics["ez_f1"])) if patient_metrics["ez_f1"] else 0.0,
        "patient_macro_auprc_ez": float(np.mean(patient_metrics["auprc_ez"])) if patient_metrics["auprc_ez"] else 0.0,
        "patient_macro_ez_mrr": float(np.mean(patient_metrics["ez_mrr"])) if patient_metrics["ez_mrr"] else 0.0,
        "top1_is_ez_rate": float(np.mean(patient_metrics["top1_is_ez"])) if patient_metrics["top1_is_ez"] else 0.0,
    }

    by_center_rows = []
    if "center" in ledger.columns:
        for center, center_df in ledger.groupby("center", sort=True):
            center_metrics = {key: [] for key in patient_metrics}
            for _, group in center_df.groupby("patient_id", sort=False):
                add_patient_metrics(group, center_metrics)
            by_center_rows.append(
                {
                    "center": center,
                    "n_patients": float(center_df["patient_id"].nunique()),
                    "n_channels_total": float(len(center_df)),
                    "patient_macro_f1": float(np.mean(center_metrics["macro_f1"])) if center_metrics["macro_f1"] else 0.0,
                    "patient_macro_ez_f1": float(np.mean(center_metrics["ez_f1"])) if center_metrics["ez_f1"] else 0.0,
                    "patient_macro_auprc_ez": float(np.mean(center_metrics["auprc_ez"])) if center_metrics["auprc_ez"] else 0.0,
                    "patient_macro_ez_mrr": float(np.mean(center_metrics["ez_mrr"])) if center_metrics["ez_mrr"] else 0.0,
                    "top1_is_ez_rate": float(np.mean(center_metrics["top1_is_ez"])) if center_metrics["top1_is_ez"] else 0.0,
                }
            )
    return summary, pd.DataFrame(by_center_rows)


def _load_original_summary(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as fin:
        return json.load(fin)


def _check_mixed_splits(ledger: pd.DataFrame, allow_mixed: bool) -> None:
    if "split_role" not in ledger.columns:
        return
    roles = set(ledger["split_role"].astype(str).str.lower().unique())
    if len(roles) > 1 and not allow_mixed:
        raise ValueError(
            f"Ledger contains multiple split_roles: {sorted(roles)}. "
            f"Pass --allow_mixed_splits to override, but prefer --split_role test for evaluation."
        )


def build_audit(
    ledger: pd.DataFrame,
    summary: dict[str, float],
    *,
    original_summary: dict[str, Any] | None = None,
    run_args: dict[str, Any] | None = None,
) -> dict[str, Any]:
    original_summary = original_summary or {}
    run_args = run_args or {}
    diffs = {}
    for key in ("patient_macro_f1", "patient_macro_ez_f1", "patient_macro_auprc_ez", "patient_macro_ez_mrr"):
        if key in original_summary:
            diffs[key] = float(summary.get(key, 0.0)) - float(original_summary.get(key, 0.0))
    success = True
    if diffs:
        success = all(abs(value) <= 1e-5 for value in diffs.values())
    split_roles_present = (
        sorted(set(ledger["split_role"].astype(str).str.lower().unique()))
        if "split_role" in ledger.columns
        else ["unknown"]
    )
    return {
        "n_patients": int(summary.get("n_patients", 0)),
        "n_records": int(summary.get("n_records", 0)),
        "n_channels_total": int(summary.get("n_channels_total", 0)),
        "n_centers": int(summary.get("n_centers", 0)),
        "positive_label": run_args.get("positive_label", "unknown"),
        "drop_high_ez_fraction_lzu": run_args.get("drop_high_ez_fraction_lzu", "unknown"),
        "split_strategy": run_args.get("split_strategy", "unknown"),
        "n_splits": run_args.get("n_splits", "unknown"),
        "random_seed": run_args.get("random_seed", "unknown"),
        "split_roles_present": split_roles_present,
        "metric_reproduction_success": bool(success),
        "metric_diff_vs_original_summary": diffs,
        "ledger_columns": list(ledger.columns),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a patient-channel prediction ledger.")
    parser.add_argument("--ledger_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--original_summary_json", type=str, default=None)
    parser.add_argument("--run_args_json", type=str, default=None)
    parser.add_argument("--allow_mixed_splits", action="store_true", default=False)
    args = parser.parse_args()

    ledger_path = Path(args.ledger_csv)
    output_dir = Path(args.output_dir) if args.output_dir else ledger_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    ledger = pd.read_csv(ledger_path)
    _check_mixed_splits(ledger, allow_mixed=args.allow_mixed_splits)
    ledger = normalize_ledger_dataframe(ledger)
    summary, by_center = summarize_ledger_dataframe(ledger)
    original_summary = _load_original_summary(Path(args.original_summary_json)) if args.original_summary_json else {}
    run_args = _load_original_summary(Path(args.run_args_json)) if args.run_args_json else {}
    audit = build_audit(ledger, summary, original_summary=original_summary, run_args=run_args)

    pd.DataFrame([summary]).to_csv(output_dir / "patient_channel_ledger_summary.csv", index=False)
    by_center.to_csv(output_dir / "patient_channel_ledger_by_center.csv", index=False)
    with open(output_dir / "ledger_eval_audit.json", "w", encoding="utf-8") as fout:
        json.dump(audit, fout, indent=2, ensure_ascii=False, sort_keys=True)
    print(f"Wrote {output_dir / 'patient_channel_ledger_summary.csv'}")
    print(f"Wrote {output_dir / 'patient_channel_ledger_by_center.csv'}")
    print(f"Wrote {output_dir / 'ledger_eval_audit.json'}")


if __name__ == "__main__":
    main()
