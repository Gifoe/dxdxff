from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neuroez_c.clean_nez_utils import add_label_encoding_columns, apply_allowed_subject_filter, json_safe


def _patient_rows(
    df: pd.DataFrame,
    *,
    method: str,
    score_col: str,
    pred_col: str | None = None,
    diagnostic_oracle_k: bool | None = None,
    no_leak: bool | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for subject_id, group in df.groupby("subject_id", sort=False):
        target_col = "clinical_true_ez" if "clinical_true_ez" in group.columns else "true_ez"
        y = pd.to_numeric(group[target_col], errors="coerce").fillna(0).astype(int).to_numpy()
        scores = pd.to_numeric(group[score_col], errors="coerce").fillna(0.0).astype(float).to_numpy()
        if pred_col and pred_col in group.columns:
            pred = pd.to_numeric(group[pred_col], errors="coerce").fillna(0).astype(int).to_numpy()
        else:
            k = int(y.sum())
            pred = np.zeros_like(y)
            if k > 0:
                order = np.argsort(-scores, kind="mergesort")
                pred[order[:k]] = 1
        true_count = int(y.sum())
        if true_count <= 0:
            continue
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        ez_f1 = 0.0 if (2 * tp + fp + fn) == 0 else (2.0 * tp) / float(2 * tp + fp + fn)
        nez_f1 = 0.0 if (2 * tn + fn + fp) == 0 else (2.0 * tn) / float(2 * tn + fn + fp)
        macro_f1 = 0.5 * (ez_f1 + nez_f1)
        recall = tp / max(tp + fn, 1)
        order = np.argsort(-scores, kind="mergesort")
        ez_positions = np.where(y[order] == 1)[0]
        mrr = 0.0 if ez_positions.size == 0 else 1.0 / float(ez_positions[0] + 1)
        auprc = float(average_precision_score(y, scores)) if np.unique(y).size > 1 else 0.0
        rows.append(
            {
                "method": method,
                "fold_idx": int(pd.to_numeric(group["fold_idx"], errors="coerce").dropna().iloc[0]),
                "subject_id": str(subject_id),
                "center": str(group["center"].iloc[0]) if "center" in group.columns else "unknown",
                "n_channels": int(len(group)),
                "n_true_ez": int(y.sum()),
                "n_true_nez": int((y == 0).sum()),
                "n_pred_ez": int(pred.sum()),
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "tn": int(tn),
                "patient_macro_f1": float(macro_f1),
                "patient_ez_f1": float(ez_f1),
                "patient_nez_f1": float(nez_f1),
                "patient_macro_auprc_ez": float(auprc),
                "patient_macro_ez_mrr": float(mrr),
                "top1_is_ez": float(y[order[0]] == 1) if order.size else 0.0,
                "recall_at_true_count": float(recall),
                "diagnostic_oracle_k": bool(diagnostic_oracle_k) if diagnostic_oracle_k is not None else bool(pred_col is None or pred_col == "predicted_by_oracle_k"),
                "no_leak": bool(no_leak) if no_leak is not None else bool(pred_col == "predicted_by_kcal"),
            }
        )
    return pd.DataFrame(rows)


def _summarize(patient_rows: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if patient_rows.empty:
        return pd.DataFrame(columns=group_cols)
    metric_cols = [
        "patient_macro_f1",
        "patient_ez_f1",
        "patient_nez_f1",
        "patient_macro_auprc_ez",
        "patient_macro_ez_mrr",
        "top1_is_ez",
        "recall_at_true_count",
    ]
    out = patient_rows.groupby(group_cols, sort=True, dropna=False)[metric_cols].mean().reset_index()
    out["n_patients"] = patient_rows.groupby(group_cols, sort=True, dropna=False)["subject_id"].nunique().to_numpy()
    return out


def evaluate_clean_nez_pipeline(
    ledger: str | Path,
    output_dir: str | Path,
    *,
    v3_ledger: str | Path | None = None,
    allowed_subjects_ledger: str | Path | None = None,
    allowed_subjects_file: str | Path | None = None,
    require_n_patients: int | None = None,
    label_encoding_mode: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    main = pd.read_csv(ledger)
    main["subject_id"] = main["subject_id"].astype(str)
    main = add_label_encoding_columns(main, label_encoding_mode=label_encoding_mode)
    main, filter_audit = apply_allowed_subject_filter(
        main,
        allowed_subjects_ledger=allowed_subjects_ledger,
        allowed_subjects_file=allowed_subjects_file,
        require_n_patients=require_n_patients,
    )
    datasets: list[tuple[str, pd.DataFrame, str, str | None, bool, bool]] = []
    v3_subject_set_equals_main = None
    if v3_ledger:
        v3 = pd.read_csv(v3_ledger)
        v3["subject_id"] = v3["subject_id"].astype(str)
        v3 = add_label_encoding_columns(v3, label_encoding_mode=label_encoding_mode)
        v3, v3_filter_audit = apply_allowed_subject_filter(
            v3,
            allowed_subjects_ledger=allowed_subjects_ledger,
            allowed_subjects_file=allowed_subjects_file,
            require_n_patients=require_n_patients,
        )
        main_subjects = set(main["subject_id"].astype(str).unique())
        v3_subjects = set(v3["subject_id"].astype(str).unique())
        v3_subject_set_equals_main = main_subjects == v3_subjects
        if not v3_subject_set_equals_main:
            raise ValueError(
                "subject_id set mismatch between main ledger and V3 ledger: "
                f"missing_in_v3={sorted(main_subjects - v3_subjects)[:20]}, "
                f"extra_in_v3={sorted(v3_subjects - main_subjects)[:20]}"
            )
        if "predicted_ez" not in v3.columns:
            raise ValueError("V3 ledger must contain predicted_ez so predicted baseline and oracle-K diagnostic are not conflated.")
        datasets.append(("V3 baseline predicted_ez", v3, "score_ez_probability", "predicted_ez", False, True))
        datasets.append(("V3 baseline oracle-K diagnostic", v3, "score_ez_probability", None, True, False))
    else:
        v3_filter_audit = {}
    if "base_suspicious_logit" in main.columns:
        datasets.append(("CleanNEZ base score oracle-K diagnostic", main, "base_suspicious_logit", None, True, False))
    if "raw_dist_onset_z" in main.columns:
        datasets.append(("RawBrainBERT-NEZDistance oracle-K diagnostic", main, "raw_dist_onset_z", None, True, False))
    if "final_suspicious_logit" in main.columns:
        datasets.append(("CleanNEZ + RawDistance + SetTopo oracle-K diagnostic", main, "final_suspicious_logit", "predicted_by_oracle_k" if "predicted_by_oracle_k" in main.columns else None, True, False))
    if "predicted_by_kcal" in main.columns:
        datasets.append(("CleanNEZ + RawDistance + SetTopo + KCal no-leak", main, "final_suspicious_logit", "predicted_by_kcal", False, True))

    patient_df = pd.concat(
        [
            _patient_rows(
                frame,
                method=method,
                score_col=score_col,
                pred_col=pred_col,
                diagnostic_oracle_k=diagnostic_oracle_k,
                no_leak=no_leak,
            )
            for method, frame, score_col, pred_col, diagnostic_oracle_k, no_leak in datasets
        ],
        ignore_index=True,
    ) if datasets else pd.DataFrame()
    summary = _summarize(patient_df, ["method"])
    by_fold = _summarize(patient_df, ["method", "fold_idx"])
    by_center = _summarize(patient_df, ["method", "center"])
    if not by_center.empty and not summary.empty:
        worst = by_center.groupby("method")["patient_macro_f1"].min().to_dict()
        best = by_center.groupby("method")["patient_macro_f1"].max().to_dict()
        summary["worst_center_f1"] = summary["method"].map(worst).fillna(summary["patient_macro_f1"])
        summary["center_gap_f1"] = summary["method"].map({k: best[k] - worst[k] for k in worst}).fillna(0.0)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "metrics_summary.csv", index=False)
    summary.to_json(output_dir / "metrics_summary.json", orient="records", indent=2)
    by_fold.to_csv(output_dir / "metrics_by_fold.csv", index=False)
    by_center.to_csv(output_dir / "metrics_by_center.csv", index=False)
    patient_df.to_csv(output_dir / "patient_level_metrics.csv", index=False)
    delta: dict[str, Any] = {}
    baseline = summary[summary["method"].eq("V3 baseline predicted_ez")] if not summary.empty else pd.DataFrame()
    if not baseline.empty:
        base = baseline.iloc[0]
        for _, row in summary.iterrows():
            if row["method"] == "V3 baseline predicted_ez":
                continue
            delta[str(row["method"])] = {
                "delta_patient_macro_f1": float(row["patient_macro_f1"] - base["patient_macro_f1"]),
                "delta_mrr": float(row["patient_macro_ez_mrr"] - base["patient_macro_ez_mrr"]),
            }
    with (output_dir / "delta_vs_v3.json").open("w", encoding="utf-8") as fout:
        json.dump(json_safe(delta), fout, indent=2, ensure_ascii=False, sort_keys=True)
    audit = {
        **filter_audit,
        "v3_filter_audit": v3_filter_audit,
        "methods": [item[0] for item in datasets],
        "n_patient_rows": int(len(patient_df)),
        "macro_f1_formula": "0.5*(ez_f1+nez_f1)",
        "v3_subject_set_equals_main": bool(v3_subject_set_equals_main) if v3_subject_set_equals_main is not None else None,
        "label_encoding_mode": str(main["label_encoding_mode"].iloc[0]) if "label_encoding_mode" in main.columns and not main.empty else None,
        "ez_label_value": int(main["ez_label_value"].iloc[0]) if "ez_label_value" in main.columns and not main.empty else None,
        "nez_label_value": int(main["nez_label_value"].iloc[0]) if "nez_label_value" in main.columns and not main.empty else None,
        "evaluation_target": "clinical_true_ez",
        "raw_binary_label_used_as_metric_target": False,
    }
    with (output_dir / "evaluation_audit.json").open("w", encoding="utf-8") as fout:
        json.dump(json_safe(audit), fout, indent=2, ensure_ascii=False, sort_keys=True)
    return summary, audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate CleanNEZ RawBB SetTopo KCal ledgers.")
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--v3-ledger", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--allowed-subjects-ledger", default=None)
    parser.add_argument("--allowed-subjects-file", default=None)
    parser.add_argument("--require-n-patients", type=int, default=None)
    parser.add_argument("--label-encoding-mode", choices=["ez1", "ez0"], default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    _, audit = evaluate_clean_nez_pipeline(
        args.ledger,
        args.output_dir,
        v3_ledger=args.v3_ledger,
        allowed_subjects_ledger=args.allowed_subjects_ledger,
        allowed_subjects_file=args.allowed_subjects_file,
        require_n_patients=args.require_n_patients,
        label_encoding_mode=args.label_encoding_mode,
    )
    print(json.dumps(json_safe(audit), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
