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


def _safe_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _patient_rows(df: pd.DataFrame, *, method: str, score_col: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for subject_id, group in df.groupby("subject_id", sort=True):
        group = group.copy()
        y = pd.to_numeric(group["true_ez"], errors="coerce").fillna(0).astype(int).to_numpy()
        scores = pd.to_numeric(group[score_col], errors="coerce").fillna(0.0).astype(float).to_numpy()
        true_count = int(y.sum())
        if true_count <= 0:
            continue
        order = np.argsort(-scores, kind="mergesort")
        pred = np.zeros_like(y)
        pred[order[:true_count]] = 1
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)
        ez_positions = np.where(y[order] == 1)[0]
        mrr = 0.0 if ez_positions.size == 0 else 1.0 / float(ez_positions[0] + 1)
        auprc = float(average_precision_score(y, scores)) if np.unique(y).size > 1 else 0.0
        top3_recall = float(y[order[: min(3, len(order))]].sum() / max(true_count, 1))
        top5_recall = float(y[order[: min(5, len(order))]].sum() / max(true_count, 1))
        rows.append(
            {
                "method": method,
                "subject_id": str(subject_id),
                "center": str(group["center"].iloc[0]) if "center" in group.columns else "unknown",
                "fold_idx": int(group["fold_idx"].iloc[0]) if "fold_idx" in group.columns else -1,
                "patient_macro_f1": float(f1),
                "patient_macro_precision": float(precision),
                "patient_macro_recall": float(recall),
                "patient_macro_ez_f1": float(f1),
                "patient_macro_auprc_ez": float(auprc),
                "patient_macro_ez_mrr": float(mrr),
                "top1_is_ez_rate": float(y[order[0]] == 1) if order.size else 0.0,
                "ez_recall_at_true_count": float(recall),
                "macro_top3_recall": top3_recall,
                "macro_top5_recall": top5_recall,
            }
        )
    return pd.DataFrame(rows)


def _summarize(patient_rows: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if patient_rows.empty:
        return pd.DataFrame(columns=group_cols)
    metric_cols = [
        "patient_macro_f1",
        "patient_macro_precision",
        "patient_macro_recall",
        "patient_macro_ez_f1",
        "patient_macro_auprc_ez",
        "patient_macro_ez_mrr",
        "top1_is_ez_rate",
        "ez_recall_at_true_count",
        "macro_top3_recall",
        "macro_top5_recall",
    ]
    summary = patient_rows.groupby(group_cols, sort=True, dropna=False)[metric_cols].mean().reset_index()
    summary["n_patients"] = patient_rows.groupby(group_cols, sort=True, dropna=False)["subject_id"].nunique().to_numpy()
    return summary


def _add_center_gap(summary: pd.DataFrame, by_center: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    worst = {}
    gap = {}
    if not by_center.empty:
        for method, group in by_center.groupby("method", sort=False):
            values = group["patient_macro_f1"].astype(float).to_numpy()
            worst[method] = float(values.min()) if values.size else 0.0
            gap[method] = float(values.max() - values.min()) if values.size else 0.0
    out["worst_center_f1"] = out["method"].map(worst).fillna(out["patient_macro_f1"])
    out["center_gap_f1"] = out["method"].map(gap).fillna(0.0)
    return out


def _method_from_corrected_path(path: Path) -> str:
    name = path.stem
    if name.startswith("corrected_oof_"):
        name = name[len("corrected_oof_") :]
    return name


def evaluate_hnc_outputs(v3_ledger: str | Path, corrected_dir: str | Path, output_dir: str | Path) -> pd.DataFrame:
    v3_ledger = Path(v3_ledger)
    corrected_dir = Path(corrected_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets: list[tuple[str, pd.DataFrame, str]] = [("V3", pd.read_csv(v3_ledger), "score_ez_probability")]
    for path in sorted(corrected_dir.glob("corrected_oof_rawbrainbert_hnc_*.csv")):
        frame = pd.read_csv(path)
        score_col = "final_score_ez_probability" if "final_score_ez_probability" in frame.columns else "score_ez_probability"
        datasets.append((_method_from_corrected_path(path), frame, score_col))

    patient_frames = [_patient_rows(frame, method=method, score_col=score_col) for method, frame, score_col in datasets]
    patient_df = pd.concat(patient_frames, ignore_index=True) if patient_frames else pd.DataFrame()
    summary = _summarize(patient_df, ["method"])
    by_center = _summarize(patient_df, ["method", "center"])
    by_fold = _summarize(patient_df, ["method", "fold_idx"])
    summary = _add_center_gap(summary, by_center)
    summary.to_csv(output_dir / "rawbrainbert_hnc_metrics_summary.csv", index=False)
    summary.to_json(output_dir / "rawbrainbert_hnc_metrics_summary.json", orient="records", indent=2)
    by_center.to_csv(output_dir / "rawbrainbert_hnc_metrics_by_center.csv", index=False)
    by_fold.to_csv(output_dir / "rawbrainbert_hnc_metrics_by_fold.csv", index=False)

    baseline = summary[summary["method"].eq("V3")]
    delta: dict[str, Any] = {}
    if not baseline.empty:
        base = baseline.iloc[0]
        for _, row in summary.iterrows():
            if row["method"] == "V3":
                continue
            delta[str(row["method"])] = {
                "delta_patient_macro_f1": float(row["patient_macro_f1"] - base["patient_macro_f1"]),
                "delta_mrr": float(row["patient_macro_ez_mrr"] - base["patient_macro_ez_mrr"]),
                "delta_top1_is_ez_rate": float(row["top1_is_ez_rate"] - base["top1_is_ez_rate"]),
                "delta_worst_center_f1": float(row["worst_center_f1"] - base["worst_center_f1"]),
            }
    with (output_dir / "rawbrainbert_hnc_delta_vs_v3.json").open("w", encoding="utf-8") as fout:
        json.dump(delta, fout, indent=2, ensure_ascii=False, sort_keys=True)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate V3-RawBrainBERT-HNC corrected ledgers.")
    parser.add_argument("--v3-ledger", required=True)
    parser.add_argument("--corrected-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    summary = evaluate_hnc_outputs(args.v3_ledger, args.corrected_dir, args.output_dir)
    print(summary.to_json(orient="records"))


if __name__ == "__main__":
    main()
