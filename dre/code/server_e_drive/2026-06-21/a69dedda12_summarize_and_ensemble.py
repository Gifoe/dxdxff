from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_recall_fscore_support, roc_auc_score, average_precision_score


def load_summary(run_dir: Path) -> dict | None:
    p = run_dir / "heldout_summary_neuroez_v3.json"
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    d["config"] = run_dir.name
    d["output_dir"] = str(run_dir)
    return d


def select_topk(scores: np.ndarray, k: int, valid_mask: np.ndarray) -> np.ndarray:
    pred = np.zeros(scores.shape[0], dtype=bool)
    valid_idx = np.where(valid_mask)[0]
    if valid_idx.size == 0:
        return pred
    k = max(1, min(int(k), int(valid_idx.size)))
    order = valid_idx[np.argsort(scores[valid_idx])[::-1]]
    pred[order[:k]] = True
    return pred


def reciprocal_rank(y_true: np.ndarray, scores: np.ndarray) -> float:
    if y_true.size == 0 or int((y_true == 1).sum()) == 0:
        return 0.0
    order = np.argsort(scores)[::-1]
    positive_ranks = np.where(y_true[order] == 1)[0]
    return float(1.0 / float(positive_ranks[0] + 1)) if positive_ranks.size else 0.0


def evaluate_channel_df(df: pd.DataFrame) -> dict:
    patient_metrics: Dict[str, List[float]] = {k: [] for k in [
        "macro_f1", "ez_f1", "ez_recall", "ez_precision", "auroc_ez", "auprc_ez", "ez_mrr", "ez_recall_at_true_count"
    ]}
    pooled_y = []
    pooled_score = []
    pooled_pred_nez = []
    pooled_y_nez = []
    for sid, g in df.groupby("subject_id", sort=True):
        y_ez = g["true_ez"].to_numpy(dtype=int)
        y_nez = 1 - y_ez
        score_ez = g["score_ez_probability"].to_numpy(dtype=float)
        valid = np.ones_like(y_ez, dtype=bool)
        pred_ez = select_topk(score_ez, int(y_ez.sum()), valid)
        pred_nez = (~pred_ez).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(y_nez, pred_nez, labels=[1,0], zero_division=0)
        patient_metrics["macro_f1"].append(float(f1_score(y_nez, pred_nez, average="macro", zero_division=0)))
        patient_metrics["ez_precision"].append(float(p[1]))
        patient_metrics["ez_recall"].append(float(r[1]))
        patient_metrics["ez_f1"].append(float(f1[1]))
        patient_metrics["ez_recall_at_true_count"].append(float(((y_ez == 1) & pred_ez).sum() / max(int(y_ez.sum()), 1)))
        patient_metrics["ez_mrr"].append(reciprocal_rank(y_ez, score_ez))
        if np.unique(y_ez).size > 1:
            patient_metrics["auroc_ez"].append(float(roc_auc_score(y_ez, score_ez)))
            patient_metrics["auprc_ez"].append(float(average_precision_score(y_ez, score_ez)))
        pooled_y.append(y_ez)
        pooled_score.append(score_ez)
        pooled_y_nez.append(y_nez)
        pooled_pred_nez.append(pred_nez)
    summary = {
        "patient_macro_f1": float(np.mean(patient_metrics["macro_f1"])) if patient_metrics["macro_f1"] else 0.0,
        "patient_macro_ez_f1": float(np.mean(patient_metrics["ez_f1"])) if patient_metrics["ez_f1"] else 0.0,
        "patient_macro_ez_recall": float(np.mean(patient_metrics["ez_recall"])) if patient_metrics["ez_recall"] else 0.0,
        "patient_macro_ez_precision": float(np.mean(patient_metrics["ez_precision"])) if patient_metrics["ez_precision"] else 0.0,
        "patient_macro_auroc_ez": float(np.mean(patient_metrics["auroc_ez"])) if patient_metrics["auroc_ez"] else 0.0,
        "patient_macro_auprc_ez": float(np.mean(patient_metrics["auprc_ez"])) if patient_metrics["auprc_ez"] else 0.0,
        "patient_macro_ez_mrr": float(np.mean(patient_metrics["ez_mrr"])) if patient_metrics["ez_mrr"] else 0.0,
        "patient_macro_ez_recall_at_true_count": float(np.mean(patient_metrics["ez_recall_at_true_count"])) if patient_metrics["ez_recall_at_true_count"] else 0.0,
    }
    if pooled_y:
        y_ez_all = np.concatenate(pooled_y)
        score_ez_all = np.concatenate(pooled_score)
        y_nez_all = np.concatenate(pooled_y_nez)
        pred_nez_all = np.concatenate(pooled_pred_nez)
        summary.update({
            "pooled_macro_f1": float(f1_score(y_nez_all, pred_nez_all, average="macro", zero_division=0)),
            "pooled_auroc_ez": float(roc_auc_score(y_ez_all, score_ez_all)) if np.unique(y_ez_all).size > 1 else 0.0,
            "pooled_auprc_ez": float(average_precision_score(y_ez_all, score_ez_all)) if np.unique(y_ez_all).size > 1 else 0.0,
        })
    return summary


def read_channels(run_dir: Path) -> pd.DataFrame | None:
    files = sorted(run_dir.glob("test_channel_predictions_neuroez_v2_fold_*.csv"))
    if not files:
        return None
    dfs = [pd.read_csv(p) for p in files]
    df = pd.concat(dfs, ignore_index=True)
    needed = {"fold_idx", "subject_id", "channel_name", "true_ez", "score_ez_probability"}
    if not needed.issubset(df.columns):
        return None
    return df


def ensemble(run_dirs: list[Path], out_dir: Path, tag: str) -> dict | None:
    frames = []
    key_cols = ["fold_idx", "subject_id", "channel_name"]
    truth_cols = ["true_ez", "true_nez"]
    for idx, d in enumerate(run_dirs):
        ch = read_channels(d)
        if ch is None:
            return None
        keep = key_cols + [c for c in truth_cols if c in ch.columns] + ["score_ez_probability"]
        ch = ch[keep].copy()
        ch = ch.rename(columns={"score_ez_probability": f"score_{idx}"})
        if idx > 0:
            ch = ch.drop(columns=[c for c in truth_cols if c in ch.columns])
        frames.append(ch)
    merged = frames[0]
    for ch in frames[1:]:
        merged = merged.merge(ch, on=key_cols, how="inner")
    score_cols = [c for c in merged.columns if c.startswith("score_")]
    merged["score_ez_probability"] = merged[score_cols].mean(axis=1)
    summary = evaluate_channel_df(merged)
    summary["config"] = tag
    summary["members"] = ";".join(d.name for d in run_dirs)
    out_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_dir / f"{tag}_channel_predictions.csv", index=False)
    (out_dir / f"{tag}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--top_n", type=int, default=3)
    args = ap.parse_args()
    root = Path(args.root)
    summaries = []
    for d in sorted(root.iterdir()):
        if d.is_dir():
            s = load_summary(d)
            if s is not None:
                summaries.append(s)
    if not summaries:
        print("no completed summaries found")
        return
    df = pd.DataFrame(summaries)
    cols = [c for c in [
        "config", "output_dir", "patient_macro_f1", "patient_macro_ez_f1", "patient_macro_auroc_ez", "patient_macro_auprc_ez", "patient_macro_ez_mrr", "pooled_macro_f1", "pooled_auroc_ez", "pooled_auprc_ez"
    ] if c in df.columns]
    df = df.sort_values("patient_macro_f1", ascending=False)
    df[cols].to_csv(root / "a0_to_a6_grid_summary.csv", index=False)
    print(df[cols].to_string(index=False))

    top_dirs = [Path(p) for p in df.head(int(args.top_n))["output_dir"].tolist()]
    ens_dir = root / "A7_ensembles"
    ens = ensemble(top_dirs, ens_dir, f"a7_top{len(top_dirs)}_score_average")
    if ens is not None:
        pd.DataFrame([ens]).to_csv(root / "a7_ensemble_summary.csv", index=False)
        print("\nA7 ensemble:")
        print(json.dumps(ens, indent=2))


if __name__ == "__main__":
    main()
