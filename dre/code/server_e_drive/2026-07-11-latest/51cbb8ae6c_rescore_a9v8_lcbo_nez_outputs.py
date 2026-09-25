"""Rescore NEZ LCBO outputs while separating validation selection from exploration."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

GAMMAS = [-0.20, -0.10, -0.05, 0.00, 0.02, 0.05, 0.10, 0.15]
ALPHAS = [0.05, 0.10, 0.20]
BETAS = [-0.10, -0.05, 0.00, 0.05, 0.10]


def _resolved_args(run_dir: Path) -> dict[str, object]:
    path = run_dir / "resolved_run_args.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}; rescore cannot infer positive-label or teacher semantics safely."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _positive_to_nez(values: np.ndarray, positive_label: str) -> np.ndarray:
    return values.astype(float) if positive_label == "nez" else 1.0 - values.astype(float)


def _z(values: pd.Series) -> pd.Series:
    array = values.astype(float)
    std = float(array.std(ddof=0))
    return (array - float(array.mean())) / std if std > 1e-8 else pd.Series(0.0, index=values.index)


def _scores(frame: pd.DataFrame, gamma: float, *, positive_label: str = "nez") -> np.ndarray:
    if {"logits_broad", "logits_core"} - set(frame.columns):
        raise RuntimeError("Predictions must contain logits_broad and logits_core.")
    logits = frame["logits_broad"].astype(float) + float(gamma) * frame["logits_core"].astype(float)
    positive = 1.0 / (1.0 + np.exp(-np.clip(logits.to_numpy(), -40.0, 40.0)))
    return _positive_to_nez(positive, positive_label)


def _anchor_scores(frame: pd.DataFrame, alpha: float, beta: float, *, positive_label: str = "nez") -> np.ndarray:
    teacher_col = "teacher_nez_score" if positive_label == "nez" else "a9v3_oof_score"
    required = {teacher_col, "score_broad", "score_core", "subject_id"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"Teacher-anchor rescore is missing columns: {missing}")
    work = frame.copy()
    work["_teacher_z"] = work.groupby("subject_id")[teacher_col].transform(_z)
    work["_broad_z"] = work.groupby("subject_id")["score_broad"].transform(_z)
    work["_core_z"] = work.groupby("subject_id")["score_core"].transform(_z)
    logits = work["_teacher_z"] + float(alpha) * work["_broad_z"] + float(beta) * work["_core_z"]
    positive = 1.0 / (1.0 + np.exp(-np.clip(logits.to_numpy(), -40.0, 40.0)))
    return _positive_to_nez(positive, positive_label)


def _select_threshold(frame: pd.DataFrame, scores: np.ndarray) -> tuple[float, float]:
    work = frame.assign(_score=scores)
    best = (0.5, -np.inf)
    for threshold in np.linspace(0.01, 0.99, 99):
        values = [
            f1_score(group["true_nez"].astype(int), group["_score"] >= threshold, average="macro", zero_division=0)
            for _, group in work.groupby("subject_id")
        ]
        score = float(np.mean(values)) if values else 0.0
        if score > best[1] + 1e-12 or (abs(score - best[1]) <= 1e-12 and abs(threshold - 0.5) < abs(best[0] - 0.5)):
            best = (float(threshold), score)
    return best


def _metrics(frame: pd.DataFrame, scores: np.ndarray, threshold: float | np.ndarray) -> dict[str, float]:
    scores = np.asarray(scores, dtype=float)
    work = frame.assign(_score=scores)
    thresholds = np.full(len(work), float(threshold), dtype=float) if np.isscalar(threshold) else np.asarray(threshold, dtype=float)
    if thresholds.shape[0] != len(work):
        raise ValueError("Per-row threshold vector must align with prediction rows.")
    work["_threshold"] = thresholds
    patient_f1 = []
    patient_ez_mrr = []
    for _, group in work.groupby("subject_id"):
        y = group["true_nez"].astype(int).to_numpy()
        patient_f1.append(f1_score(y, group["_score"].to_numpy() >= group["_threshold"].to_numpy(), average="macro", zero_division=0))
        order = np.argsort(group["_score"].to_numpy())
        ez = 1 - y
        hits = np.flatnonzero(ez[order] == 1)
        patient_ez_mrr.append(1.0 / (int(hits[0]) + 1) if hits.size else 0.0)
    y = work["true_nez"].astype(int).to_numpy()
    ez_score = 1.0 - scores
    return {
        "n_patients": int(work["subject_id"].nunique()),
        "n_channels": int(len(work)),
        "threshold": float(np.mean(thresholds)) if thresholds.size else 0.5,
        "patient_macro_f1": float(np.mean(patient_f1)) if patient_f1 else 0.0,
        "patient_macro_ez_mrr": float(np.mean(patient_ez_mrr)) if patient_ez_mrr else 0.0,
        "pooled_macro_f1": float(f1_score(y, scores >= threshold, average="macro", zero_division=0)),
        "pooled_ez_auroc": float(roc_auc_score(1 - y, ez_score)) if np.unique(y).size > 1 else 0.0,
        "pooled_ez_auprc": float(average_precision_score(1 - y, ez_score)) if np.unique(y).size > 1 else 0.0,
    }


def run(run_dir: Path) -> None:
    resolved = _resolved_args(run_dir)
    positive_label = str(resolved.get("positive_label", "nez")).lower()
    if positive_label not in {"ez", "nez"}:
        raise RuntimeError(f"Unsupported positive_label={positive_label!r}.")
    teacher_mode = str(resolved.get("teacher_mode", "legacy_global_oof")).lower()
    val_paths = sorted(run_dir.glob("val_channel_predictions_neuroez_v2_fold_*.csv"))
    test_paths = sorted(run_dir.glob("test_channel_predictions_neuroez_v2_fold_*.csv"))
    if not val_paths or len(val_paths) != len(test_paths):
        raise RuntimeError("Need matching validation and test prediction files for each completed fold.")
    fold_rows = []
    posthoc_rows = []
    selected_frames = []
    for val_path, test_path in zip(val_paths, test_paths):
        val = pd.read_csv(val_path)
        test = pd.read_csv(test_path)
        fold = int(test["fold_idx"].iloc[0])
        candidates = []
        for gamma in GAMMAS:
            val_scores = _scores(val, gamma, positive_label=positive_label)
            threshold, val_f1 = _select_threshold(val, val_scores)
            candidates.append((val_f1, -abs(gamma - 0.10), "logit_fusion", gamma, np.nan, np.nan, threshold))
            exploratory = _metrics(test, _scores(test, gamma, positive_label=positive_label), threshold)
            posthoc_rows.append({"fold_idx": fold, "method": "logit_fusion", "gamma": gamma, "alpha": np.nan, "beta": np.nan, "selection_role": "posthoc_exploratory", **exploratory})
        if teacher_mode != "physiology_only":
            for alpha in ALPHAS:
                for beta in BETAS:
                    val_scores = _anchor_scores(val, alpha, beta, positive_label=positive_label)
                    threshold, val_f1 = _select_threshold(val, val_scores)
                    candidates.append((val_f1, -abs(alpha - 0.10) - abs(beta), "teacher_anchor", np.nan, alpha, beta, threshold))
                    exploratory = _metrics(test, _anchor_scores(test, alpha, beta, positive_label=positive_label), threshold)
                    posthoc_rows.append({"fold_idx": fold, "method": "teacher_anchor", "gamma": np.nan, "alpha": alpha, "beta": beta, "selection_role": "posthoc_exploratory", **exploratory})
        _, _, method, gamma, alpha, beta, threshold = max(candidates)
        test_scores = _scores(test, gamma, positive_label=positive_label) if method == "logit_fusion" else _anchor_scores(test, alpha, beta, positive_label=positive_label)
        metrics = _metrics(test, test_scores, threshold)
        fold_rows.append({"fold_idx": fold, "method": method, "gamma": gamma, "alpha": alpha, "beta": beta, "selection_role": "validation_selected", **metrics})
        selected_frames.append(test.assign(score_nez_rescored=test_scores, selected_method=method, selected_gamma=gamma, selected_alpha=alpha, selected_beta=beta, selected_threshold=threshold))
    selected = pd.concat(selected_frames, ignore_index=True)
    summary = {
        "selection_role": "validation_selected",
        "n_folds": len(fold_rows),
        "n_patients": int(selected["subject_id"].nunique()),
        "mean_patient_macro_f1": float(np.mean([row["patient_macro_f1"] for row in fold_rows])),
        "mean_patient_macro_ez_mrr": float(np.mean([row["patient_macro_ez_mrr"] for row in fold_rows])),
    }
    by_center = []
    for center, group in selected.groupby("center"):
        by_center.append({"center": center, "selection_role": "validation_selected", **_metrics(group, group["score_nez_rescored"].to_numpy(), group["selected_threshold"].to_numpy())})
    pd.DataFrame([summary] + posthoc_rows).to_csv(run_dir / "rescore_summary.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(run_dir / "rescore_by_fold.csv", index=False)
    pd.DataFrame(by_center).to_csv(run_dir / "rescore_by_center.csv", index=False)
    (run_dir / "rescore_audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.run_dir)


if __name__ == "__main__":
    main()
