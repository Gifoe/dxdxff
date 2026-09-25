"""Fold-safe score-level center router audit.

For each fold, uses held-out VALIDATION patients to select the best per-center
lambda that combines baseline and candidate score_ez via:

    score_ez_fused = (1 - lambda) * score_ez_baseline + lambda * score_ez_candidate

The chosen lambda is then applied to TEST patients of the same fold.  Results
are aggregated across folds.

When val predictions are unavailable for a fold, the script falls back to an
ORACLE upper bound (selecting lambda on test labels) and marks the output with
``router_mode=oracle_test_upper_bound``.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


_CENTER_NAMES: dict[int, str] = {0: "HUP", 1: "LZU", 2: "multicenter", 3: "pediatric"}


def _patient_macro_f1(df: pd.DataFrame) -> float:
    """Compute patient-macro F1 (NEZ-positive) from a per-channel dataframe."""
    if "true_nez" not in df.columns or "predicted_nez" not in df.columns:
        return float("nan")
    scores: list[float] = []
    for subject_id, grp in df.groupby("subject_id"):
        y_true = grp["true_nez"].values.astype(int)
        y_pred = grp["predicted_nez"].values.astype(int)
        if len(np.unique(y_true)) < 2:
            continue
        scores.append(float(f1_score(y_true, y_pred, average="macro", zero_division=0)))
    return float(np.mean(scores)) if scores else float("nan")


def _read_channel_csvs(pred_dir: Path, split: str) -> list[pd.DataFrame]:
    """Read all ``{split}_channel_predictions_neuroez_v2_fold_*.csv``."""
    pattern = f"{split}_channel_predictions_neuroez_v2_fold_*.csv"
    paths = sorted(pred_dir.glob(pattern))
    if not paths:
        return []
    return [pd.read_csv(p) for p in paths]


def _best_lambda_per_center(
    df_base: pd.DataFrame,
    df_cand: pd.DataFrame,
    lambdas: list[float],
) -> dict[int, float]:
    """Sweep lambdas and return best per center (by patient_macro_f1)."""
    best: dict[int, float] = {}
    for center_id, center_name in sorted(_CENTER_NAMES.items()):
        mask_base = df_base["center"].str.lower() == center_name.lower()
        mask_cand = df_cand["center"].str.lower() == center_name.lower()
        if not mask_base.any() and not mask_cand.any():
            best[center_id] = 0.0
            continue
        # use whichever source has the patients
        df_use = df_base if mask_base.any() else df_cand
        mask_use = mask_base if mask_base.any() else mask_cand
        subjects = df_use.loc[mask_use, "subject_id"].unique()
        best_f1 = -1.0
        best_lam = 0.0
        for lam in lambdas:
            f1s = []
            for sid in subjects:
                sid_mask_base = df_base["subject_id"] == sid
                sid_mask_cand = df_cand["subject_id"] == sid
                if not sid_mask_base.any() or not sid_mask_cand.any():
                    continue
                base_row = df_base[sid_mask_base]
                cand_row = df_cand[sid_mask_cand]
                # merge on channel_name to align scores
                merged = base_row[["channel_name", "true_nez", "true_ez"]].merge(
                    cand_row[["channel_name", "score_ez_probability"]],
                    on="channel_name",
                    how="inner",
                    suffixes=("_base", "_cand"),
                )
                if merged.empty:
                    continue
                score_ez_base = base_row.set_index("channel_name")["score_ez_probability"]
                score_ez_cand = cand_row.set_index("channel_name")["score_ez_probability"]
                common = score_ez_base.index.intersection(score_ez_cand.index)
                if len(common) == 0:
                    continue
                fused = (1.0 - lam) * score_ez_base[common].values + lam * score_ez_cand[common].values
                # threshold: top-k where k = true EZ count
                true_ez_count = int(base_row["true_ez"].iloc[0]) if "true_ez" in base_row.columns else int(base_row["true_ez"].sum())
                if true_ez_count <= 0:
                    continue
                k = min(true_ez_count, len(common))
                top_indices = np.argsort(fused)[-k:]
                pred_ez = np.zeros(len(common), dtype=int)
                pred_ez[top_indices] = 1
                pred_nez = 1 - pred_ez
                y_nez = base_row.set_index("channel_name").loc[common, "true_nez"].values.astype(int) if "true_nez" in base_row.columns else 1 - base_row.set_index("channel_name").loc[common, "true_ez"].values.astype(int)
                if len(np.unique(y_nez)) < 2:
                    continue
                f1s.append(float(f1_score(y_nez, pred_nez, average="macro", zero_division=0)))
            if f1s:
                mean_f1 = float(np.mean(f1s))
                if mean_f1 > best_f1:
                    best_f1 = mean_f1
                    best_lam = lam
        best[center_id] = best_lam
    return best


def _apply_router(
    df_base: pd.DataFrame,
    df_cand: pd.DataFrame,
    center_lambdas: dict[int, float],
) -> pd.DataFrame:
    """Apply per-center lambdas to channel-level scores and produce predictions."""
    rows: list[dict[str, Any]] = []
    all_subjects = set(df_base["subject_id"].unique()) | set(df_cand["subject_id"].unique())
    for sid in sorted(all_subjects):
        base_sid = df_base[df_base["subject_id"] == sid]
        cand_sid = df_cand[df_cand["subject_id"] == sid]
        if base_sid.empty or cand_sid.empty:
            # use whichever is available
            use_df = base_sid if not base_sid.empty else cand_sid
            lam = 0.0
            score_ez_final = use_df["score_ez_probability"].values
            channels = use_df["channel_name"].values
            true_ez = use_df["true_ez"].values
            true_nez = use_df["true_nez"].values if "true_nez" in use_df.columns else 1 - true_ez
            center = use_df["center"].iloc[0]
            fold = use_df["fold_idx"].iloc[0] if "fold_idx" in use_df.columns else -1
        else:
            center = base_sid["center"].iloc[0]
            fold = base_sid["fold_idx"].iloc[0] if "fold_idx" in base_sid.columns else -1
            channel_map_base = base_sid.set_index("channel_name")
            channel_map_cand = cand_sid.set_index("channel_name")
            common = channel_map_base.index.intersection(channel_map_cand.index)
            if len(common) == 0:
                continue
            score_base = channel_map_base.loc[common, "score_ez_probability"].values
            score_cand = channel_map_cand.loc[common, "score_ez_probability"].values
            center_id = next((cid for cid, name in _CENTER_NAMES.items() if name.lower() == str(center).lower()), 4)
            lam = center_lambdas.get(center_id, 0.0)
            score_ez_final = (1.0 - lam) * score_base + lam * score_cand
            channels = common.values
            true_ez = channel_map_base.loc[common, "true_ez"].values
            true_nez = channel_map_base.loc[common, "true_nez"].values if "true_nez" in channel_map_base.columns else 1 - true_ez

        k = int(true_ez.sum())
        if k > 0 and k < len(score_ez_final):
            top_idx = np.argsort(score_ez_final)[-k:]
            pred_ez = np.zeros(len(score_ez_final), dtype=int)
            pred_ez[top_idx] = 1
        else:
            pred_ez = (score_ez_final >= 0.5).astype(int)
        pred_nez = 1 - pred_ez

        for i, ch in enumerate(channels):
            rows.append({
                "subject_id": sid,
                "center": center,
                "fold_idx": fold,
                "channel_name": ch,
                "lambda": lam,
                "true_ez": true_ez[i],
                "true_nez": true_nez[i],
                "score_ez_fused": score_ez_final[i],
                "predicted_ez": pred_ez[i],
                "predicted_nez": pred_nez[i],
            })
    return pd.DataFrame(rows)


def _compute_center_metrics(df: pd.DataFrame) -> dict[str, float]:
    """Compute per-center patient_macro_f1."""
    metrics: dict[str, float] = {}
    for center_name in _CENTER_NAMES.values():
        center_df = df[df["center"].str.lower() == center_name.lower()]
        if center_df.empty:
            metrics[f"patient_macro_f1_{center_name}"] = float("nan")
            continue
        f1s = []
        for sid, grp in center_df.groupby("subject_id"):
            y = grp["true_nez"].values.astype(int)
            p = grp["predicted_nez"].values.astype(int)
            if len(np.unique(y)) < 2:
                continue
            f1s.append(float(f1_score(y, p, average="macro", zero_division=0)))
        metrics[f"patient_macro_f1_{center_name}"] = float(np.mean(f1s)) if f1s else float("nan")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Fold-safe score-level center router audit.")
    parser.add_argument("--baseline_pred_dir", required=True, type=Path)
    parser.add_argument("--candidate_pred_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--lambdas", type=str, default="0,0.25,0.5,0.75,1")
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    lambdas = [float(x.strip()) for x in args.lambdas.split(",") if x.strip()]

    # ---------- load per-fold data ----------
    val_base_frames = _read_channel_csvs(args.baseline_pred_dir, "val")
    test_base_frames = _read_channel_csvs(args.baseline_pred_dir, "test")
    val_cand_frames = _read_channel_csvs(args.candidate_pred_dir, "val")
    test_cand_frames = _read_channel_csvs(args.candidate_pred_dir, "test")

    n_folds = len(test_base_frames)
    if n_folds == 0:
        print("[center-router] No test channel CSVs found — aborting.")
        return

    has_val = (
        len(val_base_frames) == n_folds
        and len(val_cand_frames) == n_folds
    )

    if not has_val:
        print("[center-router] Val predictions incomplete — falling back to oracle upper bound on test labels.")
        router_mode = "oracle_test_upper_bound"
    else:
        router_mode = "fold_safe_val_selected"
        print(f"[center-router] Fold-safe mode active — {n_folds} folds with val predictions.")

    all_router_rows: list[pd.DataFrame] = []
    all_lambdas: dict[int, list[float]] = defaultdict(list)
    per_fold_metrics: list[dict[str, Any]] = []

    for fold_idx in range(n_folds):
        test_base = test_base_frames[fold_idx]
        test_cand = test_cand_frames[fold_idx]

        if router_mode == "fold_safe_val_selected":
            val_base = val_base_frames[fold_idx]
            val_cand = val_cand_frames[fold_idx]
            center_lambdas = _best_lambda_per_center(val_base, val_cand, lambdas)
        else:
            center_lambdas = _best_lambda_per_center(test_base, test_cand, lambdas)

        for cid, lam in center_lambdas.items():
            all_lambdas[cid].append(lam)

        router_df = _apply_router(test_base, test_cand, center_lambdas)
        router_df["fold_idx"] = fold_idx
        all_router_rows.append(router_df)

        fold_metrics = _compute_center_metrics(router_df)
        overall_f1s = []
        for sid, grp in router_df.groupby("subject_id"):
            y = grp["true_nez"].values.astype(int)
            p = grp["predicted_nez"].values.astype(int)
            if len(np.unique(y)) < 2:
                continue
            overall_f1s.append(float(f1_score(y, p, average="macro", zero_division=0)))
        fold_metrics["patient_macro_f1_overall"] = float(np.mean(overall_f1s)) if overall_f1s else float("nan")
        fold_metrics["fold_idx"] = fold_idx
        fold_metrics["router_mode"] = router_mode
        for cid, name in _CENTER_NAMES.items():
            fold_metrics[f"lambda_{name}"] = center_lambdas.get(cid, float("nan"))
        per_fold_metrics.append(fold_metrics)

    # ---------- aggregate ----------
    router_all = pd.concat(all_router_rows, ignore_index=True)
    all_metrics = _compute_center_metrics(router_all)
    overall_f1s_all = []
    for sid, grp in router_all.groupby("subject_id"):
        y = grp["true_nez"].values.astype(int)
        p = grp["predicted_nez"].values.astype(int)
        if len(np.unique(y)) < 2:
            continue
        overall_f1s_all.append(float(f1_score(y, p, average="macro", zero_division=0)))
    all_metrics["patient_macro_f1_overall"] = float(np.mean(overall_f1s_all)) if overall_f1s_all else float("nan")
    all_metrics["router_mode"] = router_mode
    for cid, name in _CENTER_NAMES.items():
        lam_vals = all_lambdas.get(cid, [])
        all_metrics[f"lambda_{name}_mean"] = float(np.mean(lam_vals)) if lam_vals else float("nan")

    # ---------- output ----------
    summary_out = output_dir / ("center_router_summary.json" if router_mode != "oracle_test_upper_bound" else "oracle_center_router_upper_bound.json")
    per_fold_out = output_dir / "center_router_by_fold.csv"
    patient_delta_out = output_dir / "center_router_patient_delta.csv"

    with open(summary_out, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)

    pd.DataFrame(per_fold_metrics).to_csv(per_fold_out, index=False)

    # patient-level delta vs baseline
    baseline_test_all = pd.concat(test_base_frames, ignore_index=True) if test_base_frames else pd.DataFrame()
    if not baseline_test_all.empty:
        baseline_f1s = {}
        for sid, grp in baseline_test_all.groupby("subject_id"):
            y = grp["true_nez"].values.astype(int)
            p = grp["predicted_nez"].values.astype(int)
            if len(np.unique(y)) < 2:
                continue
            baseline_f1s[sid] = float(f1_score(y, p, average="macro", zero_division=0))
        delta_rows = []
        for sid, grp in router_all.groupby("subject_id"):
            y = grp["true_nez"].values.astype(int)
            p = grp["predicted_nez"].values.astype(int)
            if len(np.unique(y)) < 2:
                continue
            router_f1 = float(f1_score(y, p, average="macro", zero_division=0))
            base_f1 = baseline_f1s.get(sid, float("nan"))
            delta_rows.append({
                "subject_id": sid,
                "center": grp["center"].iloc[0],
                "baseline_patient_macro_f1": base_f1,
                "router_patient_macro_f1": router_f1,
                "delta": router_f1 - base_f1,
            })
        pd.DataFrame(delta_rows).to_csv(patient_delta_out, index=False)

    print(f"[center-router] Mode={router_mode} | overall_f1={all_metrics['patient_macro_f1_overall']:.4f}")
    for cid, name in sorted(_CENTER_NAMES.items()):
        print(f"  {name}: lambda_mean={all_metrics.get(f'lambda_{name}_mean', 'nan')}  f1={all_metrics.get(f'patient_macro_f1_{name}', 'nan')}")
    print(f"[center-router] Outputs written to {output_dir}")


if __name__ == "__main__":
    main()
