"""Validation-selected teacher-anchor rescore for A9v8 LCBO outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_recall_fscore_support

REQUIRED_COLUMNS = {
    "subject_id",
    "fold_idx",
    "center",
    "true_ez",
    "score_eval",
    "score_broad",
    "score_core",
    "a9v3_oof_score",
}
A9V3 = {
    "patient_macro_f1": 0.642714,
    "patient_macro_ez_f1": 0.470591,
    "patient_macro_auprc_ez": 0.518635,
    "patient_macro_ez_mrr": 0.715136,
    "top1_is_ez_rate": 0.600000,
}
DEFAULT_ALPHA_GRID = [0.02, 0.05, 0.10, 0.20, 0.30, 0.50]
DEFAULT_BETA_GRID = [-0.20, -0.10, -0.05, 0.00, 0.05, 0.10, 0.20]
DEFAULT_GAMMA_GRID = [-0.20, -0.10, -0.05, 0.00, 0.02, 0.05, 0.10, 0.15]
EPS = 1e-7


def _parse_grid(text: str, default: list[float]) -> list[float]:
    if text is None:
        return list(default)
    stripped = str(text).strip()
    if not stripped:
        return []
    return [float(item.strip()) for item in stripped.split(",") if item.strip()]


def _safe_logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values.astype(float), EPS, 1.0 - EPS)
    return np.log(clipped / (1.0 - clipped))


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values.astype(float)))


def _patient_zscore(values: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    values = values.astype(float)
    std = np.nanstd(values)
    if not np.isfinite(std) or std < eps:
        return np.zeros_like(values, dtype=float)
    return (values - np.nanmean(values)) / max(std, eps)


def _select_topk(scores: np.ndarray, k: int) -> np.ndarray:
    pred = np.zeros(scores.shape[0], dtype=bool)
    if scores.size == 0:
        return pred
    k = max(1, min(int(k), int(scores.size)))
    pred[np.argsort(scores)[::-1][:k]] = True
    return pred


def _mrr(y_ez: np.ndarray, scores: np.ndarray) -> float:
    if y_ez.size == 0 or int(y_ez.sum()) == 0:
        return 0.0
    order = np.argsort(scores)[::-1]
    positive = np.where(y_ez[order] == 1)[0]
    return float(1.0 / float(positive[0] + 1)) if positive.size else 0.0


def _passes_main_gate(row: pd.Series | dict[str, float]) -> bool:
    return (
        float(row.get("patient_macro_f1", 0.0)) > A9V3["patient_macro_f1"]
        and float(row.get("patient_macro_ez_f1", 0.0)) > A9V3["patient_macro_ez_f1"]
        and float(row.get("patient_macro_auprc_ez", 0.0)) > A9V3["patient_macro_auprc_ez"]
        and float(row.get("patient_macro_ez_mrr", 0.0)) >= A9V3["patient_macro_ez_mrr"]
        and float(row.get("top1_is_ez_rate", 0.0)) >= A9V3["top1_is_ez_rate"]
    )


def candidate_scores_for_channels(
    channels: pd.DataFrame,
    *,
    alpha_grid: Iterable[float],
    beta_grid: Iterable[float],
    gamma_grid: Iterable[float],
    include_raw_score_eval: bool = True,
    include_teacher_only: bool = True,
    include_logit_broad_core: bool = False,
) -> dict[str, np.ndarray]:
    scores: dict[str, np.ndarray] = {}
    if include_teacher_only:
        scores["teacher_only"] = channels["a9v3_oof_score"].astype(float).to_numpy()
    if include_raw_score_eval:
        scores["raw_score_eval"] = channels["score_eval"].astype(float).to_numpy()

    z_teacher = np.zeros(len(channels), dtype=float)
    z_broad = np.zeros(len(channels), dtype=float)
    z_core = np.zeros(len(channels), dtype=float)
    for _, idx in channels.groupby(["fold_idx", "subject_id"], sort=False, dropna=False).indices.items():
        idx_arr = np.asarray(idx, dtype=int)
        group = channels.iloc[idx_arr]
        z_teacher[idx_arr] = _patient_zscore(group["a9v3_oof_score"].astype(float).to_numpy())
        z_broad[idx_arr] = _patient_zscore(group["score_broad"].astype(float).to_numpy())
        z_core[idx_arr] = _patient_zscore(group["score_core"].astype(float).to_numpy())
    for alpha in alpha_grid:
        for beta in beta_grid:
            scores[f"teacher_plus_broad_core_a{float(alpha):.2f}_b{float(beta):+.2f}"] = (
                z_teacher + float(alpha) * z_broad + float(beta) * z_core
            )

    if include_logit_broad_core:
        if "logits_broad" in channels.columns and "logits_core" in channels.columns:
            broad_logit = channels["logits_broad"].astype(float).to_numpy()
            core_logit = channels["logits_core"].astype(float).to_numpy()
        else:
            broad_logit = _safe_logit(channels["score_broad"].astype(float).to_numpy())
            core_logit = _safe_logit(channels["score_core"].astype(float).to_numpy())
        for gamma in gamma_grid:
            scores[f"logit_broad_core_g{float(gamma):+.2f}"] = _sigmoid(broad_logit + float(gamma) * core_logit)
    return scores


def _variant_meta(score_variant: str) -> dict[str, float | str]:
    if score_variant == "teacher_only":
        return {"variant_family": "teacher_only", "alpha": np.nan, "beta": np.nan, "gamma": np.nan}
    if score_variant == "raw_score_eval":
        return {"variant_family": "raw_score_eval", "alpha": np.nan, "beta": np.nan, "gamma": np.nan}
    if score_variant.startswith("logit_broad_core_g"):
        return {
            "variant_family": "logit_broad_core",
            "alpha": np.nan,
            "beta": np.nan,
            "gamma": float(score_variant.rsplit("g", 1)[1]),
        }
    if score_variant.startswith("teacher_plus_broad_core_a"):
        tail = score_variant.replace("teacher_plus_broad_core_a", "")
        alpha_text, beta_text = tail.split("_b", 1)
        return {
            "variant_family": "teacher_plus_broad_core",
            "alpha": float(alpha_text),
            "beta": float(beta_text),
            "gamma": np.nan,
        }
    return {"variant_family": "unknown", "alpha": np.nan, "beta": np.nan, "gamma": np.nan}


def _patient_rows_for_score(channels: pd.DataFrame, scores: np.ndarray) -> list[dict[str, float | str | int]]:
    rows: list[dict[str, float | str | int]] = []
    work = channels.copy()
    work["_score"] = scores.astype(float)
    for keys, group in work.groupby(["config", "fold_idx", "subject_id"], sort=False, dropna=False):
        config, fold_idx, subject_id = keys
        y_ez = group["true_ez"].astype(int).to_numpy()
        y_nez = 1 - y_ez
        patient_scores = group["_score"].astype(float).to_numpy()
        pred_ez = _select_topk(patient_scores, int(y_ez.sum()))
        pred_nez = (~pred_ez).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(y_nez, pred_nez, labels=[1, 0], zero_division=0)
        rows.append(
            {
                "config": str(config),
                "fold_idx": int(fold_idx),
                "subject_id": str(subject_id),
                "center": str(group["center"].iloc[0]),
                "patient_macro_f1": float(f1_score(y_nez, pred_nez, average="macro", zero_division=0)),
                "patient_macro_ez_f1": float(f1[1]),
                "patient_macro_auprc_ez": float(average_precision_score(y_ez, patient_scores)) if np.unique(y_ez).size > 1 else 0.0,
                "patient_macro_ez_mrr": _mrr(y_ez, patient_scores),
                "top1_is_ez": float(y_ez[int(np.argmax(patient_scores))] == 1) if patient_scores.size else 0.0,
            }
        )
    return rows


def _summarize(rows: list[dict[str, float | str | int]], group_cols: list[str]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame()
    metric_cols = [
        "patient_macro_f1",
        "patient_macro_ez_f1",
        "patient_macro_auprc_ez",
        "patient_macro_ez_mrr",
        "top1_is_ez",
    ]
    out = df.groupby(group_cols, dropna=False, sort=True)[metric_cols].mean().reset_index()
    out = out.rename(columns={"top1_is_ez": "top1_is_ez_rate"})
    out["n_patients"] = df.groupby(group_cols, dropna=False, sort=True)["subject_id"].nunique().to_numpy()
    return out


def _evaluate_grid(channels: pd.DataFrame, **candidate_kwargs: object) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, float | str | int]] = []
    candidates = candidate_scores_for_channels(channels, **candidate_kwargs)
    for variant, scores in candidates.items():
        meta = _variant_meta(variant)
        patient_rows = _patient_rows_for_score(channels, scores)
        for row in patient_rows:
            row.update(
                {
                    "variant_family": meta["variant_family"],
                    "score_variant": variant,
                    "alpha": meta["alpha"],
                    "beta": meta["beta"],
                    "gamma": meta["gamma"],
                }
            )
        rows.extend(patient_rows)
    grid = _summarize(rows, ["config", "fold_idx", "variant_family", "score_variant", "alpha", "beta", "gamma"])
    return grid, pd.DataFrame(rows)


def _discover_config_frames(root: Path, split: str) -> pd.DataFrame:
    patterns = (
        [f"{split}_channel_predictions_neuroez_v2_fold_*.csv", f"*{split}*channel_predictions*.csv"]
    )
    frames: list[pd.DataFrame] = []
    seen: set[Path] = set()
    for pattern in patterns:
        for path in sorted(root.rglob(pattern)):
            if path in seen:
                continue
            seen.add(path)
            df = pd.read_csv(path)
            missing = sorted(REQUIRED_COLUMNS - set(df.columns))
            if missing:
                raise RuntimeError(f"{path} is missing required columns: {missing}")
            config = path.parent.name if path.parent != root else root.name
            df = df.copy()
            df["config"] = config
            frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No {split} channel prediction files found under {root}")
    return pd.concat(frames, ignore_index=True)


def _add_selection_deltas(val_grid: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (config, fold_idx), group in val_grid.groupby(["config", "fold_idx"], sort=False, dropna=False):
        teacher = group[group["score_variant"] == "teacher_only"]
        if teacher.empty:
            raise RuntimeError(f"Missing teacher_only validation candidate for config={config}, fold={fold_idx}")
        base = teacher.iloc[0]
        for _, row in group.iterrows():
            item = row.to_dict()
            deltas = {
                "val_delta_f1": float(row["patient_macro_f1"] - base["patient_macro_f1"]),
                "val_delta_ez_f1": float(row["patient_macro_ez_f1"] - base["patient_macro_ez_f1"]),
                "val_delta_auprc_ez": float(row["patient_macro_auprc_ez"] - base["patient_macro_auprc_ez"]),
                "val_delta_mrr": float(row["patient_macro_ez_mrr"] - base["patient_macro_ez_mrr"]),
                "val_delta_top1": float(row["top1_is_ez_rate"] - base["top1_is_ez_rate"]),
            }
            item.update(deltas)
            vals = list(deltas.values())
            item["val_pass_count"] = int(sum(v >= 0.0 for v in vals))
            item["val_min_delta"] = float(min(vals))
            rows.append(item)
    return pd.DataFrame(rows)


def _select_params(val_grid: pd.DataFrame, selection_policy: str) -> pd.DataFrame:
    if selection_policy != "gate_margin":
        raise ValueError(f"Unsupported selection_policy: {selection_policy}")
    val = _add_selection_deltas(val_grid)
    priority = {"teacher_only": 3, "raw_score_eval": 2, "teacher_plus_broad_core": 1, "logit_broad_core": 0}
    val["variant_priority"] = val["variant_family"].map(priority).fillna(-1).astype(int)
    selected = []
    sort_cols = [
        "val_pass_count",
        "val_min_delta",
        "patient_macro_f1",
        "patient_macro_ez_f1",
        "patient_macro_auprc_ez",
        "patient_macro_ez_mrr",
        "top1_is_ez_rate",
        "variant_priority",
    ]
    for _, group in val.groupby(["config", "fold_idx"], sort=False, dropna=False):
        row = group.sort_values(sort_cols, ascending=[False] * len(sort_cols)).iloc[0]
        selected.append(
            {
                "config": row["config"],
                "fold_idx": int(row["fold_idx"]),
                "selected_variant_family": row["variant_family"],
                "selected_score_variant": row["score_variant"],
                "selected_alpha": row["alpha"],
                "selected_beta": row["beta"],
                "selected_gamma": row["gamma"],
                "val_patient_macro_f1": row["patient_macro_f1"],
                "val_patient_macro_ez_f1": row["patient_macro_ez_f1"],
                "val_patient_macro_auprc_ez": row["patient_macro_auprc_ez"],
                "val_patient_macro_ez_mrr": row["patient_macro_ez_mrr"],
                "val_top1_is_ez_rate": row["top1_is_ez_rate"],
                "val_delta_f1": row["val_delta_f1"],
                "val_delta_ez_f1": row["val_delta_ez_f1"],
                "val_delta_auprc_ez": row["val_delta_auprc_ez"],
                "val_delta_mrr": row["val_delta_mrr"],
                "val_delta_top1": row["val_delta_top1"],
                "val_pass_count": int(row["val_pass_count"]),
                "val_min_delta": row["val_min_delta"],
            }
        )
    return pd.DataFrame(selected)


def _selected_test_rows(test_patient_rows: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    selected_rows = []
    for _, row in selected.iterrows():
        mask = (
            (test_patient_rows["config"] == row["config"])
            & (test_patient_rows["fold_idx"].astype(int) == int(row["fold_idx"]))
            & (test_patient_rows["score_variant"] == row["selected_score_variant"])
        )
        part = test_patient_rows[mask].copy()
        for key in ("selected_variant_family", "selected_score_variant", "selected_alpha", "selected_beta", "selected_gamma"):
            part[key] = row[key]
        selected_rows.append(part)
    return pd.concat(selected_rows, ignore_index=True) if selected_rows else pd.DataFrame()


def _failure_audit(by_fold: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in by_fold.iterrows():
        tags = []
        if float(row["patient_macro_ez_f1"]) < A9V3["patient_macro_ez_f1"]:
            tags.append("low_ez_f1")
        if float(row["patient_macro_auprc_ez"]) < A9V3["patient_macro_auprc_ez"]:
            tags.append("low_auprc")
        if float(row["patient_macro_ez_mrr"]) < A9V3["patient_macro_ez_mrr"]:
            tags.append("low_mrr")
        if float(row["top1_is_ez_rate"]) < A9V3["top1_is_ez_rate"]:
            tags.append("low_top1")
        item = row.to_dict()
        item.update(
            {
                "delta_to_gate_f1": float(row["patient_macro_f1"] - A9V3["patient_macro_f1"]),
                "delta_to_gate_ez_f1": float(row["patient_macro_ez_f1"] - A9V3["patient_macro_ez_f1"]),
                "delta_to_gate_auprc_ez": float(row["patient_macro_auprc_ez"] - A9V3["patient_macro_auprc_ez"]),
                "delta_to_gate_mrr": float(row["patient_macro_ez_mrr"] - A9V3["patient_macro_ez_mrr"]),
                "delta_to_gate_top1": float(row["top1_is_ez_rate"] - A9V3["top1_is_ez_rate"]),
                "weakness_tag": ";".join(tags),
            }
        )
        rows.append(item)
    return pd.DataFrame(rows)


def run_validation_selected_rescore(
    run_dir: str | Path,
    *,
    alpha_grid: Iterable[float] = DEFAULT_ALPHA_GRID,
    beta_grid: Iterable[float] = DEFAULT_BETA_GRID,
    gamma_grid: Iterable[float] = DEFAULT_GAMMA_GRID,
    selection_policy: str = "gate_margin",
    expected_patient_count: int = 90,
    include_raw_score_eval: bool = True,
    include_teacher_only: bool = True,
    include_logit_broad_core: bool = False,
) -> dict[str, pd.DataFrame]:
    root = Path(run_dir)
    val_channels = _discover_config_frames(root, "val")
    test_channels = _discover_config_frames(root, "test")
    candidate_kwargs = {
        "alpha_grid": list(alpha_grid),
        "beta_grid": list(beta_grid),
        "gamma_grid": list(gamma_grid),
        "include_raw_score_eval": include_raw_score_eval,
        "include_teacher_only": include_teacher_only,
        "include_logit_broad_core": include_logit_broad_core,
    }
    val_grid, _ = _evaluate_grid(val_channels, **candidate_kwargs)
    test_grid, test_patient_grid = _evaluate_grid(test_channels, **candidate_kwargs)
    selected = _select_params(val_grid, selection_policy)
    selected_patients = _selected_test_rows(test_patient_grid, selected)

    by_fold = _summarize(
        selected_patients.to_dict("records"),
        ["config", "fold_idx", "selected_variant_family", "selected_score_variant", "selected_alpha", "selected_beta", "selected_gamma"],
    )
    by_center = _summarize(selected_patients.to_dict("records"), ["config", "center"])
    summary = _summarize(selected_patients.to_dict("records"), ["config"])
    if not summary.empty:
        summary["passes_main_gate"] = summary.apply(_passes_main_gate, axis=1)

    failure = _failure_audit(by_fold)
    observed_patient_count = int(selected_patients["subject_id"].nunique()) if not selected_patients.empty else 0
    best_config = None
    if not summary.empty:
        ranked = summary.assign(
            gate_score=(
                summary["patient_macro_f1"]
                + summary["patient_macro_ez_f1"]
                + summary["patient_macro_auprc_ez"]
                + summary["patient_macro_ez_mrr"]
                + summary["top1_is_ez_rate"]
            )
        ).sort_values("gate_score", ascending=False)
        best_config = str(ranked.iloc[0]["config"])
    audit = {
        "expected_patient_count": int(expected_patient_count),
        "observed_patient_count": observed_patient_count,
        "n_folds": int(selected["fold_idx"].nunique()) if not selected.empty else 0,
        "missing_columns": {},
        "selection_policy": selection_policy,
        "a9v3_baseline": A9V3,
        "passes_main_gate": bool(summary["passes_main_gate"].any()) if "passes_main_gate" in summary.columns else False,
        "best_config_by_gate_margin": best_config,
        "fold2_metrics": by_fold[by_fold["fold_idx"].astype(int) == 2].to_dict("records"),
        "fold4_metrics": by_fold[by_fold["fold_idx"].astype(int) == 4].to_dict("records"),
        "patient_count_warning": observed_patient_count != int(expected_patient_count),
    }

    summary.to_csv(root / "validation_selected_teacher_anchor_summary.csv", index=False)
    by_fold.to_csv(root / "validation_selected_teacher_anchor_by_fold.csv", index=False)
    by_center.to_csv(root / "validation_selected_teacher_anchor_by_center.csv", index=False)
    selected_patients.to_csv(root / "validation_selected_teacher_anchor_patient_rows.csv", index=False)
    selected.to_csv(root / "validation_selected_teacher_anchor_selected_params.csv", index=False)
    val_grid.to_csv(root / "validation_selected_teacher_anchor_val_grid.csv", index=False)
    test_grid.to_csv(root / "validation_selected_teacher_anchor_test_grid.csv", index=False)
    failure.to_csv(root / "validation_selected_teacher_anchor_fold_failure_audit.csv", index=False)
    (root / "validation_selected_teacher_anchor_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    return {
        "summary": summary,
        "by_fold": by_fold,
        "by_center": by_center,
        "patient_rows": selected_patients,
        "selected_params": selected,
        "val_grid": val_grid,
        "test_grid": test_grid,
        "fold_failure_audit": failure,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validation-selected teacher-anchor rescore for A9v8 LCBO outputs.")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--alpha_grid", default="0.02,0.05,0.10,0.20,0.30,0.50")
    parser.add_argument("--beta_grid", default="-0.20,-0.10,-0.05,0.00,0.05,0.10,0.20")
    parser.add_argument("--gamma_grid", default="-0.20,-0.10,-0.05,0.00,0.02,0.05,0.10,0.15")
    parser.add_argument("--selection_policy", default="gate_margin", choices=["gate_margin"])
    parser.add_argument("--expected_patient_count", type=int, default=90)
    parser.add_argument("--include_raw_score_eval", action="store_true")
    parser.add_argument("--include_teacher_only", action="store_true")
    parser.add_argument("--include_logit_broad_core", action="store_true")
    args = parser.parse_args()
    result = run_validation_selected_rescore(
        args.run_dir,
        alpha_grid=_parse_grid(args.alpha_grid, DEFAULT_ALPHA_GRID),
        beta_grid=_parse_grid(args.beta_grid, DEFAULT_BETA_GRID),
        gamma_grid=_parse_grid(args.gamma_grid, DEFAULT_GAMMA_GRID),
        selection_policy=args.selection_policy,
        expected_patient_count=args.expected_patient_count,
        include_raw_score_eval=args.include_raw_score_eval,
        include_teacher_only=args.include_teacher_only,
        include_logit_broad_core=args.include_logit_broad_core,
    )
    print(
        "Wrote validation-selected teacher-anchor outputs: "
        f"{len(result['summary'])} configs, {len(result['by_fold'])} fold rows."
    )


if __name__ == "__main__":
    main()
