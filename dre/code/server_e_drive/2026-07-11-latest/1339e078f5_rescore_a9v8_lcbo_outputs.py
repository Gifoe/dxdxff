"""Rescore A9v8 LCBO outputs with logit fusion and teacher-anchored variants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_recall_fscore_support

GAMMA_GRID = [-0.20, -0.10, -0.05, 0.0, 0.02, 0.05, 0.10, 0.15]
ALPHA_GRID = [0.02, 0.05, 0.10, 0.20, 0.30, 0.50]
BETA_GRID = [-0.20, -0.10, -0.05, 0.00, 0.05, 0.10, 0.20]
EPS = 1e-7
A9V3_EXPECTED = {
    "patient_macro_f1": 0.642714,
    "patient_macro_ez_f1": 0.470591,
    "patient_macro_auprc_ez": 0.518635,
    "patient_macro_ez_mrr": 0.715136,
    "top1_is_ez_rate": 0.600000,
}


def _safe_logit(probs: np.ndarray) -> np.ndarray:
    clipped = np.clip(probs.astype(float), EPS, 1.0 - EPS)
    return np.log(clipped / (1.0 - clipped))


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = values.astype(float)
    return 1.0 / (1.0 + np.exp(-values))


def patient_zscore(values: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    values = values.astype(float)
    mean = np.nanmean(values)
    std = np.nanstd(values)
    if not np.isfinite(std) or std < eps:
        return np.zeros_like(values, dtype=float)
    return (values - mean) / max(std, eps)


def _load_resolved_args(root: Path) -> dict[str, object]:
    path = root / "resolved_run_args.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}; rescore cannot infer positive-label or teacher semantics safely."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _positive_label(args: dict[str, object]) -> str:
    value = str(args.get("positive_label", "ez")).lower()
    if value not in {"ez", "nez"}:
        raise RuntimeError(f"Unsupported positive_label={value!r}; expected ez or nez.")
    return value


def _positive_to_ez(scores: np.ndarray, positive_label: str) -> np.ndarray:
    return scores.astype(float) if positive_label == "ez" else 1.0 - scores.astype(float)


def passes_main_gate(row: dict[str, float] | pd.Series) -> bool:
    return (
        float(row.get("patient_macro_f1", 0.0)) > A9V3_EXPECTED["patient_macro_f1"]
        and float(row.get("patient_macro_ez_f1", 0.0)) > A9V3_EXPECTED["patient_macro_ez_f1"]
        and float(row.get("patient_macro_auprc_ez", 0.0)) > A9V3_EXPECTED["patient_macro_auprc_ez"]
        and float(row.get("patient_macro_ez_mrr", 0.0)) >= A9V3_EXPECTED["patient_macro_ez_mrr"]
        and float(row.get("top1_is_ez_rate", 0.0)) >= A9V3_EXPECTED["top1_is_ez_rate"]
    )


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


def _patient_metrics_for_score(
    df: pd.DataFrame,
    *,
    score_values: np.ndarray,
) -> list[dict[str, float | str | int]]:
    rows: list[dict[str, float | str | int]] = []
    work = df.copy()
    work["_score"] = score_values.astype(float)
    for keys, group in work.groupby(["config", "fold_idx", "subject_id"], sort=False, dropna=False):
        config, fold_idx, subject_id = keys
        y_ez = group["true_ez"].astype(int).to_numpy()
        y_nez = 1 - y_ez
        scores = group["_score"].astype(float).to_numpy()
        pred_ez = _select_topk(scores, int(y_ez.sum()))
        pred_nez = (~pred_ez).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(y_nez, pred_nez, labels=[1, 0], zero_division=0)
        auprc_ez = float(average_precision_score(y_ez, scores)) if np.unique(y_ez).size > 1 else 0.0
        rows.append(
            {
                "config": str(config),
                "fold_idx": int(fold_idx),
                "subject_id": str(subject_id),
                "center": str(group["center"].iloc[0]) if "center" in group.columns else "unknown",
                "patient_macro_f1": float(f1_score(y_nez, pred_nez, average="macro", zero_division=0)),
                "patient_macro_ez_f1": float(f1[1]),
                "patient_macro_auprc_ez": auprc_ez,
                "patient_macro_ez_mrr": _mrr(y_ez, scores),
                "top1_is_ez": float(y_ez[int(np.argmax(scores))] == 1) if scores.size else 0.0,
            }
        )
    return rows


def _summarize_patient_rows(
    rows: Iterable[dict[str, float | str | int]],
    group_cols: list[str],
) -> pd.DataFrame:
    df = pd.DataFrame(list(rows))
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
    for col in ("alpha", "beta", "gamma"):
        if col not in out.columns:
            out[col] = np.nan
    if "top1_is_ez_rate" in out.columns:
        out["beats_a9v3_f1"] = out["patient_macro_f1"] > A9V3_EXPECTED["patient_macro_f1"]
        out["beats_a9v3_ez_f1"] = out["patient_macro_ez_f1"] > A9V3_EXPECTED["patient_macro_ez_f1"]
        out["beats_a9v3_auprc"] = out["patient_macro_auprc_ez"] > A9V3_EXPECTED["patient_macro_auprc_ez"]
        out["beats_a9v3_mrr"] = out["patient_macro_ez_mrr"] >= A9V3_EXPECTED["patient_macro_ez_mrr"]
        out["beats_a9v3_top1"] = out["top1_is_ez_rate"] >= A9V3_EXPECTED["top1_is_ez_rate"]
        out["passes_main_gate"] = out.apply(passes_main_gate, axis=1)
    return out


def _load_channel_predictions(run_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(run_dir.rglob("test_channel_predictions_neuroez_v2_fold_*.csv")):
        df = pd.read_csv(path)
        if "true_ez" not in df.columns:
            continue
        df["config"] = path.parent.name if path.parent != run_dir else run_dir.name
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No test_channel_predictions_neuroez_v2_fold_*.csv found under {run_dir}")
    return pd.concat(frames, ignore_index=True)


def _logit_pair(channels: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, str]:
    if "logits_broad" in channels.columns and "logits_core" in channels.columns:
        return (
            channels["logits_broad"].astype(float).to_numpy(),
            channels["logits_core"].astype(float).to_numpy(),
            "logit",
        )
    if "score_broad" in channels.columns and "score_core" in channels.columns:
        return (
            _safe_logit(channels["score_broad"].astype(float).to_numpy()),
            _safe_logit(channels["score_core"].astype(float).to_numpy()),
            "logit_from_probability",
        )
    raise KeyError("Missing logits_broad/logits_core and score_broad/score_core for LCBO rescore.")


def _append_rows(
    out: list[dict[str, float | str | int]],
    channels: pd.DataFrame,
    *,
    scores: np.ndarray,
    variant_family: str,
    score_variant: str,
    score_space: str,
    alpha: float | None = None,
    beta: float | None = None,
    gamma: float | None = None,
) -> None:
    rows = _patient_metrics_for_score(channels, score_values=scores)
    for row in rows:
        row.update(
            {
                "variant_family": variant_family,
                "score_name": score_variant,
                "score_variant": score_variant,
                "score_space": score_space,
                "fusion_space": score_space,
                "alpha": np.nan if alpha is None else float(alpha),
                "beta": np.nan if beta is None else float(beta),
                "gamma": np.nan if gamma is None else float(gamma),
            }
        )
    out.extend(rows)


def _build_standard_patient_rows(channels: pd.DataFrame, *, positive_label: str) -> list[dict[str, float | str | int]]:
    rows: list[dict[str, float | str | int]] = []
    for col in ("score_eval", "score_broad", "score_core", "a9v3_oof_score"):
        if col in channels.columns:
            _append_rows(
                rows,
                channels,
                scores=_positive_to_ez(channels[col].astype(float).to_numpy(), positive_label),
                variant_family="raw_score",
                score_variant=col,
                score_space="probability",
            )
    broad_logit, core_logit, score_space = _logit_pair(channels)
    for gamma in GAMMA_GRID:
        _append_rows(
            rows,
            channels,
            scores=_positive_to_ez(_sigmoid(broad_logit + float(gamma) * core_logit), positive_label),
            variant_family="logit_broad_core",
            score_variant=f"logit_broad_plus_gamma_core_g{gamma:+.2f}",
            score_space=score_space,
            gamma=gamma,
        )
    return rows


def _build_teacher_anchor_patient_rows(
    channels: pd.DataFrame,
    *,
    positive_label: str,
    teacher_mode: str,
) -> list[dict[str, float | str | int]]:
    if teacher_mode == "physiology_only":
        return []
    teacher_col = "teacher_nez_score" if positive_label == "nez" else "a9v3_oof_score"
    if teacher_col not in channels.columns or not np.isfinite(channels[teacher_col].astype(float)).all():
        raise RuntimeError(f"Teacher-anchor rescore requires finite {teacher_col} for teacher_mode={teacher_mode}.")
    rows: list[dict[str, float | str | int]] = []
    grouped = channels.groupby(["config", "fold_idx", "subject_id"], sort=False, dropna=False)
    z_teacher = np.zeros(len(channels), dtype=float)
    z_broad = np.zeros(len(channels), dtype=float)
    z_core = np.zeros(len(channels), dtype=float)
    for _, idx in grouped.indices.items():
        idx_arr = np.asarray(idx, dtype=int)
        z_teacher[idx_arr] = patient_zscore(channels.iloc[idx_arr][teacher_col].astype(float).to_numpy())
        z_broad[idx_arr] = patient_zscore(channels.iloc[idx_arr]["score_broad"].astype(float).to_numpy())
        z_core[idx_arr] = patient_zscore(channels.iloc[idx_arr]["score_core"].astype(float).to_numpy())

    _append_rows(
        rows,
        channels,
        scores=z_teacher,
        variant_family="teacher_anchor",
        score_variant="teacher_only",
        score_space="patient_zscore_probability",
    )
    for alpha in ALPHA_GRID:
        _append_rows(
            rows,
            channels,
            scores=z_teacher + float(alpha) * z_broad,
            variant_family="teacher_anchor",
            score_variant=f"teacher_plus_broad_a{alpha:.2f}",
            score_space="patient_zscore_probability",
            alpha=alpha,
        )
        for beta in BETA_GRID:
            _append_rows(
                rows,
                channels,
                scores=z_teacher + float(alpha) * z_broad + float(beta) * z_core,
                variant_family="teacher_anchor",
                score_variant=f"teacher_plus_broad_core_a{alpha:.2f}_b{beta:+.2f}",
                score_space="patient_zscore_probability",
                alpha=alpha,
                beta=beta,
            )
    return rows


def _metric_dict(row: pd.Series) -> dict[str, float]:
    return {key: float(row[key]) for key in A9V3_EXPECTED if key in row}


def _audit_reproduction(root: Path, summary: pd.DataFrame, expected_patient_count: int) -> dict[str, object]:
    audit: dict[str, object] = {"expected_patient_count": int(expected_patient_count)}
    n_patients = int(summary["n_patients"].max()) if "n_patients" in summary.columns and not summary.empty else 0
    audit["n_unique_subjects"] = n_patients
    audit["patient_count_warning"] = bool(n_patients != int(expected_patient_count))

    teacher = summary[summary["score_variant"] == "a9v3_oof_score"]
    if not teacher.empty:
        observed = _metric_dict(teacher.iloc[0])
        abs_diff = {key: abs(observed.get(key, 0.0) - A9V3_EXPECTED[key]) for key in A9V3_EXPECTED}
        audit["a9v3_teacher_reproduction_warning"] = any(v > 1e-3 for v in abs_diff.values())
        audit["a9v3_teacher_reproduction"] = {
            "observed": observed,
            "expected": A9V3_EXPECTED,
            "abs_diff": abs_diff,
        }

    eval_warnings = []
    eval_rows = summary[summary["score_variant"] == "score_eval"] if "score_variant" in summary.columns else pd.DataFrame()
    for _, row in eval_rows.iterrows():
        config = str(row["config"])
        summary_path = root / config / "heldout_summary_neuroez_v3.json"
        if not summary_path.exists():
            continue
        heldout = json.loads(summary_path.read_text(encoding="utf-8"))
        observed = _metric_dict(row)
        expected = {key: float(heldout[key]) for key in A9V3_EXPECTED if key in heldout}
        abs_diff = {key: abs(observed.get(key, 0.0) - expected.get(key, 0.0)) for key in expected}
        warn = any(v > 1e-3 for v in abs_diff.values())
        if warn:
            eval_warnings.append(
                {
                    "score_eval_reproduction_warning": True,
                    "config": config,
                    "observed": observed,
                    "heldout_summary": expected,
                    "abs_diff": abs_diff,
                }
            )
    audit["score_eval_reproduction_warnings"] = eval_warnings
    audit["score_eval_reproduction_warning"] = bool(eval_warnings)
    return audit


def _write_best_variants(root: Path, teacher_summary: pd.DataFrame) -> None:
    def best(metric: str) -> dict[str, object]:
        if teacher_summary.empty or metric not in teacher_summary.columns:
            return {}
        row = teacher_summary.sort_values(metric, ascending=False).iloc[0]
        return row.where(pd.notna(row), None).to_dict()

    passing = teacher_summary[teacher_summary["passes_main_gate"] == True] if "passes_main_gate" in teacher_summary.columns else pd.DataFrame()
    if not passing.empty:
        ranked = passing.assign(
            _rank_score=(
                passing["patient_macro_f1"]
                + passing["patient_macro_ez_f1"]
                + passing["patient_macro_auprc_ez"]
                + passing["patient_macro_ez_mrr"]
            )
        ).sort_values("_rank_score", ascending=False)
        recommended = ranked.head(3).drop(columns=["_rank_score"]).where(pd.notna(ranked.head(3).drop(columns=["_rank_score"])), None).to_dict("records")
    else:
        recommended = []
    payload = {
        "best_by_f1": best("patient_macro_f1"),
        "best_by_ez_f1": best("patient_macro_ez_f1"),
        "best_by_auprc": best("patient_macro_auprc_ez"),
        "best_by_mrr": best("patient_macro_ez_mrr"),
        "best_passing_main_gate": passing.where(pd.notna(passing), None).to_dict("records"),
        "recommended_stage1c_training_configs": recommended,
    }
    (root / "lcbo_teacher_anchor_best_variants.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_outputs(root: Path, prefix: str, rows: list[dict[str, float | str | int]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    patient_df = pd.DataFrame(rows)
    group_cols = ["config", "variant_family", "score_variant", "score_space", "alpha", "beta", "gamma"]
    summary = _summarize_patient_rows(rows, group_cols)
    by_center = _summarize_patient_rows(rows, group_cols + ["center"])
    by_fold = _summarize_patient_rows(rows, group_cols + ["fold_idx"])
    for frame in (summary, by_center, by_fold):
        if not frame.empty and "score_variant" in frame.columns and "score_name" not in frame.columns:
            frame["score_name"] = frame["score_variant"]
    summary.to_csv(root / f"{prefix}_summary.csv", index=False)
    by_center.to_csv(root / f"{prefix}_by_center.csv", index=False)
    by_fold.to_csv(root / f"{prefix}_by_fold.csv", index=False)
    patient_df.to_csv(root / f"{prefix}_patient_rows.csv", index=False)
    return summary, by_center, by_fold, patient_df


def rescore_run_dir(
    run_dir: str | Path,
    *,
    expected_patient_count: int = 90,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    root = Path(run_dir)
    resolved = _load_resolved_args(root)
    positive_label = _positive_label(resolved)
    teacher_mode = str(resolved.get("teacher_mode", "legacy_global_oof")).lower()
    if teacher_mode not in {"physiology_only", "legacy_global_oof", "nested_outer_oof"}:
        raise RuntimeError(f"Unsupported teacher_mode={teacher_mode!r}.")
    channels = _load_channel_predictions(root)
    standard_rows = _build_standard_patient_rows(channels, positive_label=positive_label)
    teacher_rows = _build_teacher_anchor_patient_rows(
        channels,
        positive_label=positive_label,
        teacher_mode=teacher_mode,
    )
    summary, by_center, by_fold, _ = _write_outputs(root, "lcbo_rescore", standard_rows)
    teacher_summary, teacher_by_center, teacher_by_fold, _ = _write_outputs(root, "lcbo_teacher_anchor_rescore", teacher_rows) if teacher_rows else (pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    _write_best_variants(root, teacher_summary)
    audit = _audit_reproduction(root, summary, expected_patient_count)
    (root / "lcbo_rescore_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return pd.concat([summary, teacher_summary], ignore_index=True), pd.concat([by_center, teacher_by_center], ignore_index=True), pd.concat([by_fold, teacher_by_fold], ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rescore A9v8 LCBO outputs with logit fusion and teacher-anchor variants.")
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--expected_patient_count", type=int, default=90)
    args = parser.parse_args()
    summary, by_center, by_fold = rescore_run_dir(args.run_dir, expected_patient_count=args.expected_patient_count)
    print(f"Wrote {len(summary)} summary rows, {len(by_center)} center rows, {len(by_fold)} fold rows under {args.run_dir}")


if __name__ == "__main__":
    main()
