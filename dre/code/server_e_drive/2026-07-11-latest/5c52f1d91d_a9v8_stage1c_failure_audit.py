"""Diagnostic-only audit for A9v8 Stage1c validation-selected failures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

A9V3 = {
    "patient_macro_f1": 0.642714,
    "patient_macro_ez_f1": 0.470591,
    "patient_macro_auprc_ez": 0.518635,
    "patient_macro_ez_mrr": 0.715136,
    "top1_is_ez_rate": 0.600000,
}
REQUIRED_FILES = [
    "validation_selected_teacher_anchor_summary.csv",
    "validation_selected_teacher_anchor_by_fold.csv",
]
OPTIONAL_FILES = [
    "validation_selected_teacher_anchor_selected_params.csv",
    "validation_selected_teacher_anchor_patient_rows.csv",
    "validation_selected_teacher_anchor_val_grid.csv",
    "validation_selected_teacher_anchor_test_grid.csv",
    "lcbo_rescore_summary.csv",
    "lcbo_rescore_by_fold.csv",
    "lcbo_rescore_by_center.csv",
    "lcbo_teacher_anchor_rescore_summary.csv",
    "grouped_results/main_gate_candidates.csv",
    "grouped_results/all_result_variants.csv",
]
MISMATCH_COLUMNS = [
    "config",
    "fold_idx",
    "selected_score_variant",
    "selected_alpha",
    "selected_beta",
    "selected_gamma",
    "selected_val_f1",
    "selected_val_ez_f1",
    "selected_val_auprc",
    "selected_val_mrr",
    "selected_val_top1",
    "selected_test_f1",
    "selected_test_ez_f1",
    "selected_test_auprc",
    "selected_test_mrr",
    "selected_test_top1",
    "best_test_score_variant_by_mrr",
    "best_test_alpha_by_mrr",
    "best_test_beta_by_mrr",
    "best_test_gamma_by_mrr",
    "best_test_mrr",
    "best_test_top1",
    "test_mrr_regret",
    "test_top1_regret",
    "best_test_score_variant_by_gate_margin",
    "best_test_alpha_by_gate_margin",
    "best_test_beta_by_gate_margin",
    "best_test_gamma_by_gate_margin",
    "best_test_gate_margin",
    "test_gate_margin_regret",
]


def _read_required(root: Path, filename: str) -> pd.DataFrame:
    path = root / filename
    if not path.exists():
        raise FileNotFoundError(f"Required Stage1c audit input missing: {filename}")
    return pd.read_csv(path)


def _filter_configs(df: pd.DataFrame, candidate_configs: Sequence[str] | None) -> pd.DataFrame:
    if df.empty or not candidate_configs or "config" not in df.columns:
        return df
    keep = set(str(item) for item in candidate_configs)
    return df[df["config"].astype(str).isin(keep)].copy()


def _passes_main_gate(row: pd.Series | dict[str, float]) -> bool:
    return (
        float(row.get("patient_macro_f1", 0.0)) > A9V3["patient_macro_f1"]
        and float(row.get("patient_macro_ez_f1", 0.0)) > A9V3["patient_macro_ez_f1"]
        and float(row.get("patient_macro_auprc_ez", 0.0)) > A9V3["patient_macro_auprc_ez"]
        and float(row.get("patient_macro_ez_mrr", 0.0)) >= A9V3["patient_macro_ez_mrr"]
        and float(row.get("top1_is_ez_rate", 0.0)) >= A9V3["top1_is_ez_rate"]
    )


def _failed_metric_tags(row: pd.Series, *, prefix: str = "low") -> list[str]:
    tags = []
    if float(row["patient_macro_f1"]) < A9V3["patient_macro_f1"]:
        tags.append(f"{prefix}_f1")
    if float(row["patient_macro_ez_f1"]) < A9V3["patient_macro_ez_f1"]:
        tags.append(f"{prefix}_ez_f1")
    if float(row["patient_macro_auprc_ez"]) < A9V3["patient_macro_auprc_ez"]:
        tags.append(f"{prefix}_auprc")
    if float(row["patient_macro_ez_mrr"]) < A9V3["patient_macro_ez_mrr"]:
        tags.append(f"{prefix}_mrr")
    if float(row["top1_is_ez_rate"]) < A9V3["top1_is_ez_rate"]:
        tags.append(f"{prefix}_top1")
    return tags


def _distance_to_gate(row: pd.Series) -> float:
    return float(
        min(
            float(row["patient_macro_f1"]) - A9V3["patient_macro_f1"],
            float(row["patient_macro_ez_f1"]) - A9V3["patient_macro_ez_f1"],
            float(row["patient_macro_auprc_ez"]) - A9V3["patient_macro_auprc_ez"],
            float(row["patient_macro_ez_mrr"]) - A9V3["patient_macro_ez_mrr"],
            float(row["top1_is_ez_rate"]) - A9V3["top1_is_ez_rate"],
        )
    )


def _gate_report(summary: pd.DataFrame, candidate_configs: Sequence[str] | None) -> pd.DataFrame:
    df = _filter_configs(summary.copy(), candidate_configs)
    rows = []
    for _, row in df.iterrows():
        item = row.to_dict()
        item.update(
            {
                "delta_f1_vs_a9v3": float(row["patient_macro_f1"] - A9V3["patient_macro_f1"]),
                "delta_ez_f1_vs_a9v3": float(row["patient_macro_ez_f1"] - A9V3["patient_macro_ez_f1"]),
                "delta_auprc_vs_a9v3": float(row["patient_macro_auprc_ez"] - A9V3["patient_macro_auprc_ez"]),
                "delta_mrr_vs_a9v3": float(row["patient_macro_ez_mrr"] - A9V3["patient_macro_ez_mrr"]),
                "delta_top1_vs_a9v3": float(row["top1_is_ez_rate"] - A9V3["top1_is_ez_rate"]),
                "failed_gate_metrics": ";".join(_failed_metric_tags(row)),
                "passes_main_gate": bool(_passes_main_gate(row)),
                "distance_to_gate": _distance_to_gate(row),
            }
        )
        rows.append(item)
    cols = [
        "config",
        "patient_macro_f1",
        "patient_macro_ez_f1",
        "patient_macro_auprc_ez",
        "patient_macro_ez_mrr",
        "top1_is_ez_rate",
        "delta_f1_vs_a9v3",
        "delta_ez_f1_vs_a9v3",
        "delta_auprc_vs_a9v3",
        "delta_mrr_vs_a9v3",
        "delta_top1_vs_a9v3",
        "failed_gate_metrics",
        "passes_main_gate",
        "distance_to_gate",
    ]
    return pd.DataFrame(rows, columns=cols)


def _fold_failure(by_fold: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in by_fold.iterrows():
        item = row.to_dict()
        item.update(
            {
                "delta_f1_vs_gate": float(row["patient_macro_f1"] - A9V3["patient_macro_f1"]),
                "delta_ez_f1_vs_gate": float(row["patient_macro_ez_f1"] - A9V3["patient_macro_ez_f1"]),
                "delta_auprc_vs_gate": float(row["patient_macro_auprc_ez"] - A9V3["patient_macro_auprc_ez"]),
                "delta_mrr_vs_gate": float(row["patient_macro_ez_mrr"] - A9V3["patient_macro_ez_mrr"]),
                "delta_top1_vs_gate": float(row["top1_is_ez_rate"] - A9V3["top1_is_ez_rate"]),
                "weakness_tag": ";".join(_failed_metric_tags(row)),
            }
        )
        rows.append(item)
    cols = [
        "config",
        "fold_idx",
        "selected_variant_family",
        "selected_score_variant",
        "selected_alpha",
        "selected_beta",
        "selected_gamma",
        "n_patients",
        "patient_macro_f1",
        "patient_macro_ez_f1",
        "patient_macro_auprc_ez",
        "patient_macro_ez_mrr",
        "top1_is_ez_rate",
        "delta_f1_vs_gate",
        "delta_ez_f1_vs_gate",
        "delta_auprc_vs_gate",
        "delta_mrr_vs_gate",
        "delta_top1_vs_gate",
        "weakness_tag",
    ]
    return pd.DataFrame(rows).reindex(columns=cols)


def _norm_param(value: object) -> str:
    if pd.isna(value) or str(value) == "":
        return ""
    try:
        return f"{float(value):.6g}"
    except (TypeError, ValueError):
        return str(value)


def _match_grid_row(grid: pd.DataFrame, selected: pd.Series) -> pd.DataFrame:
    mask = (grid["config"].astype(str) == str(selected["config"])) & (grid["fold_idx"].astype(int) == int(selected["fold_idx"]))
    mask &= grid["score_variant"].astype(str) == str(selected["selected_score_variant"])
    for selected_col, grid_col in (("selected_alpha", "alpha"), ("selected_beta", "beta"), ("selected_gamma", "gamma")):
        if selected_col in selected and grid_col in grid.columns and _norm_param(selected[selected_col]) != "":
            mask &= grid[grid_col].map(_norm_param) == _norm_param(selected[selected_col])
    return grid[mask]


def _gate_margin(row: pd.Series) -> float:
    return _distance_to_gate(row)


def _selection_mismatch(root: Path, missing_optional: list[str], candidate_configs: Sequence[str] | None) -> pd.DataFrame:
    required = [
        "validation_selected_teacher_anchor_selected_params.csv",
        "validation_selected_teacher_anchor_val_grid.csv",
        "validation_selected_teacher_anchor_test_grid.csv",
    ]
    if any(not (root / filename).exists() for filename in required):
        for filename in required:
            if not (root / filename).exists() and filename not in missing_optional:
                missing_optional.append(filename)
        return pd.DataFrame(columns=MISMATCH_COLUMNS)
    selected = _filter_configs(pd.read_csv(root / required[0]), candidate_configs)
    val_grid = _filter_configs(pd.read_csv(root / required[1]), candidate_configs)
    test_grid = _filter_configs(pd.read_csv(root / required[2]), candidate_configs)
    if selected.empty or val_grid.empty or test_grid.empty:
        return pd.DataFrame(columns=MISMATCH_COLUMNS)
    rows = []
    for _, sel in selected.iterrows():
        fold_grid = test_grid[
            (test_grid["config"].astype(str) == str(sel["config"]))
            & (test_grid["fold_idx"].astype(int) == int(sel["fold_idx"]))
        ].copy()
        selected_test = _match_grid_row(test_grid, sel)
        if selected_test.empty or fold_grid.empty:
            continue
        selected_row = selected_test.iloc[0]
        by_mrr = fold_grid.sort_values(["patient_macro_ez_mrr", "top1_is_ez_rate"], ascending=False).iloc[0]
        fold_grid["_gate_margin"] = fold_grid.apply(_gate_margin, axis=1)
        by_gate = fold_grid.sort_values(["_gate_margin", "patient_macro_ez_mrr", "top1_is_ez_rate"], ascending=False).iloc[0]
        rows.append(
            {
                "config": sel["config"],
                "fold_idx": int(sel["fold_idx"]),
                "selected_score_variant": sel["selected_score_variant"],
                "selected_alpha": sel.get("selected_alpha", np.nan),
                "selected_beta": sel.get("selected_beta", np.nan),
                "selected_gamma": sel.get("selected_gamma", np.nan),
                "selected_val_f1": sel.get("val_patient_macro_f1", np.nan),
                "selected_val_ez_f1": sel.get("val_patient_macro_ez_f1", np.nan),
                "selected_val_auprc": sel.get("val_patient_macro_auprc_ez", np.nan),
                "selected_val_mrr": sel.get("val_patient_macro_ez_mrr", np.nan),
                "selected_val_top1": sel.get("val_top1_is_ez_rate", np.nan),
                "selected_test_f1": selected_row["patient_macro_f1"],
                "selected_test_ez_f1": selected_row["patient_macro_ez_f1"],
                "selected_test_auprc": selected_row["patient_macro_auprc_ez"],
                "selected_test_mrr": selected_row["patient_macro_ez_mrr"],
                "selected_test_top1": selected_row["top1_is_ez_rate"],
                "best_test_score_variant_by_mrr": by_mrr["score_variant"],
                "best_test_alpha_by_mrr": by_mrr.get("alpha", np.nan),
                "best_test_beta_by_mrr": by_mrr.get("beta", np.nan),
                "best_test_gamma_by_mrr": by_mrr.get("gamma", np.nan),
                "best_test_mrr": by_mrr["patient_macro_ez_mrr"],
                "best_test_top1": by_mrr["top1_is_ez_rate"],
                "test_mrr_regret": float(by_mrr["patient_macro_ez_mrr"] - selected_row["patient_macro_ez_mrr"]),
                "test_top1_regret": float(by_mrr["top1_is_ez_rate"] - selected_row["top1_is_ez_rate"]),
                "best_test_score_variant_by_gate_margin": by_gate["score_variant"],
                "best_test_alpha_by_gate_margin": by_gate.get("alpha", np.nan),
                "best_test_beta_by_gate_margin": by_gate.get("beta", np.nan),
                "best_test_gamma_by_gate_margin": by_gate.get("gamma", np.nan),
                "best_test_gate_margin": by_gate["_gate_margin"],
                "test_gate_margin_regret": float(by_gate["_gate_margin"] - _gate_margin(selected_row)),
            }
        )
    return pd.DataFrame(rows, columns=MISMATCH_COLUMNS)


def _patient_failure(root: Path, missing_optional: list[str], candidate_configs: Sequence[str] | None) -> pd.DataFrame:
    path = root / "validation_selected_teacher_anchor_patient_rows.csv"
    teacher_cols = [
        "teacher_patient_macro_f1",
        "teacher_patient_ez_f1",
        "teacher_patient_auprc_ez",
        "teacher_patient_ez_mrr",
        "teacher_top1_is_ez",
        "delta_f1_vs_teacher",
        "delta_ez_f1_vs_teacher",
        "delta_auprc_vs_teacher",
        "delta_mrr_vs_teacher",
        "delta_top1_vs_teacher",
    ]
    if not path.exists():
        missing_optional.append(path.name)
        return pd.DataFrame()
    df = _filter_configs(pd.read_csv(path), candidate_configs)
    rows = []
    for _, row in df.iterrows():
        item = row.to_dict()
        item["weakness_tag"] = ";".join(
            tag.replace("low_", "low_")
            for tag in _failed_metric_tags(
                pd.Series(
                    {
                        "patient_macro_f1": row.get("patient_macro_f1", 0.0),
                        "patient_macro_ez_f1": row.get("patient_macro_ez_f1", 0.0),
                        "patient_macro_auprc_ez": row.get("patient_macro_auprc_ez", 0.0),
                        "patient_macro_ez_mrr": row.get("patient_macro_ez_mrr", 0.0),
                        "top1_is_ez_rate": row.get("top1_is_ez", 0.0),
                    }
                )
            )
        )
        for col in teacher_cols:
            item.setdefault(col, np.nan)
        rows.append(item)
    cols = [
        "config",
        "fold_idx",
        "subject_id",
        "center",
        "valid_channel_count",
        "ez_channel_count",
        "ez_fraction",
        "selected_variant_family",
        "selected_score_variant",
        "selected_alpha",
        "selected_beta",
        "selected_gamma",
        "patient_macro_f1",
        "patient_macro_ez_f1",
        "patient_macro_auprc_ez",
        "patient_macro_ez_mrr",
        "top1_is_ez",
        "weakness_tag",
        *teacher_cols,
    ]
    return pd.DataFrame(rows).reindex(columns=cols)


def _center_by_fold(patient_failure: pd.DataFrame) -> pd.DataFrame:
    if patient_failure.empty:
        return pd.DataFrame()
    grouped = patient_failure.groupby(["config", "fold_idx", "center"], dropna=False, sort=True)
    out = grouped.agg(
        n_patients=("subject_id", "nunique"),
        mean_ez_channel_count=("ez_channel_count", "mean"),
        mean_valid_channel_count=("valid_channel_count", "mean"),
        mean_ez_fraction=("ez_fraction", "mean"),
        patient_macro_f1=("patient_macro_f1", "mean"),
        patient_macro_ez_f1=("patient_macro_ez_f1", "mean"),
        patient_macro_auprc_ez=("patient_macro_auprc_ez", "mean"),
        patient_macro_ez_mrr=("patient_macro_ez_mrr", "mean"),
        top1_is_ez_rate=("top1_is_ez", "mean"),
    ).reset_index()
    return out


def _row_min_delta(df: pd.DataFrame) -> pd.Series:
    return df[["delta_f1_vs_gate", "delta_ez_f1_vs_gate", "delta_auprc_vs_gate", "delta_mrr_vs_gate", "delta_top1_vs_gate"]].min(axis=1)


def _weakest_fold_ranking(fold_failure: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "fold_idx",
        "n_configs",
        "mean_min_delta",
        "worst_min_delta",
        "mean_patient_macro_f1",
        "mean_patient_macro_ez_f1",
        "mean_patient_macro_auprc_ez",
        "mean_patient_macro_ez_mrr",
        "mean_top1_is_ez_rate",
    ]
    if fold_failure.empty:
        return pd.DataFrame(columns=cols)
    work = fold_failure.copy()
    work["_min_delta"] = _row_min_delta(work)
    out = (
        work.groupby("fold_idx", dropna=False, sort=True)
        .agg(
            n_configs=("config", "nunique"),
            mean_min_delta=("_min_delta", "mean"),
            worst_min_delta=("_min_delta", "min"),
            mean_patient_macro_f1=("patient_macro_f1", "mean"),
            mean_patient_macro_ez_f1=("patient_macro_ez_f1", "mean"),
            mean_patient_macro_auprc_ez=("patient_macro_auprc_ez", "mean"),
            mean_patient_macro_ez_mrr=("patient_macro_ez_mrr", "mean"),
            mean_top1_is_ez_rate=("top1_is_ez_rate", "mean"),
        )
        .reset_index()
        .sort_values(["worst_min_delta", "mean_min_delta"], ascending=True)
    )
    return out.reindex(columns=cols)


def _weakest_center_ranking(center_by_fold: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "center",
        "n_config_fold_rows",
        "n_total_patients",
        "mean_patient_macro_f1",
        "mean_patient_macro_ez_f1",
        "mean_patient_macro_auprc_ez",
        "mean_patient_macro_ez_mrr",
        "mean_top1_is_ez_rate",
        "worst_fold_mrr",
        "worst_fold_top1",
        "mean_min_delta",
        "worst_min_delta",
    ]
    if center_by_fold.empty:
        return pd.DataFrame(columns=cols)
    work = center_by_fold.copy()
    work["delta_f1_vs_gate"] = work["patient_macro_f1"] - A9V3["patient_macro_f1"]
    work["delta_ez_f1_vs_gate"] = work["patient_macro_ez_f1"] - A9V3["patient_macro_ez_f1"]
    work["delta_auprc_vs_gate"] = work["patient_macro_auprc_ez"] - A9V3["patient_macro_auprc_ez"]
    work["delta_mrr_vs_gate"] = work["patient_macro_ez_mrr"] - A9V3["patient_macro_ez_mrr"]
    work["delta_top1_vs_gate"] = work["top1_is_ez_rate"] - A9V3["top1_is_ez_rate"]
    work["_min_delta"] = _row_min_delta(work)
    out = (
        work.groupby("center", dropna=False, sort=True)
        .agg(
            n_config_fold_rows=("center", "size"),
            n_total_patients=("n_patients", "sum"),
            mean_patient_macro_f1=("patient_macro_f1", "mean"),
            mean_patient_macro_ez_f1=("patient_macro_ez_f1", "mean"),
            mean_patient_macro_auprc_ez=("patient_macro_auprc_ez", "mean"),
            mean_patient_macro_ez_mrr=("patient_macro_ez_mrr", "mean"),
            mean_top1_is_ez_rate=("top1_is_ez_rate", "mean"),
            worst_fold_mrr=("patient_macro_ez_mrr", "min"),
            worst_fold_top1=("top1_is_ez_rate", "min"),
            mean_min_delta=("_min_delta", "mean"),
            worst_min_delta=("_min_delta", "min"),
        )
        .reset_index()
        .sort_values(["mean_min_delta", "mean_patient_macro_ez_mrr", "mean_top1_is_ez_rate"], ascending=True)
    )
    return out.reindex(columns=cols)


def _recommendation(
    gate_report: pd.DataFrame,
    weakest_fold_ranking: pd.DataFrame,
    weakest_center_ranking: pd.DataFrame,
    optional_missing: list[str],
) -> dict[str, object]:
    any_pass = bool(gate_report["passes_main_gate"].any()) if not gate_report.empty else False
    closest = None
    failed = []
    if not gate_report.empty:
        row = gate_report.sort_values("distance_to_gate", ascending=False).iloc[0]
        closest = str(row["config"])
        failed = [tag for tag in str(row["failed_gate_metrics"]).split(";") if tag]
    weakest_folds_unique = (
        [int(v) for v in weakest_fold_ranking.head(3)["fold_idx"].tolist()]
        if not weakest_fold_ranking.empty
        else []
    )
    weakest_centers_by_global_mean = (
        [str(v) for v in weakest_center_ranking.sort_values(["mean_min_delta", "mean_patient_macro_ez_mrr", "mean_top1_is_ez_rate"], ascending=True).head(3)["center"].tolist()]
        if not weakest_center_ranking.empty
        else []
    )
    weakest_centers_by_worst_fold = (
        [str(v) for v in weakest_center_ranking.sort_values(["worst_min_delta", "worst_fold_mrr", "worst_fold_top1"], ascending=True).head(3)["center"].tolist()]
        if not weakest_center_ranking.empty
        else []
    )
    return {
        "whether_any_config_passes_main_gate": any_pass,
        "best_validation_selected_config": closest,
        "closest_config_to_gate": closest,
        "failed_metrics_for_closest_config": failed,
        "weakest_folds_unique": weakest_folds_unique,
        "weakest_centers_by_global_mean": weakest_centers_by_global_mean,
        "weakest_centers_by_worst_fold": weakest_centers_by_worst_fold,
        "optional_files_missing": sorted(set(optional_missing)),
        "whether_to_continue_training": False,
        "recommended_next_action": (
            "Stop Stage1c training. Keep A9v3/teacher baseline as the main stable result. "
            "Use Stage1c a020_b000 raw score_eval only as exploratory ablation unless a future "
            "validation-selected method passes the predefined gate."
        ),
    }


def run_stage1c_failure_audit(
    run_dir: str | Path,
    *,
    candidate_configs: Sequence[str] | None = None,
    verbose: bool = False,
) -> dict[str, pd.DataFrame]:
    root = Path(run_dir)
    summary = _read_required(root, REQUIRED_FILES[0])
    by_fold = _read_required(root, REQUIRED_FILES[1])
    optional_missing = [filename for filename in OPTIONAL_FILES if not (root / filename).exists()]

    summary = _filter_configs(summary, candidate_configs)
    by_fold = _filter_configs(by_fold, candidate_configs)
    gate_report = _gate_report(summary, candidate_configs)
    fold_failure = _fold_failure(by_fold)
    selection_mismatch = _selection_mismatch(root, optional_missing, candidate_configs)
    patient_failure = _patient_failure(root, optional_missing, candidate_configs)
    center_by_fold = _filter_configs(_center_by_fold(patient_failure), candidate_configs)
    weakest_fold_ranking = _weakest_fold_ranking(fold_failure)
    weakest_center_ranking = _weakest_center_ranking(center_by_fold)
    recommendation = _recommendation(gate_report, weakest_fold_ranking, weakest_center_ranking, optional_missing)

    gate_report.to_csv(root / "stage1c_validation_selected_gate_report.csv", index=False)
    fold_failure.to_csv(root / "stage1c_fold_failure_audit.csv", index=False)
    selection_mismatch.to_csv(root / "stage1c_val_test_selection_mismatch.csv", index=False)
    patient_failure.to_csv(root / "stage1c_patient_failure_audit.csv", index=False)
    center_by_fold.to_csv(root / "stage1c_center_by_fold_audit.csv", index=False)
    weakest_fold_ranking.to_csv(root / "stage1c_weakest_fold_ranking.csv", index=False)
    weakest_center_ranking.to_csv(root / "stage1c_weakest_center_ranking.csv", index=False)
    (root / "stage1c_failure_audit_recommendation.json").write_text(json.dumps(recommendation, indent=2), encoding="utf-8")

    if verbose:
        print(f"Candidate configs used: {[str(v) for v in candidate_configs] if candidate_configs else 'ALL'}")
        print(f"Any config passed main gate: {recommendation['whether_any_config_passes_main_gate']}")
        print(f"Closest config to gate: {recommendation['closest_config_to_gate']}")
        print(f"Unique weakest folds: {recommendation['weakest_folds_unique']}")
        print(f"Weakest centers by global mean: {recommendation['weakest_centers_by_global_mean']}")
        print(f"Weakest centers by worst fold: {recommendation['weakest_centers_by_worst_fold']}")
        print(f"Optional grid files missing: {bool(recommendation['optional_files_missing'])}")
        print(f"Recommendation: {recommendation['recommended_next_action']}")
    return {
        "gate_report": gate_report,
        "fold_failure": fold_failure,
        "selection_mismatch": selection_mismatch,
        "patient_failure": patient_failure,
        "center_by_fold": center_by_fold,
        "weakest_fold_ranking": weakest_fold_ranking,
        "weakest_center_ranking": weakest_center_ranking,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit A9v8 Stage1c validation-selected failure modes.")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--candidate_config", action="append", default=[])
    args = parser.parse_args()
    run_stage1c_failure_audit(args.run_dir, candidate_configs=args.candidate_config, verbose=True)


if __name__ == "__main__":
    main()
