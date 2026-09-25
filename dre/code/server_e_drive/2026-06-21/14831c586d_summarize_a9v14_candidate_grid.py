from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


A9V3_GATE = {
    "patient_macro_f1": (">", 0.642714),
    "patient_macro_ez_f1": (">", 0.470591),
    "patient_macro_auprc_ez": (">", 0.518635),
    "patient_macro_ez_mrr": (">=", 0.715136),
    "top1_is_ez_rate": (">=", 0.600000),
}

AAAI_TARGET_070 = {
    "patient_macro_f1": (">=", 0.700),
    "patient_macro_ez_f1": (">=", 0.520),
    "patient_macro_auprc_ez": (">=", 0.560),
    "patient_macro_ez_mrr": (">=", 0.740),
    "top1_is_ez_rate": (">=", 0.650),
}

KEY_METRICS = [
    "patient_macro_f1",
    "patient_macro_ez_f1",
    "patient_macro_auprc_ez",
    "patient_macro_ez_mrr",
    "top1_is_ez_rate",
]

CENTER_FIELDS = [
    "center_gap_f1",
    "worst_center_f1",
    "best_center_f1",
    "center_hup_patient_macro_f1",
    "center_lzu_patient_macro_f1",
    "center_multicenter_patient_macro_f1",
    "center_pediatric_patient_macro_f1",
]

DIAGNOSTIC_FIELDS = [
    "rank_changed_fraction",
    "top1_changed_fraction",
    "mean_abs_reranker_delta",
    "mean_abs_consistency_delta",
    "mean_abs_local_delta",
    "parse_failure_rate",
    "consistency_feature_nan_count",
]

SUMMARY_COLUMNS = [
    "config_name",
    *KEY_METRICS,
    "patient_macro_accuracy",
    "patient_macro_balanced_accuracy",
    "patient_weighted_f1",
    "patient_macro_nez_f1",
    "patient_macro_ez_precision",
    "patient_macro_ez_recall",
    "macro_topk_recall",
    "rank_robust_composite",
    *CENTER_FIELDS,
    *DIAGNOSTIC_FIELDS,
    "n_patient_rows",
    "n_unique_subjects",
    "positive_label",
    "drop_high_ez_fraction_lzu",
    "passes_a9v3_gate",
    "passes_aaai_target_070",
    "protocol_violation",
    "failure_reasons",
]


def _warn(message: str) -> None:
    print(f"[A9v14 summarize][WARN] {message}", file=sys.stderr)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    if path.suffix.lower() == ".json":
        return _read_json(path)
    df = pd.read_csv(path)
    return df.iloc[0].to_dict() if not df.empty else {}


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if pd.notna(out) else default


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _passes(row: dict[str, Any], criteria: dict[str, tuple[str, float]]) -> bool:
    for key, (op, threshold) in criteria.items():
        value = _as_float(row.get(key))
        if op == ">" and not value > threshold:
            return False
        if op == ">=" and not value >= threshold:
            return False
    return True


def rank_robust_composite(row: dict[str, Any]) -> float:
    return float(
        0.20 * _as_float(row.get("patient_macro_f1"))
        + 0.25 * _as_float(row.get("patient_macro_ez_f1"))
        + 0.20 * _as_float(row.get("patient_macro_auprc_ez"))
        + 0.20 * _as_float(row.get("patient_macro_ez_mrr"))
        + 0.10 * _as_float(row.get("top1_is_ez_rate"))
        + 0.05 * _as_float(row.get("worst_center_f1"))
        - 0.05 * _as_float(row.get("center_gap_f1"))
    )


def _prediction_frame(run_dir: Path, stem: str) -> pd.DataFrame:
    paths = sorted(run_dir.glob(f"{stem}_fold_*.csv"))
    if not paths:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)


def _center_fields(patient_rows: pd.DataFrame) -> dict[str, float]:
    if patient_rows.empty or "center" not in patient_rows.columns or "patient_macro_f1" not in patient_rows.columns:
        return {key: 0.0 for key in CENTER_FIELDS}
    grouped = patient_rows.groupby("center", dropna=False)["patient_macro_f1"].mean(numeric_only=True)
    center_means = {str(center).strip().lower(): float(value) for center, value in grouped.items() if pd.notna(value)}
    if not center_means:
        return {key: 0.0 for key in CENTER_FIELDS}
    worst = min(center_means.values())
    best = max(center_means.values())
    out = {
        "center_gap_f1": float(best - worst) if len(center_means) >= 2 else 0.0,
        "worst_center_f1": float(worst),
        "best_center_f1": float(best),
    }
    for center in ("hup", "lzu", "multicenter", "pediatric"):
        out[f"center_{center}_patient_macro_f1"] = float(center_means.get(center, 0.0))
    return out


def _ledger_diagnostics(run_dir: Path) -> dict[str, float]:
    path = run_dir / "a9v14_prediction_ledger.csv"
    if not path.exists():
        return {}
    ledger = pd.read_csv(path)
    out: dict[str, float] = {}
    for key in ("rank_changed", "reranker_delta", "consistency_delta", "local_delta"):
        if key not in ledger.columns:
            continue
        values = pd.to_numeric(ledger[key], errors="coerce")
        if key == "rank_changed":
            out["rank_changed_fraction"] = float(values.mean()) if not values.empty else 0.0
        else:
            out[f"mean_abs_{key}"] = float(values.abs().mean()) if not values.empty else 0.0
    if "top1_before" in ledger.columns and "top1_after" in ledger.columns:
        before = pd.to_numeric(ledger["top1_before"], errors="coerce")
        after = pd.to_numeric(ledger["top1_after"], errors="coerce")
        out["top1_changed_fraction"] = float((before != after).mean()) if len(before) else 0.0
    return out


def _failure_reasons(row: dict[str, Any], baseline: dict[str, Any] | None, center_df: pd.DataFrame) -> list[str]:
    reasons: list[str] = []
    config = str(row.get("config_name", ""))
    if bool(row.get("protocol_violation", False)):
        reasons.append("protocol_violation")
    if "M1" in config and (
        _as_float(row.get("rank_changed_fraction")) == 0.0
        or _as_float(row.get("mean_abs_reranker_delta")) == 0.0
    ):
        reasons.append("reranker_no_effect")
    if baseline:
        if _as_float(row.get("patient_macro_f1")) < _as_float(baseline.get("patient_macro_f1")) - 0.005:
            reasons.append("hurts_a9v3_macro_f1")
        if _as_float(row.get("patient_macro_ez_f1")) < _as_float(baseline.get("patient_macro_ez_f1")) - 0.005:
            reasons.append("hurts_ez_f1")
        if (
            _as_float(row.get("patient_macro_ez_mrr")) < _as_float(baseline.get("patient_macro_ez_mrr")) - 0.005
            and _as_float(row.get("top1_is_ez_rate")) < _as_float(baseline.get("top1_is_ez_rate")) - 0.005
        ):
            reasons.append("hurts_mrr_top1")
        if "M3" in config:
            for center in ("lzu", "pediatric"):
                key = f"center_{center}_patient_macro_f1"
                if _as_float(row.get(key)) < _as_float(baseline.get(key)) - 0.03:
                    reasons.append("local_branch_hurts_lzu_or_pediatric")
                    break
        ez_cov_improves = (
            _as_float(row.get("patient_macro_ez_f1")) > _as_float(baseline.get("patient_macro_ez_f1")) + 0.005
            or _as_float(row.get("patient_macro_auprc_ez")) > _as_float(baseline.get("patient_macro_auprc_ez")) + 0.005
        )
        rank_drops = (
            _as_float(row.get("patient_macro_ez_mrr")) < _as_float(baseline.get("patient_macro_ez_mrr")) - 0.005
            or _as_float(row.get("top1_is_ez_rate")) < _as_float(baseline.get("top1_is_ez_rate")) - 0.005
        )
        if "M4" in config and ez_cov_improves and rank_drops:
            reasons.append("mixture_improves_coverage_hurts_top1")
        rank_improves = (
            _as_float(row.get("patient_macro_ez_mrr")) > _as_float(baseline.get("patient_macro_ez_mrr")) + 0.005
            or _as_float(row.get("top1_is_ez_rate")) > _as_float(baseline.get("top1_is_ez_rate")) + 0.005
        )
        ez_cov_not_improved = (
            _as_float(row.get("patient_macro_ez_f1")) <= _as_float(baseline.get("patient_macro_ez_f1")) + 0.005
            and _as_float(row.get("patient_macro_auprc_ez")) <= _as_float(baseline.get("patient_macro_auprc_ez")) + 0.005
        )
        if rank_improves and ez_cov_not_improved:
            reasons.append("core_reranker_only")
    diag_path = Path(str(row.get("_run_dir", ""))) / "source_propagation_diagnostic_summary.json"
    if diag_path.exists():
        diag = _read_json(diag_path)
        if diag.get("diagnostic_status") == "insufficient_features":
            reasons.append("insufficient_propagation_features")
    if int(_as_float(row.get("n_patient_rows"))) == 0 and not diag_path.exists():
        reasons.append("missing_patient_predictions")
    return sorted(set(reasons))


def _summarize_run(run_dir: Path) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    config = _read_json(run_dir / "a9v14_config.json")
    effective_args = _read_json(run_dir / "a9v14_effective_args.json")
    run_args = _read_json(run_dir / "run_args_b0_pruned.json")
    summary = _read_summary(run_dir / "heldout_summary_neuroez_v3.csv") or _read_summary(run_dir / "heldout_summary_neuroez_v3.json")
    patient_rows = _prediction_frame(run_dir, "test_patient_predictions_neuroez_v2")
    channel_rows = _prediction_frame(run_dir, "test_channel_predictions_neuroez_v2")

    row: dict[str, Any] = {"config_name": run_dir.name, "_run_dir": str(run_dir)}
    for source in (summary, run_args, effective_args, config):
        for key, value in source.items():
            row.setdefault(key, value)
    row.update(_center_fields(patient_rows))
    row.update({key: value for key, value in _ledger_diagnostics(run_dir).items() if key not in row or not pd.notna(row[key])})

    row["macro_topk_recall"] = _as_float(row.get("macro_topk_recall", row.get("ez_recall_at_true_count", 0.0)))
    row["n_patient_rows"] = int(len(patient_rows)) if not patient_rows.empty else int(_as_float(row.get("n_patient_rows")))
    if not patient_rows.empty and "subject_id" in patient_rows.columns:
        row["n_unique_subjects"] = int(patient_rows["subject_id"].nunique())
    else:
        row["n_unique_subjects"] = int(_as_float(row.get("n_patients", row.get("n_unique_subjects"))))
    row["positive_label"] = str(row.get("positive_label", "")).lower()
    row["drop_high_ez_fraction_lzu"] = _as_bool(row.get("drop_high_ez_fraction_lzu", False))
    is_smoke = int(_as_float(row.get("max_outer_folds", 0))) > 0
    row["protocol_violation"] = False
    if "M5_SourcePropagationDiagnostic" not in run_dir.name:
        row["protocol_violation"] = not (
            row["positive_label"] == "ez"
            and not row["drop_high_ez_fraction_lzu"]
            and (is_smoke or int(row["n_unique_subjects"]) in {0, 90})
        )
        if int(row["n_unique_subjects"]) == 0 and not channel_rows.empty:
            row["protocol_violation"] = True
    row["passes_a9v3_gate"] = _passes(row, A9V3_GATE)
    row["passes_aaai_target_070"] = _passes(row, AAAI_TARGET_070)
    row["rank_robust_composite"] = rank_robust_composite(row)
    for key in SUMMARY_COLUMNS:
        row.setdefault(key, 0.0 if key not in {"config_name", "positive_label", "failure_reasons"} else "")

    fold_df = pd.read_csv(run_dir / "heldout_fold_summary_neuroez_v3.csv") if (run_dir / "heldout_fold_summary_neuroez_v3.csv").exists() else pd.DataFrame()
    if not fold_df.empty:
        fold_df.insert(0, "config_name", run_dir.name)
    center_df = pd.DataFrame([{"config_name": run_dir.name, **_center_fields(patient_rows)}])
    return row, fold_df, center_df


def _recommend(summary_df: pd.DataFrame, baseline: dict[str, Any] | None) -> dict[str, Any]:
    if summary_df.empty:
        return {"recommended_next_step": "No completed configs found."}
    sortable = summary_df.copy()
    for key in ["rank_robust_composite", *KEY_METRICS, "worst_center_f1", "center_gap_f1"]:
        sortable[key] = pd.to_numeric(sortable[key], errors="coerce").fillna(0.0)
    non_diag = sortable[~sortable["config_name"].astype(str).str.contains("M5_SourcePropagationDiagnostic", na=False)]
    table = non_diag if not non_diag.empty else sortable
    best_overall = table.sort_values("rank_robust_composite", ascending=False, kind="mergesort").iloc[0].to_dict()
    best_rank = table.sort_values(["patient_macro_ez_mrr", "top1_is_ez_rate"], ascending=False, kind="mergesort").iloc[0].to_dict()
    best_ez = table.sort_values(["patient_macro_ez_f1", "patient_macro_auprc_ez"], ascending=False, kind="mergesort").iloc[0].to_dict()
    best_center = table.sort_values(["worst_center_f1", "center_gap_f1"], ascending=[False, True], kind="mergesort").iloc[0].to_dict()

    next_step = "Run source-to-propagation diagnostic and stop tuning loss until failure mode is understood."
    if baseline:
        m1m2 = table[table["config_name"].astype(str).str.contains("M1M2_RankConsistency", na=False)]
        if not m1m2.empty:
            row = m1m2.iloc[0].to_dict()
            improved = sum(_as_float(row.get(key)) > _as_float(baseline.get(key)) + 0.005 for key in KEY_METRICS)
            if bool(row.get("passes_a9v3_gate", False)) and improved >= 3:
                next_step = "Recommend A9v14-CRR using M1+M2; verify on locked full grid before paper use."
        m1_rows = table[table["config_name"].astype(str).str.contains("M1_", na=False)]
        if not m1_rows.empty:
            row = m1_rows.sort_values("rank_robust_composite", ascending=False).iloc[0].to_dict()
            rank_improves = (
                _as_float(row.get("patient_macro_ez_mrr")) > _as_float(baseline.get("patient_macro_ez_mrr")) + 0.005
                or _as_float(row.get("top1_is_ez_rate")) > _as_float(baseline.get("top1_is_ez_rate")) + 0.005
            )
            ez_cov_not = (
                _as_float(row.get("patient_macro_ez_f1")) <= _as_float(baseline.get("patient_macro_ez_f1")) + 0.005
                and _as_float(row.get("patient_macro_auprc_ez")) <= _as_float(baseline.get("patient_macro_auprc_ez")) + 0.005
            )
            if rank_improves and ez_cov_not:
                next_step = "M1 behaves like a core reranker; add M4 mixture before claiming EZ coverage gains."
        m2 = table[table["config_name"].astype(str).str.contains("M2_ConsistencyOnly", na=False)]
        if not m2.empty:
            row = m2.iloc[0].to_dict()
            ez_improves = (
                _as_float(row.get("patient_macro_ez_f1")) > _as_float(baseline.get("patient_macro_ez_f1")) + 0.005
                or _as_float(row.get("patient_macro_auprc_ez")) > _as_float(baseline.get("patient_macro_auprc_ez")) + 0.005
            )
            rank_hurts = (
                _as_float(row.get("patient_macro_ez_mrr")) < _as_float(baseline.get("patient_macro_ez_mrr")) - 0.005
                or _as_float(row.get("top1_is_ez_rate")) < _as_float(baseline.get("top1_is_ez_rate")) - 0.005
            )
            if ez_improves and rank_hurts:
                next_step = "M2 improves coverage but hurts first-rank behavior; add core/first-rank auxiliary."
        m3 = table[table["config_name"].astype(str).str.contains("M3|Local", regex=True, na=False)]
        if not m3.empty and m3["failure_reasons"].astype(str).str.contains("local_branch_hurts_lzu_or_pediatric").any():
            next_step = "Disable M3 local branch for now; it degrades LZU or pediatric robustness."

    return {
        "best_overall_config": best_overall.get("config_name", ""),
        "best_rank_config": best_rank.get("config_name", ""),
        "best_ez_coverage_config": best_ez.get("config_name", ""),
        "best_robust_center_config": best_center.get("config_name", ""),
        "recommended_next_step": next_step,
        "best_overall_row": best_overall,
    }


def summarize(root: Path) -> dict[str, Path]:
    root = Path(root)
    run_dirs = sorted(path for path in root.iterdir() if path.is_dir()) if root.exists() else []
    raw_rows: list[dict[str, Any]] = []
    fold_frames: list[pd.DataFrame] = []
    center_frames: list[pd.DataFrame] = []
    for run_dir in run_dirs:
        try:
            row, fold_df, center_df = _summarize_run(run_dir)
        except Exception as exc:
            _warn(f"{run_dir}: {exc}")
            row = {"config_name": run_dir.name, "_run_dir": str(run_dir), "failure_reasons": f"summarizer_error:{exc}"}
            for key in SUMMARY_COLUMNS:
                row.setdefault(key, "")
            fold_df = pd.DataFrame()
            center_df = pd.DataFrame()
        raw_rows.append(row)
        if not fold_df.empty:
            fold_frames.append(fold_df)
        if not center_df.empty:
            center_frames.append(center_df)

    baseline = next((row for row in raw_rows if row.get("config_name") == "A9v3_Reproduce"), None)
    rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    for row in raw_rows:
        reasons = _failure_reasons(row, baseline, pd.DataFrame())
        existing = str(row.get("failure_reasons", ""))
        if existing:
            reasons.extend([item for item in existing.split(";") if item])
        reasons = sorted(set(reasons))
        row["failure_reasons"] = ";".join(reasons)
        rows.append(row)
        if reasons:
            failure_rows.append({"config_name": row.get("config_name", ""), "failure_reasons": row["failure_reasons"]})

    summary_df = pd.DataFrame(rows)
    if summary_df.empty:
        summary_df = pd.DataFrame(columns=SUMMARY_COLUMNS)
    for key in SUMMARY_COLUMNS:
        if key not in summary_df.columns:
            summary_df[key] = ""
    summary_df = summary_df[SUMMARY_COLUMNS]
    if not summary_df.empty:
        summary_df = summary_df.sort_values(
            by=["passes_a9v3_gate", "rank_robust_composite", "patient_macro_f1", "patient_macro_ez_f1", "center_gap_f1"],
            ascending=[False, False, False, False, True],
            kind="mergesort",
        )

    fold_df = pd.concat(fold_frames, ignore_index=True) if fold_frames else pd.DataFrame()
    center_df = pd.concat(center_frames, ignore_index=True) if center_frames else pd.DataFrame(columns=["config_name", *CENTER_FIELDS])
    failure_df = pd.DataFrame(failure_rows, columns=["config_name", "failure_reasons"])
    best = _recommend(summary_df, baseline)

    paths = {
        "summary_csv": root / "a9v14_candidate_grid_summary.csv",
        "summary_json": root / "a9v14_candidate_grid_summary.json",
        "best": root / "a9v14_candidate_best.json",
        "failure_report": root / "a9v14_candidate_failure_report.json",
        "center_summary": root / "a9v14_candidate_center_summary.csv",
        "fold_summary": root / "a9v14_candidate_fold_summary.csv",
    }
    root.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(paths["summary_csv"], index=False)
    paths["summary_json"].write_text(
        json.dumps(summary_df.to_dict(orient="records"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    paths["best"].write_text(json.dumps(best, indent=2, ensure_ascii=False), encoding="utf-8")
    paths["failure_report"].write_text(
        json.dumps(failure_df.to_dict(orient="records"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    center_df.to_csv(paths["center_summary"], index=False)
    fold_df.to_csv(paths["fold_summary"], index=False)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize A9v14 candidate module grid outputs.")
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    paths = summarize(args.root)
    print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2))


if __name__ == "__main__":
    main()
