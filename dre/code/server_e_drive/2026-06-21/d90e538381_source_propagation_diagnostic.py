from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_TIME_FEATURES = (
    "onset_latency",
    "activation_time",
    "time_to_peak",
    "early_high_gamma",
    "late_high_gamma",
    "early_line_length",
    "late_line_length",
    "early_to_late_ratio",
    "lagged_corr_outward",
    "lagged_corr_inward",
    "directed_early_to_late_connectivity",
)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _resolve_config_dir(root: Path, config_name: str) -> Path:
    if str(config_name).upper() == "BEST":
        best_path = root / "a9v14_candidate_best.json"
        if not best_path.exists():
            raise FileNotFoundError(f"BEST requested but missing {best_path}")
        best = json.loads(best_path.read_text(encoding="utf-8"))
        config_name = str(best.get("best_overall_config") or best.get("config_name") or "")
        if not config_name:
            raise ValueError(f"Could not resolve BEST config from {best_path}")
    return root / config_name


def _load_ledger(config_dir: Path) -> pd.DataFrame:
    candidates = [
        config_dir / "a9v14_prediction_ledger.csv",
        config_dir / "patient_channel_ledger.csv",
    ]
    candidates.extend(sorted(config_dir.glob("test_channel_predictions_neuroez_v2_fold_*.csv")))
    existing = [path for path in candidates if path.exists()]
    if not existing:
        raise FileNotFoundError(f"No prediction ledger or test channel CSVs found under {config_dir}")
    frames = [pd.read_csv(path) for path in existing if path.suffix.lower() == ".csv"]
    return pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]


def _score_column(df: pd.DataFrame) -> str:
    for col in ("final_score", "score_eval", "score_ez_probability", "score_ez_final", "base_score_a9v3"):
        if col in df.columns:
            return col
    raise ValueError("Ledger has no usable EZ score column for propagation diagnostic.")


def _label_column(df: pd.DataFrame) -> str:
    for col in ("y_true", "label_ez", "true_ez"):
        if col in df.columns:
            return col
    raise ValueError("Ledger has no EZ label column for propagation diagnostic.")


def _write_insufficient(config_dir: Path, ledger: pd.DataFrame, missing: list[str]) -> dict[str, Any]:
    config_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "diagnostic_status": "insufficient_features",
        "missing_required_fields": missing,
        "available_columns": list(ledger.columns),
        "n_rows": int(len(ledger)),
    }
    (config_dir / "source_propagation_missing_features.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    (config_dir / "insufficient_feature_report.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    summary = {
        "diagnostic_status": "insufficient_features",
        "missing_required_fields": missing,
        "n_rows": int(len(ledger)),
    }
    (config_dir / "source_propagation_diagnostic_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    pd.DataFrame().to_csv(config_dir / "source_propagation_diagnostic_by_patient.csv", index=False)
    pd.DataFrame().to_csv(config_dir / "source_propagation_diagnostic_by_center.csv", index=False)
    return summary


def _add_fp_tp_tn_flags(df: pd.DataFrame, score_col: str, label_col: str) -> pd.DataFrame:
    work = df.copy()
    if "patient_id" not in work.columns:
        work["patient_id"] = work.get("subject_id", "unknown").astype(str)
    work["_score"] = pd.to_numeric(work[score_col], errors="coerce").fillna(0.0)
    work["_label"] = pd.to_numeric(work[label_col], errors="coerce").fillna(0.0)
    pred_flags = []
    for _, group in work.groupby("patient_id", sort=False):
        k = int(max(round(float(group["_label"].sum())), 1))
        ranks = group["_score"].rank(method="first", ascending=False)
        pred_flags.extend((ranks <= k).astype(int).tolist())
    work["_pred"] = pred_flags
    work["_bucket"] = np.select(
        [
            (work["_label"] <= 0.5) & (work["_pred"] == 1),
            (work["_label"] > 0.5) & (work["_pred"] == 1),
            (work["_label"] > 0.5) & (work["_pred"] == 0),
        ],
        ["FP", "TP", "FN"],
        default="TN",
    )
    return work


def _mannwhitney(values_a: pd.Series, values_b: pd.Series) -> float | None:
    try:
        from scipy.stats import mannwhitneyu
    except Exception:
        return None
    a = pd.to_numeric(values_a, errors="coerce").dropna()
    b = pd.to_numeric(values_b, errors="coerce").dropna()
    if len(a) == 0 or len(b) == 0:
        return None
    return float(mannwhitneyu(a, b, alternative="two-sided").pvalue)


def _compute_diagnostic(config_dir: Path, ledger: pd.DataFrame) -> dict[str, Any]:
    score_col = _score_column(ledger)
    label_col = _label_column(ledger)
    work = _add_fp_tp_tn_flags(ledger, score_col, label_col)
    work["late_to_early_hg_ratio"] = pd.to_numeric(work["late_high_gamma"], errors="coerce") / pd.to_numeric(work["early_high_gamma"], errors="coerce").replace(0, np.nan)
    work["late_to_early_ll_ratio"] = pd.to_numeric(work["late_line_length"], errors="coerce") / pd.to_numeric(work["early_line_length"], errors="coerce").replace(0, np.nan)
    work["propagation_like_score"] = (
        pd.to_numeric(work["onset_latency"], errors="coerce").rank(pct=True)
        + pd.to_numeric(work["late_to_early_hg_ratio"], errors="coerce").rank(pct=True)
        + pd.to_numeric(work["late_to_early_ll_ratio"], errors="coerce").rank(pct=True)
    ) / 3.0
    by_patient = (
        work.groupby(["patient_id", "_bucket"], dropna=False)
        .agg(
            n_channels=("channel_name", "count"),
            mean_onset_latency=("onset_latency", "mean"),
            mean_time_to_peak=("time_to_peak", "mean"),
            mean_late_to_early_hg_ratio=("late_to_early_hg_ratio", "mean"),
            mean_late_to_early_ll_ratio=("late_to_early_ll_ratio", "mean"),
            mean_propagation_like_score=("propagation_like_score", "mean"),
        )
        .reset_index()
        .rename(columns={"_bucket": "prediction_bucket"})
    )
    if "center" in work.columns:
        by_center = (
            work.groupby(["center", "_bucket"], dropna=False)
            .agg(
                n_channels=("channel_name", "count"),
                mean_onset_latency=("onset_latency", "mean"),
                mean_propagation_like_score=("propagation_like_score", "mean"),
            )
            .reset_index()
            .rename(columns={"_bucket": "prediction_bucket"})
        )
    else:
        by_center = pd.DataFrame()

    fp = work[work["_bucket"].eq("FP")]
    tp = work[work["_bucket"].eq("TP")]
    summary = {
        "diagnostic_status": "computed",
        "n_rows": int(len(work)),
        "n_false_positive": int(len(fp)),
        "n_true_positive": int(len(tp)),
        "fp_late_activation_rate": float((pd.to_numeric(fp["onset_latency"], errors="coerce") > pd.to_numeric(work["onset_latency"], errors="coerce").median()).mean()) if len(fp) else 0.0,
        "fp_high_late_amplitude_rate": float((pd.to_numeric(fp["late_to_early_hg_ratio"], errors="coerce") > 1.0).mean()) if len(fp) else 0.0,
        "fp_propagation_like_rate": float((pd.to_numeric(fp["propagation_like_score"], errors="coerce") > 0.67).mean()) if len(fp) else 0.0,
        "tp_early_activation_rate": float((pd.to_numeric(tp["onset_latency"], errors="coerce") <= pd.to_numeric(work["onset_latency"], errors="coerce").median()).mean()) if len(tp) else 0.0,
    }
    p_value = _mannwhitney(fp["onset_latency"] if len(fp) else pd.Series(dtype=float), tp["onset_latency"] if len(tp) else pd.Series(dtype=float))
    if p_value is not None:
        summary["mannwhitney_fp_vs_tp_onset_latency_p"] = p_value
    else:
        summary["mannwhitney_status"] = "skipped_scipy_unavailable_or_empty_groups"

    config_dir.mkdir(parents=True, exist_ok=True)
    by_patient.to_csv(config_dir / "source_propagation_diagnostic_by_patient.csv", index=False)
    by_center.to_csv(config_dir / "source_propagation_diagnostic_by_center.csv", index=False)
    (config_dir / "source_propagation_missing_features.json").write_text(
        json.dumps({"diagnostic_status": "computed", "missing_required_fields": []}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (config_dir / "source_propagation_diagnostic_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    return summary


def run_source_propagation_diagnostic(root: str | Path, config_name: str) -> dict[str, Any]:
    root = Path(root)
    config_dir = _resolve_config_dir(root, config_name)
    ledger = _load_ledger(config_dir)
    missing = [field for field in REQUIRED_TIME_FEATURES if field not in ledger.columns]
    if missing:
        return _write_insufficient(config_dir, ledger, missing)
    return _compute_diagnostic(config_dir, ledger)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run A9v14 source-to-propagation diagnostic from prediction ledger.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config_name", type=str, required=True)
    args = parser.parse_args()
    summary = run_source_propagation_diagnostic(args.root, args.config_name)
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default))


if __name__ == "__main__":
    main()
