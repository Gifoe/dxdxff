#!/usr/bin/env python3
"""Compare frozen P2-Q10 and the LZU-only adapter without conflating optimization and efficacy."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from neuroez_c.p2_q10_lzu_adapter_protocol import canonicalize_channel_frame, evaluate_channel_ledger


METRICS = [
    "patient_macro_f1", "patient_macro_ez_f1", "patient_macro_auprc_ez",
    "patient_macro_ez_mrr", "predicted_ez_count_mae", "truek_macro_f1", "truek_ez_f1",
]
SUMMARY_METRICS = [metric for metric in METRICS if not metric.startswith("truek_")]


def _resolve_base_ledger(root: Path) -> pd.DataFrame:
    candidates = [
        root / "p2_q10_adapter_base_oof_channel_ledger.csv",
        root / "p23_channel_predictions.csv",
        root / "outer_test_channel_predictions.csv",
    ]
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        raise FileNotFoundError(
            f"No frozen P2-Q10 channel ledger found under {root}; expected one of "
            + ", ".join(candidate.name for candidate in candidates)
        )
    frame = pd.read_csv(path)
    if "base_nez_logit" not in frame.columns:
        for source in ("direct_nez_logit", "final_nez_logit"):
            if source in frame.columns:
                frame["base_nez_logit"] = frame[source]
                break
    return canonicalize_channel_frame(frame)


def _paired_bootstrap(frame: pd.DataFrame, metric: str, repeats: int, seed: int, center: str | None = None) -> dict:
    selected = frame if center is None else frame[frame.center.eq(center)]
    if selected.empty:
        return {"metric": metric, "center": center or "overall", "mean_delta": np.nan, "CI_2.5": np.nan,
                "CI_97.5": np.nan, "probability_delta_gt_zero": np.nan, "n_patients_improved": 0,
                "n_patients_worsened": 0, "n_delta_gt_0.05": 0, "n_delta_lt_minus_0.05": 0}
    base_column, adapter_column = f"{metric}_base", f"{metric}_adapter"
    missing = [column for column in (base_column, adapter_column) if column not in selected]
    if missing:
        raise RuntimeError(f"Missing paired bootstrap columns for {metric}: {missing}")
    delta = selected[adapter_column].to_numpy(float) - selected[base_column].to_numpy(float)
    rng = np.random.default_rng(seed)
    draws = np.asarray([delta[rng.integers(0, len(delta), len(delta))].mean() for _ in range(repeats)])
    return {"metric": metric, "center": center or "overall", "mean_delta": float(delta.mean()),
            "CI_2.5": float(np.quantile(draws, .025)), "CI_97.5": float(np.quantile(draws, .975)),
            "probability_delta_gt_zero": float((draws > 0).mean()),
            "n_patients_improved": int((delta > 0).sum()), "n_patients_worsened": int((delta < 0).sum()),
            "n_delta_gt_0.05": int((delta > .05).sum()), "n_delta_lt_minus_0.05": int((delta < -.05).sum())}


def _performance_status(*, optimization_status: str, minimum_pass: bool) -> str:
    if optimization_status == "OPTIMIZATION_INSUFFICIENT_STEPS":
        return optimization_status
    if optimization_status != "OPTIMIZATION_EFFECTIVE":
        return "OPTIMIZATION_NO_OP"
    return "OPTIMIZATION_EFFECTIVE_WITH_GAIN" if minimum_pass else "OPTIMIZATION_EFFECTIVE_BUT_NO_PERFORMANCE_GAIN"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_root", required=True)
    parser.add_argument("--adapter_root", required=True)
    parser.add_argument("--bootstrap_repeats", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    base_root, adapter_root = Path(args.base_root), Path(args.adapter_root)

    base_ledger = _resolve_base_ledger(base_root)
    adapter_ledger = canonicalize_channel_frame(pd.read_csv(adapter_root / "p2_lzu_adapter_oof_channel_ledger.csv"))
    completed_folds = set(adapter_ledger.outer_fold.astype(int).unique())
    base_ledger = base_ledger[base_ledger.outer_fold.astype(int).isin(completed_folds)].copy()
    if set(base_ledger.subject_id.astype(str)) != set(adapter_ledger.subject_id.astype(str)):
        missing = sorted(set(adapter_ledger.subject_id.astype(str)) - set(base_ledger.subject_id.astype(str)))
        extra = sorted(set(base_ledger.subject_id.astype(str)) - set(adapter_ledger.subject_id.astype(str)))
        raise RuntimeError(f"Base/adapter subjects differ within completed folds: missing={missing}, extra={extra}")
    base_patients, base_summary, base_fold, base_center = evaluate_channel_ledger(base_ledger, logit_column="base_nez_logit")
    adapter_patients, adapter_summary, adapter_fold, adapter_center = evaluate_channel_ledger(adapter_ledger, logit_column="adapted_nez_logit")
    base_truek, _, _, _ = evaluate_channel_ledger(base_ledger, logit_column="base_nez_logit", truek=True)
    adapter_truek, _, _, _ = evaluate_channel_ledger(adapter_ledger, logit_column="adapted_nez_logit", truek=True)

    keys = ["subject_id", "center", "outer_fold"]
    paired = base_patients.merge(adapter_patients, on=keys, suffixes=("_base", "_adapter"), validate="one_to_one")
    if len(paired) != len(base_patients) or len(paired) != len(adapter_patients):
        raise RuntimeError("Base/adapter patient ledgers do not align exactly")
    paired = paired.merge(
        base_truek[keys + ["patient_macro_f1", "patient_macro_ez_f1"]].rename(columns={
            "patient_macro_f1": "truek_macro_f1_base", "patient_macro_ez_f1": "truek_ez_f1_base"}),
        on=keys, validate="one_to_one",
    ).merge(
        adapter_truek[keys + ["patient_macro_f1", "patient_macro_ez_f1"]].rename(columns={
            "patient_macro_f1": "truek_macro_f1_adapter", "patient_macro_ez_f1": "truek_ez_f1_adapter"}),
        on=keys, validate="one_to_one",
    )
    for metric in METRICS:
        paired[f"delta_{metric}"] = paired[f"{metric}_adapter"] - paired[f"{metric}_base"]
    paired.to_csv(adapter_root / "p2_lzu_adapter_vs_base_by_patient.csv", index=False)

    comparisons = []
    for level, base_frame, adapter_frame, merge_keys in (
        ("overall", base_summary, adapter_summary, []),
        ("by_fold", base_fold, adapter_fold, ["outer_fold"]),
        ("by_center", base_center, adapter_center, ["center"]),
    ):
        if merge_keys:
            merged = base_frame.merge(adapter_frame, on=merge_keys, suffixes=("_base", "_adapter"), validate="one_to_one")
        else:
            merged = pd.concat([base_frame.add_suffix("_base"), adapter_frame.add_suffix("_adapter")], axis=1)
        for metric in SUMMARY_METRICS:
            merged[f"delta_{metric}"] = merged[f"{metric}_adapter"] - merged[f"{metric}_base"]
        merged.to_csv(adapter_root / f"p2_lzu_adapter_vs_base_{level}.csv", index=False)
        comparisons.append(merged.assign(level=level))
    comparison = pd.concat(comparisons, ignore_index=True)
    comparison.to_csv(adapter_root / "p2_lzu_adapter_vs_base_summary.csv", index=False)

    grouped = paired.assign(center_group=np.where(paired.center.eq("lzu"), "lzu", "non_lzu"))
    bootstrap = [_paired_bootstrap(paired, metric, args.bootstrap_repeats, args.seed) for metric in METRICS]
    bootstrap += [_paired_bootstrap(grouped.rename(columns={"center": "original_center", "center_group": "center"}), metric, args.bootstrap_repeats, args.seed, center)
                  for center in ("lzu", "non_lzu") for metric in METRICS]
    pd.DataFrame(bootstrap).to_csv(adapter_root / "p2_lzu_adapter_paired_bootstrap.csv", index=False)

    optimization = json.loads((adapter_root / "p2_lzu_adapter_optimization_audit.json").read_text(encoding="utf-8"))
    invariance = json.loads((adapter_root / "non_lzu_invariance_audit.json").read_text(encoding="utf-8"))
    residual = pd.read_csv(adapter_root / "p2_lzu_adapter_residual_diagnostics.csv")
    overall = comparison[comparison.level.eq("overall")].iloc[0]
    lzu_row = comparison[(comparison.level.eq("by_center")) & comparison.center.eq("lzu")].iloc[0]
    lzu_fold = paired[paired.center.eq("lzu")].groupby("outer_fold", as_index=False).agg(
        base_f1=("patient_macro_f1_base", "mean"), adapter_f1=("patient_macro_f1_adapter", "mean"))
    lzu_fold["delta_f1"] = lzu_fold.adapter_f1 - lzu_fold.base_f1

    mean_abs_delta = float(np.average(residual.mean_abs_adapter_delta, weights=residual.get("n_channels", pd.Series(np.ones(len(residual))))))
    p95_abs_delta = float(residual.p95_abs_adapter_delta.max())
    saturation_rate = float(residual.saturation_rate.max())
    saturation_status = "RESIDUAL_SATURATION_FAILURE" if saturation_rate >= .20 else "RESIDUAL_SATURATION_WARNING" if saturation_rate >= .05 else "PASSED"
    folds_non_decreasing = int((lzu_fold.delta_f1 >= 0).sum())
    full_five_fold = int(adapter_ledger.outer_fold.nunique()) == 5
    lzu_auprc_delta = float(lzu_row.delta_patient_macro_auprc_ez)
    minimum_pass = bool(
        full_five_fold and optimization.get("all_folds_optimization_passed") and invariance.get("passed")
        and float(overall.patient_macro_f1_adapter) >= .634
        and float(overall.delta_patient_macro_f1) >= .005
        and float(lzu_row.delta_patient_macro_f1) >= .02
        and folds_non_decreasing >= 3 and lzu_auprc_delta >= -.005
        and mean_abs_delta < .10 and p95_abs_delta < .25 and saturation_rate < .05
    )
    strong_pass = bool(minimum_pass and float(overall.patient_macro_f1_adapter) >= .640
                       and float(lzu_row.delta_patient_macro_f1) >= .05 and folds_non_decreasing >= 4)
    optimization_status = str(optimization.get("optimization_status", "MISSING"))
    performance_status = _performance_status(optimization_status=optimization_status, minimum_pass=minimum_pass)
    admission = {
        "optimization_status": optimization_status,
        "performance_status": performance_status,
        "non_lzu_invariance_status": "PASSED" if invariance.get("passed") else "FAILED",
        "threshold_freeze_status": "PASSED" if bool(adapter_ledger.get("threshold_refit_after_adapter", pd.Series([True])).eq(False).all()) else "FAILED",
        "full_five_fold_run": full_five_fold,
        "overall_formal_f1": float(overall.patient_macro_f1_adapter),
        "overall_formal_f1_delta": float(overall.delta_patient_macro_f1),
        "lzu_formal_f1_delta": float(lzu_row.delta_patient_macro_f1),
        "lzu_ez_auprc_delta": lzu_auprc_delta,
        "folds_lzu_formal_non_decreasing": folds_non_decreasing,
        "mean_abs_adapter_delta": mean_abs_delta,
        "p95_abs_adapter_delta": p95_abs_delta,
        "saturation_rate": saturation_rate,
        "residual_safety_status": saturation_status,
        "minimum_pass": minimum_pass,
        "strong_pass": strong_pass,
    }
    (adapter_root / "p2_lzu_adapter_final_status.json").write_text(json.dumps(admission, indent=2), encoding="utf-8")
    report = (
        "# P2-Q10 LZU Adapter Optimization Report\n\n"
        "Formal predictions retain the frozen P2-Q10 validation threshold. True-K metrics are diagnostic only.\n\n"
        "```json\n" + json.dumps(admission, indent=2) + "\n```\n"
    )
    (adapter_root / "P2_Q10_LZU_ADAPTER_OPTIMIZATION_REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps(admission, indent=2))


if __name__ == "__main__":
    main()
