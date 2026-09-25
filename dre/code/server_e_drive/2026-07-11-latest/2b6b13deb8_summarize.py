from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from outcome_hifos.bootstrap import paired_patient_bootstrap
from outcome_hifos.metrics import compute_patient_metrics

from .plots import calibration_table


REQUIRED_PREDICTION_COLUMNS = {
    "subject_id",
    "center",
    "outcome",
    "variant",
    "seed",
    "outer_fold_idx",
    "role",
    "raw_logit",
    "calibrated_probability",
    "selected_threshold",
    "predicted",
    "calibration_method",
    "threshold_source",
    "checkpoint_path",
    "fold_ledger_hash",
    "config_hash",
    "cohort_id",
    "subject_set_hash",
}


def _validate_prediction_protocol(predictions: pd.DataFrame, allowed_roles: set[str]) -> None:
    missing = sorted(REQUIRED_PREDICTION_COLUMNS - set(predictions.columns))
    if missing:
        raise ValueError(f"Prediction protocol is missing required columns: {missing}")
    invalid_roles = sorted(set(predictions["role"].astype(str)) - set(allowed_roles))
    if invalid_roles:
        raise ValueError(f"Summary accepts only roles {sorted(allowed_roles)}, got roles {invalid_roles}.")
    invalid_sources = sorted(set(predictions["threshold_source"].astype(str)) - {"inner_oof"})
    if invalid_sources:
        raise ValueError(f"Prediction threshold_source must be inner_oof, got {invalid_sources}.")
    expected = (predictions["calibrated_probability"].to_numpy() >= predictions["selected_threshold"].to_numpy()).astype(int)
    if not np.array_equal(expected, predictions["predicted"].to_numpy(dtype=int)):
        raise ValueError("Stored predicted labels do not match calibrated_probability and selected_threshold.")


def _metric_row(group: pd.DataFrame, **identity: Any) -> dict[str, Any]:
    probability = group["calibrated_probability"].to_numpy()
    bundle = compute_patient_metrics(group["outcome"].to_numpy(), probability, predicted=group["predicted"].to_numpy())
    reference = compute_patient_metrics(group["outcome"].to_numpy(), probability, 0.5)
    return {
        **identity,
        **bundle.values,
        "macro_f1_at_0_5": reference.values["macro_f1"],
        "patient_count": int(group["subject_id"].nunique()),
        "confusion_matrix": json.dumps(bundle.confusion_matrix),
        "undefined_reasons": json.dumps(bundle.undefined_reasons, sort_keys=True),
    }


def comparison_cohort_status(predictions: pd.DataFrame, candidate: str, reference: str) -> str:
    candidate_rows = predictions[predictions["variant"] == candidate]
    reference_rows = predictions[predictions["variant"] == reference]
    if candidate_rows.empty or reference_rows.empty:
        return "UNAVAILABLE_VARIANT"
    keys = ["cohort_id", "fold_ledger_hash", "subject_set_hash"]
    candidate_provenance = {tuple(row) for row in candidate_rows[keys].drop_duplicates().itertuples(index=False, name=None)}
    reference_provenance = {tuple(row) for row in reference_rows[keys].drop_duplicates().itertuples(index=False, name=None)}
    return "VALID" if candidate_provenance == reference_provenance and len(candidate_provenance) == 1 else "INVALID_COHORT_MISMATCH"


def _go_no_go(metrics: pd.DataFrame, *, comparison_status: dict[tuple[str, str], str] | None = None, loco_collapse: bool = False, shortcut_risk: bool = False) -> list[str]:
    scores = metrics.set_index("variant")["macro_f1"].to_dict() if not metrics.empty else {}
    flags = []
    if shortcut_risk:
        flags.append("SHORTCUT_RISK")
    comparisons = [
        ("H5_ANCHORED_CORE", "H4_MULTI_CORE", "NO_VISIBLE_ANCHOR_GAIN"),
        ("H6_ANCHORED_UOT_DESC", "H5_ANCHORED_CORE", "NO_VISIBLE_UOT_GAIN"),
        ("H7_TRANSPORT_GRAPH", "H6_ANCHORED_UOT_DESC", "PREFER_DESCRIPTOR_MODEL"),
        ("H8_RECURRENCE", "H7_TRANSPORT_GRAPH", "NO_VISIBLE_RECURRENCE_GAIN"),
        ("H9_FM_INTERSECTION", "H8_FEATURE_INTERSECTION", "NO_VISIBLE_FM_GAIN"),
        ("H10_LATE_FUSION", "H8_FEATURE_INTERSECTION", "NO_VISIBLE_FUSION_GAIN"),
    ]
    for candidate, reference, flag in comparisons:
        status = (comparison_status or {}).get((candidate, reference), "UNAVAILABLE_COMPARISON")
        if status == "VALID" and candidate in scores and reference in scores and float(scores[candidate]) <= float(scores[reference]):
            flags.append(flag)
    if loco_collapse:
        flags.append("CENTER_GENERALIZATION_FAILURE")
    return flags


def summarize_experiment(
    predictions: pd.DataFrame,
    failures: pd.DataFrame,
    output_dir: str | Path,
    *,
    run_manifest: dict[str, Any],
    evaluation_roles: tuple[str, ...] = ("outer_test",),
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if predictions.empty:
        raise ValueError("Cannot summarize an experiment with no held-out predictions.")
    _validate_prediction_protocol(predictions, set(evaluation_roles))
    predictions = predictions.copy().sort_values(["variant", "subject_id", "seed"], kind="stable").reset_index(drop=True)
    predictions.to_csv(output / "outcome_oof_predictions.csv", index=False)
    predictions.to_csv(output / "outcome_patient_predictions.csv", index=False)
    seed_rows = [
        _metric_row(group, variant=str(variant), seed=int(seed))
        for (variant, seed), group in predictions.groupby(["variant", "seed"], sort=True)
    ]
    metrics_by_seed = pd.DataFrame(seed_rows)
    metrics_by_seed.to_csv(output / "outcome_metrics_by_seed.csv", index=False)
    summary_rows = []
    metric_columns = [column for column in metrics_by_seed.columns if column not in {"variant", "seed", "confusion_matrix", "undefined_reasons"}]
    for variant, group in metrics_by_seed.groupby("variant", sort=True):
        row: dict[str, Any] = {"variant": str(variant), "seed_count": int(group["seed"].nunique())}
        for column in metric_columns:
            row[column] = float(group[column].mean())
            row[f"{column}_std"] = float(group[column].std(ddof=0))
        summary_rows.append(row)
    metrics_summary = pd.DataFrame(summary_rows)
    metrics_summary.to_csv(output / "outcome_metrics_summary.csv", index=False)
    metrics_summary.to_csv(output / "outcome_model_ablation.csv", index=False)

    fold_rows = []
    if "outer_fold_idx" in predictions:
        for (variant, seed, fold), group in predictions.groupby(["variant", "seed", "outer_fold_idx"], sort=True):
            fold_rows.append(_metric_row(group, variant=str(variant), seed=int(seed), outer_fold_idx=int(fold)))
    pd.DataFrame(fold_rows).to_csv(output / "outcome_metrics_by_fold.csv", index=False)
    center_rows = []
    for (variant, seed, center), group in predictions.groupby(["variant", "seed", "center"], sort=True):
        center_rows.append(_metric_row(group, variant=str(variant), seed=int(seed), center=str(center)))
    pd.DataFrame(center_rows).to_csv(output / "outcome_metrics_by_center.csv", index=False)
    calibration_table(predictions.assign(probability=predictions["calibrated_probability"])).to_csv(output / "outcome_calibration.csv", index=False)

    bootstrap_rows = []
    comparison_rows = []
    variants = sorted(predictions["variant"].unique().tolist())
    if variants:
        configured_pairs = run_manifest.get("bootstrap_pairs")
        if configured_pairs:
            pairs = [(str(pair[0]), str(pair[1])) for pair in configured_pairs]
        else:
            reference = str(run_manifest.get("bootstrap_reference", variants[0]))
            pairs = [(variant, reference) for variant in variants if variant != reference]
        for variant, reference in pairs:
            status = comparison_cohort_status(predictions, variant, reference)
            comparison_rows.append({"candidate": variant, "reference": reference, "comparison_status": status})
            if status != "VALID":
                bootstrap_rows.append({"candidate": variant, "reference": reference, "status": status})
                continue
            common_seeds = sorted(
                set(predictions.loc[predictions["variant"] == reference, "seed"].astype(int))
                & set(predictions.loc[predictions["variant"] == variant, "seed"].astype(int))
            )
            for current_seed in common_seeds:
                reference_frame = predictions[(predictions["variant"] == reference) & (predictions["seed"] == current_seed)][["subject_id", "outcome", "predicted"]]
                candidate = predictions[(predictions["variant"] == variant) & (predictions["seed"] == current_seed)][["subject_id", "outcome", "predicted"]]
                aligned = reference_frame.merge(candidate, on="subject_id", suffixes=("_reference", "_candidate"))
                if aligned.empty or set(aligned["subject_id"]) != set(reference_frame["subject_id"]) or set(aligned["subject_id"]) != set(candidate["subject_id"]):
                    continue
                if not np.array_equal(aligned["outcome_reference"], aligned["outcome_candidate"]):
                    continue
                interval = paired_patient_bootstrap(
                    aligned["outcome_reference"].to_numpy(),
                    aligned["predicted_candidate"].to_numpy(dtype=float),
                    aligned["predicted_reference"].to_numpy(dtype=float),
                    lambda y, p: compute_patient_metrics(y, p, predicted=np.asarray(p, dtype=int)).values["macro_f1"],
                    samples=int(run_manifest.get("bootstrap_samples", 500)),
                    seed=int(run_manifest.get("random_seed", 42)) + int(current_seed),
                )
                bootstrap_rows.append({"candidate": variant, "reference": reference, "seed": int(current_seed), "status": "VALID", **asdict(interval)})
    pd.DataFrame(bootstrap_rows).to_csv(output / "outcome_paired_bootstrap.csv", index=False)
    comparison_status_frame = pd.DataFrame(comparison_rows)
    comparison_status_frame.to_csv(output / "outcome_comparison_status.csv", index=False)

    failure_columns = ["variant", "seed", "outer_fold_idx", "error_type", "error"]
    if failures.empty:
        failures = pd.DataFrame(columns=failure_columns)
    failures.to_csv(output / "outcome_training_failures.csv", index=False)
    (output / "outcome_run_manifest.json").write_text(json.dumps(run_manifest, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    shortcut_audit_path = output.parent / "outcome_shortcut_audit.json"
    shortcut_risk = False
    if shortcut_audit_path.exists():
        shortcut_risk = bool(json.loads(shortcut_audit_path.read_text(encoding="utf-8")).get("shortcut_risk", False))
    status_lookup = {(str(row.candidate), str(row.reference)): str(row.comparison_status) for row in comparison_status_frame.itertuples(index=False)} if not comparison_status_frame.empty else {}
    flags = _go_no_go(metrics_summary, comparison_status=status_lookup, shortcut_risk=shortcut_risk)
    report_lines = [
        "# HiFOS-PACT Outcome Report",
        "",
        f"Protocol: `{run_manifest.get('protocol', 'unknown')}`",
        f"Held-out prediction rows: {len(predictions)}",
        f"Unique patients: {predictions['subject_id'].nunique()}",
        "",
        "## Go/No-Go flags",
        "",
    ]
    if "held_out_center" in predictions:
        report_lines.extend(["Held-out centers: " + ", ".join(sorted(predictions["held_out_center"].astype(str).unique())), ""])
    report_lines.extend([f"- `{flag}`" for flag in flags] or ["- No automatic flag can be determined from the available variants."])
    report_lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "Latent dynamical cores and channel responsibilities are model constructs, not EZ, SOZ, lesions, or causal clinical explanations.",
            "This report does not claim final performance or SOTA status from a bounded smoke run.",
        ]
    )
    (output / "outcome_final_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return {"metrics": metrics_summary.to_dict(orient="records"), "flags": flags}


__all__ = ["comparison_cohort_status", "summarize_experiment"]
