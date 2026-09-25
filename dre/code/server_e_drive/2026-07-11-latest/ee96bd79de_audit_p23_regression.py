"""Read-only P23/P2 regression audit. It never trains or rewrites model outputs."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


PROFILES = (
    "P0_CURRENT_P2", "P1_TEMPORAL", "P2_TEMPORAL_Q10",
    "P3_BOUNDED_FUSION", "P4_NOISE_AWARE", "P5_FULL", "P23_LITE",
)
METRICS = (
    "patient_macro_f1", "patient_oracle_macro_f1", "patient_macro_auprc_ez",
    "patient_macro_auprc_nez", "patient_macro_ez_mrr", "ez_recall_at_true_count",
    "patient_macro_ez_recall_at_true_count",
)


def _number(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _run_dir(root: Path, profile: str) -> Path:
    return root / profile / "seed_42"


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.is_file() else pd.DataFrame()


def _summary(root: Path, profile: str) -> dict[str, object] | None:
    path = _run_dir(root, profile) / "p23_overall_summary.csv"
    frame = _read_csv(path)
    if frame.empty:
        return None
    row = dict(frame.iloc[0])
    row["profile"] = profile
    row["run_dir"] = str(path.parent)
    row["n_heldout_patients"] = _number(row.get("n_patient_rows", row.get("n_unique_subjects")))
    return row


def _flatten_summary(rows: list[dict[str, object]]) -> pd.DataFrame:
    result = []
    for row in rows:
        flattened = {"profile": row["profile"], "run_dir": row["run_dir"]}
        for metric in METRICS:
            flattened[metric] = _number(row.get(metric))
        flattened["n_heldout_patients"] = _number(row.get("n_heldout_patients"))
        flattened["training_protocol"] = row.get("training_protocol", "")
        flattened["threshold_source"] = row.get("threshold_source", "")
        flattened["checkpoint_selection"] = row.get("checkpoint_selection", "")
        result.append(flattened)
    return pd.DataFrame(result)


def _pair_flip_fraction(frame: pd.DataFrame) -> float:
    required = {"subject_id", "direct_nez_logit", "final_nez_logit"}
    if frame.empty or not required.issubset(frame.columns):
        return float("nan")
    total = flips = 0
    for _, patient in frame.groupby("subject_id"):
        direct = patient["direct_nez_logit"].to_numpy(dtype=float)
        final = patient["final_nez_logit"].to_numpy(dtype=float)
        finite = np.isfinite(direct) & np.isfinite(final)
        direct, final = direct[finite], final[finite]
        if direct.size < 2:
            continue
        for left, right in combinations(range(direct.size), 2):
            before = direct[left] - direct[right]
            after = final[left] - final[right]
            if before == 0.0 or after == 0.0:
                continue
            total += 1
            flips += int(before * after < 0.0)
    return float(flips / total) if total else float("nan")


def _fusion_audit(root: Path, profiles: list[str]) -> pd.DataFrame:
    rows = []
    values = ("delta", "w_noop", "w_anchor", "w_seizure", "correction_saturation")
    for profile in profiles:
        frame = _read_csv(_run_dir(root, profile) / "p23_channel_predictions.csv")
        if frame.empty:
            continue
        for center, subset in [("all", frame), *list(frame.groupby("center"))]:
            row = {"profile": profile, "center": center, "n_channels": len(subset)}
            delta = pd.to_numeric(subset.get("delta"), errors="coerce") if "delta" in subset else pd.Series(dtype=float)
            row["mean_abs_delta"] = float(delta.abs().mean()) if not delta.empty else float("nan")
            row["delta_p95"] = float(delta.abs().quantile(0.95)) if not delta.empty else float("nan")
            for name in values[1:]:
                series = pd.to_numeric(subset.get(name), errors="coerce") if name in subset else pd.Series(dtype=float)
                row[f"mean_{name}"] = float(series.mean()) if not series.empty else float("nan")
            row["ranking_pair_flip_fraction"] = _pair_flip_fraction(subset)
            rows.append(row)
    return pd.DataFrame(rows)


def _temporal_audit(root: Path, profiles: list[str]) -> pd.DataFrame:
    rows = []
    for profile in profiles:
        frame = _read_csv(_run_dir(root, profile) / "p23_channel_predictions.csv")
        if frame.empty or "temporal_delta_norm" not in frame:
            continue
        for center, subset in [("all", frame), *list(frame.groupby("center"))]:
            row = {"profile": profile, "center": center, "n_channels": len(subset)}
            for field in ("valid_pre", "valid_onset", "valid_spread", "valid_late", "slope_valid"):
                values = pd.to_numeric(subset.get(field), errors="coerce") if field in subset else pd.Series(dtype=float)
                row[f"{field}_fraction"] = float(values.mean()) if not values.empty else float("nan")
            bins = [row.get(f"valid_{name}_fraction", float("nan")) for name in ("pre", "onset", "spread", "late")]
            row["missing_time_bin_fraction"] = float(np.mean([1.0 - value for value in bins if np.isfinite(value)])) if any(np.isfinite(bins)) else float("nan")
            delta = pd.to_numeric(subset["temporal_delta_norm"], errors="coerce")
            row["mean_temporal_delta_norm"] = float(delta.mean())
            rows.append(row)
    return pd.DataFrame(rows)


def _noisy_audit(root: Path, profiles: list[str]) -> pd.DataFrame:
    rows = []
    for profile in profiles:
        if profile not in {"P4_NOISE_AWARE", "P5_FULL"}:
            continue
        run_dir = _run_dir(root, profile)
        frame = _read_csv(run_dir / "p23_channel_predictions.csv")
        if not frame.empty and {"label_nez", "soft_target_nez", "ema_teacher_score_nez", "observed_ez_reliability"}.issubset(frame.columns):
            observed = frame[pd.to_numeric(frame["label_nez"], errors="coerce") == 0].copy()
            for center, subset in [("all", observed), *list(observed.groupby("center"))]:
                soft = pd.to_numeric(subset["soft_target_nez"], errors="coerce")
                teacher = pd.to_numeric(subset["ema_teacher_score_nez"], errors="coerce")
                reliability = pd.to_numeric(subset["observed_ez_reliability"], errors="coerce")
                rows.append({
                    "profile": profile, "audit_scope": "heldout_observed_ez", "center": center,
                    "n_channels": len(subset), "mean_soft_target_nez": float(soft.mean()),
                    "soft_target_nez_p05": float(soft.quantile(.05)), "soft_target_nez_p50": float(soft.quantile(.50)),
                    "soft_target_nez_p95": float(soft.quantile(.95)), "soft_target_gt_0_5_fraction": float((soft > .5).mean()),
                    "mean_teacher_score_nez": float(teacher.mean()), "mean_observed_ez_reliability": float(reliability.mean()),
                    "ema_self_reinforcement_status": "diagnostic_only; epoch correlation below is not causal evidence",
                })
        history = _read_csv(run_dir / "p23_train_history.csv")
        if not history.empty and "p23_predicted_nez_fraction_at_0_5" in history:
            for _, value in history.iterrows():
                rows.append({
                    "profile": profile, "audit_scope": "train_epoch", "center": "all",
                    "epoch": _number(value.get("epoch")),
                    "predicted_nez_fraction_at_0_5": _number(value.get("p23_predicted_nez_fraction_at_0_5")),
                    "soft_target_gt_0_5_fraction": _number(value.get("p23_observed_ez_soft_target_gt_0_5_fraction")),
                    "mean_observed_ez_reliability": _number(value.get("p23_mean_observed_ez_reliability")),
                    "ema_self_reinforcement_status": "diagnostic_only; inspect monotonic co-movement, not a causal claim",
                })
    return pd.DataFrame(rows)


def _write_report(root: Path, overall: pd.DataFrame, missing: list[str], baseline_ok: bool | None) -> None:
    lines = ["# P23 Regression Audit", "", "This report separates formal held-out metrics from diagnostics. It does not change thresholds or train models.", ""]
    if not overall.empty:
        lines.extend(["## Available Profiles", "", "```text", overall.to_string(index=False), "```", ""])
    if missing:
        lines.extend(["## Missing Outputs", "", ", ".join(missing), ""])
    if baseline_ok is not None:
        lines.extend(["## Baseline Gate", "", f"P0 reproduction gate: **{'passed' if baseline_ok else 'failed'}**.", ""])
    lines.extend([
        "## Protocol", "",
        "All P23 regression profiles use direct outer folds. Their checkpoint and threshold source are read from each run's audit; profile differences must not be interpreted unless those fields match the frozen P2 protocol.",
        "",
        "P4/P5 noisy-EZ fields are diagnostic only. A correlation between teacher score and soft targets is not evidence of causality or self-reinforcement by itself.",
    ])
    (root / "regression_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _lite_report(root: Path, overall: pd.DataFrame, center: pd.DataFrame) -> None:
    targets = {"oracle": .6984, "ez_auprc": .5151, "ez_mrr": .681919, "lzu_f1_floor": .568665}
    lite = overall[overall["profile"] == "P23_LITE"] if not overall.empty else pd.DataFrame()
    p0 = overall[overall["profile"] == "P0_CURRENT_P2"] if not overall.empty else pd.DataFrame()
    p1 = overall[overall["profile"] == "P1_TEMPORAL"] if not overall.empty else pd.DataFrame()
    p2 = overall[overall["profile"] == "P2_TEMPORAL_Q10"] if not overall.empty else pd.DataFrame()
    module_admission = False
    module_checks: dict[str, bool] = {}
    if not p0.empty and not p1.empty and not p2.empty:
        baseline = p0.iloc[0]
        for name, row in (("P1_TEMPORAL", p1.iloc[0]), ("P2_TEMPORAL_Q10", p2.iloc[0])):
            module_checks[f"{name}_ez_auprc_not_down"] = _number(row.get("patient_macro_auprc_ez")) >= _number(baseline.get("patient_macro_auprc_ez"))
            module_checks[f"{name}_ez_mrr_not_down"] = _number(row.get("patient_macro_ez_mrr")) >= _number(baseline.get("patient_macro_ez_mrr"))
        module_admission = all(module_checks.values())
    result = {"status": "eligible_for_fold1" if module_admission else "not_admitted", "criteria": targets, "module_admission": module_checks}
    if not lite.empty:
        row = lite.iloc[0]
        lzu = center[(center["profile"] == "P23_LITE") & (center["center"].astype(str).str.lower() == "lzu")] if not center.empty else pd.DataFrame()
        lzu_f1 = _number(lzu.iloc[0].get("patient_macro_f1")) if not lzu.empty else float("nan")
        full = int(_number(row.get("n_heldout_patients"))) == 80
        conditions = {
            "patient_oracle_gt_0_6984": _number(row.get("patient_oracle_macro_f1")) > targets["oracle"],
            "ez_auprc_gt_0_5151": _number(row.get("patient_macro_auprc_ez")) > targets["ez_auprc"],
            "ez_mrr_not_down": _number(row.get("patient_macro_ez_mrr")) >= targets["ez_mrr"],
            "lzu_f1_not_below_p2_by_more_than_0_01": lzu_f1 >= targets["lzu_f1_floor"],
        }
        result.update({
            "status": "keep" if full and all(conditions.values()) else ("reject" if full else "fold1_complete_pending_full"),
            "full_five_fold": full, "conditions": conditions, "lzu_f1": lzu_f1,
        })
    (root / "p23_lite_admission.json").write_text(json.dumps(result, indent=2, allow_nan=True), encoding="utf-8")
    lines = ["# P23_LITE Report", "", f"Status: `{result['status']}`", "", "This decision is valid only after all 80 held-out patients are present.", "", "```json", json.dumps(result, indent=2, allow_nan=True), "```", ""]
    (root / "P23_LITE_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--check-baseline", action="store_true")
    parser.add_argument("--expected-f1", type=float, default=.623402)
    parser.add_argument("--expected-oracle", type=float, default=.698361)
    parser.add_argument("--f1-tolerance", type=float, default=.015)
    parser.add_argument("--oracle-tolerance", type=float, default=.020)
    args = parser.parse_args()
    root = args.root; root.mkdir(parents=True, exist_ok=True)
    rows = [value for profile in PROFILES if (value := _summary(root, profile)) is not None]
    missing = [profile for profile in PROFILES if profile not in {row["profile"] for row in rows}]
    overall = _flatten_summary(rows)
    by_fold = []
    by_center = []
    for profile in overall.get("profile", pd.Series(dtype=str)).tolist():
        run_dir = _run_dir(root, profile)
        for record in _read_csv(run_dir / "p23_fold_summary.csv").to_dict("records"):
            by_fold.append({"profile": profile, **record})
        for record in _read_csv(run_dir / "p23_center_summary.csv").to_dict("records"):
            by_center.append({"profile": profile, **record})
    folds = pd.DataFrame(by_fold)
    centers = pd.DataFrame(by_center)
    adjacent = []
    order = [profile for profile in PROFILES if profile in set(overall.get("profile", []))]
    for previous, current in zip(order, order[1:]):
        before, after = overall[overall.profile == previous].iloc[0], overall[overall.profile == current].iloc[0]
        adjacent.append({"profile": current, "previous_profile": previous, **{f"delta_{metric}": _number(after.get(metric)) - _number(before.get(metric)) for metric in METRICS}})
    overall = overall.merge(pd.DataFrame(adjacent), on="profile", how="left") if adjacent else overall
    fusion = _fusion_audit(root, order)
    temporal = _temporal_audit(root, order)
    noisy = _noisy_audit(root, order)
    overall.to_csv(root / "profile_ablation_overall.csv", index=False)
    folds.to_csv(root / "profile_ablation_by_fold.csv", index=False)
    centers.to_csv(root / "profile_ablation_by_center.csv", index=False)
    fusion.to_csv(root / "fusion_delta_audit.csv", index=False)
    temporal.to_csv(root / "temporal_validity_audit.csv", index=False)
    noisy.to_csv(root / "noisy_ez_failure_audit.csv", index=False)
    baseline_ok = None
    if args.check_baseline:
        p0 = overall[overall.profile == "P0_CURRENT_P2"]
        if p0.empty or int(_number(p0.iloc[0].get("n_heldout_patients"))) != 80:
            baseline_ok = False
        else:
            baseline_ok = (
                abs(_number(p0.iloc[0].get("patient_macro_f1")) - args.expected_f1) <= args.f1_tolerance
                and abs(_number(p0.iloc[0].get("patient_oracle_macro_f1")) - args.expected_oracle) <= args.oracle_tolerance
            )
    _write_report(root, overall, missing, baseline_ok)
    _lite_report(root, overall, centers)
    print(overall.to_string(index=False) if not overall.empty else "No completed profiles found.")
    if args.check_baseline and not baseline_ok:
        raise SystemExit("P0 reproduction gate failed; refusing subsequent P23 regression profiles.")


if __name__ == "__main__":
    main()
