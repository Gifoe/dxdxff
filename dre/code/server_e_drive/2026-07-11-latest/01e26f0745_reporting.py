from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .utils import write_json
from .evaluation import evaluate_patient_predictions, paired_patient_delta


ACTION_COLUMNS = ["subject_id", "outer_fold", "variant", "eject_channel", "add_channel", "p_benefit", "p_harm", "pred_delta", "utility", "gate_pass", "action_order"]
PATIENT_DELTA_COLUMNS = ["subject_id", "outer_fold", "center", "anchor_patient_macro_f1",
                         "candidate_patient_macro_f1", "delta_patient_macro_f1",
                         "anchor_patient_ez_f1", "candidate_patient_ez_f1", "delta_patient_ez_f1",
                         "anchor_patient_nez_f1", "candidate_patient_nez_f1", "delta_patient_nez_f1",
                         "n_swaps", "status"]


def build_patient_delta_table(anchor: pd.DataFrame, candidate: pd.DataFrame, actions: pd.DataFrame) -> pd.DataFrame:
    left = anchor[["subject_id", "outer_fold", "center", "patient_macro_f1", "patient_ez_f1", "patient_nez_f1"]]
    right = candidate[["subject_id", "patient_macro_f1", "patient_ez_f1", "patient_nez_f1"]]
    joined = left.merge(right, on="subject_id", suffixes=("_anchor", "_candidate"), validate="one_to_one")
    output = pd.DataFrame({"subject_id": joined["subject_id"], "outer_fold": joined["outer_fold"], "center": joined["center"],
                           "anchor_patient_macro_f1": joined["patient_macro_f1_anchor"], "candidate_patient_macro_f1": joined["patient_macro_f1_candidate"],
                           "anchor_patient_ez_f1": joined["patient_ez_f1_anchor"], "candidate_patient_ez_f1": joined["patient_ez_f1_candidate"],
                           "anchor_patient_nez_f1": joined["patient_nez_f1_anchor"], "candidate_patient_nez_f1": joined["patient_nez_f1_candidate"]})
    for metric in ("patient_macro_f1", "patient_ez_f1", "patient_nez_f1"):
        output[f"delta_{metric}"] = output[f"candidate_{metric}"] - output[f"anchor_{metric}"]
    counts = actions.groupby("subject_id").size() if not actions.empty and "subject_id" in actions else pd.Series(dtype=int)
    output["n_swaps"] = output["subject_id"].map(counts).fillna(0).astype(int)
    output["status"] = np.select([output["delta_patient_macro_f1"] > 0, output["delta_patient_macro_f1"] < 0], ["improved", "harmed"], default="unchanged")
    return output[PATIENT_DELTA_COLUMNS]


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty: return "(none)"
    columns = list(frame.columns)
    lines = ["| " + " | ".join(map(str, columns)) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in frame.itertuples(index=False, name=None))
    return "\n".join(lines)


def write_variant_artifacts(root: str | Path, *, config: dict[str, Any], final_ledger: pd.DataFrame, actions: pd.DataFrame, metrics: dict[str, Any], inner_selection: dict[str, Any], manifest: dict[str, Any], pair_oof: pd.DataFrame | None = None, test_candidates: pd.DataFrame | None = None, training_log: pd.DataFrame | None = None) -> None:
    output = Path(root)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", config)
    write_json(output / "inner_selection_by_fold.json", inner_selection)
    write_json(output / "metrics_summary.json", {key: value for key, value in metrics.items() if key != "patient_rows"})
    write_json(output / "model_manifest.json", manifest)
    final_ledger.to_csv(output / "final_channel_ledger.csv", index=False)
    action_output = actions.copy()
    for column in ACTION_COLUMNS:
        if column not in action_output:
            action_output[column] = pd.Series(dtype=float if column in {"p_benefit", "p_harm", "pred_delta", "utility"} else object)
    action_output.to_csv(output / "repair_actions.csv", index=False)
    final_ledger[["subject_id", "outer_fold"]].drop_duplicates().to_csv(output / "fold_assignments.csv", index=False)
    (pair_oof if pair_oof is not None else pd.DataFrame()).to_csv(output / "pair_oof_predictions.csv", index=False)
    (test_candidates if test_candidates is not None else pd.DataFrame()).to_csv(output / "test_candidate_predictions.csv", index=False)
    (training_log if training_log is not None else pd.DataFrame()).to_csv(output / "training_log.csv", index=False)
    patient_rows = metrics.get("patient_rows", pd.DataFrame())
    if isinstance(patient_rows, pd.DataFrame):
        anchor_rows = evaluate_patient_predictions(final_ledger, "old_v3_selected")["patient_rows"]
        delta = build_patient_delta_table(anchor_rows, patient_rows, actions)
        delta.to_csv(output / "patient_delta.csv", index=False)
        delta.groupby("outer_fold", dropna=False).mean(numeric_only=True).reset_index().to_csv(output / "a12_metrics_by_fold.csv", index=False)
        delta.groupby("center", dropna=False).mean(numeric_only=True).reset_index().to_csv(output / "a12_metrics_by_center.csv", index=False)
    write_json(output / "completion.json", {"status": "success", "config_hash": manifest.get("config_hash"), "fingerprint": manifest.get("fingerprint"), "variant": manifest.get("variant")})


def patient_bootstrap(anchor: pd.DataFrame, candidate: pd.DataFrame, *, n_bootstrap: int = 1000, seed: int = 42) -> dict[str, float]:
    joined = anchor[["subject_id", "patient_macro_f1"]].merge(candidate[["subject_id", "patient_macro_f1"]], on="subject_id", suffixes=("_anchor", "_candidate"), validate="one_to_one")
    if joined.empty:
        return {"mean_delta": 0.0, "ci_low": 0.0, "ci_high": 0.0, "probability_delta_positive": 0.0, "n_patients": 0, "n_improved": 0, "n_harmed": 0, "n_unchanged": 0}
    deltas = (joined["patient_macro_f1_candidate"] - joined["patient_macro_f1_anchor"]).to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    draws = np.array([rng.choice(deltas, size=len(deltas), replace=True).mean() for _ in range(n_bootstrap)])
    return {"mean_delta": float(deltas.mean()), "ci_low": float(np.quantile(draws, 0.025)), "ci_high": float(np.quantile(draws, 0.975)), "probability_delta_positive": float(np.mean(draws > 0)), "n_patients": int(len(deltas)), "n_improved": int((deltas > 0).sum()), "n_harmed": int((deltas < 0).sum()), "n_unchanged": int((deltas == 0).sum())}


def write_comparison(root: str | Path, variant_results: dict[str, dict[str, Any]], *, run_classification: str = "synthetic") -> dict[str, Any]:
    comparison_dir = Path(root) / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    anchor_rows = variant_results.get("A12-V0", {}).get("metrics", {}).get("patient_rows", pd.DataFrame())
    rows, ci_rows = [], []
    for variant, result in variant_results.items():
        metrics = result.get("metrics", {})
        row = {"variant": variant, "status": result.get("status"), "patient_macro_f1": metrics.get("patient_macro_f1"), "patient_macro_ez_f1": metrics.get("patient_macro_ez_f1"), "action_coverage": result.get("action_coverage", 0.0)}
        rows.append(row)
        candidate = metrics.get("patient_rows")
        if variant != "A12-V0" and isinstance(anchor_rows, pd.DataFrame) and isinstance(candidate, pd.DataFrame) and {"subject_id", "patient_macro_f1"}.issubset(anchor_rows.columns) and {"subject_id", "patient_macro_f1"}.issubset(candidate.columns):
            ci_rows.append({"variant": variant, **patient_bootstrap(anchor_rows, candidate)})
    table = pd.DataFrame(rows)
    ci = pd.DataFrame(ci_rows)
    table.to_csv(comparison_dir / "a12_variant_comparison.csv", index=False)
    ci.to_csv(comparison_dir / "a12_bootstrap_ci.csv", index=False)
    table.to_csv(comparison_dir / "a12_variant_summary.csv", index=False)
    write_json(comparison_dir / "a12_variant_comparison.json", {"variants": rows, "bootstrap": ci_rows})
    write_json(comparison_dir / "best_variant_posthoc.json", {"selection_is_posthoc": True, "do_not_reuse_outer_oof_for_further_tuning": True})
    action_rows = [{"variant": variant, "n_actions": int(len(result.get("actions", pd.DataFrame()))), "n_patients_with_actions": int(result.get("actions", pd.DataFrame()).get("subject_id", pd.Series(dtype=str)).nunique())} for variant, result in variant_results.items()]
    pd.DataFrame(action_rows).to_csv(comparison_dir / "a12_action_summary.csv", index=False)
    patient_parts = []
    fold_parts, center_parts = [], []
    for variant, result in variant_results.items():
        patient = result.get("metrics", {}).get("patient_rows")
        if isinstance(patient, pd.DataFrame) and not patient.empty:
            patient_parts.append(patient[["subject_id", "patient_macro_f1"]].rename(columns={"patient_macro_f1": variant}).set_index("subject_id"))
            fold = patient.groupby("outer_fold", as_index=False).mean(numeric_only=True); fold.insert(0, "variant", variant); fold_parts.append(fold)
            center = patient.groupby("center", as_index=False).mean(numeric_only=True); center.insert(0, "variant", variant); center_parts.append(center)
    matrix = pd.concat(patient_parts, axis=1).reset_index() if patient_parts else pd.DataFrame()
    matrix.to_csv(comparison_dir / "a12_patient_delta_matrix.csv", index=False)
    (pd.concat(fold_parts, ignore_index=True) if fold_parts else pd.DataFrame()).to_csv(comparison_dir / "a12_metrics_by_fold.csv", index=False)
    (pd.concat(center_parts, ignore_index=True) if center_parts else pd.DataFrame()).to_csv(comparison_dir / "a12_metrics_by_center.csv", index=False)
    lines = ["# A12-VCSN comparison", "", f"Run classification: **{run_classification}**.", "", "This comparison is post-hoc and must not be reused for tuning.", "", "## Variant status", "", _markdown_table(table), "", "## Interpretation", ""]
    successful = table[table["status"] == "success"]
    above = successful[successful["patient_macro_f1"].fillna(-np.inf) > .70]
    lines.append("No real full five-fold OOF claim is made by this report; synthetic or partial executions are explicitly non-clinical." if above.empty else "At least one executed variant exceeds 0.70 in this output; verify full five-fold OOF provenance before any claim.")
    failed = table[table["status"].isin(["failed", "skipped"])]
    if not failed.empty: lines.extend(["", "## Skipped or failed", "", _markdown_table(failed[["variant", "status"]])])
    (comparison_dir / "final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"comparison_dir": str(comparison_dir), "rows": rows, "bootstrap": ci_rows}
