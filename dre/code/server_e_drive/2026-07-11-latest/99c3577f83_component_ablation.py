"""Shared protocol, evaluation, and reporting for Task 1 component ablations.

This module deliberately reuses the confirmatory threshold selector and the
patient-equal evaluator.  It only adapts each native OOF ledger to the common
NEZ-positive representation; it never creates a second metric definition.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from P2_V3_AAAI_ABLATIONS.metrics import evaluate, summaries
from P2_V3_AAAI_ABLATIONS.threshold_protocol import select_fold_threshold
from neuroez_c.p2_v3_fusion_protocol import (
    canonicalize_p2_fusion_ledger,
    canonicalize_v3_fusion_ledger,
    discover_fold_ledger,
)
from .protocol import sha256_file
from .provenance import git_commit, write_json


PRQ_PROFILES = {
    "P0_CURRENT_P2": {"display_name": "Base", "patient_relative": True, "temporal": False, "q10": False},
    "P1_TEMPORAL": {"display_name": "+ Temporal", "patient_relative": True, "temporal": True, "q10": False},
    "P2_TEMPORAL_Q10": {"display_name": "+ Q10 (PRQ-Net)", "patient_relative": True, "temporal": True, "q10": True},
    "P2_TEMPORAL_Q10_NO_PATIENT_RELATIVE": {"display_name": "w/o Patient-Relative Normalization", "patient_relative": False, "temporal": True, "q10": True, "p23_profile": "P2_TEMPORAL_Q10"},
}
BCR_PROFILES = {
    "BCR_BC_ONLY": {
        "display_name": "BCR Base (BCE only)",
        "objective": "BCE only", "boundary_loss": False, "coverage_loss": False,
    },
    "BCR_BOUNDARY_ONLY": {
        "display_name": "BCR Boundary-only",
        "objective": "BCE + Boundary", "boundary_loss": True, "coverage_loss": False,
    },
    "BCR_COVERAGE_ONLY": {
        "display_name": "BCR Coverage-only",
        "objective": "BCE + Coverage", "boundary_loss": False, "coverage_loss": True,
    },
    "BCR_BOUNDARY_COVERAGE": {
        "display_name": "BCR Boundary+Coverage",
        "objective": "BCE + Boundary + Coverage", "boundary_loss": True, "coverage_loss": True,
    },
}


def profile_definition(branch: str, profile: str) -> dict[str, Any]:
    table = PRQ_PROFILES if branch == "prq" else BCR_PROFILES
    if profile not in table:
        raise ValueError(f"Unknown {branch} ablation profile: {profile}")
    return {"branch": branch, "profile": profile, **table[profile]}


def parse_csv(value: str, allowed: Iterable[str]) -> list[str]:
    parsed = [item.strip().upper() for item in str(value).split(",") if item.strip()]
    unknown = sorted(set(parsed) - set(allowed))
    if unknown:
        raise ValueError(f"Unknown profiles: {unknown}")
    return parsed


def validate_component_inputs(*, subjects: str | Path, split_manifest: str | Path, outer_manifest: str | Path, require_n_patients: int) -> dict[str, Any]:
    subject_path, split_path, outer_path = map(Path, (subjects, split_manifest, outer_manifest))
    for path in (subject_path, split_path, outer_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    subject = pd.read_csv(subject_path)
    ids = set(subject.subject_id.astype(str))
    if len(ids) != require_n_patients:
        raise RuntimeError(f"Allowed-subject ledger has {len(ids)} patients, expected {require_n_patients}")
    split = pd.read_csv(split_path).rename(columns={"fold_idx": "outer_fold", "fold": "outer_fold", "split_role": "partition", "role": "partition"})
    outer = pd.read_csv(outer_path).rename(columns={"fold_idx": "outer_fold", "fold": "outer_fold", "split_role": "partition", "role": "partition"})
    for frame, name in ((split, "fixed split"), (outer, "outer fold")):
        if "subject_id" not in frame or "outer_fold" not in frame:
            raise ValueError(f"{name} manifest needs subject_id and outer_fold")
    if not ids.issubset(set(split.subject_id.astype(str))):
        raise RuntimeError("Fixed split manifest misses allowed patients")
    if "partition" not in split or not {"fit", "validation", "test"}.issubset(set(split.partition.astype(str).str.lower())):
        raise RuntimeError("Fixed split manifest must explicitly contain fit, validation, and test patients")
    if not ids.issubset(set(outer.subject_id.astype(str))):
        raise RuntimeError("Outer fold manifest misses allowed patients")
    test = outer
    if "partition" in test:
        test = test.loc[test.partition.astype(str).str.lower().isin({"test", "outer_test", "heldout"})]
    test_counts = test.groupby("subject_id").size()
    if set(test_counts.index) != ids or not test_counts.eq(1).all():
        raise RuntimeError("Every allowed patient must occur in exactly one outer-test fold")
    return {
        "status": "passed", "n_patients": len(ids), "n_outer_folds": int(test.outer_fold.nunique()),
        "subject_ledger_sha256": sha256_file(subject_path), "fixed_split_sha256": sha256_file(split_path),
        "outer_fold_sha256": sha256_file(outer_path),
    }


def evaluate_branch(*, branch: str, profile: str, seed: int, run_root: str | Path, output_root: str | Path, folds: Iterable[int]) -> dict[str, Any]:
    """Evaluate a retrained branch through the exact confirmatory primitives."""
    run_root, output = Path(run_root), Path(output_root)
    metric_root = output / "metrics"
    metric_root.mkdir(parents=True, exist_ok=True)
    canonical_rows, patient_rows, threshold_rows = [], [], []
    for fold in sorted(set(map(int, folds))):
        role_frames: dict[str, pd.DataFrame] = {}
        for role in ("validation", "test"):
            raw = pd.read_csv(discover_fold_ledger(run_root, fold, role))
            canonical = canonicalize_p2_fusion_ledger(raw, split_role=role) if branch == "prq" else canonicalize_v3_fusion_ledger(raw, split_role=role)
            # The canonical V3 adapter performs P(NEZ)=1-P(EZ).  Both branches
            # now have label_nez=1 and score_nez=P(NEZ).
            canonical["y_true"] = canonical.label_nez.astype(int)
            canonical["p_pos"] = canonical.score_nez.astype(float)
            role_frames[role] = canonical
        threshold, search = select_fold_threshold(role_frames["validation"], score_column="p_pos")
        search.assign(branch=branch, profile=profile, seed=int(seed), outer_fold=fold).to_csv(metric_root / f"threshold_search_fold_{fold}.csv", index=False)
        threshold_rows.append({"branch": branch, "profile": profile, "seed": int(seed), "outer_fold": fold, "threshold": threshold, "threshold_source": "validation_only", "grid_step": .005})
        test = role_frames["test"].copy()
        test["predicted_nez"] = (test.p_pos >= threshold).astype(int)
        test["predicted_ez"] = 1 - test.predicted_nez
        test["selected_threshold"] = threshold
        test["score_semantics"] = "P(NEZ)"
        test["label_semantics"] = "NEZ=1,EZ=0"
        test["true_count_used_for_prediction"] = False
        test["branch"] = branch; test["profile"] = profile; test["seed"] = int(seed)
        canonical_rows.append(test)
        patient_rows.append(evaluate(test, score_column="p_pos", threshold=threshold, experiment=f"{branch}:{profile}", analysis_status="RETRAINED_COMPONENT_ABLATION"))
    channel = pd.concat(canonical_rows, ignore_index=True)
    patients = pd.concat(patient_rows, ignore_index=True)
    overall, by_fold, by_center = summaries(patients)
    overall, by_fold, by_center = _attach_patient_accuracy(overall, by_fold, by_center, patients)
    for frame in (overall, by_fold, by_center, patients):
        frame["branch"] = branch; frame["profile"] = profile; frame["seed"] = int(seed)
    channel.to_csv(metric_root / "oof_channel_ledger.csv", index=False)
    patients.to_csv(metric_root / "patient_metrics.csv", index=False)
    by_fold.to_csv(metric_root / "fold_metrics.csv", index=False)
    by_center.to_csv(metric_root / "center_metrics.csv", index=False)
    overall.to_csv(metric_root / "overall_metrics.csv", index=False)
    pd.DataFrame(threshold_rows).to_csv(metric_root / "fold_thresholds.csv", index=False)
    return {"overall": overall.iloc[0].to_dict(), "n_patients": int(patients.subject_id.nunique()), "n_channels": int(len(channel))}


def write_completion(run_root: str | Path, *, payload: dict[str, Any]) -> None:
    write_json(Path(run_root) / "component_ablation_completion.json", {"status": "complete", **payload})


def completion_matches(path: str | Path, *, expected: dict[str, Any]) -> bool:
    """Fail closed when a resume marker belongs to another protocol run."""
    marker = Path(path)
    if not marker.is_file():
        return False
    observed = json.loads(marker.read_text(encoding="utf-8"))
    if observed.get("status") != "complete":
        raise RuntimeError(f"Incomplete completion marker: {marker}")
    mismatches = {key: {"expected": value, "actual": observed.get(key)} for key, value in expected.items() if observed.get(key) != value}
    if mismatches:
        raise RuntimeError(f"Refusing component-ablation resume with mismatched provenance: {mismatches}")
    return True


def _attach_patient_accuracy(
    overall: pd.DataFrame,
    by_fold: pd.DataFrame,
    by_center: pd.DataFrame,
    patients: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Keep paper accuracy available with older shared-metric installations."""
    if "patient_accuracy" not in patients.columns:
        raise RuntimeError("Patient metrics are missing patient_accuracy")

    def attach(target: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
        result = target.drop(columns=["patient_accuracy"], errors="ignore")
        available = [key for key in keys if key in result.columns and key in patients.columns]
        if available:
            accuracy = patients.groupby(available, sort=False, as_index=False)["patient_accuracy"].mean()
            return result.merge(accuracy, on=available, how="left", validate="many_to_one")
        result["patient_accuracy"] = float(patients["patient_accuracy"].mean())
        return result

    return (
        attach(overall, ["experiment"]),
        attach(by_fold, ["experiment", "outer_fold"]),
        attach(by_center, ["experiment", "center"]),
    )


def _repair_missing_patient_accuracy(metrics: Path) -> None:
    """Backfill an older component run without retraining it.

    Early component runs predate patient_accuracy in the shared evaluator.
    Their saved OOF masks are sufficient to recompute that metric exactly.
    """
    patient_path = metrics / "patient_metrics.csv"
    if not patient_path.is_file():
        return
    patients = pd.read_csv(patient_path)
    # Older runs may have a patched patient CSV while retaining stale aggregate
    # CSVs.  Rebuild aggregates below even when this column is already present.
    needs_accuracy = "patient_accuracy" not in patients.columns
    ledger_path = metrics / "oof_channel_ledger.csv"
    if needs_accuracy and not ledger_path.is_file():
        raise RuntimeError(f"Cannot backfill patient_accuracy without {ledger_path}")
    ledger = pd.read_csv(ledger_path) if ledger_path.is_file() else None
    if needs_accuracy:
        required = {"subject_id", "label_nez", "predicted_nez"}
        missing = sorted(required - set(ledger.columns))
        if missing:
            raise RuntimeError(f"Cannot backfill patient_accuracy; OOF ledger missing {missing}")
        accuracy = (
            ledger.assign(_correct=ledger["label_nez"].astype(int).eq(ledger["predicted_nez"].astype(int)))
            .groupby("subject_id", sort=False)["_correct"].mean()
            .rename("patient_accuracy")
            .reset_index()
        )
        patients = patients.merge(accuracy, on="subject_id", how="left", validate="one_to_one")
        if patients["patient_accuracy"].isna().any():
            raise RuntimeError("Cannot backfill patient_accuracy for every patient")
    if "n_channels" not in patients.columns:
        if ledger is None:
            raise RuntimeError(f"Cannot backfill n_channels without {ledger_path}")
        counts = ledger.groupby("subject_id", sort=False).size().rename("n_channels").reset_index()
        patients = patients.merge(counts, on="subject_id", how="left", validate="one_to_one")
        if patients["n_channels"].isna().any():
            raise RuntimeError("Cannot backfill n_channels for every patient")
    fraction_columns = ({} if ledger is None else {
        "true_ez_fraction": 1 - ledger["label_nez"].astype(int),
        "predicted_ez_fraction": 1 - ledger["predicted_nez"].astype(int),
    })
    for name, values in fraction_columns.items():
        if name in patients.columns:
            continue
        if ledger is None:
            raise RuntimeError(f"Cannot backfill {name} without {ledger_path}")
        fractions = values.groupby(ledger["subject_id"], sort=False).mean().rename(name).reset_index()
        patients = patients.merge(fractions, on="subject_id", how="left", validate="one_to_one")
        if patients[name].isna().any():
            raise RuntimeError(f"Cannot backfill {name} for every patient")
    if "patient_ez_ndcg" not in patients.columns:
        score_column = next((name for name in ("p_pos", "score_nez") if ledger is not None and name in ledger.columns), None)
        if score_column is None:
            # Legacy ledgers used by structural tests may only retain masks.
            # They cannot recover a ranking metric, but should not block the
            # primary F1/accuracy report.
            patients["patient_ez_ndcg"] = np.nan
        else:
            from sklearn.metrics import ndcg_score
            ranking_rows = []
            for subject_id, group in ledger.groupby("subject_id", sort=False):
                labels_ez = 1 - group["label_nez"].to_numpy(int)
                score_ez = 1.0 - group[score_column].to_numpy(float)
                ranking_rows.append({
                    "subject_id": subject_id,
                    "patient_ez_ndcg": float(ndcg_score(labels_ez[None, :], score_ez[None, :])) if labels_ez.sum() else np.nan,
                })
            patients = patients.merge(pd.DataFrame(ranking_rows), on="subject_id", how="left", validate="one_to_one")
    if "analysis_status" not in patients.columns:
        patients["analysis_status"] = "RETRAINED_COMPONENT_ABLATION"
    overall, by_fold, by_center = summaries(patients)
    overall, by_fold, by_center = _attach_patient_accuracy(overall, by_fold, by_center, patients)
    metadata = {key: patients[key].iloc[0] for key in ("branch", "profile", "seed") if key in patients.columns}
    for frame in (overall, by_fold, by_center):
        for key, value in metadata.items():
            frame[key] = value
    patients.to_csv(patient_path, index=False)
    overall.to_csv(metrics / "overall_metrics.csv", index=False)
    by_fold.to_csv(metrics / "fold_metrics.csv", index=False)
    by_center.to_csv(metrics / "center_metrics.csv", index=False)


def collect_reports(output_root: str | Path, *, profiles: dict[str, list[str]], seeds: list[int], repo: str | Path, audit: dict[str, Any]) -> dict[str, Any]:
    root = Path(output_root); report = root / "reports"; report.mkdir(parents=True, exist_ok=True)
    overall_rows, fold_rows, center_rows = [], [], []
    for branch, profile_names in profiles.items():
        for profile in profile_names:
            for seed in seeds:
                metrics = root / branch / profile / f"seed_{seed}" / "metrics"
                _repair_missing_patient_accuracy(metrics)
                required = [metrics / name for name in ("overall_metrics.csv", "fold_metrics.csv", "center_metrics.csv")]
                if not all(path.is_file() for path in required):
                    raise RuntimeError(f"Incomplete component-ablation run: {branch}/{profile}/seed_{seed}")
                overall_rows.append(pd.read_csv(required[0])); fold_rows.append(pd.read_csv(required[1])); center_rows.append(pd.read_csv(required[2]))
    by_seed = pd.concat(overall_rows, ignore_index=True)
    by_fold = pd.concat(fold_rows, ignore_index=True)
    by_center = pd.concat(center_rows, ignore_index=True)
    numeric = [column for column in by_seed if column.startswith("patient_") or column in {"predicted_ez_fraction", "ez_fraction_bias", "predicted_ez_fraction_mae"}]
    grouped = by_seed.groupby(["branch", "profile"], sort=False)
    means = grouped[numeric].mean().add_suffix("_mean")
    stds = grouped[numeric].std(ddof=1).add_suffix("_std")
    mean_std = means.join(stds).reset_index()
    by_seed.to_csv(report / "task1_component_ablation_by_seed.csv", index=False)
    by_fold.to_csv(report / "task1_component_ablation_by_fold.csv", index=False)
    by_center.to_csv(report / "task1_component_ablation_by_center.csv", index=False)
    mean_std.to_csv(report / "task1_component_ablation_mean_std.csv", index=False)
    audit = {**audit, "git_commit": git_commit(repo), "status": "passed", "seed_protocol": "42,52,62 model randomness only; fixed patient folds", "label_conversion": "PRQ P(NEZ) unchanged; BCR P(NEZ)=1-P(EZ)", "threshold_protocol": "fold-global validation-only patient-equal Macro-F1; step=0.005", "forbidden_information": {"test_labels_for_threshold": False, "true_k": False, "patient_threshold": False, "center_threshold": False}}
    write_json(report / "task1_component_ablation_audit.json", audit)
    display_rows = []
    for _, row in mean_std.iterrows():
        spec = profile_definition(str(row.branch), str(row.profile))
        display_rows.append({"Method": spec["display_name"], "Macro-F1": f"{row.patient_macro_f1_mean:.4f} +/- {row.patient_macro_f1_std:.4f}", "EZ-F1": f"{row.patient_ez_f1_mean:.4f} +/- {row.patient_ez_f1_std:.4f}", "NEZ-F1": f"{row.patient_nez_f1_mean:.4f} +/- {row.patient_nez_f1_std:.4f}", "Accuracy": f"{row.patient_accuracy_mean:.4f} +/- {row.patient_accuracy_std:.4f}", "AUROC": f"{row.patient_ez_auroc_mean:.4f} +/- {row.patient_ez_auroc_std:.4f}"})
    table = pd.DataFrame(display_rows)
    table.to_csv(report / "task1_component_ablation_paper_table.csv", index=False)
    report_md = "# Task 1 Component and Objective Ablation\n\nAll formal metrics are patient-equal. Labels are NEZ=1/EZ=0 and p_pos=P(NEZ).\n\n" + table.to_markdown(index=False) + "\n"
    (report / "task1_component_ablation.md").write_text(report_md, encoding="utf-8")
    latex = table.to_latex(index=False, escape=True, column_format="lrrrrr")
    (report / "task1_component_ablation.tex").write_text(latex, encoding="utf-8")

    # This is deliberately separate from the main component table because the
    # extra ranking and calibration diagnostics are BCR-objective specific.
    bcr_rows = []
    for _, row in mean_std.loc[mean_std.branch.eq("bcr")].iterrows():
        spec = profile_definition("bcr", str(row.profile))
        def value(metric: str) -> str:
            mean_key, std_key = f"{metric}_mean", f"{metric}_std"
            if mean_key not in row or std_key not in row:
                return "NA"
            return f"{row[mean_key]:.4f} +/- {row[std_key]:.4f}"
        bcr_rows.append({
            "Branch": "BCR",
            "Objective": spec["objective"],
            "Macro-F1": value("patient_macro_f1"),
            "EZ-F1": value("patient_ez_f1"),
            "Acc.": value("patient_accuracy"),
            "AUROC": value("patient_ez_auroc"),
            "EZ-AUPRC": value("patient_ez_auprc"),
            "NDCG-EZ": value("patient_ez_ndcg"),
            "EZ-Frac. Bias": value("ez_fraction_bias"),
        })
    bcr_table = pd.DataFrame(bcr_rows)
    bcr_table.to_csv(report / "bcr_objective_ablation_paper_table.csv", index=False)
    (report / "bcr_objective_ablation_paper_table.tex").write_text(
        bcr_table.to_latex(index=False, escape=True, column_format="llrrrrrrr"), encoding="utf-8"
    )
    return {"status": "passed", "n_runs": int(len(by_seed)), "report_dir": str(report)}
