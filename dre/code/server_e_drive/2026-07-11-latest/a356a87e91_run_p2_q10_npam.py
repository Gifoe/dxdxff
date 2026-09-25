from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch

PROJECT = Path(__file__).resolve().parents[2]
REPO = PROJECT.parent
for path in (REPO, PROJECT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from neuroez_c.task2.data import filtered_cache, load_cache, load_graph_cache
from neuroez_c.task2.clinical_target import build_clinical_target_lookup
from neuroez_c.task2.exclusions import load_exclusion_manifest
from neuroez_c.task2.functional_graph import NETWORK_PHASES
from neuroez_c.task2.metrics import FIXED_DECISION_THRESHOLD, FIXED_THRESHOLD_SOURCE, bootstrap_metrics, compute_metrics
from neuroez_c.task2.outcomes import load_outcome_table
from neuroez_c.task2.p2_adapter import P23Runtime, P2ExportAdapter, load_p2_args, locate_p2_config
from neuroez_c.task2.p2_q10_npam_model import P2Q10NPAMModel
from neuroez_c.task2.p2_target_outcome_model import P2TargetOutcomeModel
from neuroez_c.task2.profiles import get_profile, profile_names
from neuroez_c.task2.protocol import assert_checkpoint_safe, checkpoint_training_subjects, load_fold_manifest, manifest_hash, manifest_patient_keys
from neuroez_c.task2.training import NPAMSystem, class_pos_weight, fit_target_scalar_normalizer, make_loader, predict, prepare_examples, seed_everything, train_fixed_epochs, train_seizure_probe_fixed_epochs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fixed outer-CV P2-Q10-NPAM outcome prediction (success=1)")
    parser.add_argument("--profile", "--p2_q10_npam_profile", dest="profile", required=True, choices=profile_names())
    parser.add_argument("--outcome_table", default=os.getenv("DRE_TASK2_OUTCOME_TABLE", "cache://patient_index"))
    parser.add_argument("--feature_cache", default=os.getenv("DRE_TASK1_FEATURE_CACHE_PATH"), required=os.getenv("DRE_TASK1_FEATURE_CACHE_PATH") is None)
    parser.add_argument("--raw_cache", default=os.getenv("DRE_TASK1_RAW_CACHE_PATH"))
    parser.add_argument("--graph_cache", default=os.getenv("DRE_TASK2_GRAPH_CACHE"))
    parser.add_argument("--p2_checkpoint_root", default=os.getenv("DRE_TASK1_P2_CHECKPOINT_ROOT"), required=os.getenv("DRE_TASK1_P2_CHECKPOINT_ROOT") is None)
    parser.add_argument("--p2_runtime_root", default=os.getenv("DRE_TASK1_P2_RUNTIME_ROOT", str(REPO / "P23_TRN_NEZ_80")))
    parser.add_argument("--p2_config")
    parser.add_argument("--p2_training_manifest", default=os.getenv("DRE_TASK1_P2_TRAINING_MANIFEST"))
    parser.add_argument("--fold_manifest", default=os.getenv("DRE_TASK2_FOLD_MANIFEST"), required=os.getenv("DRE_TASK2_FOLD_MANIFEST") is None)
    parser.add_argument("--exclusion_manifest", default=os.getenv("DRE_TASK2_EXCLUSION_MANIFEST", str(PROJECT / "configs" / "data_exclusions.csv")))
    parser.add_argument("--protocol", choices=("quick", "outer_cv"), default="outer_cv")
    parser.add_argument("--outer_folds", type=int, default=5)
    parser.add_argument("--max_outer_folds", type=int, default=0)
    parser.add_argument("--max_patients", type=int, default=0)
    parser.add_argument("--stage_a_epochs", type=int, default=0, help="0 selects the fixed profile schedule")
    parser.add_argument("--stage_b_epochs", type=int, default=0, help="0 selects 10 for C5 and 15 for historical M5")
    parser.add_argument("--probe_epochs", type=int, default=0, help="0 selects 15 formally and 2 in quick screening")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap_repeats", type=int, default=2000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", default=os.getenv("DRE_TASK2_OUTPUT_DIR"), required=os.getenv("DRE_TASK2_OUTPUT_DIR") is None)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True, help="Resume completed folds and in-progress epochs (default: enabled)")
    return parser


def _balanced_take(subjects: Iterable[str], outcomes: pd.DataFrame, count: int) -> list[str]:
    values = sorted(map(str, subjects))
    if count <= 0 or len(values) <= count:
        return values
    labels = outcomes.set_index("patient_key")["outcome_label"].astype(int).to_dict()
    buckets = {label: [value for value in values if labels[value] == label] for label in (0, 1)}
    selected: list[str] = []
    while len(selected) < count and any(buckets.values()):
        for label in (0, 1):
            if buckets[label] and len(selected) < count:
                selected.append(buckets[label].pop(0))
    return selected


def _split_examples(examples: Sequence[dict[str, Any]], subjects: set[str]) -> list[dict[str, Any]]:
    return [example for example in examples if str(example["subject_id"]) in subjects]


def _system(runtime: P23Runtime, p2_args: Any, checkpoint: Path, profile: str, device: str) -> NPAMSystem:
    adapter = P2ExportAdapter(runtime, p2_args, checkpoint, device=device, limited_finetune=False)
    model = P2TargetOutcomeModel(int(p2_args.model_dim) * 2, profile, seizure_embedding_dim=int(p2_args.model_dim)) if profile.startswith("C") else P2Q10NPAMModel(int(p2_args.model_dim), profile)
    return NPAMSystem(adapter, model).to(device)


def _checkpoint_outer_fold(path: Path) -> int | None:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        return None
    for mapping in (payload, payload.get("metadata", {}), payload.get("config", {})):
        if isinstance(mapping, dict):
            for key in ("outer_fold", "fold", "fold_idx"):
                if key in mapping:
                    return int(mapping[key])
    return None


def _write_outer_cv_invalid(output: Path, args: Any, reason: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "P2_Q10_NPAM_REPORT.md").write_text(
        "# P2-Q10-NPAM Report\n\n"
        f"Profile: `{args.profile}`\n\n"
        "OUTER_CV_NOT_PAPER_VALID\n\n"
        f"Reason: {reason}\n",
        encoding="utf-8",
    )
    (output / "fold_protocol_audit.json").write_text(json.dumps({
        "protocol": "fixed_outer_cv_no_inner", "paper_valid": False,
        "inner_cv_used": False, "threshold_tuning_used": False,
        "early_stopping_used": False, "failure_reason": reason,
        "success_label": 1, "failure_label": 0,
    }, indent=2), encoding="utf-8")


def _metric_rows(predictions: pd.DataFrame, fold: int) -> dict[str, Any]:
    return {
        "outer_fold": int(fold),
        "decision_threshold": FIXED_DECISION_THRESHOLD,
        "threshold_source": FIXED_THRESHOLD_SOURCE,
        "inner_cv_used": False,
        **compute_metrics(
            predictions["outcome_true"],
            predictions["outcome_probability_success"],
            prediction_success=predictions["outcome_pred_05"],
        ),
    }


def _fold_resume_signature(args: Any, outer_fold: int, outer_train: set[str], outer_test: set[str], checkpoint: Path) -> str:
    payload = {
        "profile": args.profile, "outer_fold": int(outer_fold), "seed": int(args.seed),
        "stage_a_epochs": int(args.stage_a_epochs), "stage_b_epochs": int(args.stage_b_epochs), "probe_epochs": int(args.probe_epochs), "batch_size": int(args.batch_size),
        "protocol": args.protocol, "checkpoint": str(checkpoint.resolve()),
        "train": sorted(outer_train), "test": sorted(outer_test),
        "feature_cache": str(Path(args.feature_cache).expanduser().resolve()),
        "graph_cache": str(Path(args.graph_cache).expanduser().resolve()) if args.graph_cache else None,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _save_fold_bundle(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def _write_outputs(
    output: Path,
    args: Any,
    predictions: pd.DataFrame,
    fold_metrics: list[dict[str, Any]],
    thresholds: list[dict[str, Any]],
    schedules: list[dict[str, Any]],
    histories: list[pd.DataFrame],
    audits: dict[str, list[dict[str, Any]]],
    fold_audit: dict[str, Any],
    paper_valid: bool,
) -> None:
    predictions.to_csv(output / "oof_patient_predictions.csv", index=False)
    fold_frame = pd.DataFrame(fold_metrics)
    fold_frame.to_csv(output / "fold_metrics.csv", index=False)
    pd.DataFrame(thresholds).to_csv(output / "fold_thresholds.csv", index=False)
    pd.DataFrame(schedules).to_csv(output / "fixed_training_schedule.csv", index=False)
    pd.concat(histories, ignore_index=True).to_csv(output / "training_curves.csv", index=False)

    metric_columns = [column for column in fold_frame if column not in {"outer_fold", "decision_threshold", "threshold_source", "inner_cv_used"}]
    pooled = {"scope": "pooled_oof", **compute_metrics(predictions["outcome_true"], predictions["outcome_probability_success"], prediction_success=predictions["outcome_pred_05"])}
    mean = {"scope": "fold_mean", **{column: float(pd.to_numeric(fold_frame[column], errors="coerce").mean()) for column in metric_columns}}
    std = {"scope": "fold_std", **{column: float(pd.to_numeric(fold_frame[column], errors="coerce").std(ddof=1)) for column in metric_columns}}
    pd.DataFrame((pooled, mean, std)).to_csv(output / "summary_metrics.csv", index=False)
    center_rows = [
        {"center": center, **compute_metrics(group["outcome_true"], group["outcome_probability_success"], prediction_success=group["outcome_pred_05"])}
        for center, group in predictions.groupby("center")
    ]
    pd.DataFrame(center_rows).to_csv(output / "center_metrics.csv", index=False)

    bootstrap, skipped = bootstrap_metrics(
        predictions["outcome_true"], predictions["outcome_probability_success"],
        prediction_success=predictions["outcome_pred_05"], repeats=args.bootstrap_repeats, seed=args.seed,
    )
    ci_rows = []
    for column in ("success_auprc", "failure_auprc", "auroc", "macro_f1", "weighted_f1", "balanced_accuracy", "accuracy", "success_precision", "success_recall", "failure_precision", "failure_recall", "specificity_for_failure", "brier", "ece"):
        values = bootstrap[column].dropna().to_numpy() if column in bootstrap else np.asarray([])
        ci_rows.append({
            "metric": column, "mean": float(np.mean(values)) if values.size else np.nan,
            "ci_low": float(np.quantile(values, 0.025)) if values.size else np.nan,
            "ci_high": float(np.quantile(values, 0.975)) if values.size else np.nan,
            "valid_repeats": int(values.size), "skipped_single_class": int(skipped),
            "success_label": 1, "failure_label": 0, "threshold": 0.5,
            "threshold_source": FIXED_THRESHOLD_SOURCE, "inner_cv_used": False,
        })
    pd.DataFrame(ci_rows).to_csv(output / "bootstrap_ci.csv", index=False)

    filenames = {
        "patient": "patient_representation_audit.csv", "seizure": "seizure_representation_audit.csv",
        "channel": "channel_risk_audit.csv", "network": "network_burden_audit.csv",
        "phase": "phase_network_audit.csv", "gate": "graph_gate_audit.csv",
    }
    for key, filename in filenames.items():
        pd.DataFrame(audits.get(key, [])).to_csv(output / filename, index=False)
    pd.DataFrame(audits.get("feature", [])).to_csv(output / "patient_feature_table.csv", index=False)
    pd.DataFrame(audits.get("probe", [])).to_csv(output / "seizure_target_probe_audit.csv", index=False)
    pd.DataFrame(audits.get("q10", [])).to_csv(output / "q10_audit.csv", index=False)
    pd.DataFrame(audits.get("network", [])).to_csv(output / "network_feature_audit.csv", index=False)
    predictions.assign(error=lambda x: x["outcome_pred_05"] != x["outcome_true"]).sort_values(["error", "outcome_probability_success"], ascending=[False, True]).to_csv(output / "failure_case_analysis.csv", index=False)

    (output / "run_args.json").write_text(json.dumps(vars(args), indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    config_audit = {
        "profile": args.profile, "primary_profile": "C5_FULL", "final_ensemble_profile": "C6_FIXED_ENSEMBLE",
        "task1_label_semantics": {"EZ": 0, "NEZ": 1},
        "task2_label_semantics": {"failure": 0, "success": 1},
        "model_output": "outcome_probability_success",
        "derived_probability": "outcome_probability_failure = 1 - outcome_probability_success",
        "channel_internal_evidence": "failure-risk evidence",
        "success_label": 1, "failure_label": 0,
        "decision_threshold": 0.5, "threshold_source": FIXED_THRESHOLD_SOURCE,
        "training_schedule": "fixed_epochs", "inner_cv_used": False, "threshold_tuning_used": False,
        "early_stopping_used": False, "center_as_input": False, "coordinates_used": False,
        "soz_used": False, "resection_used": False, "true_k_used": False, "p2_backbones": 1,
        "network_phases": list(NETWORK_PHASES), "graph_source": "raw_aec_spearman" if get_profile(args.profile).network_stats else "none",
        "strict": bool(args.strict), "paper_valid": bool(paper_valid),
        "clinical_target_input": bool(get_profile(args.profile).clinical_target), "outcome_pos_weight": 1.0 if args.profile.startswith("C") else "historical_class_ratio",
    }
    (output / "config_audit.json").write_text(json.dumps(config_audit, indent=2), encoding="utf-8")
    (output / "fold_protocol_audit.json").write_text(json.dumps(fold_audit, indent=2, ensure_ascii=False), encoding="utf-8")

    checklist = [
        "success=1", "failure=0", "Engel I maps to success", "Engel II-IV map to failure",
        "single P2 backbone", "patient-wise outer folds", "fixed fold manifest", "no inner CV",
        "no validation selection", "fixed epochs", "no early stopping", "fixed threshold 0.5",
        "no calibration", "outer-train normalizer only", "P2 train/test leakage checked",
        "checkpoint fold metadata checked", "three raw-network phases", "declared exclusion applied",
        "center excluded from model", "coordinates excluded", "SOZ/resection excluded", "true-K excluded",
        "patient-equal loss weighting", "final checkpoint saved", "one OOF prediction per evaluated patient",
    ]
    report = ["# P2 Clinical-Target Outcome Report" if args.profile.startswith("C") else "# P2-Q10-NPAM Report", "", f"Profile: `{args.profile}`", f"Protocol: `{args.protocol}`", f"Patients: {len(predictions)}", "", "## Protocol checklist"]
    report.extend(f"{index}. {item}" for index, item in enumerate(checklist, 1))
    report.extend(("", "QUICK_SCREENING_NOT_PAPER_VALID" if args.protocol == "quick" else ""))
    if predictions["outer_fold"].nunique() != 5:
        report.extend(("", "FULL_OUTER_CV_NOT_RUN"))
    if not paper_valid:
        report.extend(("", "OUTER_CV_NOT_PAPER_VALID"))
    diagnostic_flags=[]
    pooled_metrics=compute_metrics(predictions["outcome_true"],predictions["outcome_probability_success"],prediction_success=predictions["outcome_pred_05"])
    if args.profile=="C0_TARGET_ONLY" and pooled_metrics["accuracy"]>=.75: diagnostic_flags.append("POTENTIAL_TARGET_LABEL_LEAKAGE")
    q10_frame=pd.DataFrame(audits.get("q10",[]))
    if not q10_frame.empty and q10_frame["old_q10_std"].mean()<.01: diagnostic_flags.append("OLD_Q10_DEGENERATE")
    if "new_q10_std" in q10_frame and q10_frame["new_q10_std"].mean()<.03 and get_profile(args.profile).cross_seizure: diagnostic_flags.append("NEW_Q10_LOW_VARIANCE")
    probe_frame=pd.DataFrame(audits.get("probe",[]))
    if get_profile(args.profile).cross_seizure and "target_auroc" in probe_frame and probe_frame["target_auroc"].mean()<.65: diagnostic_flags.append("SEIZURE_PROBE_WEAK")
    if diagnostic_flags: report.extend(("","## Diagnostic flags",*(f"- {flag}" for flag in diagnostic_flags)))
    (output / ("P2_TARGET_OUTCOME_REPORT.md" if args.profile.startswith("C") else "P2_Q10_NPAM_REPORT.md")).write_text("\n".join(line for line in report if line is not None) + "\n", encoding="utf-8")


def main() -> int:
    args = build_parser().parse_args()
    selected_profile=get_profile(args.profile)
    if args.stage_a_epochs <= 0: args.stage_a_epochs=selected_profile.outcome_epochs
    if args.stage_b_epochs <= 0: args.stage_b_epochs=10 if args.profile=="C5_FULL" else 15
    if args.probe_epochs <= 0: args.probe_epochs=2 if args.protocol=="quick" else 15
    output = Path(args.output_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    if args.protocol == "outer_cv" and args.outer_folds != 5:
        _write_outer_cv_invalid(output, args, "formal outer_cv requires exactly five folds")
        raise ValueError("Formal outer_cv protocol requires --outer_folds 5")
    if args.protocol == "outer_cv" and args.strict and not args.p2_training_manifest:
        _write_outer_cv_invalid(output, args, "P2 training manifest was not supplied")
        raise ValueError("Strict outer_cv requires --p2_training_manifest")
    seed_everything(args.seed)

    exclusions = load_exclusion_manifest(args.exclusion_manifest)
    feature = filtered_cache(load_cache(args.feature_cache), exclusions)
    outcomes, _ = load_outcome_table(args.outcome_table, cache=feature)
    outcomes = outcomes[outcomes["outcome_group"].isin(["success", "failure"])].copy()
    fixed_cohort = manifest_patient_keys(args.fold_manifest)
    outcomes = outcomes[outcomes["patient_key"].isin(fixed_cohort)].reset_index(drop=True)
    folds = load_fold_manifest(args.fold_manifest, outcomes, strict=args.strict)
    cohort = set(folds["patient_key"])
    outcomes = outcomes[outcomes["patient_key"].isin(cohort)].reset_index(drop=True)
    outcome_lookup = outcomes.set_index("patient_key")["outcome_label"].astype(int).to_dict()
    clinical_lookup, clinical_alignment, clinical_distribution = build_clinical_target_lookup(feature, sorted(cohort), strict=args.strict)
    clinical_alignment["outcome_label"]=clinical_alignment["patient_key"].map(outcome_lookup)
    clinical_alignment.to_csv(output / "clinical_target_alignment_audit.csv", index=False)
    clinical_distribution.to_csv(output / "clinical_target_distribution_audit.csv", index=False)

    profile = get_profile(args.profile)
    graph_lookup = load_graph_cache(args.graph_cache)
    if profile.network_stats and not graph_lookup:
        raise ValueError(f"{args.profile} requires --graph_cache")
    runtime = P23Runtime(args.p2_runtime_root)
    p2_args = load_p2_args(locate_p2_config(args.p2_checkpoint_root, args.p2_config), feature_cache=args.feature_cache)

    fold_values = [int(value) for value in sorted(folds["outer_fold"].unique())[: args.max_outer_folds or None]]
    prediction_frames: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    schedules: list[dict[str, Any]] = []
    histories: list[pd.DataFrame] = []
    all_audits: dict[str, list[dict[str, Any]]] = defaultdict(list)
    fold_details = []

    for outer_fold in fold_values:
        outer_test = set(folds.loc[folds["outer_fold"] == outer_fold, "patient_key"])
        outer_train = cohort - outer_test
        if args.max_patients:
            test_count = max(2, args.max_patients // 4)
            outer_test = set(_balanced_take(outer_test, outcomes, test_count))
            outer_train = set(_balanced_take(outer_train, outcomes, max(4, args.max_patients - len(outer_test))))
        if outer_train & outer_test:
            raise ValueError(f"Outer fold {outer_fold} train/test overlap")
        checkpoint = Path(args.p2_checkpoint_root).expanduser() / f"fold_{int(outer_fold)}" / "best_model.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        fold_dir = output / f"fold_{int(outer_fold)}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        fold_signature = _fold_resume_signature(args, int(outer_fold), outer_train, outer_test, checkpoint)
        fold_bundle_path = fold_dir / "fold_result.pkl"
        if args.resume and fold_bundle_path.exists():
            with fold_bundle_path.open("rb") as handle:
                bundle = pickle.load(handle)
            if bundle.get("resume_signature") == fold_signature:
                print(f"[{args.profile}] fold {outer_fold}/{len(fold_values)}: already complete, skipping", flush=True)
                prediction_frames.append(bundle["predictions"])
                metric_rows.append(bundle["metric_row"])
                threshold_rows.append(bundle["threshold_row"])
                schedules.append(bundle["schedule_row"])
                histories.extend(bundle["histories"])
                for key, values in bundle["audits"].items():
                    all_audits[key].extend(values)
                fold_details.append(bundle["fold_detail"])
                continue
            print(f"[{args.profile}] fold {outer_fold}: stale fold result ignored", flush=True)
        print(
            f"[{args.profile}] fold {outer_fold}/{len(fold_values)}: "
            f"train={len(outer_train)} test={len(outer_test)} device={args.device}",
            flush=True,
        )
        metadata_fold = _checkpoint_outer_fold(checkpoint)
        metadata_match = metadata_fold == int(outer_fold)
        if args.protocol == "outer_cv" and args.strict and not metadata_match:
            _write_outer_cv_invalid(output, args, f"P2 checkpoint fold metadata mismatch: expected {outer_fold}, found {metadata_fold}")
            raise ValueError(f"P2 checkpoint outer_fold mismatch: expected {outer_fold}, found {metadata_fold}")
        if args.protocol == "quick" and metadata_fold not in (None, 0, int(outer_fold)):
            raise ValueError(f"Quick checkpoint fold metadata is incompatible: {metadata_fold}")
        leakage_verified = False
        if args.p2_training_manifest:
            p2_train = checkpoint_training_subjects(args.p2_training_manifest, int(outer_fold))
            try:
                assert_checkpoint_safe(p2_train, outer_test)
            except ValueError as error:
                if args.protocol == "outer_cv":
                    _write_outer_cv_invalid(output, args, str(error))
                raise
            leakage_verified = True

        examples, _ = prepare_examples(runtime, feature, p2_args, normalizer_subjects=sorted(outer_train), output_subjects=sorted(outer_train | outer_test))
        train_loader = make_loader(_split_examples(examples, outer_train), runtime, outcome_lookup, graph_lookup, batch_size=args.batch_size, shuffle=True, seed=args.seed + int(outer_fold), clinical_target_lookup=clinical_lookup)
        test_loader = make_loader(_split_examples(examples, outer_test), runtime, outcome_lookup, graph_lookup, batch_size=args.batch_size, shuffle=False, seed=args.seed, clinical_target_lookup=clinical_lookup)
        system = _system(runtime, p2_args, checkpoint, args.profile, args.device)
        pos_weight = 1.0 if args.profile.startswith("C") else class_pos_weight([outcome_lookup[patient] for patient in outer_train])
        probe_history=train_seizure_probe_fixed_epochs(system,train_loader,device=args.device,epochs=args.probe_epochs,seed=args.seed+500+int(outer_fold),progress_checkpoint=fold_dir/"seizure_probe_progress.pt",resume=args.resume,resume_signature=f"{fold_signature}:probe:{args.probe_epochs}")
        fit_target_scalar_normalizer(system, train_loader, args.device)

        print(f"[{args.profile}] fold {outer_fold}: Stage A start ({args.stage_a_epochs} epochs)", flush=True)
        history_a = train_fixed_epochs(system, train_loader, device=args.device, pos_weight=pos_weight, epochs=args.stage_a_epochs, seed=args.seed + 1000 + int(outer_fold), head_lr=1e-3, progress_prefix=f"{args.profile} fold {outer_fold} Stage A", progress_checkpoint=fold_dir / "stage_a_progress.pt", resume=args.resume, resume_signature=f"{fold_signature}:stage_a")
        fold_histories = []
        if not probe_history.empty: fold_histories.append(probe_history.assign(outer_fold=int(outer_fold),stage="probe",inner_cv_used=False))
        fold_histories.append(history_a.assign(outer_fold=int(outer_fold), stage="A", inner_cv_used=False))
        histories.extend(fold_histories)
        stage_b = 0
        if profile.limited_finetune:
            system.p2_adapter.set_limited_finetune(True)
            print(f"[{args.profile}] fold {outer_fold}: Stage B start ({args.stage_b_epochs} epochs)", flush=True)
            history_b = train_fixed_epochs(system, train_loader, device=args.device, pos_weight=pos_weight, epochs=args.stage_b_epochs, seed=args.seed + 2000 + int(outer_fold), head_lr=5e-4, p2_lr=5e-5, progress_prefix=f"{args.profile} fold {outer_fold} Stage B", progress_checkpoint=fold_dir / "stage_b_progress.pt", resume=args.resume, resume_signature=f"{fold_signature}:stage_b")
            stage_b_history = history_b.assign(outer_fold=int(outer_fold), stage="B", inner_cv_used=False)
            fold_histories.append(stage_b_history)
            histories.append(stage_b_history)
            stage_b = int(args.stage_b_epochs)

        final_payload={
            "model_state_dict": system.state_dict(), "outer_fold": int(outer_fold), "profile": args.profile,
            "stage_a_epochs": int(args.stage_a_epochs), "stage_b_epochs": stage_b,
            "success_label": 1, "failure_label": 0, "decision_threshold": 0.5,
            "training_schedule": "fixed_epochs", "inner_cv_used": False,
        }
        torch.save(final_payload, fold_dir / ("final_model.pt" if args.profile.startswith("C") else "final_npam.pt"))

        print(f"[{args.profile}] fold {outer_fold}: training complete; running outer-test inference", flush=True)
        predictions, audits = predict(system, test_loader, args.device)
        for row in audits.get("feature", []): row["outer_fold"]=int(outer_fold)
        for key in ("probe","q10","network"):
            for row in audits.get(key, []): row["outer_fold"]=int(outer_fold)
        predictions["outer_fold"] = int(outer_fold)
        predictions["seed"] = int(args.seed)
        predictions["profile"] = args.profile
        predictions["protocol"] = args.protocol
        predictions["signal_type"] = "intracranial"
        predictions["success_label"] = 1
        predictions["failure_label"] = 0
        predictions["decision_threshold"] = 0.5
        predictions["threshold_source"] = FIXED_THRESHOLD_SOURCE
        predictions["inner_cv_used"] = False
        predictions["stage_a_epochs"] = int(args.stage_a_epochs)
        predictions["stage_b_epochs"] = stage_b
        predictions["p2_checkpoint_path"] = str(checkpoint.resolve())
        predictions["p2_checkpoint_leakage_verified"] = leakage_verified
        predictions["target_mask_available"] = True
        metric_row = _metric_rows(predictions, int(outer_fold))
        threshold_row = {"outer_fold": int(outer_fold), "decision_threshold": 0.5, "threshold_source": FIXED_THRESHOLD_SOURCE, "inner_cv_used": False}
        schedule_row = {"outer_fold": int(outer_fold), "profile": args.profile, "stage_a_epochs": int(args.stage_a_epochs), "stage_b_epochs": stage_b, "training_schedule": "fixed_epochs", "inner_cv_used": False}
        prediction_frames.append(predictions)
        metric_rows.append(metric_row)
        threshold_rows.append(threshold_row)
        schedules.append(schedule_row)
        for key, values in audits.items():
            all_audits[key].extend(values)
        fold_detail = {
            "outer_fold": int(outer_fold), "n_train": len(outer_train), "n_test": len(outer_test),
            "n_train_success": sum(outcome_lookup[p] == 1 for p in outer_train), "n_test_success": sum(outcome_lookup[p] == 1 for p in outer_test),
            "train_test_overlap": 0, "p2_checkpoint": str(checkpoint.resolve()),
            "p2_checkpoint_training_leakage_verified": leakage_verified,
            "checkpoint_outer_fold": metadata_fold, "checkpoint_outer_fold_matches": metadata_match,
        }
        fold_details.append(fold_detail)
        _save_fold_bundle(fold_bundle_path, {
            "resume_signature": fold_signature,
            "predictions": predictions,
            "metric_row": metric_row,
            "threshold_row": threshold_row,
            "schedule_row": schedule_row,
            "histories": fold_histories,
            "audits": audits,
            "fold_detail": fold_detail,
        })
        print(f"[{args.profile}] fold {outer_fold}: complete ({len(predictions)} OOF predictions saved in memory)", flush=True)

    predictions = pd.concat(prediction_frames, ignore_index=True)
    if predictions["patient_key"].duplicated().any():
        raise ValueError("Each patient must have exactly one OOF prediction")
    completed_five = predictions["outer_fold"].nunique() == 5 and set(predictions["patient_key"]) == cohort
    all_leakage = all(item["p2_checkpoint_training_leakage_verified"] for item in fold_details)
    all_metadata = all(item["checkpoint_outer_fold_matches"] for item in fold_details)
    paper_valid = bool(args.protocol == "outer_cv" and completed_five and args.strict and all_leakage and all_metadata)
    fold_audit = {
        "protocol": "fixed_outer_cv_no_inner", "n_outer_folds": int(predictions["outer_fold"].nunique()),
        "n_patients": int(len(predictions)), "n_success": int(predictions["outcome_true"].sum()),
        "n_failure": int((1 - predictions["outcome_true"]).sum()),
        "fold_sizes": {str(item["outer_fold"]): item["n_test"] for item in fold_details},
        "fold_success_counts": {str(item["outer_fold"]): item["n_test_success"] for item in fold_details},
        "fold_failure_counts": {str(item["outer_fold"]): item["n_test"] - item["n_test_success"] for item in fold_details},
        "train_test_overlap": {str(item["outer_fold"]): item["train_test_overlap"] for item in fold_details},
        "manifest_hash": manifest_hash(folds), "inner_cv_used": False, "threshold_tuning_used": False,
        "early_stopping_used": False, "success_label": 1, "failure_label": 0,
        "folds": fold_details, "paper_valid": paper_valid,
    }
    _write_outputs(output, args, predictions, metric_rows, threshold_rows, schedules, histories, all_audits, fold_audit, paper_valid)
    print(json.dumps({"profile": args.profile, "protocol": args.protocol, "n_predictions": len(predictions), "outer_folds": fold_values, "output_dir": str(output.resolve()), "paper_valid": paper_valid}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
