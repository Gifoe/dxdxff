#!/usr/bin/env python3
"""Run independently trained BCR-Net profiles without test-label selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from neuroez_c.v3_qbc_profiles import get_v3_qbc_profile
from neuroez_c.v3_qbc_protocol import file_sha256, read_allowed_subjects, validate_v3_qbc_protocol
from neuroez_c.v3_qbc_reporting import build_v3_qbc_reports


PROFILES = [
    "BCR_BC_ONLY",
    "BCR_BOUNDARY_ONLY",
    "BCR_COVERAGE_ONLY",
    "BCR_BOUNDARY_COVERAGE",
]
RETIRED_BCR_CONFIG_KEYS = {"model_family", "score_semantics"}
RETIRED_BCR_CONFIG_PREFIXES = ("v3_qbc_q10_",)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-run-args",
        default=str(REPO / "b0_m1_A9v3_search_20260624" / "A9v3_s5_anchor_w002_rank005_m005_all90_posEZ" / "run_args_b0_pruned.json"),
    )
    parser.add_argument("--window-cache-path", "--window_cache_path", dest="window_cache_path", required=True)
    parser.add_argument("--allowed-subjects-ledger", "--allowed_subjects_ledger", dest="allowed_subjects_ledger", required=True)
    parser.add_argument("--outer-fold-manifest", "--fixed_fold_manifest", dest="outer_fold_manifest", required=True)
    parser.add_argument("--fixed-split-manifest", "--fixed_split_manifest", dest="fixed_split_manifest", default="")
    parser.add_argument("--loco-mode", "--loco_mode", dest="loco_mode", action="store_true")
    parser.add_argument("--selected-outer-fold", "--selected_outer_fold", dest="selected_outer_fold", type=int, default=0)
    parser.add_argument("--output-root", dest="output_root", default="")
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", default="")
    parser.add_argument("--v3_qbc_profile", dest="single_profile", choices=PROFILES)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--profiles", default=",".join(PROFILES))
    parser.add_argument("--require-n-patients", "--require_n_patients", dest="require_n_patients", type=int, default=80)
    parser.add_argument("--seed", "--random_seed", dest="seed", type=int, default=42)
    parser.add_argument("--max-outer-folds", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--epochs", type=int, default=0)
    # Keep the final BCR training budget explicit.  The base run arguments
    # remain the source of truth unless a protocol runner supplies overrides.
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--min-epochs-before-early-stop", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--patient-batch-size", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=-1)
    parser.add_argument("--device", default="")
    parser.add_argument("--bcr-checkpoint", default="", help="Optional checkpoint inspection only; final BCR runs retrain from scratch.")
    parser.add_argument("--allow-bcr-non-strict-debug", action="store_true")
    return parser


def _flag(name: str) -> str:
    aliases = {
        "allowed_subjects_ledger": "allowed-subjects-ledger",
        "allowed_subjects_file": "allowed-subjects-file",
        "require_n_patients": "require-n-patients",
    }
    return "--" + aliases.get(name, name)


def _command(python: str, values: dict[str, Any]) -> list[str]:
    command = [python, str(REPO / "run_neuroez_c.py")]
    value_boole = {"drop_high_ez_fraction_lzu", "use_patient_relative_z"}
    store_true_only = {"allow_partial_fixed_test_manifest"}
    for key in sorted(values):
        value = values[key]
        if value is None or value == "":
            continue
        name = _flag(key)
        if isinstance(value, bool):
            if key in value_boole:
                command.extend([name, str(value).lower()])
            elif key in store_true_only:
                if value:
                    command.append(name)
            else:
                command.append(name if value else "--no-" + key)
        else:
            command.extend([name, str(value)])
    return command


def _sanitize_effective_config(values: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Remove historical fields that are invalid for the final Q10-free BCR."""
    effective = dict(values)
    removed: dict[str, Any] = {}
    for key in list(effective):
        if key in RETIRED_BCR_CONFIG_KEYS or key.startswith(RETIRED_BCR_CONFIG_PREFIXES):
            removed[key] = effective.pop(key)
    return effective, removed


def _write_standardized_artifacts(output: Path, *, profile: Any) -> None:
    """Expose a stable per-fold artifact contract without changing checkpoints."""
    thresholds = pd.read_csv(output / "fold_thresholds.csv")
    for row in thresholds.itertuples(index=False):
        fold_dir = output / f"fold_{int(row.outer_fold)}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        native_checkpoint = fold_dir / "best_bcr_boundary_coverage.pt"
        if not native_checkpoint.is_file():
            raise FileNotFoundError(
                f"BCR fold {row.outer_fold} is missing its selected checkpoint: {native_checkpoint}"
            )
        shutil.copy2(native_checkpoint, fold_dir / "best_model.pt")
        history = output / "training_loss_components.csv"
        if not history.is_file():
            raise FileNotFoundError(f"BCR run is missing its training history: {history}")
        shutil.copy2(history, fold_dir / "training_history.csv")
        (fold_dir / "selected_threshold.json").write_text(
            json.dumps({
                "outer_fold": int(row.outer_fold),
                "selected_threshold": float(row.threshold),
                "threshold_source": str(row.threshold_source),
                "simple_q10": False,
                "boundary_loss": bool(profile.use_boundary),
                "coverage_loss": bool(profile.use_coverage),
            }, indent=2),
            encoding="utf-8",
        )


def main() -> None:
    args = _parser().parse_args()
    if args.bcr_checkpoint:
        # This inspection is optional. Final BCR-Net runs retrain from
        # scratch, so a missing inspection helper must not block training.
        from neuroez_c.bcr_checkpoint import inspect_bcr_checkpoint_path

        report = inspect_bcr_checkpoint_path(args.bcr_checkpoint)
        if report["legacy_q10_keys"] and not args.allow_bcr_non_strict_debug:
            raise RuntimeError(
                "Legacy BCR-Q10 checkpoint is incompatible with final BCR-Net; retrain it. "
                f"keys={report['legacy_q10_keys']}"
            )
        if report["legacy_q10_keys"]:
            print("[BCR-Net][DEBUG] non-strict checkpoint inspection: " + json.dumps(report), flush=True)
    base_path = Path(args.base_run_args)
    base = json.loads(base_path.read_text(encoding="utf-8"))
    profiles = [args.single_profile] if args.single_profile else [
        item.strip().upper() for item in args.profiles.split(",") if item.strip()
    ]
    unknown = sorted(set(profiles) - set(PROFILES))
    if unknown:
        raise ValueError(f"Unknown profiles: {unknown}")
    allowed = read_allowed_subjects(args.allowed_subjects_ledger)
    protocol = validate_v3_qbc_protocol(
        patient_ids=allowed,
        allowed_subjects_path=args.allowed_subjects_ledger,
        outer_fold_manifest_path=args.outer_fold_manifest,
        require_n_patients=args.require_n_patients,
        seed=args.seed,
        fixed_split_manifest_path=args.fixed_split_manifest or None,
        allow_partial_test_manifest=args.loco_mode,
    )
    if not args.output_root and not args.output_dir:
        raise ValueError("Provide --output_dir for one profile or --output-root for a profile set")
    root = Path(args.output_root or args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    for profile_name in profiles:
        profile = get_v3_qbc_profile(profile_name)
        output = Path(args.output_dir) if args.output_dir and len(profiles) == 1 else root / profile_name
        output.mkdir(parents=True, exist_ok=True)
        old_args = output / "run_args.json"
        if old_args.is_file():
            old_profile = json.loads(old_args.read_text(encoding="utf-8")).get("v3_qbc_profile", "")
            if old_profile and old_profile not in PROFILES:
                raise RuntimeError(
                    f"Existing output contains retired BCR-Q10 profile {old_profile!r}; "
                    "use a new output directory and retrain final BCR-Net."
                )
        effective = dict(base)
        effective.update({
            "config_name": f"{profile_name}_seed{args.seed}",
            "output_dir": str(output),
            "window_cache_path": args.window_cache_path,
            "allowed_subjects_ledger": args.allowed_subjects_ledger,
            "require_n_patients": args.require_n_patients,
            "v3_qbc_outer_fold_manifest": args.outer_fold_manifest,
            "fixed_split_manifest": args.fixed_split_manifest,
            "selected_outer_fold": args.selected_outer_fold,
            "allow_partial_fixed_test_manifest": bool(args.loco_mode),
            "random_seed": args.seed,
            "split_strategy": "5fold",
            "n_splits": 5,
            "positive_label": "ez",
            "drop_high_ez_fraction_lzu": False,
            "use_v3_qbc": True,
            "v3_qbc_profile": profile_name,
            "v3_qbc_boundary_weight": 0.05,
            "v3_qbc_coverage_weight": 0.08,
            "use_physics_dynamics": True,
            "temporal_pooling": "mean",
            "record_pooling": "mean",
            "loss_mode": "patient_balanced_bce",
            "patient_loss_weighting": "uniform",
            "use_negative_anchor_head": True,
            "negative_anchor_loss_weight": 0.02,
            "use_ez_ranking_loss": False,
            "ez_ranking_loss_weight": 0.0,
            "use_n6_dual_view_ema": False,
            "use_two_expert_router": False,
            "use_feature_separated_two_expert": False,
            "use_a9v8_lcbo": False,
            "use_broad_ez_mil_loss": False,
            "use_diffusion_residual": False,
            "use_view_gated_fusion": False,
            "use_hard_topk_loss": False,
            "group_robust_mode": "none",
            "use_teacher_anchor_eval": False,
            "teacher_anchor_apply_to_train_loss": False,
            "use_patient_context_reranker": False,
            "use_multi_seizure_consistency": False,
            "use_shaft_local_residual": False,
            "use_clinical_mixture_head": False,
            "pretrain_masked_windows": False,
            "max_outer_folds": args.max_outer_folds,
            "dry_run_config_only": args.dry_run,
        })
        if args.epochs > 0:
            effective["epochs"] = int(args.epochs)
        if args.patience > 0:
            effective["patience"] = int(args.patience)
        if args.min_epochs_before_early_stop > 0:
            effective["min_epochs_before_early_stop"] = int(args.min_epochs_before_early_stop)
        if args.batch_size > 0:
            effective["batch_size"] = int(args.batch_size)
        if args.patient_batch_size > 0:
            effective["patient_batch_size"] = int(args.patient_batch_size)
        if args.num_workers >= 0:
            effective["num_workers"] = int(args.num_workers)
        if args.device:
            effective["device"] = str(args.device)
        effective, removed_legacy_config = _sanitize_effective_config(effective)
        audit = {
            **protocol,
            "profile": profile_name,
            "enabled_modules": {
                "bcr_backbone": True,
                "negative_anchor": True,
                "boundary_loss": profile.use_boundary,
                "coverage_loss": profile.use_coverage,
            },
            "objective": {
                "bce_loss": True,
                "simple_q10": False,
                "boundary_loss": profile.use_boundary,
                "coverage_loss": profile.use_coverage,
                "boundary_weight": 0.05,
                "coverage_weight": 0.08,
            },
            "disabled_modules": [
                "HNC", "raw", "P2", "P23", "SCOPE", "LCBO", "two_expert",
                "broad_ez_mil", "diffusion", "view_gated_fusion", "group_DRO",
            ],
            "feature_cache_path": args.window_cache_path,
            "feature_cache_hash": file_sha256(args.window_cache_path) if Path(args.window_cache_path).is_file() else None,
            "base_run_args_path": str(base_path),
            "base_run_args_hash": file_sha256(base_path),
            "training_budget": {
                "epochs": effective.get("epochs"),
                "patience": effective.get("patience"),
                "min_epochs_before_early_stop": effective.get("min_epochs_before_early_stop"),
                "batch_size": effective.get("batch_size"),
                "patient_batch_size": effective.get("patient_batch_size"),
                "num_workers": effective.get("num_workers"),
                "device": effective.get("device"),
            },
            "normalizer_protocol": "fit-fold-only",
            "removed_legacy_config": removed_legacy_config,
        }
        (output / "bcr_config_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
        (output / "bcr_protocol_audit.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
        (output / "run_args.json").write_text(json.dumps(effective, indent=2, sort_keys=True), encoding="utf-8")
        (output / "resolved_config.json").write_text(json.dumps({
            **effective,
            "simple_q10": False,
            "boundary_loss": profile.use_boundary,
            "coverage_loss": profile.use_coverage,
            "boundary_weight": 0.05,
            "coverage_weight": 0.08,
        }, indent=2, sort_keys=True), encoding="utf-8")
        command = _command(args.python, effective)
        retired_arguments = [
            argument for argument in command
            if argument.startswith("--v3_qbc_q10_")
        ]
        if retired_arguments:
            raise RuntimeError(
                f"Final BCR command contains retired Q10 arguments: {retired_arguments}"
            )
        (output / "command.json").write_text(json.dumps(command, indent=2), encoding="utf-8")
        complete = (output / "formal_summary.csv").is_file() and (output / "truek_summary.csv").is_file()
        if args.skip_existing and complete:
            _write_standardized_artifacts(output, profile=profile)
            print(f"[BCR-Net] skip complete profile {profile_name}", flush=True)
            continue
        expected_folds = int(args.max_outer_folds or 5)
        saved_test = list(output.glob("test_channel_predictions_neuroez_v2_fold_*.csv"))
        saved_val = list(output.glob("val_channel_predictions_neuroez_v2_fold_*.csv"))
        predictions_complete = len(saved_test) == expected_folds and len(saved_val) == expected_folds
        if args.skip_existing and predictions_complete and not args.dry_run:
            result = build_v3_qbc_reports(output)
            _write_standardized_artifacts(output, profile=profile)
            print(
                f"[BCR-Net] recovered complete saved predictions for {profile_name}: "
                f"{json.dumps(result, sort_keys=True)}",
                flush=True,
            )
            continue
        try:
            subprocess.run(command, cwd=REPO, check=True)
        except subprocess.CalledProcessError as error:
            # Preserve native Windows/CUDA exit codes for the outer resumable
            # component runner instead of converting every failure to code 1.
            raise SystemExit(error.returncode) from error
        # A partial fold smoke run intentionally has no five-fold summary yet.
        # Its raw validation/test ledgers are consumed by the common evaluator.
        if not args.dry_run and int(args.max_outer_folds) in {0, 5}:
            result = build_v3_qbc_reports(output)
            _write_standardized_artifacts(output, profile=profile)
            print(f"[V3-QBC] {profile_name}: {json.dumps(result, sort_keys=True)}", flush=True)


if __name__ == "__main__":
    main()
