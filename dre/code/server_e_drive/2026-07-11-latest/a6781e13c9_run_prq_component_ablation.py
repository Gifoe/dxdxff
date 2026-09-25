#!/usr/bin/env python3
"""Retrain PRQ-Net P0/P1/P2 and no-patient-relative component ablations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from task1_confirmatory.component_ablation import PRQ_PROFILES, completion_matches, evaluate_branch, parse_csv, profile_definition, validate_component_inputs, write_completion
from task1_confirmatory.provenance import canonical_json_sha256, file_sha256, write_branch_provenance, write_json


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--base-run-args", required=True); p.add_argument("--window-cache-path", required=True)
    p.add_argument("--allowed-subjects-ledger", required=True); p.add_argument("--fixed-split-manifest", required=True); p.add_argument("--outer-fold-manifest", required=True)
    p.add_argument("--output-root", required=True); p.add_argument("--seeds", default="42,52,62"); p.add_argument("--profiles", default=",".join(PRQ_PROFILES))
    p.add_argument("--max-outer-folds", type=int, default=0); p.add_argument("--dry-run", action="store_true"); p.add_argument("--skip-existing", action="store_true"); p.add_argument("--python", default=sys.executable)
    p.add_argument("--native-retries", type=int, default=2, help="Retry only Windows native access-violation exits; P23 resumes completed outer folds.")
    return p


def _run_with_native_resume(command: list[str], *, retries: int) -> None:
    """Retry a driver-level Windows crash without changing the experiment state."""
    if retries < 0:
        raise ValueError("native-retries must be non-negative")
    # Windows may surface a CUDA/native abort as either the access-violation
    # code or a generic unsigned/signed -1. Both are safe to retry because
    # P23 stores completed outer folds before the next fold begins.
    recoverable_native_exit_codes = {
        3221225477,  # 0xC0000005 unsigned
        -1073741519, # 0xC0000005 signed
        4294967295,  # 0xFFFFFFFF unsigned (-1)
        -1,
    }
    for attempt in range(retries + 1):
        try:
            subprocess.run(command, cwd=REPO, check=True)
            return
        except subprocess.CalledProcessError as error:
            if error.returncode not in recoverable_native_exit_codes or attempt >= retries:
                raise
            print(
                f"[PRQ-Ablation] Native CUDA/driver access violation; resuming the same output "
                f"({attempt + 1}/{retries}).",
                flush=True,
            )
            time.sleep(10)
    raise AssertionError("unreachable")


def main() -> None:
    args = parser().parse_args(); root = Path(args.output_root); base_path = Path(args.base_run_args)
    if not base_path.is_file(): raise FileNotFoundError(base_path)
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]; profiles = parse_csv(args.profiles, PRQ_PROFILES)
    audit = validate_component_inputs(subjects=args.allowed_subjects_ledger, split_manifest=args.fixed_split_manifest, outer_manifest=args.outer_fold_manifest, require_n_patients=80)
    write_json(root / "audit" / "prq_preflight.json", {**audit, "base_args_sha256": file_sha256(base_path), "profiles": [profile_definition("prq", p) for p in profiles]})
    base = json.loads(base_path.read_text(encoding="utf-8")); entry = REPO / "P23_TRN_NEZ_80" / "run_neuroez_c.py"; folds = range(1, (args.max_outer_folds or 5) + 1)
    for profile in profiles:
        spec = profile_definition("prq", profile); native_profile = spec.get("p23_profile", profile)
        for seed in seeds:
            run = root / "prq" / profile / f"seed_{seed}"; marker = run / "component_ablation_completion.json"
            resume_contract = {"branch": "prq", "profile": profile, "seed": seed, "base_args_sha256": file_sha256(base_path), "cohort_sha256": audit["subject_ledger_sha256"], "partition_sha256": audit["fixed_split_sha256"], "feature_cache_sha256": file_sha256(args.window_cache_path)}
            if args.skip_existing and completion_matches(marker, expected=resume_contract): continue
            effective = {**base, "output_dir": str(run), "config_name": f"PRQ_{profile}_seed{seed}", "window_cache_path": args.window_cache_path, "allowed_subjects_ledger": args.allowed_subjects_ledger, "fixed_split_manifest": args.fixed_split_manifest, "require_n_patients": 80, "cohort_mode": "sensitivity80", "positive_label": "nez", "score_semantics": "nez_probability", "use_p23_trn_nez": True, "p23_profile": native_profile, "p23_direct_outer_only": True, "p23_regression_protocol": True, "p23_use_p2_loss": True, "use_patient_relative_z": bool(spec["patient_relative"]), "random_seed": seed, "model_seed": seed, "outer_split_seed": 42, "inner_split_seed": 42, "split_strategy": "5fold", "n_splits": 5, "max_outer_folds": int(args.max_outer_folds), "use_cane_path_cp_nez": False, "use_n6_dual_view_ema": False, "use_two_expert_router": False, "use_feature_separated_two_expert": False, "use_a9v8_lcbo": False, "use_broad_ez_mil_loss": False, "use_diffusion_residual": False, "use_view_gated_fusion": False, "use_hard_topk_loss": False, "use_negative_anchor_head": False, "use_ez_ranking_loss": False}
            run.mkdir(parents=True, exist_ok=True); write_json(run / "effective_run_args.json", effective)
            # Reuse the parser-aware launcher so BooleanOptionalAction flags are
            # rendered correctly. Only the two predeclared component switches are
            # admitted beyond the frozen P2 reference contract.
            command = [args.python, str(REPO / "scripts" / "task1_confirmatory" / "launch_p2.py"), "--entrypoint", str(entry), "--base-args", str(base_path), "--overrides-json", str(run / "effective_run_args.json"), "--command-output", str(run / "command.json"), "--argument-coverage-output", str(run / "argument_coverage.json"), "--reference-effective-config-output", str(run / "reference_effective_config.json"), "--effective-config-output", str(run / "effective_config.json"), "--effective-config-diff-output", str(run / "effective_config_diff.json"), "--experiment-type", "component_ablation", "--allow-component-ablation"]
            if args.dry_run: command.append("--dry-run")
            write_json(run / "protocol_audit.json", {**audit, "profile": spec, "score_semantics": "P(NEZ)", "label_semantics": "NEZ=1,EZ=0", "threshold_protocol": "validation-only"})
            if args.dry_run: print(json.dumps({"profile": profile, "seed": seed, "command": command})); continue
            _run_with_native_resume(command, retries=int(args.native_retries))
            result = evaluate_branch(branch="prq", profile=profile, seed=seed, run_root=run, output_root=run, folds=folds)
            hashes = {"effective_config_sha256": canonical_json_sha256(json.loads((run / "effective_config.json").read_text(encoding="utf-8"))), "cohort_sha256": audit["subject_ledger_sha256"], "partition_sha256": audit["fixed_split_sha256"], "feature_cache_sha256": file_sha256(args.window_cache_path)}
            write_branch_provenance(root=run, method_id="PRQ_COMPONENT_ABLATION", profile_id=profile, seed=seed, repo=REPO, folds=list(folds), **hashes)
            write_completion(run, payload={"branch": "prq", "profile": profile, "seed": seed, "result": result, "base_args_sha256": file_sha256(base_path), **hashes})


if __name__ == "__main__": main()
