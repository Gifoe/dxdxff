#!/usr/bin/env python3
"""Formal retrained Task 1 PRQ/BCR component and objective ablations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from task1_confirmatory.component_ablation import BCR_PROFILES, PRQ_PROFILES, collect_reports, completion_matches, evaluate_branch, parse_csv, profile_definition, validate_component_inputs, write_completion
from task1_confirmatory.provenance import canonical_json_sha256, file_sha256, write_branch_provenance, write_json


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--p2-base-run-args", required=True); p.add_argument("--v3-base-run-args", required=True)
    p.add_argument("--window-cache-path", required=True); p.add_argument("--allowed-subjects-ledger", required=True)
    p.add_argument("--fixed-split-manifest", required=True); p.add_argument("--outer-fold-manifest", required=True)
    p.add_argument("--confirmatory-reference-root", default="")
    p.add_argument("--output-root", required=True); p.add_argument("--seeds", default="42,52,62")
    p.add_argument("--prq-profiles", default=",".join(PRQ_PROFILES)); p.add_argument("--bcr-profiles", default=",".join(BCR_PROFILES))
    p.add_argument("--skip-prq", action="store_true", help="Reuse existing PRQ outputs; run only the requested BCR profiles.")
    p.add_argument("--max-outer-folds", type=int, default=0); p.add_argument("--dry-run", action="store_true"); p.add_argument("--skip-existing", action="store_true"); p.add_argument("--python", default=sys.executable)
    p.add_argument("--native-retries", type=int, default=2)
    return p


def _run(command: list[str]) -> None:
    print("[ComponentAblation] " + " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO, check=True)


def _run_resumable(command: list[str], *, retries: int) -> None:
    recoverable = {3221225477, -1073741519, 4294967295, -1}
    for attempt in range(retries + 1):
        try:
            _run(command)
            return
        except subprocess.CalledProcessError as error:
            if error.returncode not in recoverable or attempt >= retries:
                raise
            print(
                f"[ComponentAblation] Native CUDA/Windows exit {error.returncode}; "
                f"retrying ({attempt + 1}/{retries}).",
                flush=True,
            )
            time.sleep(10)


def _bcr(args: argparse.Namespace, *, profile: str, seed: int, audit: dict) -> None:
    root = Path(args.output_root) / "bcr" / profile / f"seed_{seed}"; marker = root / "component_ablation_completion.json"
    resume_contract = {"branch": "bcr", "profile": profile, "seed": seed, "v3_base_args_sha256": file_sha256(args.v3_base_run_args), "cohort_sha256": audit["subject_ledger_sha256"], "partition_sha256": audit["fixed_split_sha256"], "feature_cache_sha256": file_sha256(args.window_cache_path)}
    if args.skip_existing and completion_matches(marker, expected=resume_contract): return
    root.mkdir(parents=True, exist_ok=True)
    spec = profile_definition("bcr", profile)
    write_json(root / "protocol_audit.json", {**audit, "profile": spec, "label_semantics_native": "EZ=1", "label_semantics_unified": "NEZ=1,EZ=0", "score_conversion": "P(NEZ)=1-P(EZ)"})
    command = [args.python, str(REPO / "scripts" / "run_v3_qbc.py"), "--base-run-args", args.v3_base_run_args, "--window-cache-path", args.window_cache_path, "--allowed-subjects-ledger", args.allowed_subjects_ledger, "--outer-fold-manifest", args.outer_fold_manifest, "--fixed-split-manifest", args.fixed_split_manifest, "--output_dir", str(root), "--v3_qbc_profile", profile, "--seed", str(seed), "--max-outer-folds", str(args.max_outer_folds)]
    if args.skip_existing:
        command.append("--skip-existing")
    if args.dry_run:
        command.append("--dry-run"); write_json(root / "command.json", command); print(json.dumps({"branch": "bcr", "profile": profile, "seed": seed, "command": command})); return
    _run_resumable(command, retries=int(args.native_retries))
    folds = range(1, (args.max_outer_folds or 5) + 1)
    result = evaluate_branch(branch="bcr", profile=profile, seed=seed, run_root=root, output_root=root, folds=folds)
    hashes = {"effective_config_sha256": canonical_json_sha256(json.loads((root / "run_args.json").read_text(encoding="utf-8"))), "cohort_sha256": audit["subject_ledger_sha256"], "partition_sha256": audit["fixed_split_sha256"], "feature_cache_sha256": file_sha256(args.window_cache_path)}
    write_branch_provenance(root=root, method_id="BCR_QBC_COMPONENT_ABLATION", profile_id=profile, seed=seed, repo=REPO, folds=list(folds), **hashes)
    write_completion(root, payload={"branch": "bcr", "profile": profile, "seed": seed, "result": result, "v3_base_args_sha256": file_sha256(args.v3_base_run_args), **hashes})


def _reproduction(root: Path, reference: str, profiles: dict[str, list[str]], seeds: list[int]) -> None:
    rows = []
    if not reference:
        write_json(root / "reports" / "full_profile_reproduction_check.json", {"status": "REFERENCE_NOT_PROVIDED"}); return
    reference_root = Path(reference)
    for branch, full_profile, model in (("prq", "P2_TEMPORAL_Q10", "PRQ-Net"), ("bcr", "BCR_BOUNDARY_COVERAGE", "BCR-Net")):
        if full_profile not in profiles[branch]: continue
        for seed in seeds:
            candidate = pd.read_csv(root / branch / full_profile / f"seed_{seed}" / "metrics" / "overall_metrics.csv").iloc[0]
            reference_csv = reference_root / "pooled_cv" / f"seed_{seed}" / "metrics" / "overall.csv"
            if not reference_csv.is_file():
                raise FileNotFoundError(f"Reference overall metrics missing: {reference_csv}")
            base = pd.read_csv(reference_csv); name = "experiment" if "experiment" in base else "model"
            expected = base.loc[base[name].eq(model)]
            if len(expected) != 1: raise RuntimeError(f"Reference lacks unique {model} row for seed {seed}")
            expected_row = expected.iloc[0]
            for metric in ("patient_macro_f1", "patient_ez_f1", "patient_nez_f1", "patient_ez_auroc"):
                value, ref = float(candidate[metric]), float(expected_row[metric])
                rows.append({"branch": branch, "profile": full_profile, "seed": seed, "metric": metric, "reference_value": ref, "ablation_value": value, "absolute_difference": abs(value - ref), "tolerance": 1e-6, "passed": abs(value - ref) <= 1e-6})
    frame = pd.DataFrame(rows); frame.to_csv(root / "reports" / "full_profile_reproduction_check.csv", index=False)
    if not frame.empty and not frame.passed.all():
        raise RuntimeError("Full-profile reproduction check failed")


def main() -> None:
    args = parser().parse_args()
    for path in (args.p2_base_run_args, args.v3_base_run_args, args.window_cache_path):
        if not Path(path).is_file(): raise FileNotFoundError(path)
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    if seeds != sorted(set(seeds)) or not seeds: raise ValueError("Seeds must be unique")
    prq = [] if args.skip_prq else parse_csv(args.prq_profiles, PRQ_PROFILES)
    bcr = parse_csv(args.bcr_profiles, BCR_PROFILES)
    audit = validate_component_inputs(subjects=args.allowed_subjects_ledger, split_manifest=args.fixed_split_manifest, outer_manifest=args.outer_fold_manifest, require_n_patients=80)
    audit.update({"p2_base_args_sha256": file_sha256(args.p2_base_run_args), "v3_base_args_sha256": file_sha256(args.v3_base_run_args), "feature_cache_sha256": file_sha256(args.window_cache_path), "prq_profiles": [profile_definition("prq", item) for item in prq], "bcr_profiles": [profile_definition("bcr", item) for item in bcr]})
    root = Path(args.output_root); write_json(root / "audit" / "component_preflight.json", audit)
    if not args.skip_prq:
        prq_command = [args.python, str(REPO / "scripts" / "run_prq_component_ablation.py"), "--base-run-args", args.p2_base_run_args, "--window-cache-path", args.window_cache_path, "--allowed-subjects-ledger", args.allowed_subjects_ledger, "--fixed-split-manifest", args.fixed_split_manifest, "--outer-fold-manifest", args.outer_fold_manifest, "--output-root", str(root), "--seeds", args.seeds, "--profiles", ",".join(prq), "--max-outer-folds", str(args.max_outer_folds), "--python", args.python]
        prq_command.extend(["--native-retries", str(args.native_retries)])
        if args.dry_run: prq_command.append("--dry-run")
        if args.skip_existing: prq_command.append("--skip-existing")
        _run(prq_command)
    for profile in bcr:
        for seed in seeds: _bcr(args, profile=profile, seed=seed, audit=audit)
    if args.dry_run: return
    profiles = {"prq": prq, "bcr": bcr}
    if args.max_outer_folds in {0, 5}:
        _reproduction(root, args.confirmatory_reference_root, profiles, seeds)
        print(json.dumps(collect_reports(root, profiles=profiles, seeds=seeds, repo=REPO, audit=audit), indent=2))
    else:
        write_json(root / "reports" / "partial_smoke.json", {"status": "partial_smoke_complete", "max_outer_folds": args.max_outer_folds, "formal_summary_generated": False})


if __name__ == "__main__": main()
