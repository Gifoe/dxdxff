#!/usr/bin/env python3
"""Run the declared Task 1 AAAI training units sequentially and resumably.

PRQ-Net's completed pooled 5-fold runs are immutable inputs.  This command
re-trains only final BCR-Net and the LOCO model units, then derives CDEL from
the matching PRQ/BCR ledgers.  SEEGformer is intentionally not scheduled.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from neuroez_c.p2_v3_fusion_protocol import discover_fold_ledger
from task1_aaai.training_plan import (
    FINAL_BCR_PROFILE,
    FORMAL_SEEDS,
    LOCO_TRAINED_MODELS,
    bcr_command,
    final_bcr_seed_root,
    final_cdel_seed_root,
    load_plan,
    prq_seed_root,
    timeconv_command,
    validate_bcr_runtime_contract,
    validate_plan,
)
from task1_confirmatory.config import load_config
from task1_confirmatory.evaluate import evaluate_pair
from task1_confirmatory.orchestrator import prepare_manifests, run_loco
from task1_confirmatory.protocol import sha256_file
from task1_confirmatory.provenance import (
    P2_METHOD_ID,
    P2_PROFILE_ID,
    V3_METHOD_ID,
    V3_PROFILE_ID,
    canonical_json_sha256,
    validate_branch_provenance,
    write_branch_provenance,
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _append_job(output: Path, payload: dict[str, Any]) -> None:
    path = output / "logs" / "training_jobs.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")


def _run(command: list[str], *, cwd: Path, output: Path, job: str, dry_run: bool) -> None:
    record = {"job": job, "command": command, "status": "started", "timestamp": time.time()}
    _append_job(output, record)
    print(f"[Task1-AAAI][START] {job}", flush=True)
    print("[Task1-AAAI][CMD] " + subprocess.list2cmdline(command), flush=True)
    if dry_run:
        _append_job(output, {**record, "status": "dry_run"})
        return
    log = output / "logs" / f"{job}.log"
    with log.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            stream.write(line)
            stream.flush()
            print(line, end="", flush=True)
        code = process.wait()
    if code != 0:
        _append_job(output, {**record, "status": "failed", "returncode": code, "log": str(log)})
        raise subprocess.CalledProcessError(code, command)
    _append_job(output, {**record, "status": "complete", "log": str(log)})
    print(f"[Task1-AAAI][COMPLETE] {job}", flush=True)


def _require_file(path: str | Path, label: str) -> Path:
    value = Path(path)
    if not value.is_file():
        raise FileNotFoundError(f"{label} not found: {value}")
    return value


def _confirmatory_config(plan: dict[str, Any]) -> dict[str, Any]:
    config = dict(load_config(plan["confirmatory_config"]))
    config.update({
        "repo_root": str(plan["repo_root"]),
        "output_root": str(plan["output_root"]),
        "p2_reference_root": str(prq_seed_root(plan, 42)),
        "v3_reference_args": str(plan["bcr_base_args"]),
        "cohort_ledger": str(plan["cohort_manifest"]),
        "fixed_fold_manifest": str(plan["outer_fold_manifest"]),
        "feature_cache": str(plan["feature_cache"]),
        "raw_cache": str(plan["raw_cache"]),
        "require_n_patients": 80,
        "seeds": list(FORMAL_SEEDS),
        "bcr_training": dict(plan["bcr_training"]),
        "_resume_allowed": True,
        "_overwrite_invalid_checkpoints": False,
    })
    return config


def _verify_prq_reuse(plan: dict[str, Any]) -> dict[str, Any]:
    """Ensure every reused PRQ branch is a full frozen five-fold artifact."""
    rows = []
    for seed in FORMAL_SEEDS:
        root = prq_seed_root(plan, seed)
        if not root.is_dir():
            raise FileNotFoundError(f"Completed PRQ branch missing for seed {seed}: {root}")
        for fold in range(1, 6):
            for role in ("validation", "test"):
                path = discover_fold_ledger(root, fold, role)
                if not path.is_file():
                    raise FileNotFoundError(f"PRQ seed {seed} fold {fold} missing {role} ledger: {path}")
        provenance = validate_branch_provenance(
            root,
            expected_method_id=P2_METHOD_ID,
            expected_profile_id=P2_PROFILE_ID,
            expected_seed=seed,
            expected_held_out_center="",
        )
        rows.append({"seed": seed, "root": str(root), "provenance": provenance})
    return {"status": "passed", "prq_reused": True, "branches": rows}


def _bcr_complete(root: Path) -> bool:
    return (
        (root / "formal_summary.csv").is_file()
        and (root / "truek_summary.csv").is_file()
        and all((root / f"test_channel_predictions_neuroez_v2_fold_{fold}.csv").is_file() for fold in range(1, 6))
        and all((root / f"val_channel_predictions_neuroez_v2_fold_{fold}.csv").is_file() for fold in range(1, 6))
    )


def _write_bcr_provenance(plan: dict[str, Any], *, root: Path, seed: int, partition: Path) -> None:
    if (root / "training_provenance.json").is_file():
        return
    base = _require_file(plan["bcr_base_args"], "BCR base arguments")
    effective_hash = canonical_json_sha256({
        "base_args_sha256": sha256_file(base), "seed": int(seed),
        "partition_sha256": sha256_file(partition), "profile": V3_PROFILE_ID,
        "held_out_center": "",
    })
    write_branch_provenance(
        root=root, method_id=V3_METHOD_ID, profile_id=V3_PROFILE_ID,
        effective_config_sha256=effective_hash, cohort_sha256=sha256_file(plan["cohort_manifest"]),
        partition_sha256=sha256_file(partition), feature_cache_sha256=sha256_file(plan["feature_cache"]),
        seed=seed, repo=plan["repo_root"], held_out_center="", folds=list(range(1, 6)),
    )


def _run_final_bcr(plan: dict[str, Any], *, partition: Path, output: Path, dry_run: bool, resume: bool) -> None:
    for seed in FORMAL_SEEDS:
        root = final_bcr_seed_root(plan, seed)
        if resume and _bcr_complete(root):
            _write_bcr_provenance(plan, root=root, seed=seed, partition=partition)
            print(f"[Task1-AAAI] skip completed final BCR seed {seed}", flush=True)
            continue
        command = bcr_command(plan, seed=seed, output_dir=root, partition=partition)
        if resume:
            command.append("--skip-existing")
        _run(command, cwd=Path(plan["repo_root"]), output=output, job=f"final_bcr_seed_{seed}", dry_run=dry_run)
        if not dry_run:
            if not _bcr_complete(root):
                raise RuntimeError(f"Final BCR seed {seed} did not produce all validation/test ledgers")
            _write_bcr_provenance(plan, root=root, seed=seed, partition=partition)


def _run_final_cdel(plan: dict[str, Any], *, output: Path, dry_run: bool, resume: bool) -> None:
    for seed in FORMAL_SEEDS:
        target = final_cdel_seed_root(plan, seed)
        if resume and (target / "metrics" / "overall.csv").is_file():
            print(f"[Task1-AAAI] skip completed final CDEL seed {seed}", flush=True)
            continue
        if dry_run:
            _append_job(output, {"job": f"final_cdel_seed_{seed}", "status": "dry_run", "prq": str(prq_seed_root(plan, seed)), "bcr": str(final_bcr_seed_root(plan, seed))})
            continue
        result = evaluate_pair(
            p2_root=prq_seed_root(plan, seed), v3_root=final_bcr_seed_root(plan, seed),
            folds=range(1, 6), output_dir=target, analysis_status="PRIMARY_CONFIRMATORY_FINAL_BCR_RETRAIN",
            require_provenance=True, expected_seed=seed, expected_held_out_center="",
        )
        _write_json(target / "CDEL_FINAL_AUDIT.json", {**result, "formula": "0.80*P(NEZ|PRQ)+0.20*P(NEZ|BCR)", "training_performed": False})
        _append_job(output, {"job": f"final_cdel_seed_{seed}", "status": "complete", "output": str(target)})


def _run_loco_baselines(plan: dict[str, Any], *, manifests: dict[str, Any], output: Path, dry_run: bool, resume: bool) -> None:
    for center, audit in manifests["loco"]["held_out"].items():
        split = Path(audit["manifest_path"])
        for seed in FORMAL_SEEDS:
            root = output / "loco" / f"heldout_{center}" / f"seed_{seed}" / "baselines"
            rbf = root / "rbf_svm"
            rbf_command = [
                str(plan["python"]), str(Path(plan["repo_root"]) / "scripts" / "task1_aaai" / "run_loco_rbf_svm.py"),
                "--feature-cache", str(plan["feature_cache"]), "--loco-manifest", str(split),
                "--output-dir", str(rbf), "--seed", str(seed),
            ]
            if resume:
                rbf_command.append("--resume")
            _run(rbf_command, cwd=Path(plan["repo_root"]), output=output, job=f"loco_rbf_{center}_seed_{seed}", dry_run=dry_run)
            timeconv = root / "timeconv_cnn"
            if resume and (timeconv / "oof_ledgers" / "omni_timeconv_cnn" / f"seed_{seed}_channel_oof.csv").is_file():
                print(f"[Task1-AAAI] skip completed LOCO TimeConv-CNN {center} seed {seed}", flush=True)
                continue
            _run(
                timeconv_command(plan, seed=seed, split_manifest=split, output_dir=timeconv),
                cwd=Path(plan["repo_root"]), output=output,
                job=f"loco_timeconv_{center}_seed_{seed}", dry_run=dry_run,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--stage", choices=["audit", "final_bcr", "final_cdel", "loco", "all"], default="all")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    plan = load_plan(args.plan)
    validate_plan(plan)
    bcr_runtime = validate_bcr_runtime_contract(plan["repo_root"])
    output = Path(plan["output_root"])
    output.mkdir(parents=True, exist_ok=True)
    _require_file(plan["confirmatory_config"], "Confirmatory configuration")
    _require_file(plan["bcr_base_args"], "BCR base arguments")
    _require_file(plan["feature_cache"], "Feature cache")
    _require_file(plan["raw_cache"], "Raw cache")
    _require_file(plan["cohort_manifest"], "80-patient cohort manifest")
    _require_file(plan["outer_fold_manifest"], "Outer-fold manifest")
    _require_file(plan["timeconv_entrypoint"], "TimeConv-CNN entrypoint")
    _write_json(output / "configs" / "training_plan_effective.json", plan)
    prq = _verify_prq_reuse(plan)
    config = _confirmatory_config(plan)
    manifests = prepare_manifests(config, output_root=output)
    _write_json(output / "audit" / "training_input_audit.json", {
        "status": "passed", "prq": prq, "loco_models": list(LOCO_TRAINED_MODELS),
        "excluded_models": ["SEEGformer"], "manifest_audit": manifests,
        "bcr_profile": FINAL_BCR_PROFILE, "bcr_training": plan["bcr_training"],
        "bcr_runtime": bcr_runtime,
    })
    if args.stage == "audit":
        print(json.dumps({"status": "passed", "stage": "audit", "output": str(output)}, indent=2)); return
    partition = Path(manifests["partition"])
    if args.stage in {"final_bcr", "all"}:
        _run_final_bcr(plan, partition=partition, output=output, dry_run=args.dry_run, resume=args.resume)
    if args.stage in {"final_cdel", "all"}:
        _run_final_cdel(plan, output=output, dry_run=args.dry_run, resume=args.resume)
    if args.stage in {"loco", "all"}:
        # PRQ/BCR are independently retrained for each held-out center and
        # seed. CDEL is generated by the existing locked evaluator.
        result = run_loco(config, output_root=output, seeds=list(FORMAL_SEEDS), held_out_center=None,
                          dry_run=args.dry_run, smoke=False, skip_completed=args.resume, model="all")
        _write_json(output / "loco" / "prq_bcr_cdel_loco_summary.json", {"status": "scheduled" if args.dry_run else "passed", "results": result})
        _run_loco_baselines(plan, manifests=manifests, output=output, dry_run=args.dry_run, resume=args.resume)
    _write_json(output / "reports" / "TRAINING_PLAN_STATUS.json", {
        "status": "dry_run" if args.dry_run else "completed", "stage": args.stage,
        "prq_reused_from": str(plan["prq_reuse_root"]), "final_bcr_profile": FINAL_BCR_PROFILE,
        "loco_models": list(LOCO_TRAINED_MODELS), "excluded_models": ["SEEGformer"],
    })
    print(json.dumps({"status": "passed", "stage": args.stage, "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
