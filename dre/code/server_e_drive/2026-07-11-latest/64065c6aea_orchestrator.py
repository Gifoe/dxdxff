"""Execute and resume P2/V3/fusion confirmatory units without model rewrites."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

from .config import resolve_path
from .evaluate import evaluate_pair
from .protocol import build_loco_manifests, build_reference_partition, sha256_file
from .provenance import (
    P2_METHOD_ID, P2_PROFILE_ID, V3_METHOD_ID, V3_PROFILE_ID,
    canonical_json_sha256, validate_branch_provenance, write_branch_provenance,
)
from .registry import append_registry


def _p2_patient_batch_size(held_out_center: str) -> int:
    """Match one patient per available center for center-balanced batches."""
    return 3 if held_out_center else 4


def _cached_hash(config: dict[str, Any], path: Path) -> str:
    cache = config.setdefault("_artifact_sha256", {})
    key = str(path.resolve())
    if key not in cache:
        cache[key] = sha256_file(path)
    return str(cache[key])


def _completed_with_provenance(
    root: Path, *, folds: list[int], method_id: str, profile_id: str,
    effective_config_sha256: str | None = None, cohort_sha256: str | None = None,
    partition_sha256: str | None = None, feature_cache_sha256: str | None = None,
) -> bool:
    ledgers = all(
        all(
            (root / f"{stem}_channel_predictions_neuroez_v2_fold_{fold}.csv").is_file()
            or (root / f"fold_{fold}" / f"{stem}_channel_predictions_neuroez_v2_fold_{fold}.csv").is_file()
            for stem in ("val", "test")
        ) for fold in folds
    )
    if not ledgers:
        return False
    try:
        provenance = validate_branch_provenance(
            root, expected_method_id=method_id, expected_profile_id=profile_id,
        )
    except RuntimeError:
        return False
    expected = {
        "effective_config_sha256": effective_config_sha256,
        "cohort_sha256": cohort_sha256,
        "partition_sha256": partition_sha256,
        "feature_cache_sha256": feature_cache_sha256,
    }
    return all(value is None or provenance.get(key) == value for key, value in expected.items())


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run(command: list[str], *, cwd: Path, log_path: Path, registry: Path, record: dict[str, Any], dry_run: bool) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = _now(); clock = time.perf_counter()
    row = dict(record, status="running", start_time=start, command=json.dumps(command), log_path=str(log_path))
    append_registry(registry, row)
    label = " | ".join(
        str(record.get(key, "")) for key in
        ("experiment_type", "model", "seed", "outer_fold", "held_out_center")
        if str(record.get(key, ""))
    )
    print(f"[Confirmatory][START] {label}", flush=True)
    print(f"[Confirmatory][LOG] {log_path}", flush=True)
    try:
        with log_path.open("w", encoding="utf-8") as stream:
            process = subprocess.Popen(
                command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                stream.write(line)
                stream.flush()
                print(line, end="", flush=True)
            return_code = process.wait()
            if return_code != 0:
                raise subprocess.CalledProcessError(return_code, command)
    except (subprocess.CalledProcessError, OSError) as exc:
        tail = ""
        try:
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            tail = "\n".join(lines[-80:])
        except OSError:
            tail = "<log unavailable>"
        message = f"{exc}\n\nLast {min(80, len(tail.splitlines()))} lines of {log_path}:\n{tail}"
        append_registry(registry, {**row, "status": "failed", "end_time": _now(), "elapsed_seconds": time.perf_counter() - clock, "error_message": message})
        print(f"[Confirmatory][FAILED] {label}", flush=True)
        raise RuntimeError(message) from exc
    elapsed = time.perf_counter() - clock
    append_registry(registry, {**row, "status": "dry_run" if dry_run else "complete", "end_time": _now(), "elapsed_seconds": elapsed})
    print(f"[Confirmatory][COMPLETE] {label} | elapsed={elapsed / 60.0:.1f} min", flush=True)


def _json(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8"); return path


def _enforce_branch_reuse(
    root: Path, *, contract_path: Path, expected_contract: dict[str, Any], allow_resume: bool,
    overwrite_invalid: bool = False,
) -> None:
    if not root.exists() or not any(root.iterdir()):
        return
    if not allow_resume:
        raise RuntimeError(f"--no_resume requires a fresh branch directory: {root}")
    valid = contract_path.is_file()
    observed = json.loads(contract_path.read_text(encoding="utf-8")) if valid else None
    valid = valid and observed == expected_contract
    if valid:
        return
    if not overwrite_invalid:
        reason = "missing launch contract" if observed is None else "launch contract mismatch"
        raise RuntimeError(f"Refusing invalid checkpoint/output reuse under {root}: {reason}")
    archive = root.parent / "invalid_checkpoint_archive"
    archive.mkdir(parents=True, exist_ok=True)
    suffix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = archive / f"{root.name}_{suffix}"
    shutil.move(str(root), str(destination))


def _run_models(
    *, config: dict, output: Path, partition: Path, outer_manifest: Path, seed: int, max_outer_folds: int,
    experiment_type: str, held_out_center: str, dry_run: bool, skip_completed: bool, selected_outer_fold: int | None = None, smoke: bool = False,
    model: str = "all",
) -> tuple[Path | None, Path | None]:
    repo = Path(config["repo_root"]); registry = Path(config["output_root"]) / "registry" / "run_registry.csv"
    p2_root, v3_root = output / "p2_q10", output / "v3_qbc"
    p2_base = resolve_path(config, "p2_reference_run_args" if "p2_reference_run_args" in config else "p2_base_args")
    p2_entry = resolve_path(config, "true_p2_entrypoint" if "true_p2_entrypoint" in config else "p2_entrypoint")
    v3_base = resolve_path(config, "v3_reference_args" if "v3_reference_args" in config else "v3_base_args")
    v3_entry = resolve_path(config, "v3_entrypoint")
    cohort = resolve_path(config, "cohort_ledger"); cache = resolve_path(config, "feature_cache")
    folds = [int(selected_outer_fold)] if selected_outer_fold else list(range(1, max_outer_folds + 1))
    cohort_hash = _cached_hash(config, cohort)
    partition_hash = _cached_hash(config, partition)
    feature_hash = _cached_hash(config, cache)
    need_p2 = model in {"all", "p2", "fusion"}
    need_v3 = model in {"all", "v3", "fusion"}
    overrides = {
        "output_dir": str(p2_root), "window_cache_path": str(cache),
        "fixed_split_manifest": str(partition), "require_n_patients": int(config["require_n_patients"]),
        "config_name": f"P2_Q10_CONFIRM_{experiment_type}_seed{seed}",
        "model_seed": int(seed), "random_seed": int(seed), "outer_split_seed": int(config.get("outer_split_seed", 42)),
        "max_outer_folds": int(max_outer_folds), "selected_outer_fold": int(selected_outer_fold or 0),
        "patient_batch_size": _p2_patient_batch_size(held_out_center),
        "dry_run_config_only": bool(dry_run), "skip_existing": False,
    }
    if held_out_center:
        overrides["allow_partial_fixed_test_manifest"] = True
    if smoke:
        overrides.update({"epochs": max(5, int(config.get("smoke_epochs", 5))), "patience": 1, "min_epochs_before_early_stop": 5})
    audit_dir = output / "audit"
    overrides_path = _json(audit_dir / "p2_overrides.json", overrides)
    p2_effective_path = audit_dir / "p2_effective_config.json"
    p2_command = [
        sys.executable, str(repo / "scripts" / "task1_confirmatory" / "launch_p2.py"),
        "--entrypoint", str(p2_entry), "--base-args", str(p2_base), "--overrides-json", str(overrides_path),
        "--command-output", str(audit_dir / "p2_command.json"),
        "--argument-coverage-output", str(audit_dir / "p2_argument_coverage.json"),
        "--reference-effective-config-output", str(audit_dir / "p2_reference_effective_config.json"),
        "--effective-config-output", str(p2_effective_path),
        "--effective-config-diff-output", str(audit_dir / "p2_effective_config_diff.json"),
        "--experiment-type", experiment_type, "--strict-arg-coverage",
    ]
    if dry_run: p2_command.append("--dry-run")
    # The hash is deterministic before launch and is checked again after launch.
    from .provenance import effective_config, load_entrypoint_parser
    p2_effective = effective_config(json.loads(p2_base.read_text(encoding="utf-8")), overrides, load_entrypoint_parser(p2_entry))
    p2_effective_hash = canonical_json_sha256(p2_effective)
    p2_contract = {
        "method_id": P2_METHOD_ID, "profile_id": P2_PROFILE_ID,
        "effective_config_sha256": p2_effective_hash, "cohort_sha256": cohort_hash,
        "partition_sha256": partition_hash, "feature_cache_sha256": feature_hash,
        "seed": int(seed), "held_out_center": held_out_center,
    }
    p2_contract_path = audit_dir / "p2_launch_contract.json"
    _enforce_branch_reuse(
        p2_root, contract_path=p2_contract_path, expected_contract=p2_contract,
        allow_resume=bool(config.get("_resume_allowed", skip_completed)),
        overwrite_invalid=bool(config.get("_overwrite_invalid_checkpoints", False)),
    )
    _json(p2_contract_path, p2_contract)
    p2_done = _completed_with_provenance(
        p2_root, folds=folds, method_id=P2_METHOD_ID, profile_id=P2_PROFILE_ID,
        effective_config_sha256=p2_effective_hash, cohort_sha256=cohort_hash,
        partition_sha256=partition_hash, feature_cache_sha256=feature_hash,
    )
    if need_p2 and not (skip_completed and p2_done):
        _run(p2_command, cwd=repo, log_path=output / "logs" / "p2.log", registry=registry, dry_run=dry_run, record={"experiment_type": experiment_type, "model": "P2-Q10", "seed": seed, "outer_fold": "all" if max_outer_folds > 1 else 1, "held_out_center": held_out_center, "config_path": str(p2_base), "config_sha256": sha256_file(p2_base), "cohort_sha256": cohort_hash, "fold_manifest_sha256": partition_hash})
        if not dry_run:
            write_branch_provenance(root=p2_root, method_id=P2_METHOD_ID, profile_id=P2_PROFILE_ID, effective_config_sha256=p2_effective_hash, cohort_sha256=cohort_hash, partition_sha256=partition_hash, feature_cache_sha256=feature_hash, seed=seed, repo=repo, held_out_center=held_out_center, folds=folds)
    v3_effective_hash = canonical_json_sha256({"base_args_sha256": sha256_file(v3_base), "seed": seed, "partition_sha256": partition_hash, "profile": V3_PROFILE_ID, "held_out_center": held_out_center})
    v3_contract = {
        "method_id": V3_METHOD_ID, "profile_id": V3_PROFILE_ID,
        "effective_config_sha256": v3_effective_hash, "cohort_sha256": cohort_hash,
        "partition_sha256": partition_hash, "feature_cache_sha256": feature_hash,
        "seed": int(seed), "held_out_center": held_out_center,
    }
    v3_contract_path = audit_dir / "v3_launch_contract.json"
    _enforce_branch_reuse(
        v3_root, contract_path=v3_contract_path, expected_contract=v3_contract,
        allow_resume=bool(config.get("_resume_allowed", skip_completed)),
        overwrite_invalid=bool(config.get("_overwrite_invalid_checkpoints", False)),
    )
    _json(v3_contract_path, v3_contract)
    v3_done = _completed_with_provenance(v3_root, folds=folds, method_id=V3_METHOD_ID, profile_id=V3_PROFILE_ID, effective_config_sha256=v3_effective_hash, cohort_sha256=cohort_hash, partition_sha256=partition_hash, feature_cache_sha256=feature_hash)
    if need_v3 and not (skip_completed and v3_done):
        command = [sys.executable, str(v3_entry), "--base-run-args", str(v3_base), "--v3_qbc_profile", "BCR_BOUNDARY_COVERAGE", "--window_cache_path", str(cache), "--allowed_subjects_ledger", str(cohort), "--fixed_fold_manifest", str(outer_manifest), "--fixed_split_manifest", str(partition), "--require_n_patients", str(config["require_n_patients"]), "--output_dir", str(v3_root), "--random_seed", str(seed), "--max-outer-folds", str(max_outer_folds), "--selected_outer_fold", str(selected_outer_fold or 0)]
        # The final BCR budget is predeclared by the orchestration config.  Do
        # not inherit the temporary short-early-stop settings used by earlier
        # component ablations.
        bcr_budget = dict(config.get("bcr_training", {}))
        for config_key, flag in (
            ("epochs", "--epochs"),
            ("patience", "--patience"),
            ("min_epochs_before_early_stop", "--min-epochs-before-early-stop"),
            ("batch_size", "--batch-size"),
            ("patient_batch_size", "--patient-batch-size"),
            ("num_workers", "--num-workers"),
        ):
            value = bcr_budget.get(config_key)
            if value is not None:
                command.extend([flag, str(int(value))])
        if bcr_budget.get("device"):
            command.extend(["--device", str(bcr_budget["device"])])
        if held_out_center: command.append("--loco_mode")
        if smoke:
            command.extend(["--epochs", str(max(5, int(config.get("smoke_epochs", 5)))), "--patience", "1"])
        if dry_run: command.append("--dry-run")
        _run(command, cwd=repo, log_path=output / "logs" / "bcr.log", registry=registry, dry_run=dry_run, record={"experiment_type": experiment_type, "model": "BCR-Net", "seed": seed, "outer_fold": "all" if max_outer_folds > 1 else 1, "held_out_center": held_out_center, "config_path": str(v3_base), "config_sha256": sha256_file(v3_base), "cohort_sha256": sha256_file(cohort), "fold_manifest_sha256": sha256_file(partition)})
        if not dry_run:
            write_branch_provenance(root=v3_root, method_id=V3_METHOD_ID, profile_id=V3_PROFILE_ID, effective_config_sha256=v3_effective_hash, cohort_sha256=cohort_hash, partition_sha256=partition_hash, feature_cache_sha256=feature_hash, seed=seed, repo=repo, held_out_center=held_out_center, folds=folds)
    return (p2_root if need_p2 else None), (v3_root if need_v3 else None)


def prepare_manifests(config: dict, *, output_root: Path) -> dict:
    audit_dir = output_root / "audit"; audit_dir.mkdir(parents=True, exist_ok=True)
    partition = build_reference_partition(cohort_ledger=resolve_path(config, "cohort_ledger"), outer_fold_manifest=resolve_path(config, "fixed_fold_manifest"), p2_reference_root=resolve_path(config, "p2_reference_root"), output_path=audit_dir / "fixed_partition_manifest.csv", require_n_patients=int(config["require_n_patients"]))
    loco = build_loco_manifests(cohort_ledger=resolve_path(config, "cohort_ledger"), output_dir=audit_dir, split_seed=int(config["loco_split_seed"]))
    _json(audit_dir / "manifest_audit.json", {"status": "passed", "reference_partition": partition, "loco": loco})
    return {"partition": Path(partition["partition_path"]), "loco": loco}


def run_pooled(config: dict, *, output_root: Path, seeds: list[int], outer_fold: int | None, dry_run: bool, smoke: bool, skip_completed: bool, model: str = "all") -> list[dict]:
    manifests = prepare_manifests(config, output_root=output_root)
    results = []
    count = 1 if outer_fold is not None else 5
    for seed in seeds:
        root = output_root / ("smoke" if smoke else "pooled_cv") / f"seed_{seed}"
        p2, v3 = _run_models(config=config, output=root, partition=manifests["partition"], outer_manifest=resolve_path(config, "fixed_fold_manifest"), seed=seed, max_outer_folds=count, experiment_type="pooled_cv_smoke" if smoke else "pooled_cv", held_out_center="", dry_run=dry_run, skip_completed=skip_completed, selected_outer_fold=outer_fold, smoke=smoke, model=model)
        if not dry_run and not smoke and count == 5 and p2 and v3:
            results.append(evaluate_pair(
                p2_root=p2, v3_root=v3, folds=range(1, 6), output_dir=root,
                analysis_status="PRIMARY_CONFIRMATORY", require_provenance=bool(config.get("require_branch_provenance", False)),
                expected_seed=seed, expected_held_out_center="",
            ))
    return results


def run_loco(config: dict, *, output_root: Path, seeds: list[int], held_out_center: str | None, dry_run: bool, smoke: bool, skip_completed: bool, model: str = "all") -> list[dict]:
    manifests = prepare_manifests(config, output_root=output_root)
    centers = [held_out_center] if held_out_center else ["hup", "lzu", "multicenter", "pediatric"]
    results = []
    for center in centers:
        partition = Path(manifests["loco"]["held_out"][center]["manifest_path"])
        root_base = output_root / ("smoke" if smoke else "loco") / f"heldout_{center}"
        # V3 validates outer-test membership from this one-fold manifest.
        for seed in seeds:
            root = root_base / f"seed_{seed}"
            p2, v3 = _run_models(config=config, output=root, partition=partition, outer_manifest=partition, seed=seed, max_outer_folds=1, experiment_type="loco_smoke" if smoke else "loco", held_out_center=center, dry_run=dry_run, skip_completed=skip_completed, smoke=smoke, model=model)
            if not dry_run and not smoke and p2 and v3:
                results.append(evaluate_pair(
                    p2_root=p2, v3_root=v3, folds=[1], output_dir=root,
                    analysis_status="PRIMARY_CONFIRMATORY_LOCO", require_provenance=bool(config.get("require_branch_provenance", False)),
                    expected_seed=seed, expected_held_out_center=center,
                ))
    return results
