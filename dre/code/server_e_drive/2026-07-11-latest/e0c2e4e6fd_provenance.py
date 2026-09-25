"""Fail-closed model identity and effective-configuration contracts."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


P2_METHOD_ID = "P23_TRN_NEZ_80:P2_TEMPORAL_Q10"
P2_PROFILE_ID = "P2_TEMPORAL_Q10"
V3_METHOD_ID = "BCR_NET"
V3_PROFILE_ID = "BCR_BOUNDARY_COVERAGE"

# These values are derived after parsing and were persisted by the historical
# runner. They are not independent model knobs, but their values are audited.
DERIVED_METADATA = {
    "model_family": "b0_pruned_ez_backbone",
    "score_semantics": "nez_probability",
}

COMMON_ALLOWED_DIFFERENCES = {
    "output_dir", "config_name", "model_seed", "random_seed", "selected_outer_fold",
    "fixed_split_manifest", "max_outer_folds", "device", "dry_run_config_only",
    "skip_existing", "allowed_subjects_ledger",
}
LOCO_ALLOWED_DIFFERENCES = COMMON_ALLOWED_DIFFERENCES | {
    "allow_partial_fixed_test_manifest", "patient_batch_size",
}


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(repo: str | Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "UNAVAILABLE"


def load_entrypoint_parser(entrypoint: str | Path) -> argparse.ArgumentParser:
    path = Path(entrypoint).resolve()
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(f"confirmatory_entry_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load entrypoint parser: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    parser = module.build_parser()
    if not isinstance(parser, argparse.ArgumentParser):
        raise TypeError(f"{path} build_parser() did not return ArgumentParser")
    return parser


def parser_actions(parser: argparse.ArgumentParser) -> dict[str, argparse.Action]:
    actions: dict[str, argparse.Action] = {}
    for action in parser._actions:
        prior = actions.get(action.dest)
        if prior is None or isinstance(action, argparse.BooleanOptionalAction):
            actions[action.dest] = action
    return actions


def audit_argument_coverage(
    base_args: dict[str, Any], parser: argparse.ArgumentParser, *, strict: bool,
) -> dict[str, Any]:
    actions = parser_actions(parser)
    consumed = sorted(key for key in base_args if key in actions)
    ignored: list[str] = []
    unknown: list[str] = []
    invalid_derived: dict[str, Any] = {}
    for key in sorted(set(base_args) - set(actions)):
        if key in DERIVED_METADATA and base_args[key] == DERIVED_METADATA[key]:
            ignored.append(key)
        else:
            unknown.append(key)
            if key in DERIVED_METADATA:
                invalid_derived[key] = {"actual": base_args[key], "expected": DERIVED_METADATA[key]}
    passed = not unknown
    report = {
        "n_base_args": len(base_args),
        "n_consumed_args": len(consumed),
        "n_ignored_metadata_args": len(ignored),
        "n_unknown_model_args": len(unknown),
        "consumed_args": consumed,
        "ignored_metadata_args": ignored,
        "unknown_model_args": unknown,
        "invalid_derived_metadata": invalid_derived,
        "passed": passed,
    }
    if strict and not passed:
        raise RuntimeError("P2 strict argument coverage failed: " + json.dumps(report, sort_keys=True))
    return report


def effective_config(
    base_args: dict[str, Any], overrides: dict[str, Any], parser: argparse.ArgumentParser,
) -> dict[str, Any]:
    actions = parser_actions(parser)
    merged = {**base_args, **overrides}
    return {key: merged[key] for key in sorted(merged) if key in actions and merged[key] is not None}


def audit_effective_config_diff(
    reference: dict[str, Any], candidate: dict[str, Any], *, experiment_type: str,
) -> dict[str, Any]:
    allowed_names = LOCO_ALLOWED_DIFFERENCES if experiment_type.startswith("loco") else COMMON_ALLOWED_DIFFERENCES
    all_keys = sorted(set(reference) | set(candidate))
    differences = {
        key: {"reference": reference.get(key, "<MISSING>"), "candidate": candidate.get(key, "<MISSING>")}
        for key in all_keys if reference.get(key, "<MISSING>") != candidate.get(key, "<MISSING>")
    }
    allowed = {key: value for key, value in differences.items() if key in allowed_names}
    forbidden = {key: value for key, value in differences.items() if key not in allowed_names}
    missing = sorted(key for key in reference if key not in candidate)
    extra = sorted(key for key in candidate if key not in reference)
    missing_forbidden = [key for key in missing if key not in allowed_names]
    extra_forbidden = [key for key in extra if key not in allowed_names]
    report = {
        "experiment_type": experiment_type,
        "allowed_difference_fields": sorted(allowed_names),
        "allowed_differences": allowed,
        "forbidden_differences": forbidden,
        "missing_reference_fields": missing,
        "extra_confirmatory_fields": extra,
        "forbidden_missing_reference_fields": missing_forbidden,
        "forbidden_extra_confirmatory_fields": extra_forbidden,
        "reference_effective_config_sha256": canonical_json_sha256(reference),
        "effective_config_sha256": canonical_json_sha256(candidate),
        "passed": not forbidden and not missing_forbidden and not extra_forbidden,
    }
    return report


def audit_original_p2(config: dict[str, Any], *, output_dir: str | Path) -> dict[str, Any]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    entrypoint = Path(str(config["true_p2_entrypoint"]))
    run_args_path = Path(str(config["p2_reference_run_args"]))
    reference_root = Path(str(config["p2_reference_root"]))
    errors: list[str] = []
    if not entrypoint.is_file(): errors.append(f"true entrypoint missing: {entrypoint}")
    if not run_args_path.is_file(): errors.append(f"reference run args missing: {run_args_path}")
    if not reference_root.is_dir(): errors.append(f"reference root missing: {reference_root}")
    args: dict[str, Any] = {}
    if run_args_path.is_file():
        args = json.loads(run_args_path.read_text(encoding="utf-8"))
        expected = {
            "use_p23_trn_nez": True,
            "p23_profile": str(config["expected_p2_method_id"]).split(":")[-1],
            "p23_direct_outer_only": True,
            "p23_regression_protocol": True,
            "p23_use_p2_loss": True,
            "use_cane_path_cp_nez": False,
        }
        for key, value in expected.items():
            if args.get(key) != value:
                errors.append(f"reference {key}={args.get(key)!r}, expected {value!r}")
    if "CANE_PATH" in str(entrypoint).upper() or "P2_CANE_PATH" in str(entrypoint).upper():
        errors.append("CANE-PATH entrypoint is forbidden for P2-Q10")
    if entrypoint.name != "run_neuroez_c.py" or entrypoint.parent.name != "P23_TRN_NEZ_80":
        errors.append("true P2 entrypoint must be P23_TRN_NEZ_80/run_neuroez_c.py")
    method_id = P2_METHOD_ID
    if str(config.get("expected_p2_method_id")) != method_id:
        errors.append(f"expected_p2_method_id must be {method_id}")
    evidence = [str(path) for path in (
        run_args_path, reference_root / "p23_protocol_audit.json",
        reference_root / "P23_TRN_SENSITIVITY80_REPORT.md",
        reference_root / "p23_resume" / "outer_1" / "direct_outer_complete.pt",
    ) if path.exists()]
    payload = {
        "status": "passed" if not errors else "P2_ORIGINAL_ENTRYPOINT_UNRESOLVED",
        "reference_root": str(reference_root),
        "reference_run_args": str(run_args_path),
        "true_entrypoint": str(entrypoint),
        "true_method_id": method_id,
        "true_profile": args.get("p23_profile"),
        "true_loss_mode": "p23_use_p2_loss/cane_path_cp_ranking_loss",
        "persisted_legacy_loss_mode": args.get("loss_mode"),
        "true_config_name": args.get("config_name"),
        "true_checkpoint_pattern": "fold_{fold}/best_model.pt",
        "true_git_commit": "UNRESOLVED_IN_REFERENCE_ARTIFACT",
        "evidence_files": evidence,
        "confidence": "high" if not errors else "insufficient",
        "unresolved_fields": ["historical_git_commit"],
        "errors": errors,
    }
    (target / "p2_original_provenance.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if errors:
        raise RuntimeError("P2_ORIGINAL_ENTRYPOINT_UNRESOLVED: " + "; ".join(errors))
    return payload


def write_branch_provenance(
    *, root: str | Path, method_id: str, profile_id: str, effective_config_sha256: str,
    cohort_sha256: str, partition_sha256: str, feature_cache_sha256: str,
    seed: int, repo: str | Path, held_out_center: str = "", folds: list[int],
) -> dict[str, Any]:
    branch = Path(root)
    checkpoint_rows = []
    commit = git_commit(repo)
    for fold in folds:
        # PRQ stores ``best_model.pt`` while BCR stores its explicitly named
        # boundary/coverage checkpoint.  The legacy V3 name is accepted only for
        # historical provenance inspection, never as a final BCR identity.
        # exactly one known name rather than guessing from a broad glob.
        fold_dir = branch / f"fold_{fold}"
        standard_checkpoint = fold_dir / "best_model.pt"
        bcr_checkpoint = fold_dir / "best_bcr_boundary_coverage.pt"
        if bcr_checkpoint.is_file():
            # The BCR runner also exposes best_model.pt as a standardized
            # copy for downstream tools.  It is not a second independently
            # selected checkpoint.  Verify that copy before recording the
            # native BCR checkpoint as provenance evidence.
            if standard_checkpoint.is_file() and file_sha256(standard_checkpoint) != file_sha256(bcr_checkpoint):
                raise RuntimeError(
                    f"BCR standardized checkpoint differs from native checkpoint for fold {fold}: {fold_dir}"
                )
            candidates = [bcr_checkpoint]
        else:
            candidates = [
                path for path in (
                    standard_checkpoint,
                    fold_dir / "best_b0_pruned_model.pth",
                ) if path.is_file()
            ]
        if len(candidates) != 1:
            raise RuntimeError(f"Expected one checkpoint for fold {fold} under {branch}; found {candidates}")
        checkpoint_hash = file_sha256(candidates[0])
        metadata = {
            "status": "complete", "method_id": method_id, "profile_id": profile_id,
            "effective_config_sha256": effective_config_sha256, "checkpoint_sha256": checkpoint_hash,
            "cohort_sha256": cohort_sha256, "partition_sha256": partition_sha256,
            "feature_cache_sha256": feature_cache_sha256, "git_commit": commit,
            "seed": int(seed), "fold": int(fold), "held_out_center": held_out_center,
        }
        metadata_path = candidates[0].with_suffix(candidates[0].suffix + ".metadata.json")
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        checkpoint_rows.append({
            "fold": fold, "path": str(candidates[0]), "sha256": checkpoint_hash,
            "metadata_path": str(metadata_path), "metadata_sha256": file_sha256(metadata_path),
        })
    payload = {
        "status": "passed", "method_id": method_id, "profile_id": profile_id,
        "effective_config_sha256": effective_config_sha256,
        "checkpoint_sha256": canonical_json_sha256(checkpoint_rows),
        "checkpoints": checkpoint_rows, "git_commit": commit, "seed": int(seed),
        "partition_sha256": partition_sha256, "cohort_sha256": cohort_sha256,
        "feature_cache_sha256": feature_cache_sha256, "held_out_center": held_out_center,
    }
    path = branch / "training_provenance.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def validate_branch_provenance(
    root: str | Path, *, expected_method_id: str, expected_profile_id: str,
    expected_seed: int | None = None, expected_held_out_center: str | None = None,
) -> dict[str, Any]:
    path = Path(root) / "training_provenance.json"
    if not path.is_file():
        raise RuntimeError(f"Missing branch provenance: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    errors = []
    if payload.get("status") != "passed": errors.append("status is not passed")
    if payload.get("method_id") != expected_method_id: errors.append("method_id mismatch")
    if payload.get("profile_id") != expected_profile_id: errors.append("profile_id mismatch")
    if expected_seed is not None and int(payload.get("seed", -1)) != int(expected_seed):
        errors.append("seed mismatch")
    if expected_held_out_center is not None and str(payload.get("held_out_center", "")) != expected_held_out_center:
        errors.append("held_out_center mismatch")
    for key in ("effective_config_sha256", "checkpoint_sha256", "cohort_sha256", "partition_sha256", "feature_cache_sha256"):
        if not str(payload.get(key, "")).strip(): errors.append(f"missing {key}")
    checkpoint_rows = payload.get("checkpoints")
    if not isinstance(checkpoint_rows, list) or not checkpoint_rows:
        errors.append("missing checkpoints")
    else:
        actual_rows = []
        for row in checkpoint_rows:
            checkpoint = Path(str(row.get("path", "")))
            if not checkpoint.is_file():
                errors.append(f"checkpoint missing: {checkpoint}")
                continue
            actual_hash = file_sha256(checkpoint)
            if actual_hash != row.get("sha256"):
                errors.append(f"checkpoint hash mismatch: {checkpoint}")
            metadata_path = Path(str(row.get("metadata_path", "")))
            if not metadata_path.is_file() or file_sha256(metadata_path) != row.get("metadata_sha256"):
                errors.append(f"checkpoint metadata missing or mismatched: {metadata_path}")
            else:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                required = {
                    "method_id": expected_method_id, "profile_id": expected_profile_id,
                    "checkpoint_sha256": actual_hash, "fold": int(row["fold"]),
                }
                if any(metadata.get(key) != value for key, value in required.items()):
                    errors.append(f"checkpoint metadata contract mismatch: {metadata_path}")
            actual_rows.append({
                "fold": int(row["fold"]), "path": str(checkpoint), "sha256": actual_hash,
                "metadata_path": str(metadata_path), "metadata_sha256": row.get("metadata_sha256"),
            })
        if len(actual_rows) == len(checkpoint_rows) and canonical_json_sha256(actual_rows) != payload.get("checkpoint_sha256"):
            errors.append("checkpoint aggregate hash mismatch")
    if errors:
        raise RuntimeError(f"Branch provenance failed for {root}: {errors}")
    return payload


def write_json(path: str | Path, value: Any) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return target
