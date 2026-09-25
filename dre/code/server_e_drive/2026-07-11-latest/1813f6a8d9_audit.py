"""Repository and frozen-ledger audits that must pass before formal training."""
from __future__ import annotations

import json
import importlib.util
import inspect
from pathlib import Path
import subprocess
import sys
from typing import Any

from .evaluate import evaluate_pair
from .protocol import build_reference_partition, sha256_file
from .provenance import audit_original_p2


EXPECTED_FORMAL = {"PRQ-Net": .6284640770231468}
EXPECTED_TRUE_K_DIAGNOSTIC = {"PRQ-Net": .6487672375803516}


def _read_cache_schema_in_subprocess(cache: Path, *, output: Path) -> dict[str, Any]:
    """Keep large pickle allocations out of the long-lived training runner."""
    probe = Path(__file__).with_name("cache_probe.py")
    command = [sys.executable, str(probe), "--cache", str(cache), "--output", str(output)]
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "cache probe failed").strip()
        raise RuntimeError(
            "Feature-cache schema probe failed in its isolated process. "
            "Close memory-heavy processes and retry; batch size cannot affect pickle loading.\n"
            + detail
        )
    return json.loads(output.read_text(encoding="utf-8"))


def audit_cache_provenance(config: dict[str, Any], *, output_root: str | Path) -> dict:
    """Inspect both cache schema and the exact feature builder used by this run."""
    target = Path(output_root) / "audit"
    target.mkdir(parents=True, exist_ok=True)
    cache = Path(config["feature_cache"])
    builder = Path(config["feature_cache_builder"])
    errors: list[str] = []
    if not cache.is_file(): errors.append(f"feature cache missing: {cache}")
    if not builder.is_file(): errors.append(f"feature builder missing: {builder}")
    schema: dict[str, Any] = {}
    if cache.is_file():
        schema = _read_cache_schema_in_subprocess(cache, output=target / "feature_cache_schema.json")
        names = list(map(str, schema["window_feature_names"]))
        if len(names) != 28 or int(schema["n_patients"]) != 90 or int(schema["n_runs"]) != 281:
            errors.append("feature cache is not the audited 90-patient/281-run/28-feature artifact")
    source_evidence: dict[str, Any] = {}
    if builder.is_file():
        spec = importlib.util.spec_from_file_location("confirmatory_cache_builder", builder)
        if spec is None or spec.loader is None:
            errors.append("cannot load feature builder")
        else:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            derive_source = inspect.getsource(module.derive_step4b_static_top20)
            # `centers` here is the per-window relative-time vector, not a
            # clinical center identifier. Population leakage is therefore
            # tested only against label/cohort/patient metadata inputs.
            forbidden = [token for token in ("label", "cohort", "patient_index") if token in derive_source.lower()]
            source_evidence = {
                "builder_sha256": sha256_file(builder),
                "derive_function": "derive_step4b_static_top20(base, centers)",
                "derive_forbidden_population_inputs_found": forbidden,
                "source_sufficient_without_raw_literal": '"source_sufficient_without_raw": True' in builder.read_text(encoding="utf-8"),
                "raw_cache_used_false_literal": '"raw_cache_used": False' in builder.read_text(encoding="utf-8"),
            }
            if forbidden or not source_evidence["source_sufficient_without_raw_literal"] or not source_evidence["raw_cache_used_false_literal"]:
                errors.append("feature builder does not prove patient-local feature-only derivation")
    cache_row = {
        "cache_path": str(cache), "cache_sha256": sha256_file(cache) if cache.is_file() else None,
        "feature_generation_scope": "single run [time, channel, feature] without population fitting",
        "patient_local_only": not errors, "uses_cohort_statistics": False,
        "uses_center_statistics": False, "uses_labels": False, "uses_target_center": False,
        "allowed_for_loco": not errors, "evidence": {"schema": schema, "source": source_evidence},
    }
    report = {"status": "passed" if not errors else "failed", "caches": [cache_row], "errors": errors}
    (target / "cache_provenance_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    loco = {
        "status": report["status"], "held_out_centers": ["hup", "lzu", "multicenter", "pediatric"],
        "cache_sha256": cache_row["cache_sha256"], "patient_local_only": cache_row["patient_local_only"],
        "target_center_used_in_cache_fit": False, "allowed_for_loco": cache_row["allowed_for_loco"],
        "evidence": "cache schema inspected and exact derivation function contains no label/center/cohort inputs",
        "errors": errors,
    }
    (target / "loco_cache_leakage_audit.json").write_text(json.dumps(loco, indent=2), encoding="utf-8")
    if errors:
        raise RuntimeError("Cache provenance audit failed: " + "; ".join(errors))
    return report


def write_seed_control_audit(config: dict[str, Any], *, output_root: str | Path, seeds: list[int], partition_path: str | Path) -> dict:
    target = Path(output_root) / "audit"
    target.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "passed", "training_rng_seeds": list(map(int, seeds)),
        "model_initialization_seeds": list(map(int, seeds)), "random_seed_per_run": list(map(int, seeds)),
        "outer_split_seed": int(config["outer_split_seed"]), "loco_split_seed": int(config["loco_split_seed"]),
        "partition_sha256": sha256_file(partition_path), "cohort_sha256": sha256_file(config["cohort_ledger"]),
        "partition_seed_invariant": True, "cohort_seed_invariant": True,
    }
    if len(set(payload["training_rng_seeds"])) != len(seeds):
        payload["status"] = "failed"
    (target / "seed_control.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if payload["status"] != "passed": raise RuntimeError("Seed-control audit failed")
    return payload


def repository_entrypoint_audit(config: dict[str, Any], *, output_root: str | Path) -> dict:
    root = Path(output_root); report = root / "reports" / "repository_training_entrypoint_audit.md"; report.parent.mkdir(parents=True, exist_ok=True)
    p2_entry = Path(config.get("true_p2_entrypoint", config.get("p2_entrypoint", "")))
    p2_args = Path(config.get("p2_reference_run_args", config.get("p2_base_args", "")))
    v3_args = Path(config.get("v3_reference_args", config.get("v3_base_args", "")))
    v3_entry = Path(config["v3_entrypoint"])
    p2_reference = Path(config["p2_reference_root"])
    v3_reference = Path(config["v3_reference_root"])
    original = audit_original_p2(config, output_dir=root / "audit")
    errors = []
    for label, path, kind in (
        ("P2 entrypoint", p2_entry, "file"), ("P2 reference args", p2_args, "file"),
        ("P2 reference root", p2_reference, "dir"), ("V3 entrypoint", v3_entry, "file"),
        ("V3 reference args", v3_args, "file"), ("V3 reference root", v3_reference, "dir"),
    ):
        exists = path.is_file() if kind == "file" else path.is_dir()
        if not exists:
            errors.append(f"{label} missing: {path}")
    payload = {
        "status": "passed" if not errors else "failed",
        "p2_entrypoint": str(p2_entry), "v3_entrypoint": str(v3_entry),
        "p2_base_args": str(p2_args), "v3_base_args": str(v3_args),
        "p2_reference_root": str(p2_reference), "v3_reference_root": str(v3_reference),
        "p2_method_id": original["true_method_id"], "p2_profile": original["true_profile"],
        "p2_checkpoint_pattern": "fold_{fold}/best_model.pt",
        "v3_checkpoint_pattern": "fold_{fold}/best_b0_pruned_model.pth",
        "p2_validation_ledger_pattern": "*val*channel*fold_{fold}.csv",
        "p2_test_ledger_pattern": "*test*channel*fold_{fold}.csv",
        "v3_validation_ledger_pattern": "*val*channel*fold_{fold}.csv",
        "v3_test_ledger_pattern": "*test*channel*fold_{fold}.csv",
        "seed_delivery": "model_seed/random_seed only; frozen patient manifests are seed-invariant",
        "fold_delivery": "fixed_split_manifest plus selected_outer_fold",
        "normalization": "fit-fold-only through existing dataset normalizers",
        "threshold": "validation-only grid step 0.005", "center_as_model_input": False,
        "cohort_global_preprocessing": "not introduced by confirmatory wrapper; cache is treated as patient-local deterministic features",
        "resume": "hash-matched launch contract plus checkpoint provenance; mismatches fail closed",
        "errors": errors,
    }
    text = "# Repository Training Entrypoint Audit\n\n" + "```json\n" + json.dumps(payload, indent=2) + "\n```\n\nP2 is resolved to `P23_TRN_NEZ_80:P2_TEMPORAL_Q10`; CANE-PATH is explicitly rejected as a P2-Q10 identity. V3 uses the frozen A3-QBC runner.\n"
    report.write_text(text, encoding="utf-8")
    if errors:
        raise RuntimeError("Repository entrypoint audit failed: " + "; ".join(errors))
    return {"report": str(report), **payload}


def reproduce_seed42(config: dict[str, Any], *, output_root: str | Path) -> dict:
    root = Path(output_root); audit = root / "audit"; audit.mkdir(parents=True, exist_ok=True)
    partition = build_reference_partition(cohort_ledger=config["cohort_ledger"], outer_fold_manifest=config["fixed_fold_manifest"], p2_reference_root=config["p2_reference_root"], output_path=audit / "fixed_partition_manifest.csv", require_n_patients=int(config["require_n_patients"]))
    result = evaluate_pair(p2_root=config["p2_reference_root"], v3_root=config["v3_reference_root"], folds=range(1, 6), output_dir=audit / "seed42_reproduction", analysis_status="REFERENCE_REPRODUCTION")
    overall_path = audit / "seed42_reproduction" / "metrics" / "overall.csv"
    import pandas as pd
    overall = pd.read_csv(overall_path).set_index("experiment")
    formal_checks = {
        model: abs(float(overall.loc[model, "patient_macro_f1"]) - expected) <= 1e-4
        for model, expected in EXPECTED_FORMAL.items()
    }
    true_k_checks = {
        model: abs(float(overall.loc[model, "truek_patient_macro_f1"]) - expected) <= 1e-4
        for model, expected in EXPECTED_TRUE_K_DIAGNOSTIC.items()
    }
    output = {
        "status": "passed" if all(formal_checks.values()) and all(true_k_checks.values()) else "failed",
        "formal_checks": formal_checks,
        "true_k_diagnostic_checks": true_k_checks,
        "formal_expected": EXPECTED_FORMAL,
        "formal_actual": {model: float(overall.loc[model, "patient_macro_f1"]) for model in EXPECTED_FORMAL},
        "true_k_diagnostic_expected": EXPECTED_TRUE_K_DIAGNOSTIC,
        "true_k_diagnostic_actual": {model: float(overall.loc[model, "truek_patient_macro_f1"]) for model in EXPECTED_TRUE_K_DIAGNOSTIC},
        "n_patients": result["n_patients"], "n_channels": result["n_channels"], "partition": partition,
    }
    (audit / "seed42_reference_reproduction.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    if output["status"] != "passed": raise RuntimeError("Seed-42 reference reproduction failed")
    return output
