from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch

PROJECT = Path(__file__).resolve().parents[2]
REPO = PROJECT.parent
PIPELINE_VERSION = "ngbr_full_v2"
for location in (REPO, PROJECT):
    if str(location) not in sys.path: sys.path.insert(0, str(location))

from neuroez_c.task2.clinical_target import build_clinical_target_lookup
from neuroez_c.task2.data import filtered_cache, load_cache
from neuroez_c.task2.exclusions import load_exclusion_manifest
from neuroez_c.task2.ngbr.audit import BiomarkerCache, cache_signature, file_identity, stable_seed
from neuroez_c.task2.ngbr.biomarker_consensus import build_joint_maps
from neuroez_c.task2.ngbr.channel_alignment import align_p2_to_raw, audit_raw_feature_alignment
from neuroez_c.task2.ngbr.epileptogenicity_index import ALGORITHM_VERSION as EI_VERSION, compute_ei
from neuroez_c.task2.ngbr.evaluation import biomarker_nez_association, bootstrap_ci, metric_table, univariate_audit
from neuroez_c.task2.functional_graph import normalize_channel_name
from neuroez_c.task2.ngbr.hfo_detection import ALGORITHM_VERSION as HFO_VERSION, compute_hfo, hfo_eligibility
from neuroez_c.task2.ngbr.ictal_propagation import ALGORITHM_VERSION as PROP_VERSION, compute_propagation
from neuroez_c.task2.ngbr.low_entropy import ALGORITHM_VERSION as ENTROPY_VERSION, compute_low_entropy
from neuroez_c.task2.ngbr.negative_controls import CONTROLS, permute_channel_map, permute_outcomes, permute_target
from neuroez_c.task2.ngbr.neural_fragility import ALGORITHM_VERSION as FRAGILITY_VERSION, FragilityConfig, compute_fragility
from neuroez_c.task2.ngbr.outcome_model import RIDGE_C, coefficient_rows, fit_outcome_model
from neuroez_c.task2.ngbr.p2_nez_map import permute_nez, records_from_p2_exports
from neuroez_c.task2.ngbr.patient_aggregation import CORE_FEATURES, HFO_FEATURES, aggregate_patients
from neuroez_c.task2.ngbr.preprocessing import hfo_fold_eligible
from neuroez_c.task2.ngbr.schema import RawRunRecord, adapt_feature_record, adapt_raw_record, inspect_cache_schema, validate_target
from neuroez_c.task2.ngbr.seizure_aggregation import aggregate_seizure, top_fraction_set
from neuroez_c.task2.outcomes import load_outcome_table
from neuroez_c.task2.p2_adapter import P23Runtime, P2ExportAdapter, load_p2_args, locate_p2_config
from neuroez_c.task2.protocol import assert_checkpoint_safe, checkpoint_training_subjects, load_fold_manifest, manifest_hash, manifest_patient_keys
from neuroez_c.task2.training import make_loader, prepare_examples, seed_everything


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="NGBR-Full fixed ridge outcome model")
    value.add_argument("--feature_cache", default=os.getenv("DRE_TASK1_FEATURE_CACHE_PATH"), required=os.getenv("DRE_TASK1_FEATURE_CACHE_PATH") is None)
    value.add_argument("--raw_cache", default=os.getenv("DRE_TASK1_RAW_CACHE_PATH"), required=os.getenv("DRE_TASK1_RAW_CACHE_PATH") is None)
    value.add_argument("--outcome_table", default=os.getenv("DRE_TASK2_OUTCOME_TABLE", "cache://patient_index"))
    value.add_argument("--fold_manifest", default=os.getenv("DRE_TASK2_FOLD_MANIFEST"), required=os.getenv("DRE_TASK2_FOLD_MANIFEST") is None)
    value.add_argument("--exclusion_manifest", default=os.getenv("DRE_TASK2_EXCLUSION_MANIFEST", str(PROJECT / "configs" / "data_exclusions.csv")))
    value.add_argument("--p2_checkpoint_root", default=os.getenv("DRE_TASK1_P2_CHECKPOINT_ROOT"), required=os.getenv("DRE_TASK1_P2_CHECKPOINT_ROOT") is None)
    value.add_argument("--p2_runtime_root", default=os.getenv("DRE_TASK1_P2_RUNTIME_ROOT", str(REPO / "P23_TRN_NEZ_80")))
    value.add_argument("--p2_config"); value.add_argument("--p2_training_manifest", default=os.getenv("DRE_TASK1_P2_TRAINING_MANIFEST"))
    value.add_argument("--output_dir", default=os.getenv("DRE_TASK2_NGBR_OUTPUT_DIR"), required=os.getenv("DRE_TASK2_NGBR_OUTPUT_DIR") is None)
    value.add_argument("--biomarker_cache_dir", default=os.getenv("DRE_TASK2_NGBR_CACHE_DIR"), required=os.getenv("DRE_TASK2_NGBR_CACHE_DIR") is None)
    value.add_argument("--seed", type=int, default=42); value.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu"); value.add_argument("--batch_size", type=int, default=8); value.add_argument("--bootstrap_repeats", type=int, default=2000)
    value.add_argument("--disable_hfo", action="store_true"); value.add_argument("--negative_control", choices=CONTROLS, default="none")
    value.add_argument("--audit_only", action="store_true", help="schema/alignment/fold audit without expensive biomarker extraction"); value.add_argument("--audit_biomarkers", action="store_true", help="when used with --audit_only, also compute biomarker coverage"); value.add_argument("--strict", action="store_true"); value.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    value.add_argument("--max_outer_folds", type=int, default=0); value.add_argument("--max_patients", type=int, default=0)
    value.add_argument("--fragility_lambda", type=float, default=1e-5); value.add_argument("--fragility_angles", type=int, default=128)
    for family in ("fragility", "ei", "propagation", "entropy", "hfo", "p2"): value.add_argument(f"--recompute_{family}", action="store_true")
    return value


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_suffix(path.suffix + ".tmp"); temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8"); temporary.replace(path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_suffix(path.suffix + ".tmp"); frame.to_csv(temporary, index=False); temporary.replace(path)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_suffix(path.suffix + ".tmp"); temporary.write_text(value, encoding="utf-8"); temporary.replace(path)


def _balanced_take(subjects: Iterable[str], labels: dict[str, int], count: int) -> list[str]:
    values = sorted(map(str, subjects))
    if count <= 0 or len(values) <= count: return values
    buckets = {label: [p for p in values if labels[p] == label] for label in (0, 1)}; selected: list[str] = []
    while len(selected) < count and any(buckets.values()):
        for label in (0, 1):
            if buckets[label] and len(selected) < count: selected.append(buckets[label].pop(0))
    return selected


def _checkpoint_fold(path: Path) -> int | None:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    for mapping in (payload, payload.get("metadata", {}) if isinstance(payload, dict) else {}):
        if isinstance(mapping, dict):
            for key in ("outer_fold", "fold", "fold_idx"):
                if key in mapping: return int(mapping[key])
    return None


def _extract_p2(adapter: P2ExportAdapter, loader: Any) -> list[dict[str, Any]]:
    adapter.eval(); records = []
    with torch.no_grad():
        for batch in loader:
            export = {key: item.detach().cpu() for key, item in adapter(batch).items() if torch.is_tensor(item)}
            for index, subject in enumerate(batch["subject_id"]):
                channels = int(batch["channel_mask"][index].sum()); names = list(batch["canonical_channels"][index])[:channels]
                records.append({"patient_key": str(subject), "center": str(batch["center"][index]), "channel_names": names, "clinical_target_mask": batch["clinical_target_mask"][index:index + 1, :channels].cpu().bool(), "channel_mask": batch["channel_mask"][index:index + 1, :channels].cpu().bool(), "final_nez_logit": export["final_nez_logit"][index:index + 1, :channels], "direct_nez_logit": export.get("direct_nez_logit", None)[index:index + 1, :channels] if export.get("direct_nez_logit") is not None else None})
    return records


def _load_p2_fold(args: argparse.Namespace, fold: int, train: set[str], test: set[str], runtime: P23Runtime, p2_args: Any, feature_cache: dict, labels: dict[str, int], target_lookup: dict) -> tuple[list, pd.DataFrame, dict]:
    checkpoint = Path(args.p2_checkpoint_root) / f"fold_{fold}" / "best_model.pt"
    if not checkpoint.exists(): raise FileNotFoundError(checkpoint)
    metadata = _checkpoint_fold(checkpoint); metadata_match = metadata == fold
    if args.strict and not metadata_match: raise ValueError(f"P2 checkpoint fold mismatch: expected={fold}, found={metadata}")
    manifest_safe = False
    if args.p2_training_manifest:
        assert_checkpoint_safe(checkpoint_training_subjects(args.p2_training_manifest, fold), test); manifest_safe = True
    elif args.strict: raise ValueError("Strict NGBR requires --p2_training_manifest")
    cache_root = Path(args.biomarker_cache_dir) / "p2"; cache_root.mkdir(parents=True, exist_ok=True); path = cache_root / f"fold_{fold}.pt"
    signature_payload = {"feature_input": file_identity(args.feature_cache), "checkpoint": file_identity(checkpoint), "fold": fold, "train": sorted(train), "test": sorted(test), "algorithm_version": "frozen_p2_final_nez_v1", "negative_control": args.negative_control}
    signature = hashlib.sha256(json.dumps(signature_payload, sort_keys=True).encode()).hexdigest(); exports = None
    if args.resume and not args.recompute_p2 and path.exists():
        cached = torch.load(path, map_location="cpu", weights_only=False)
        if cached.get("signature") == signature: exports = cached["exports"]
    if exports is None:
        examples, _ = prepare_examples(runtime, feature_cache, p2_args, normalizer_subjects=sorted(train), output_subjects=sorted(train | test))
        loader = make_loader(examples, runtime, labels, None, batch_size=args.batch_size, shuffle=False, seed=args.seed, clinical_target_lookup=target_lookup)
        adapter = P2ExportAdapter(runtime, p2_args, checkpoint, device=args.device, limited_finetune=False)
        if any(parameter.requires_grad for parameter in adapter.parameters()): raise RuntimeError("NGBR requires fully frozen P2")
        exports = _extract_p2(adapter, loader); temporary = path.with_suffix(".tmp"); torch.save({"signature": signature, "signature_payload": signature_payload, "exports": exports}, temporary); temporary.replace(path); del adapter
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    records, table = records_from_p2_exports(exports)
    audit = {"outer_fold": fold, "checkpoint": str(checkpoint.resolve()), "checkpoint_outer_fold": metadata, "checkpoint_outer_fold_matches": metadata_match, "training_manifest_leakage_verified": manifest_safe, "P2_frozen": True, "P2_fold_safe": bool(metadata_match and manifest_safe), "n_records": len(records)}
    return records, table.assign(outer_fold=fold), audit


def _compute_biomarkers(args: argparse.Namespace, records: list[RawRunRecord], cache: BiomarkerCache) -> tuple[dict, dict[str, pd.DataFrame]]:
    all_maps: dict[tuple[str, str], dict[str, Any]] = {}; tables: dict[str, list] = {name: [] for name in ("fragility_channel", "fragility_quality", "fragility_window", "ei_channel", "ei_audit", "propagation_channel", "propagation_audit", "entropy_channel", "entropy_audit", "hfo_channel", "hfo_event", "hfo_audit")}
    fragility_config = FragilityConfig(ridge_lambda=args.fragility_lambda, angle_count=args.fragility_angles)
    families = {
        "fragility": (FRAGILITY_VERSION, {"window_sec": .25, "step_sec": .125, "ridge_lambda": args.fragility_lambda, "angle_count": args.fragility_angles}, args.recompute_fragility),
        "ei": (EI_VERSION, {"low_band": [4, 12], "high_band": [12, 127], "threshold": 3, "persistence_sec": .25, "tau": 5}, args.recompute_ei),
        "propagation": (PROP_VERSION, {"window_sec": .25, "step_sec": .05, "threshold": 3, "persistence_sec": .25}, args.recompute_propagation),
        "entropy": (ENTROPY_VERSION, {"preictal": [-10, -1], "subwindow_sec": 1, "artifact_mad": 8}, args.recompute_entropy),
        "hfo": (HFO_VERSION, {"ripple": [80, 250], "fast_ripple": [250, 500], "coincidence_sec": .1, "disabled": args.disable_hfo}, args.recompute_hfo),
    }
    for position, record in enumerate(records, 1):
        print(f"[NGBR biomarkers] {position}/{len(records)} {record.patient_key}/{record.seizure_id}", flush=True); local = {}
        for family, (version, parameters, recompute) in families.items():
            signature, payload = cache_signature(args.raw_cache, patient=record.patient_key, seizure=record.seizure_id, sampling_rate=record.sampling_rate, algorithm_version=version, parameters=parameters, negative_control=args.negative_control)
            value = cache.load(family, record.patient_key, record.seizure_id, signature, recompute=recompute)
            if value is None:
                if family == "fragility": value = compute_fragility(record, fragility_config)
                elif family == "ei": value = compute_ei(record)
                elif family == "propagation": value = compute_propagation(record)
                elif family == "entropy": value = compute_low_entropy(record)
                else: value = compute_hfo(record, disabled=args.disable_hfo)
                cache.save(family, record.patient_key, record.seizure_id, signature, payload, value)
            local[family] = value
        fragility, fragility_channel, fragility_quality = local["fragility"]; ei, ei_channel, ei_audit = local["ei"]; propagation, propagation_channel, propagation_audit = local["propagation"]; entropy, entropy_channel, entropy_audit = local["entropy"]; hfo_result = local["hfo"]
        if args.negative_control == "CHANNEL_PERMUTATION":
            for offset, (name, values) in enumerate((("fragility", fragility), ("ei", ei), ("propagation", propagation), ("entropy", entropy), ("hfo", hfo_result.score))):
                permuted = permute_channel_map(values, record.valid_channel_mask, stable_seed(args.seed, record.patient_key, record.seizure_id, name));
                if name == "fragility": fragility = permuted
                elif name == "ei": ei = permuted
                elif name == "propagation": propagation = permuted
                elif name == "entropy": entropy = permuted
                else:
                    hfo_result.score = permuted
                    hfo_result.hub = permute_channel_map(hfo_result.hub, record.valid_channel_mask, stable_seed(args.seed, record.patient_key, record.seizure_id, name))
            for frame, column, values in ((fragility_channel, "fragility", fragility), (ei_channel, "ei", ei), (propagation_channel, "propagation", propagation), (entropy_channel, "low_entropy", entropy), (hfo_result.channel_table, "hfo", hfo_result.score)):
                if not frame.empty:
                    mapping = dict(zip(record.channel_names, values)); frame[column] = frame["channel"].map(mapping)
        all_maps[(record.patient_key, record.seizure_id)] = {"fragility": fragility, "ei": ei, "propagation": propagation, "entropy": entropy, "hfo": hfo_result.score, "hfo_hub": hfo_result.hub, "propagation_audit": propagation_audit, "fragility_quality": fragility_quality, "ei_audit": ei_audit, "entropy_audit": entropy_audit, "hfo_audit": hfo_result.quality}
        window_audit = fragility_quality.pop("window_audit", [])
        tables["fragility_channel"].append(fragility_channel); tables["fragility_quality"].append(fragility_quality); tables["fragility_window"].extend(window_audit); tables["ei_channel"].append(ei_channel); tables["ei_audit"].append(ei_audit); tables["propagation_channel"].append(propagation_channel); tables["propagation_audit"].append(propagation_audit); tables["entropy_channel"].append(entropy_channel); tables["entropy_audit"].append(entropy_audit); tables["hfo_channel"].append(hfo_result.channel_table); tables["hfo_event"].append(hfo_result.event_table); tables["hfo_audit"].append(hfo_result.quality)
    frames = {key: (pd.concat(value, ignore_index=True) if value and isinstance(value[0], pd.DataFrame) and any(not x.empty for x in value) else pd.DataFrame([x for x in value if isinstance(x, dict)])) for key, value in tables.items()}
    return all_maps, frames


def _coverage(frame: pd.DataFrame, column: str) -> float:
    return float(frame[column].astype(bool).mean()) if not frame.empty and column in frame else 0.0


def main() -> int:
    args = parser().parse_args(); output = Path(args.output_dir).expanduser(); output.mkdir(parents=True, exist_ok=True); seed_everything(args.seed); _atomic_json(output / "run_args.json", vars(args))
    exclusions = load_exclusion_manifest(args.exclusion_manifest); feature_cache = filtered_cache(load_cache(args.feature_cache), exclusions); raw_cache = filtered_cache(load_cache(args.raw_cache), exclusions)
    _atomic_json(output / "ngbr_feature_cache_schema.json", inspect_cache_schema(feature_cache, "feature")); _atomic_json(output / "ngbr_raw_cache_schema.json", inspect_cache_schema(raw_cache, "raw"))
    outcomes, _ = load_outcome_table(args.outcome_table, cache=feature_cache); outcomes = outcomes[outcomes.outcome_group.isin(["success", "failure"])]
    outcomes = outcomes[outcomes.patient_key.isin(manifest_patient_keys(args.fold_manifest))].copy(); folds = load_fold_manifest(args.fold_manifest, outcomes, strict=args.strict); cohort = set(folds.patient_key); outcomes = outcomes[outcomes.patient_key.isin(cohort)].copy(); labels = outcomes.set_index("patient_key").outcome_label.astype(int).to_dict()
    if args.max_patients: cohort = set(_balanced_take(cohort, labels, args.max_patients)); outcomes = outcomes[outcomes.patient_key.isin(cohort)].copy(); folds = folds[folds.patient_key.isin(cohort)].copy()
    target_lookup, target_alignment, target_distribution = build_clinical_target_lookup(feature_cache, sorted(cohort), strict=args.strict)
    if args.negative_control == "TARGET_PERMUTATION":
        target_lookup = copy.deepcopy(target_lookup)
        for patient, entry in target_lookup.items():
            mask = np.asarray(entry.get("clinical_target_mask", entry.get("mask")), dtype=bool); valid = np.ones(mask.shape, dtype=bool); permuted = permute_target(mask, valid, stable_seed(args.seed, patient, "target"))
            if "clinical_target_mask" in entry: entry["clinical_target_mask"] = permuted
            if "mask" in entry: entry["mask"] = permuted
    feature_records, feature_axis = [], []
    for record in feature_cache["run_records"]:
        if str(record.get("subject_id")) in cohort: value, audit = adapt_feature_record(record, target_lookup); validate_target(value); feature_records.append(value); feature_axis.append(audit)
    raw_records, raw_axis = [], []
    for record in raw_cache["run_records"]:
        if str(record.get("subject_id")) in cohort: value, audit = adapt_raw_record(record, target_lookup); validate_target(value); raw_records.append(value); raw_axis.append(audit)
    alignment = audit_raw_feature_alignment(feature_records, raw_records, strict=args.strict); _atomic_csv(output / "ngbr_feature_axis_audit.csv", pd.DataFrame(feature_axis)); _atomic_csv(output / "ngbr_raw_axis_audit.csv", pd.DataFrame(raw_axis)); _atomic_csv(output / "ngbr_channel_alignment_audit.csv", alignment); _atomic_csv(output / "ngbr_sampling_rate_audit.csv", pd.DataFrame([{"patient_key": x.patient_key, "seizure_id": x.seizure_id, "sampling_rate": x.sampling_rate, "ripple_eligible": x.sampling_rate >= 1000, "fast_ripple_eligible": x.sampling_rate >= 2000, "hfo_eligible": hfo_eligibility(x)[0], "hfo_invalid_reason": hfo_eligibility(x)[1]} for x in raw_records])); _atomic_csv(output / "clinical_target_alignment_audit.csv", target_alignment); _atomic_csv(output / "clinical_target_distribution_audit.csv", target_distribution)
    biomarker_maps: dict = {}; biomarker_tables = {key: pd.DataFrame() for key in ("fragility_channel", "fragility_quality", "fragility_window", "ei_channel", "ei_audit", "propagation_channel", "propagation_audit", "entropy_channel", "entropy_audit", "hfo_channel", "hfo_event", "hfo_audit")}
    if not args.audit_only or args.audit_biomarkers:
        biomarker_maps, biomarker_tables = _compute_biomarkers(args, raw_records, BiomarkerCache(args.biomarker_cache_dir, args.resume))
        file_map = {"fragility_channel": "ngbr_fragility_channel_map.csv", "fragility_quality": "ngbr_fragility_quality_audit.csv", "fragility_window": "ngbr_fragility_window_audit.csv", "ei_channel": "ngbr_ei_channel_map.csv", "ei_audit": "ngbr_ei_detection_audit.csv", "propagation_channel": "ngbr_propagation_channel_map.csv", "propagation_audit": "ngbr_propagation_seizure_audit.csv", "entropy_channel": "ngbr_low_entropy_channel_map.csv", "entropy_audit": "ngbr_low_entropy_quality_audit.csv", "hfo_channel": "ngbr_hfo_channel_map.csv", "hfo_event": "ngbr_hfo_event_audit.csv", "hfo_audit": "ngbr_hfo_quality_audit.csv"}
        for key, name in file_map.items(): _atomic_csv(output / name, biomarker_tables[key])
    fold_values = sorted(map(int, folds.outer_fold.unique()))[:args.max_outer_folds or None]; p2_safety_preview = []
    for fold in fold_values:
        checkpoint = Path(args.p2_checkpoint_root) / f"fold_{fold}" / "best_model.pt"; test = set(folds.loc[folds.outer_fold == fold, "patient_key"]); metadata = _checkpoint_fold(checkpoint) if checkpoint.exists() else None; safe = False
        if checkpoint.exists() and args.p2_training_manifest: assert_checkpoint_safe(checkpoint_training_subjects(args.p2_training_manifest, fold), test); safe = metadata == fold
        if args.strict and (not checkpoint.exists() or metadata != fold or not args.p2_training_manifest): raise ValueError(f"P2_FOLD_SAFETY_FAILED: fold={fold}, checkpoint_exists={checkpoint.exists()}, metadata={metadata}")
        p2_safety_preview.append({"outer_fold": fold, "checkpoint_exists": checkpoint.exists(), "checkpoint_outer_fold": metadata, "checkpoint_outer_fold_matches": metadata == fold, "training_manifest_leakage_verified": bool(args.p2_training_manifest), "P2_fold_safe": safe})
    if args.audit_only:
        _atomic_json(output / "ngbr_p2_fold_safety_audit.json", p2_safety_preview); protocol = {"pipeline": "NGBR-Full", "fixed_outer_folds": 5, "completed_outer_folds": 0, "success_label": 1, "failure_label": 0, "outer_test_used_for_selection": False, "threshold": .5, "ridge_C": RIDGE_C, "center_as_input": False, "patient_id_as_input": False, "channel_count_as_input": False, "seizure_count_as_input": False, "P2_fold_safe": all(x["P2_fold_safe"] for x in p2_safety_preview), "P2_frozen": True, "train_test_overlap": 0, "manifest_hash": manifest_hash(folds), "negative_control": args.negative_control, "preprocessing_fold_safe": True, "paper_valid": False, "audit_only": True}; _atomic_json(output / "ngbr_protocol_audit.json", protocol); _atomic_text(output / "NGBR_OUTCOME_REPORT.md", "# NGBR-Full Audit\n\nAUDIT_ONLY_NO_OUTCOME_MODEL_RUN\n"); return 0
    runtime = P23Runtime(args.p2_runtime_root); p2_args = load_p2_args(locate_p2_config(args.p2_checkpoint_root, args.p2_config), feature_cache=args.feature_cache); raw_by_patient: dict[str, list[RawRunRecord]] = {}
    for record in raw_records: raw_by_patient.setdefault(record.patient_key, []).append(record)
    prediction_frames = []; fold_metrics = []; coefficient = []; preprocessing = []; fold_details = []; p2_safety = []; assigned_p2 = []; assigned_joint = []; assigned_seizure = []; assigned_patient = []; assigned_recurrence = []; assigned_p2_alignment = []; force_recompute = any(getattr(args, f"recompute_{name}") for name in ("fragility", "ei", "propagation", "entropy", "hfo", "p2"))
    for fold in fold_values:
        test = set(folds.loc[folds.outer_fold == fold, "patient_key"]); train = cohort - test; fold_dir = output / f"fold_{fold}"; fold_dir.mkdir(exist_ok=True); result_path = fold_dir / "fold_result.pkl"
        signature = hashlib.sha256(json.dumps({"fold": fold, "train": sorted(train), "test": sorted(test), "seed": args.seed, "control": args.negative_control, "disable_hfo": args.disable_hfo, "feature": file_identity(args.feature_cache), "raw": file_identity(args.raw_cache), "fragility_lambda": args.fragility_lambda, "fragility_angles": args.fragility_angles, "algorithm_versions": {"pipeline": PIPELINE_VERSION, "fragility": FRAGILITY_VERSION, "ei": EI_VERSION, "propagation": PROP_VERSION, "entropy": ENTROPY_VERSION, "hfo": HFO_VERSION}}, sort_keys=True).encode()).hexdigest()
        if args.resume and not force_recompute and result_path.exists():
            with result_path.open("rb") as handle: bundle = pickle.load(handle)
            if bundle.get("signature") == signature:
                prediction_frames.append(bundle["prediction"]); fold_metrics.append(bundle["metric"]); coefficient.extend(bundle["coefficient"]); preprocessing.extend(bundle["preprocessing"]); fold_details.append(bundle["fold_detail"]); p2_safety.append(bundle["p2_safety"]); assigned_p2.append(bundle["assigned_p2"]); assigned_joint.append(bundle["assigned_joint"]); assigned_seizure.append(bundle["assigned_seizure"]); assigned_patient.append(bundle["assigned_patient"]); assigned_recurrence.append(bundle["assigned_recurrence"]); assigned_p2_alignment.extend(bundle.get("p2_alignment", [])); print(f"[NGBR] fold {fold}: resumed", flush=True); continue
            raise FileExistsError(f"OUTPUT_SIGNATURE_CONFLICT: {result_path} belongs to different inputs/parameters; use a new --output_dir")
        p2_records, p2_table, safety = _load_p2_fold(args, fold, train, test, runtime, p2_args, feature_cache, labels, target_lookup); p2_by_patient = {x.patient_key: x for x in p2_records}
        if args.negative_control == "NEZ_PERMUTATION":
            p2_by_patient = {key: permute_nez(value, stable_seed(args.seed, key, "nez")) for key, value in p2_by_patient.items()}
            p2_table = pd.DataFrame([{"patient_key": value.patient_key, "center": value.center, "channel": value.channel_names[index], "q_nez": value.final_nez_probability[index], "direct_nez_probability_audit": value.direct_nez_probability[index] if value.direct_nez_probability is not None else np.nan, "valid": int(value.valid_channel_mask[index]), "clinical_target": int(value.clinical_target_mask[index]), "outer_fold": fold} for value in p2_by_patient.values() for index in range(len(value.channel_names))])
        seizure_rows, joint_frames, p2_alignment_rows = [], [], []
        for patient in sorted(train | test):
            for raw in raw_by_patient.get(patient, []):
                source_p2 = p2_by_patient[patient]; p2_names = {normalize_channel_name(x) for x in source_p2.channel_names}; raw_names = {normalize_channel_name(x) for x in raw.channel_names}; p2_rate = len(p2_names & raw_names) / max(len(raw_names), 1)
                if patient in test: p2_alignment_rows.append({"alignment_type": "p2_raw", "patient_key": patient, "seizure_id": raw.seizure_id, "outer_fold": fold, "n_p2_channels": len(p2_names), "n_raw_channels": len(raw_names), "n_matched_channels": len(p2_names & raw_names), "match_rate": p2_rate, "missing_p2_channels": ";".join(sorted(raw_names - p2_names)), "missing_raw_channels": ";".join(sorted(p2_names - raw_names))})
                p2 = align_p2_to_raw(source_p2, raw, strict=args.strict); local = biomarker_maps[(raw.patient_key, raw.seizure_id)]; maps, joint = build_joint_maps(raw, p2, local["fragility"], local["ei"], local["propagation"], local["entropy"], local["hfo"]); seizure_rows.append(aggregate_seizure(maps, raw, p2, local["propagation_audit"], local["hfo_hub"])); joint_frames.append(joint)
        patient_table, definition, recurrence = aggregate_patients(seizure_rows, p2_by_patient); patient_table = patient_table.merge(outcomes[["patient_key", "outcome_label"]].rename(columns={"outcome_label": "outcome_true"}), on="patient_key", validate="one_to_one").merge(folds, on="patient_key", validate="one_to_one")
        train_frame = patient_table[patient_table.patient_key.isin(train)].reset_index(drop=True); test_frame = patient_table[patient_table.patient_key.isin(test)].reset_index(drop=True); hfo_enabled, hfo_audit = hfo_fold_eligible(train_frame); hfo_enabled = bool(hfo_enabled and not args.disable_hfo); features = list(CORE_FEATURES) + (list(HFO_FEATURES) if hfo_enabled else [])
        y = train_frame.outcome_true.to_numpy(dtype=int)
        if args.negative_control == "OUTCOME_PERMUTATION": y = permute_outcomes(y, args.seed + fold)
        fitted = fit_outcome_model(train_frame, y, features, outer_fold=fold, seed=args.seed + fold, hfo_enabled=hfo_enabled); probability = fitted.predict_probability(test_frame); prediction = test_frame[["patient_key", "center", "outer_fold", "outcome_true", "n_valid_seizures", "n_fragility_valid_seizures", "n_ei_valid_seizures", "n_entropy_valid_seizures", "n_hfo_valid_seizures"]].copy(); prediction["probability_success"] = probability; prediction["prediction"] = (probability >= .5).astype(int); prediction["seed"] = args.seed; prediction["hfo_enabled_in_fold"] = hfo_enabled
        metric = {"outer_fold": fold, "n_patients": len(prediction), **metric_table(prediction).iloc[0].drop(labels="scope").to_dict()}; coeff = coefficient_rows(fitted, fold); prep = fitted.preprocessor.audit; detail = {"outer_fold": fold, "n_train": len(train), "n_test": len(test), "train_test_overlap": len(train & test), **hfo_audit, "hfo_enabled_in_fold": hfo_enabled, "P2_fold_safe": safety["P2_fold_safe"], "preprocessing_fit_partition": "outer_train_only", "threshold": .5, "ridge_C": RIDGE_C}
        assigned_p2_fold = p2_table[p2_table.patient_key.isin(test)].copy(); assigned_joint_fold = pd.concat(joint_frames, ignore_index=True); assigned_joint_fold = assigned_joint_fold[assigned_joint_fold.patient_key.isin(test)]; assigned_seizure_fold = pd.DataFrame(seizure_rows); assigned_seizure_fold = assigned_seizure_fold[assigned_seizure_fold.patient_key.isin(test)].map(lambda x: ";".join(sorted(x)) if isinstance(x, set) else x); assigned_patient_fold = patient_table[patient_table.patient_key.isin(test)].copy(); assigned_recurrence_fold = recurrence[recurrence.patient_key.isin(test)].copy()
        bundle = {"signature": signature, "prediction": prediction, "metric": metric, "coefficient": coeff, "preprocessing": prep, "fold_detail": detail, "p2_safety": safety, "assigned_p2": assigned_p2_fold, "assigned_joint": assigned_joint_fold, "assigned_seizure": assigned_seizure_fold, "assigned_patient": assigned_patient_fold, "assigned_recurrence": assigned_recurrence_fold, "p2_alignment": p2_alignment_rows, "model": fitted}
        temporary = result_path.with_suffix(".tmp");
        with temporary.open("wb") as handle: pickle.dump(bundle, handle, pickle.HIGHEST_PROTOCOL)
        temporary.replace(result_path); prediction_frames.append(prediction); fold_metrics.append(metric); coefficient.extend(coeff); preprocessing.extend(prep); fold_details.append(detail); p2_safety.append(safety); assigned_p2.append(assigned_p2_fold); assigned_joint.append(assigned_joint_fold); assigned_seizure.append(assigned_seizure_fold); assigned_patient.append(assigned_patient_fold); assigned_recurrence.append(assigned_recurrence_fold); assigned_p2_alignment.extend(p2_alignment_rows); print(f"[NGBR] fold {fold}: complete, HFO={hfo_enabled}", flush=True)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    if predictions.patient_key.duplicated().any(): raise ValueError("NGBR OOF contains duplicate patients")
    patient_output = pd.concat(assigned_patient, ignore_index=True); joint_output = pd.concat(assigned_joint, ignore_index=True); base_alignment = alignment.assign(alignment_type="raw_feature"); _atomic_csv(output / "ngbr_channel_alignment_audit.csv", pd.concat([base_alignment, pd.DataFrame(assigned_p2_alignment)], ignore_index=True, sort=False)); _atomic_csv(output / "oof_patient_predictions.csv", predictions); _atomic_csv(output / "fold_metrics.csv", pd.DataFrame(fold_metrics)); _atomic_csv(output / "center_metrics.csv", metric_table(predictions, "center")); _atomic_csv(output / "summary_metrics.csv", metric_table(predictions)); _atomic_csv(output / "bootstrap_ci.csv", bootstrap_ci(predictions, args.bootstrap_repeats, args.seed)); _atomic_csv(output / "ngbr_p2_nez_channel_map.csv", pd.concat(assigned_p2, ignore_index=True)); _atomic_json(output / "ngbr_p2_fold_safety_audit.json", p2_safety); _atomic_csv(output / "ngbr_joint_channel_maps.csv", joint_output); _atomic_csv(output / "ngbr_seizure_features.csv", pd.concat(assigned_seizure, ignore_index=True)); _atomic_csv(output / "ngbr_patient_features.csv", patient_output); _atomic_csv(output / "ngbr_patient_feature_definition.csv", definition); _atomic_csv(output / "ngbr_cross_seizure_recurrence_audit.csv", pd.concat(assigned_recurrence, ignore_index=True)); _atomic_csv(output / "ngbr_preprocessing_audit.csv", pd.DataFrame(preprocessing)); _atomic_csv(output / "ngbr_model_coefficients.csv", pd.DataFrame(coefficient)); _atomic_csv(output / "ngbr_univariate_biomarker_audit.csv", univariate_audit(patient_output, list(CORE_FEATURES) + list(HFO_FEATURES), repeats=args.bootstrap_repeats, seed=args.seed)); _atomic_csv(output / "ngbr_biomarker_nez_association_audit.csv", biomarker_nez_association(joint_output))
    complete = len(fold_values) == 5 and set(predictions.patient_key) == cohort; p2_safe = all(x["P2_fold_safe"] for x in p2_safety); no_overlap = all(x["train_test_overlap"] == 0 for x in fold_details); paper_valid = bool(args.strict and complete and args.negative_control == "none" and p2_safe and no_overlap)
    protocol = {"pipeline": "NGBR-Full", "fixed_outer_folds": 5, "completed_outer_folds": len(fold_values), "success_label": 1, "failure_label": 0, "outer_test_used_for_selection": False, "threshold": .5, "ridge_C": RIDGE_C, "center_as_input": False, "patient_id_as_input": False, "channel_count_as_input": False, "seizure_count_as_input": False, "P2_fold_safe": p2_safe, "P2_frozen": True, "train_test_overlap": int(sum(x["train_test_overlap"] for x in fold_details)), "manifest_hash": manifest_hash(folds), "negative_control": args.negative_control, "preprocessing_fold_safe": True, "folds": fold_details, "paper_valid": paper_valid}; _atomic_json(output / "ngbr_protocol_audit.json", protocol)
    pooled = metric_table(predictions).iloc[0].to_dict(); flags = []
    coverages = {"fragility": _coverage(biomarker_tables["fragility_quality"], "fragility_valid"), "ei": _coverage(biomarker_tables["ei_audit"], "ei_valid"), "propagation": _coverage(biomarker_tables["propagation_audit"], "propagation_valid"), "entropy": _coverage(biomarker_tables["entropy_audit"], "entropy_valid")}
    if any(value < .60 for value in coverages.values()): flags.append("CORE_BIOMARKER_COVERAGE_INSUFFICIENT")
    if not biomarker_tables["fragility_quality"].empty and biomarker_tables["fragility_quality"].optimization_failure_fraction.mean() > .20: flags.append("FRAGILITY_ESTIMATION_UNSTABLE")
    if coverages["ei"] < .50: flags.append("EI_NOT_ROBUST_FOR_CURRENT_COHORT")
    if coverages["entropy"] < .70: flags.append("LOW_ENTROPY_DATA_QUALITY_INSUFFICIENT")
    hfo_patients = patient_output.loc[patient_output.n_hfo_valid_seizures > 0, "patient_key"].nunique()
    if hfo_patients < 30: flags.append("HFO_EXTENSION_UNDERPOWERED")
    if pooled.get("auroc", 0) < .60: flags.append("NO_RELIABLE_OUTCOME_SIGNAL")
    if pooled.get("auroc", 0) >= .65 and pooled.get("balanced_accuracy", 0) >= .60: flags.append("MODERATE_BIOMARKER_OUTCOME_SIGNAL")
    if pooled.get("auroc", 0) >= .72 and sum(float(x.get("auroc", 0)) > .5 for x in fold_metrics) >= 4: flags.append("PROMISING_MULTIBIOMARKER_SIGNAL")
    if pooled.get("accuracy", 0) >= .80 and pooled.get("balanced_accuracy", 0) >= .75 and pooled.get("macro_f1", 0) >= .75 and pooled.get("auroc", 0) >= .80 and pooled.get("failure_recall", 0) >= .70: flags.append("STRONG_RESULT_REQUIRES_LOCKED_OR_EXTERNAL_VALIDATION")
    status = [] if paper_valid else ["OUTER_CV_NOT_PAPER_VALID"]
    report = ["# NGBR-Full Outcome Report", "", "NGBR-Full derives channel-resolved maps of neural fragility, epileptogenicity, ictal propagation, preictal low entropy, and, when technically valid, pathological high-frequency activity.", "", "P2 NEZ is an auxiliary fold-safe localization signal. The clinical target is not interpreted as biological true EZ or guaranteed complete resection.", "", "No outer-test information is used for feature construction, preprocessing, model fitting, or threshold selection.", "", "## Status", *(f"- {value}" for value in status + flags), "", "## Coverage", *(f"- {key}: {value:.3f}" for key, value in coverages.items())]; _atomic_text(output / "NGBR_OUTCOME_REPORT.md", "\n".join(report) + "\n")
    print(json.dumps({"pipeline": "NGBR-Full", "n_predictions": len(predictions), "folds": fold_values, "output": str(output.resolve()), "paper_valid": paper_valid}, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
