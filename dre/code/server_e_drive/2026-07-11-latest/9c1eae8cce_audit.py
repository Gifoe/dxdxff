from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .functional_graph import PHASES, is_brain_channel, normalize_channel_name
from .exclusions import apply_exclusions, build_exclusion_audit, load_exclusion_manifest
from .outcomes import load_outcome_table
from .protocol import ProtocolError, load_fold_manifest, manifest_patient_keys


def _sample(record: Mapping[str, Any]) -> Mapping[str, Any]:
    value = record.get("sample")
    return value if isinstance(value, Mapping) else {}


def _record_key(record: Mapping[str, Any]) -> tuple[str, str, str]:
    sample = _sample(record)
    return str(record.get("subject_id", "")), str(record.get("run_id", "")), str(sample.get("sample_id", record.get("run_id", "")))


def _channels(record: Mapping[str, Any]) -> list[str]:
    sample = _sample(record)
    return [normalize_channel_name(value) for value in record.get("channel_names_norm", sample.get("channel_names_norm", []))]


def cache_schema(cache: Mapping[str, Any], path: str | Path, kind: str) -> dict[str, Any]:
    record = cache.get("run_records", [{}])[0] if cache.get("run_records") else {}
    sample = _sample(record)
    return {
        "cache_kind": kind,
        "source_path": str(Path(path).expanduser().resolve()),
        "top_level_keys": sorted(map(str, cache)),
        "n_run_records": len(cache.get("run_records", [])),
        "n_patient_index": len(cache.get("patient_index", {})),
        "sample_record_keys": sorted(map(str, record)),
        "sample_keys": sorted(map(str, sample)),
        "sample_array_contract": {str(key): {"shape": list(value.shape), "dtype": str(value.dtype)} for key, value in sample.items() if isinstance(value, np.ndarray)},
    }


def _phase_counts(centers: np.ndarray) -> dict[str, int]:
    return {
        "preictal": int(np.sum(centers < 0.0)),
        "onset": int(np.sum((centers >= 0.0) & (centers <= 10.0))),
        "spread": int(np.sum((centers > 10.0) & (centers <= 30.0))),
    }


def _metadata_fields(patient_entry: Mapping[str, Any]) -> set[str]:
    fields = set(map(str, patient_entry.keys()))
    channel_meta = patient_entry.get("channel_meta", [])
    if isinstance(channel_meta, list):
        for item in channel_meta:
            if isinstance(item, Mapping):
                fields.update(map(str, item.keys()))
    return fields


def audit_inputs(
    feature_cache: Mapping[str, Any],
    raw_cache: Mapping[str, Any],
    *,
    feature_path: str | Path,
    raw_path: str | Path,
    outcome_table: str | Path | None,
    fold_manifest: str | Path | None,
    p2_checkpoint_root: str | Path | None,
    output_dir: str | Path,
    exclusion_manifest: str | Path | None = None,
    strict: bool = False,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    feature_records_all = feature_cache.get("run_records", [])
    raw_records_all = raw_cache.get("run_records", [])
    exclusions = load_exclusion_manifest(exclusion_manifest)
    declared_exclusion_audit = build_exclusion_audit(exclusions, feature_records_all, raw_records_all)
    feature_records, _ = apply_exclusions(feature_records_all, exclusions)
    raw_records, _ = apply_exclusions(raw_records_all, exclusions)
    outcomes, outcome_candidates = load_outcome_table(outcome_table, cache=feature_cache)
    if fold_manifest:
        fixed_cohort = manifest_patient_keys(fold_manifest)
        outcomes = outcomes[outcomes["patient_key"].isin(fixed_cohort)].reset_index(drop=True)
    conflicts = outcomes[outcomes["outcome_group"] == "conflict"]
    unknown = outcomes[outcomes["outcome_group"] == "unknown"]
    if strict and (not conflicts.empty or not unknown.empty):
        raise ValueError(f"Strict outcome audit failed: conflict={len(conflicts)}, unknown={len(unknown)}")
    fold_rows = pd.DataFrame(columns=("patient_key", "outer_fold"))
    if fold_manifest:
        fold_rows = load_fold_manifest(fold_manifest, outcomes, strict=strict)
        cohort = set(fold_rows["patient_key"])
        feature_records = [record for record in feature_records if str(record.get("subject_id", "")) in cohort]
        raw_records = [record for record in raw_records if str(record.get("subject_id", "")) in cohort]
    feature_lookup = {_record_key(record): record for record in feature_records}
    raw_lookup = {_record_key(record): record for record in raw_records}
    keys = sorted(set(feature_lookup) | set(raw_lookup))
    feature_only = sorted(set(feature_lookup) - set(raw_lookup))
    raw_only = sorted(set(raw_lookup) - set(feature_lookup))
    duplicate_feature_keys = len(feature_lookup) != len(feature_records)
    duplicate_raw_keys = len(raw_lookup) != len(raw_records)
    if strict and (feature_only or raw_only):
        raise ValueError(
            "Strict raw/feature alignment failed; "
            f"feature_only={feature_only[:20]}, raw_only={raw_only[:20]}"
        )
    if strict and (duplicate_feature_keys or duplicate_raw_keys):
        raise ValueError("Strict audit found duplicate patient/seizure/sample cache keys")
    patient_run_counts = Counter(key[0] for key in feature_lookup)
    patient_raw_counts = Counter(key[0] for key in raw_lookup)
    outcome_lookup = outcomes.set_index("patient_key").to_dict("index")
    patient_rows = []
    audited_patients = set(fold_rows["patient_key"]) if not fold_rows.empty else set(feature_cache.get("patient_index", {})) | set(raw_cache.get("patient_index", {})) | set(outcome_lookup)
    for patient in sorted(audited_patients):
        row = outcome_lookup.get(patient, {})
        patient_entry = feature_cache.get("patient_index", {}).get(patient, {})
        canonical = patient_entry.get("canonical_channels", [])
        metadata_fields = _metadata_fields(patient_entry)
        cached_success = patient_entry.get("surgery_success")
        expected_success = row.get("outcome_group") == "success"
        patient_rows.append({
            "patient_id": patient, "patient_key": patient,
            "center": row.get("center", patient.split(":", 1)[0] if ":" in patient else "unknown"),
            "outcome_group": row.get("outcome_group", "unknown"), "outcome_label_success": row.get("outcome_label", np.nan),
            "n_feature_seizures": patient_run_counts[patient], "n_raw_seizures": patient_raw_counts[patient],
            "n_canonical_channels": len(canonical), "feature_present": patient in feature_cache.get("patient_index", {}), "raw_present": patient in raw_cache.get("patient_index", {}),
            "outcome_source": patient_entry.get("outcome_source", row.get("source", "cache")),
            "outcome_stage_field_present": bool({"engel_score", "ilae_score", "outcome_raw"} & metadata_fields),
            "follow_up_field_present": any("followup" in field.lower().replace("_", "") for field in metadata_fields),
            "surgery_success_consistent": bool(cached_success is None or bool(cached_success) == expected_success),
        })
    seizure_rows, channel_rows, phase_rows, exclusion_rows = [], [], [], []
    for key in keys:
        feature_record, raw_record = feature_lookup.get(key), raw_lookup.get(key)
        feature_channels, raw_channels = _channels(feature_record or {}), _channels(raw_record or {})
        feature_sample, raw_sample = _sample(feature_record or {}), _sample(raw_record or {})
        centers = np.asarray(feature_sample.get("window_relative_centers_sec", []), dtype=float).reshape(-1)
        raw_waveform = np.asarray(raw_sample.get("raw_waveform", []))
        channel_meta = (feature_record or {}).get("channel_meta", [])
        channel_meta_fields = {
            str(field).lower()
            for item in channel_meta if isinstance(item, Mapping)
            for field in item.keys()
        } if isinstance(channel_meta, list) else set()
        status = "matched" if feature_record is not None and raw_record is not None else "feature_only" if feature_record is not None else "raw_only"
        seizure_rows.append({
            "patient_key": key[0], "seizure_id": key[1], "sample_id": key[2], "alignment_status": status,
            "n_feature_channels": len(feature_channels), "n_raw_channels": len(raw_channels), "n_shared_channels": len(set(feature_channels) & set(raw_channels)),
            "raw_present": raw_waveform.ndim == 2 and raw_waveform.size > 0,
            "raw_shape": str(tuple(raw_waveform.shape)) if raw_waveform.size else "", "sampling_frequency": raw_sample.get("raw_temporal_sfreq", (raw_record or {}).get("sfreq")),
            "window_timing_present": bool(centers.size), "phase_assignment_present": bool(centers.size),
            "bad_channel_field_present": bool({"bad", "is_bad", "status", "good", "is_valid"} & channel_meta_fields),
            "channel_names_cross_cache_equal": feature_channels == raw_channels,
        })
        counts = _phase_counts(centers)
        phase_rows.append({"patient_key": key[0], "seizure_id": key[1], **{f"n_{phase}_windows": counts[phase] for phase in PHASES}, **{f"has_{phase}": counts[phase] > 0 for phase in PHASES}})
        for index, channel in enumerate(feature_channels):
            channel_rows.append({"patient_key": key[0], "seizure_id": key[1], "channel_index": index, "channel_id": channel, "brain_channel": is_brain_channel(channel)})
        if status != "matched":
            exclusion_rows.append({"patient_key": key[0], "seizure_id": key[1], "level": "seizure", "reason": status})
    if fold_manifest:
        missing_outcomes = sorted(set(fold_rows["patient_key"]) - set(outcomes["patient_key"]))
        if strict and missing_outcomes:
            raise ProtocolError(f"Fold patients missing outcomes: {missing_outcomes[:20]}")
        excluded = sorted(set(outcomes.loc[outcomes["outcome_group"].isin(["success", "failure"]), "patient_key"]) - set(fold_rows["patient_key"]))
        exclusion_rows.extend({"patient_key": patient, "seizure_id": "", "level": "patient", "reason": "not_in_fixed_task2_fold_manifest"} for patient in excluded)
    checkpoint_rows = []
    if p2_checkpoint_root:
        root = Path(p2_checkpoint_root).expanduser()
        for fold in sorted(fold_rows["outer_fold"].unique().tolist()) if not fold_rows.empty else range(1, 6):
            path = root / f"fold_{int(fold)}" / "best_model.pt"
            checkpoint_rows.append({"outer_fold": int(fold), "checkpoint_path": str(path.resolve()), "checkpoint_exists": path.exists(), "training_subject_manifest_verified": False, "embedding_export_verified": False})
        if strict and checkpoint_rows and not all(row["checkpoint_exists"] for row in checkpoint_rows):
            raise FileNotFoundError("One or more fold-specific P2 checkpoints are missing")
    patient_frame, seizure_frame, channel_frame, phase_frame = map(pd.DataFrame, (patient_rows, seizure_rows, channel_rows, phase_rows))
    patient_frame.to_csv(output / "task2_npam_patient_audit.csv", index=False)
    seizure_frame.to_csv(output / "task2_npam_seizure_audit.csv", index=False)
    channel_frame.to_csv(output / "task2_npam_channel_audit.csv", index=False)
    declared_exclusion_audit.to_csv(output / "task2_npam_exclusion_audit.csv", index=False)
    pd.DataFrame(exclusion_rows, columns=("patient_key", "seizure_id", "level", "reason")).to_csv(output / "task2_npam_alignment_exclusions.csv", index=False)
    phase_frame.to_csv(output / "task2_npam_phase_coverage.csv", index=False)
    pd.DataFrame(checkpoint_rows, columns=("outer_fold", "checkpoint_path", "checkpoint_exists", "training_subject_manifest_verified", "embedding_export_verified")).to_csv(output / "task2_npam_checkpoint_audit.csv", index=False)
    outcome_candidates.to_csv(output / "task2_npam_outcome_source_audit.csv", index=False)
    summary = {
        "label_semantics": {"task1_nez": 1, "task1_ez": 0, "task2_success": 1, "task2_failure": 0},
        "network_phases": list(PHASES),
        "exclusion_manifest": str(Path(exclusion_manifest).expanduser().resolve()) if exclusion_manifest else None,
        "feature_cache_schema": cache_schema(feature_cache, feature_path, "feature"),
        "raw_cache_schema": cache_schema(raw_cache, raw_path, "raw"),
        "n_outcome_patients": int(len(outcomes)),
        "n_success": int((outcomes["outcome_group"] == "success").sum()), "n_failure": int((outcomes["outcome_group"] == "failure").sum()),
        "n_unknown": int(len(unknown)), "n_conflict": int(len(conflicts)),
        "center_outcome_distribution": outcomes.groupby(["center", "outcome_group"]).size().rename("n").reset_index().to_dict("records"),
        "signal_type_distribution": dict(Counter(str(record.get("task", record.get("phase_group", "unknown"))) for record in feature_records)),
        "patient_key_unique": bool(not outcomes["patient_key"].duplicated().any()),
        "n_outcome_duplicate_patients": int(outcomes["patient_key"].duplicated(keep=False).sum()),
        "n_follow_up_fields_present": int(sum(row["follow_up_field_present"] for row in patient_rows)),
        "n_surgery_success_inconsistent": int(sum(not row["surgery_success_consistent"] for row in patient_rows if row["outcome_group"] in {"success", "failure"})),
        "n_feature_runs": len(feature_records), "n_raw_runs": len(raw_records), "n_matched_runs": int(sum(row["alignment_status"] == "matched" for row in seizure_rows)),
        "n_feature_only_runs": len(feature_only), "n_raw_only_runs": len(raw_only),
        "duplicate_feature_keys": duplicate_feature_keys, "duplicate_raw_keys": duplicate_raw_keys,
        "raw_feature_match_rate": float(sum(row["alignment_status"] == "matched" for row in seizure_rows) / max(len(keys), 1)),
        "fold_manifest_patients": int(len(fold_rows)), "fold_manifest_hash": hashlib.sha256(fold_rows.to_csv(index=False).encode()).hexdigest() if not fold_rows.empty else None,
        "p2_checkpoint_count": int(sum(row["checkpoint_exists"] for row in checkpoint_rows)),
        "forbidden_model_fields": ["coordinates", "fsaverage", "soz", "resect", "true_k", "true_ez", "center"],
        "strict_passed": bool(
            conflicts.empty
            and unknown.empty
            and not feature_only
            and not raw_only
            and not duplicate_feature_keys
            and not duplicate_raw_keys
            and all(row["checkpoint_exists"] for row in checkpoint_rows)
        ),
    }
    (output / "task2_npam_input_audit.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return summary


def chain_keys(*values: Any) -> list[str]:
    output = []
    for value in values:
        if isinstance(value, Mapping):
            output.extend(map(str, value.keys()))
    return output


__all__ = ["audit_inputs", "cache_schema"]
