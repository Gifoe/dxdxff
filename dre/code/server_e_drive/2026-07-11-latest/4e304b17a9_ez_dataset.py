from __future__ import annotations

from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import pickle
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from neuroez_c.quality import is_simple_quality_metadata


def _log(message: str) -> None:
    print(f"[NeuroEZ-C0][Data] {message}", flush=True)


def _outcome_log(message: str) -> None:
    print(f"[NeuroEZ-C0][Data][OutcomeSubset] {message}", flush=True)


_OUTCOME_GENERIC_KEYS = {"outcome_group", "outcome_raw", "outcome", "lzu_outcome_raw", "surgery_result"}
_OUTCOME_BOOL_KEYS = {"surgery_success", "success_used", "seizure_free"}
_OUTCOME_ENGEL_KEYS = {"engel_score", "engel"}
_OUTCOME_HIGH_CONFIDENCE_KEYS = (
    "outcome_group",
    "surgery_success",
    "outcome_raw",
    "outcome",
    "lzu_outcome_raw",
    "engel_score",
    "engel",
    "surgery_result",
    "seizure_free",
)
_OUTCOME_KEYS = (
    *_OUTCOME_HIGH_CONFIDENCE_KEYS,
    "success_used",
)
_NON_QUALITY_METADATA_KEYS = {
    "raw_waveform",
    "raw_temporal_sfreq",
    "raw_temporal_duration_sec",
    "raw_valid_samples",
    "raw_valid_start_sample",
    "outcome_group",
    "surgery_success",
    "outcome_raw",
    "outcome",
    "lzu_outcome_raw",
    "engel_score",
    "engel",
    "outcome_source",
    "success_used",
    "success_used_legacy_filter",
    "surgery_result",
    "seizure_free",
}


def _extract_run_records_from_cache_payload(cached_payload: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not isinstance(cached_payload, dict):
        raise ValueError("Unsupported external cache format: top-level object is not a dict.")
    run_records = cached_payload.get("run_records")
    patient_index = cached_payload.get("patient_index")
    if not isinstance(run_records, list) or not isinstance(patient_index, dict):
        raise ValueError("Unsupported external cache format: expected run_records list and patient_index dict.")
    return run_records, patient_index


def _load_external_cache(cache_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    with open(cache_path, "rb") as fin:
        cached_payload = pickle.load(fin)
    return _extract_run_records_from_cache_payload(cached_payload)


def _normalize_outcome_group(value: Any) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, (bool, np.bool_)):
        return "success" if bool(value) else "failure"
    if isinstance(value, (int, float, np.integer, np.floating)):
        value_float = float(value)
        if not math.isfinite(value_float):
            return "unknown"
        if value_float == 1.0:
            return "success"
        if value_float == 0.0:
            return "failure"
        return "unknown"

    text = str(value).strip()
    if not text:
        return "unknown"
    text_lower = text.lower()
    text_norm = text_lower.replace("_", " ").replace("-", " ").strip()
    text_compact = "".join(ch for ch in text_norm if ch.isalnum())
    if text_norm in {"nan", "none", "null", "na", "n/a", "unknown", "unk", "not documented", "not recorded"}:
        return "unknown"
    if text in {"未知", "不详", "未记录", "未記錄"}:
        return "unknown"

    success_values = {
        "success",
        "successful",
        "s",
        "yes",
        "y",
        "true",
        "1",
        "engel i",
        "engel 1",
        "seizure free",
        "i",
    }
    failure_values = {
        "failure",
        "fail",
        "failed",
        "f",
        "no",
        "n",
        "false",
        "0",
        "ii",
        "iii",
        "iv",
    }
    success_phrases = ("成功", "术后成功", "術後成功", "无发作", "無發作", "癫痫无发作", "癲癇無發作")
    failure_phrases = ("失败", "失敗", "术后失败", "術後失敗", "复发", "復發", "未控制", "仍发作", "仍發作")
    if text_norm in success_values or text_compact in {"engeli", "engel1"}:
        return "success"
    if any(phrase in text for phrase in success_phrases):
        return "success"
    if text_norm in failure_values or text_compact in {"engelii", "engeliii", "engeliv", "engel2", "engel3", "engel4"}:
        return "failure"
    if any(phrase in text for phrase in failure_phrases):
        return "failure"

    if text_compact.startswith("engel"):
        return _normalize_engel_score(value)
    return "unknown"


def _normalize_engel_score(value: Any) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, (int, float, np.integer, np.floating)):
        value_float = float(value)
        if not math.isfinite(value_float):
            return "unknown"
        if value_float == 1.0:
            return "success"
        if value_float in {2.0, 3.0, 4.0}:
            return "failure"
        return "unknown"
    text = str(value).strip()
    if not text:
        return "unknown"
    text_lower = text.lower().replace("_", " ").replace("-", " ").strip()
    text_compact = "".join(ch for ch in text_lower if ch.isalnum())
    if text_compact.startswith("engel"):
        text_compact = text_compact[len("engel") :]
    if text_compact in {"i", "1"}:
        return "success"
    if text_compact in {"ii", "iii", "iv", "2", "3", "4"}:
        return "failure"
    return "unknown"


def _normalize_outcome_value_for_key(key: str, value: Any) -> str:
    key_norm = str(key).strip().lower()
    if key_norm == "success_used":
        parsed = _normalize_outcome_group(value)
        return "success" if parsed == "success" else "unknown"
    if key_norm in _OUTCOME_ENGEL_KEYS:
        return _normalize_engel_score(value)
    if key_norm in _OUTCOME_BOOL_KEYS or key_norm in _OUTCOME_GENERIC_KEYS:
        return _normalize_outcome_group(value)
    return "unknown"


def _extract_outcome_group_from_mapping(mapping: Any) -> str:
    if not isinstance(mapping, dict):
        return "unknown"
    has_high_confidence_outcome = any(key in mapping for key in _OUTCOME_HIGH_CONFIDENCE_KEYS)
    for key in _OUTCOME_HIGH_CONFIDENCE_KEYS:
        if key not in mapping:
            continue
        group = _normalize_outcome_value_for_key(key, mapping.get(key))
        if group != "unknown":
            return group
    if has_high_confidence_outcome:
        return "unknown"
    if "success_used" in mapping:
        return _normalize_outcome_value_for_key("success_used", mapping.get("success_used"))
    return "unknown"


def _channel_meta_outcome_details(channel_meta: Any) -> tuple[str, set[str], Counter[str]]:
    if not isinstance(channel_meta, list):
        return "unknown", set(), Counter()
    groups: set[str] = set()
    source_fields: Counter[str] = Counter()
    for item in channel_meta:
        if not isinstance(item, dict):
            continue
        candidate_keys = _OUTCOME_HIGH_CONFIDENCE_KEYS
        if not any(key in item for key in _OUTCOME_HIGH_CONFIDENCE_KEYS):
            candidate_keys = ("success_used",)
        for key in candidate_keys:
            if key not in item:
                continue
            group = _normalize_outcome_value_for_key(key, item.get(key))
            if group == "unknown":
                continue
            groups.add(group)
            source_fields[str(key)] += 1
    if groups == {"success"}:
        return "success", groups, source_fields
    if groups == {"failure"}:
        return "failure", groups, source_fields
    return "unknown", groups, source_fields


def _extract_outcome_group_from_channel_meta(channel_meta: Any) -> str:
    group, _, _ = _channel_meta_outcome_details(channel_meta)
    return group


def _group_run_records_by_subject(run_records: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in run_records:
        grouped[str(record.get("subject_id"))].append(record)
    return dict(grouped)


def _extract_patient_outcome_group(
    subject_id: str,
    patient_meta: dict[str, Any],
    run_records_by_subject: dict[str, list[dict[str, Any]]] | None = None,
) -> str:
    group = _extract_outcome_group_from_mapping(patient_meta)
    if group != "unknown":
        return group
    group = _extract_outcome_group_from_channel_meta(patient_meta.get("channel_meta"))
    if group != "unknown":
        return group

    records = (run_records_by_subject or {}).get(str(subject_id), [])
    for source_name in ("metadata", "sample"):
        for record in records:
            source = record.get(source_name, {})
            group = _extract_outcome_group_from_mapping(source)
            if group != "unknown":
                return group
    for record in records:
        group = _extract_outcome_group_from_channel_meta(record.get("channel_meta"))
        if group != "unknown":
            return group
        sample = record.get("sample", {})
        if isinstance(sample, dict):
            group = _extract_outcome_group_from_channel_meta(sample.get("channel_meta"))
            if group != "unknown":
                return group
    return "unknown"


def _allowed_outcome_groups(outcome_subset: str) -> set[str]:
    subset = str(outcome_subset or "all").strip().lower()
    mapping = {
        "all": {"success", "failure", "unknown"},
        "success": {"success"},
        "failure": {"failure"},
        "success_failure": {"success", "failure"},
        "unknown": {"unknown"},
    }
    if subset not in mapping:
        raise ValueError(f"Unsupported outcome_subset={outcome_subset!r}; expected one of {sorted(mapping)}.")
    return set(mapping[subset])


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _sanitize_quality_metadata_value(value: Any) -> Any:
    if isinstance(value, dict):
        sanitized = {
            str(key): _sanitize_quality_metadata_value(child)
            for key, child in value.items()
            if str(key) not in _NON_QUALITY_METADATA_KEYS
        }
        return {key: child for key, child in sanitized.items() if is_simple_quality_metadata(child)}
    return value


def _filter_run_records_and_patient_index_by_outcome(
    run_records: list[dict[str, Any]],
    patient_index: dict[str, dict[str, Any]],
    args: Any,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    outcome_subset = str(getattr(args, "outcome_subset", "all") or "all").strip().lower()
    allowed_groups = _allowed_outcome_groups(outcome_subset)
    run_records_by_subject = _group_run_records_by_subject(run_records)

    normalized_index: dict[str, dict[str, Any]] = {}
    patient_groups: dict[str, str] = {}
    channel_meta_conflict_subjects: list[str] = []
    channel_meta_source_fields: Counter[str] = Counter()
    for subject_id, patient_meta_raw in patient_index.items():
        subject_id_str = str(subject_id)
        patient_meta = dict(patient_meta_raw or {})
        group = _extract_patient_outcome_group(subject_id_str, patient_meta, run_records_by_subject)
        channel_sources = [patient_meta.get("channel_meta")]
        for record in run_records_by_subject.get(subject_id_str, []):
            channel_sources.append(record.get("channel_meta"))
            sample = record.get("sample", {})
            if isinstance(sample, dict):
                channel_sources.append(sample.get("channel_meta"))
        for channel_source in channel_sources:
            _, channel_groups, source_fields = _channel_meta_outcome_details(channel_source)
            channel_meta_source_fields.update(source_fields)
            if {"success", "failure"}.issubset(channel_groups):
                channel_meta_conflict_subjects.append(subject_id_str)
        patient_meta["outcome_group"] = group
        patient_meta["surgery_success"] = True if group == "success" else False if group == "failure" else None
        normalized_index[subject_id_str] = patient_meta
        patient_groups[subject_id_str] = group

    filtered_index = {
        subject_id: meta for subject_id, meta in normalized_index.items() if patient_groups[subject_id] in allowed_groups
    }
    filtered_subjects = set(filtered_index)
    filtered_records = [record for record in run_records if str(record.get("subject_id")) in filtered_subjects]

    patient_counts = Counter(patient_groups.values())
    run_counts = Counter()
    for record in run_records:
        subject_id = str(record.get("subject_id"))
        run_counts[patient_groups.get(subject_id, "unknown")] += 1
    dropped_counts = Counter(
        group for subject_id, group in patient_groups.items() if subject_id not in filtered_subjects
    )
    summary: dict[str, Any] = {
        "before_patients": len(patient_index),
        "after_patients": len(filtered_index),
        "before_run_records": len(run_records),
        "after_run_records": len(filtered_records),
        "outcome_subset": outcome_subset,
        "patient_counts_by_outcome_group": {key: int(patient_counts.get(key, 0)) for key in ("success", "failure", "unknown")},
        "run_counts_by_outcome_group": {key: int(run_counts.get(key, 0)) for key in ("success", "failure", "unknown")},
        "dropped_patient_counts_by_outcome_group": {
            key: int(dropped_counts.get(key, 0)) for key in ("success", "failure", "unknown")
        },
        "missing_outcome_patients": sorted(
            subject_id for subject_id, group in patient_groups.items() if group == "unknown"
        ),
        "selected_subjects_preview": sorted(filtered_subjects)[:20],
        "channel_meta_outcome_conflict_patients": sorted(set(channel_meta_conflict_subjects)),
        "top_channel_meta_outcome_source_fields": dict(channel_meta_source_fields.most_common(20)),
        "warnings": [],
    }
    if outcome_subset != "all" and len(filtered_index) == 0:
        raise ValueError(f"outcome_subset={outcome_subset!r} selected zero patients.")
    if outcome_subset == "success" and patient_counts.get("success", 0) == 0:
        raise ValueError("outcome_subset='success' requested, but no success patients were found.")
    if outcome_subset == "failure" and patient_counts.get("failure", 0) == 0:
        raise ValueError("outcome_subset='failure' requested, but no failure patients were found.")
    if outcome_subset == "success_failure":
        missing_groups = [group for group in ("success", "failure") if patient_counts.get(group, 0) == 0]
        if missing_groups:
            summary["warnings"].append(
                f"success_failure subset is missing patient outcome group(s): {','.join(missing_groups)}"
            )
        if patient_counts.get("unknown", 0) > 0:
            summary["warnings"].append(
                f"success_failure subset excluded {int(patient_counts.get('unknown', 0))} unknown-outcome patient(s)."
            )
    if channel_meta_conflict_subjects:
        summary["warnings"].append(
            f"Detected conflicting success/failure outcome values in channel_meta for "
            f"{len(set(channel_meta_conflict_subjects))} patient(s)."
        )

    _outcome_log(json.dumps(_json_safe(summary), ensure_ascii=False, sort_keys=True))
    return filtered_records, filtered_index, summary


def _write_outcome_subset_summary_if_possible(args: Any, summary: dict[str, Any]) -> None:
    output_dir_raw = getattr(args, "output_dir", None)
    if not output_dir_raw:
        return
    try:
        output_dir = Path(str(output_dir_raw)).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "data_reader_outcome_subset_summary.json", "w", encoding="utf-8") as fout:
            json.dump(_json_safe(summary), fout, indent=2, ensure_ascii=False, sort_keys=True)
    except Exception as exc:
        _outcome_log(f"Could not write outcome subset summary: {type(exc).__name__}: {exc}")


def _patient_source_center(subject_id: str, patient_meta: dict[str, Any]) -> str:
    for key in ("source_center", "center", "source_dataset"):
        value = str(patient_meta.get(key, "")).strip().lower()
        if value:
            return value
    if ":" in subject_id:
        return subject_id.split(":", 1)[0].strip().lower()
    return ""


def _filter_high_ez_fraction_lzu(
    run_records: list[dict[str, Any]],
    patient_index: dict[str, dict[str, Any]],
    args: Any,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    enabled = bool(getattr(args, "drop_high_ez_fraction_lzu", False))
    threshold = float(getattr(args, "lzu_max_ez_fraction", 0.40))
    base_audit: dict[str, Any] = {
        "drop_high_ez_fraction_lzu_enabled": enabled,
        "drop_high_ez_fraction_lzu_threshold": float(threshold),
        "n_patients_dropped_by_high_ez_fraction": 0,
        "subjects_dropped_by_high_ez_fraction": [],
    }
    if not enabled:
        return run_records, patient_index, base_audit

    dropped: set[str] = set()
    for subject_id, patient_meta in patient_index.items():
        if _patient_source_center(str(subject_id), patient_meta) != "lzu":
            continue
        labels = np.asarray(patient_meta.get("labels", []), dtype=np.float32)
        mask = np.asarray(patient_meta.get("label_mask", np.ones(labels.shape, dtype=bool)), dtype=bool)
        valid = mask & (labels >= 0.0)
        if not np.any(valid):
            continue
        ez_fraction = float(np.sum(labels[valid] > 0.5) / max(int(np.sum(valid)), 1))
        if ez_fraction > threshold:
            dropped.add(str(subject_id))

    if not dropped:
        _log(f"LZU EZ-fraction filter enabled at {threshold:.2f}; no patients dropped.")
        return run_records, patient_index, base_audit

    filtered_records = [record for record in run_records if str(record.get("subject_id")) not in dropped]
    filtered_index = {sid: meta for sid, meta in patient_index.items() if str(sid) not in dropped}
    _log(
        f"LZU EZ-fraction filter enabled at {threshold:.2f}; "
        f"dropped {len(dropped)} patient(s), kept {len(filtered_index)} patient(s) and {len(filtered_records)} record(s)."
    )
    audit = {
        "drop_high_ez_fraction_lzu_enabled": True,
        "drop_high_ez_fraction_lzu_threshold": float(threshold),
        "n_patients_dropped_by_high_ez_fraction": len(dropped),
        "subjects_dropped_by_high_ez_fraction": sorted(dropped),
    }
    return filtered_records, filtered_index, audit


def _read_allowed_subjects(args: Any) -> list[str] | None:
    """Read allowed subjects from --allowed-subjects-ledger or --allowed-subjects-file."""
    ledger_path = getattr(args, "allowed_subjects_ledger", None) or getattr(args, "allowed-subjects-ledger", None)
    if ledger_path:
        df = pd.read_csv(Path(str(ledger_path)))
        col = "subject_id" if "subject_id" in df.columns else "patient_id"
        return sorted(df[col].astype(str).unique().tolist())
    file_path = getattr(args, "allowed_subjects_file", None) or getattr(args, "allowed-subjects-file", None)
    if file_path:
        with open(Path(str(file_path)), "r") as f:
            return sorted(line.strip() for line in f if line.strip())
    return None


def _filter_run_records_by_allowed_subjects(
    run_records: list[dict[str, Any]],
    patient_index: dict[str, dict[str, Any]],
    args: Any,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    """Apply allowed-subject filtering BEFORE build_outer_splits."""
    allowed = _read_allowed_subjects(args)
    n_before = len(patient_index)
    subjects_before = sorted(patient_index.keys())

    audit: dict[str, Any] = {
        "v3_allowed_subject_filter_used": allowed is not None,
        "v3_allowed_subjects_source": (
            str(getattr(args, "allowed_subjects_ledger", getattr(args, "allowed-subjects-file", "")))
            if allowed else ""
        ),
        "v3_n_patients_before_allowed_filter": n_before,
        "v3_require_n_patients": int(getattr(args, "require_n_patients", 0) or 0),
        "v3_split_built_after_allowed_filter": True,
    }

    if not allowed:
        audit["v3_n_patients_after_allowed_filter"] = n_before
        audit["v3_subjects_removed_by_allowed_filter"] = []
        audit["v3_subjects_missing_from_allowed_filter"] = []
        return run_records, patient_index, audit

    allowed_set = set(str(s) for s in allowed)
    existing = set(str(s) for s in subjects_before)
    removed = sorted(existing - allowed_set)
    missing = sorted(allowed_set - existing)

    filtered_records = [r for r in run_records if str(r.get("subject_id")) in allowed_set]
    filtered_index = {sid: meta for sid, meta in patient_index.items() if str(sid) in allowed_set}

    audit.update({
        "v3_n_patients_after_allowed_filter": len(filtered_index),
        "v3_subjects_removed_by_allowed_filter": removed,
        "v3_subjects_missing_from_allowed_filter": missing,
    })
    return filtered_records, filtered_index, audit


def _write_allowed_subject_filter_audit(args: Any, audit: dict[str, Any]) -> None:
    """Write allowed-subject filter audit if possible."""
    try:
        output_dir = getattr(args, "output_dir", None)
        if output_dir:
            out = Path(str(output_dir))
            out.mkdir(parents=True, exist_ok=True)
            with open(out / "v3_allowed_subject_filter_audit.json", "w", encoding="utf-8") as f:
                json.dump(audit, f, indent=2, ensure_ascii=False, sort_keys=True)
    except Exception:
        pass


def build_or_load_run_records(args: Any) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    cache_path_raw = getattr(args, "sample_cache_path", None) or getattr(args, "window_cache_path", None)
    if not cache_path_raw:
        raise ValueError("C0 ablation requires --window_cache_path pointing to a validated window cache pkl.")
    cache_path = Path(str(cache_path_raw)).expanduser()
    if not cache_path.exists():
        raise FileNotFoundError(f"External sample cache does not exist: {cache_path}")
    run_records, patient_index = _load_external_cache(cache_path)
    run_records, patient_index, outcome_summary = _filter_run_records_and_patient_index_by_outcome(
        run_records,
        patient_index,
        args,
    )
    _write_outcome_subset_summary_if_possible(args, outcome_summary)
    run_records, patient_index, high_ez_audit = _filter_high_ez_fraction_lzu(run_records, patient_index, args)
    run_records, patient_index, allowed_filter_audit = _filter_run_records_by_allowed_subjects(
        run_records, patient_index, args,
    )
    allowed_filter_audit.update(high_ez_audit)
    _write_allowed_subject_filter_audit(args, allowed_filter_audit)
    require_n = int(getattr(args, "require_n_patients", 0) or 0)
    if require_n > 0 and len(patient_index) != require_n:
        raise ValueError(
            f"require_n_patients={require_n} but after all filters got {len(patient_index)} patients."
        )
    _log(f"Loaded {len(run_records)} run records and {len(patient_index)} patients from {cache_path}.")
    return run_records, patient_index


def flatten_window_samples(
    run_records: Iterable[dict[str, Any]],
    *,
    subject_ids: Optional[Sequence[str]] = None,
    include_raw_waveform: bool = False,
) -> list[dict[str, Any]]:
    selected_subjects = set(subject_ids) if subject_ids is not None else None
    samples: list[dict[str, Any]] = []
    for run_record in run_records:
        subject_id = str(run_record["subject_id"])
        if selected_subjects is not None and subject_id not in selected_subjects:
            continue
        sample = dict(run_record["sample"])
        feature_names = sample.get("window_feature_names", run_record.get("window_feature_names"))
        quality_metadata: dict[str, Any] = {}
        for source in (run_record, sample):
            for key, value in source.items():
                if key in {
                    "sample",
                    "labels",
                    "channel_names_norm",
                    "window_features",
                    "window_adjacency",
                    "window_relative_centers_sec",
                } or key in _NON_QUALITY_METADATA_KEYS:
                    continue
                sanitized_value = _sanitize_quality_metadata_value(value)
                if is_simple_quality_metadata(sanitized_value):
                    quality_metadata.setdefault(str(key), sanitized_value)
        flattened = {
            "subject_id": subject_id,
            "run_id": str(run_record["run_id"]),
            "sample_id": str(sample.get("sample_id", run_record["run_id"])),
            "source_center": run_record.get("source_center", sample.get("source_center")),
            "center": run_record.get("center", sample.get("center")),
            "source_dataset": run_record.get("source_dataset", sample.get("source_dataset")),
            "channel_names_norm": list(run_record["channel_names_norm"]),
            "labels": np.asarray(run_record["labels"], dtype=np.float32),
            "window_features": np.asarray(sample.get("window_features", np.zeros((0, 0, 0))), dtype=np.float32),
            "window_adjacency": np.asarray(sample.get("window_adjacency", np.zeros((0, 0, 0))), dtype=np.float32),
            "window_relative_centers_sec": np.asarray(
                sample.get("window_relative_centers_sec", np.zeros((0,))),
                dtype=np.float32,
            ),
            "feature_scale_used_secs": np.asarray(sample.get("feature_scale_used_secs", np.zeros((0,))), dtype=np.float32),
            "seizure_onset_sec": sample.get("seizure_onset_sec", run_record.get("seizure_onset_sec")),
            "source_seizure_id": sample.get("source_seizure_id", run_record.get("source_seizure_id")),
            "start_sec": sample.get("start_sec", run_record.get("start_sec")),
            "end_sec": sample.get("end_sec", run_record.get("end_sec")),
            "window_feature_names": list(feature_names) if feature_names is not None else None,
            "quality_metadata": quality_metadata,
        }
        if include_raw_waveform:
            if "raw_waveform" in sample:
                flattened["raw_waveform"] = sample["raw_waveform"]
                for raw_key in (
                    "raw_temporal_sfreq",
                    "raw_temporal_duration_sec",
                    "raw_valid_samples",
                    "raw_valid_start_sample",
                ):
                    if raw_key in sample:
                        flattened[raw_key] = sample[raw_key]
            else:
                flattened["raw_waveform_missing"] = True
                flattened["raw_waveform_missing_reason"] = "missing_in_sample"
        samples.append(flattened)
    return samples


__all__ = ["build_or_load_run_records", "flatten_window_samples"]
