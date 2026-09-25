from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..functional_graph import normalize_channel_name


class COPSchemaError(ValueError): pass


def _required(mapping: Mapping[str, Any], key: str, context: str) -> Any:
    value = mapping.get(key)
    if value is None: raise COPSchemaError(f"MISSING_REQUIRED_FIELD: {context}.{key}")
    return value


@dataclass
class FeatureRunRecord:
    patient_key: str
    center: str
    seizure_id: str
    channel_names: list[str]
    window_start_sec: np.ndarray
    window_end_sec: np.ndarray
    feature_values: np.ndarray
    feature_names: list[str]
    valid_mask: np.ndarray
    clinical_target_mask: np.ndarray


@dataclass
class RawRunRecord:
    patient_key: str
    center: str
    seizure_id: str
    signal: np.ndarray
    sampling_rate: float
    onset_sample: int
    channel_names: list[str]
    valid_channel_mask: np.ndarray
    clinical_target_mask: np.ndarray


def _target_for(patient: str, channels: Sequence[str], lookup: Mapping[str, Mapping[str, Any]]) -> np.ndarray:
    entry = lookup.get(str(patient))
    if entry is None: raise COPSchemaError(f"MISSING_REQUIRED_FIELD: clinical_target_lookup[{patient}]")
    names = [normalize_channel_name(value) for value in entry["channel_names"]]
    values = np.asarray(entry.get("clinical_target_mask", entry.get("mask")), dtype=bool); mapping = dict(zip(names, values.tolist()))
    missing = [name for name in channels if normalize_channel_name(name) not in mapping]
    if missing: raise COPSchemaError(f"MISSING_REQUIRED_FIELD: target channel alignment {patient}: {missing[:5]}")
    return np.asarray([mapping[normalize_channel_name(name)] for name in channels], dtype=bool)


def adapt_feature_record(record: Mapping[str, Any], target_lookup: Mapping[str, Mapping[str, Any]]) -> tuple[FeatureRunRecord, dict[str, Any]]:
    sample = _required(record, "sample", "run_record"); patient = str(_required(record, "subject_id", "run_record")); run = str(_required(record, "run_id", patient))
    channels = list(map(str, _required(record, "channel_names_norm", run))); names = list(map(str, _required(sample, "window_feature_names", run)))
    values = np.asarray(_required(sample, "window_features", run), dtype=np.float32)
    centers = np.asarray(_required(sample, "window_relative_centers_sec", run), dtype=np.float64)
    if values.ndim != 3: raise COPSchemaError(f"MISSING_REQUIRED_FIELD: {run}.window_features must be rank 3")
    axis = "W,C,F"
    if values.shape == (len(centers), len(channels), len(names)): pass
    elif values.shape == (len(channels), len(centers), len(names)): values = values.transpose(1, 0, 2); axis = "C,W,F->W,C,F"
    else: raise COPSchemaError(f"MISSING_REQUIRED_FIELD: {run} axes inconsistent: shape={values.shape}, W={len(centers)}, C={len(channels)}, F={len(names)}")
    scales = sample.get("feature_scale_used_secs", sample.get("feature_scales_sec"))
    if scales is None: raise COPSchemaError(f"MISSING_REQUIRED_FIELD: {run}.feature_scale_used_secs")
    scale = np.asarray(scales, dtype=float).reshape(-1)
    if scale.size == 1: scale = np.repeat(scale, len(centers))
    if scale.size != len(centers): raise COPSchemaError(f"MISSING_REQUIRED_FIELD: {run} window scale length")
    start, end = centers-scale/2, centers+scale/2
    valid = np.isfinite(values).all(axis=-1)
    target = _target_for(patient, channels, target_lookup)
    if not target.any() or target.all(): raise COPSchemaError(f"MISSING_REQUIRED_FIELD: {patient} requires non-empty target and outside")
    center = str(record.get("center") or record.get("source_center") or patient.split(":", 1)[0])
    return FeatureRunRecord(patient, center, run, channels, start, end, values, names, valid, target), {"patient_key":patient,"seizure_id":run,"input_shape":list(np.shape(_required(sample,"window_features",run))),"axis_transform":axis,"n_windows":len(centers),"n_channels":len(channels),"n_features":len(names),"phase_source":"derived_from_window_center","phase_definition":"preictal:t<0|onset:0<=t<10|spread:10<=t<30|late:t>=30","explicit_window_phase_present":False,"phase_definition_status":"OK"}


def adapt_raw_record(record: Mapping[str, Any], target_lookup: Mapping[str, Mapping[str, Any]]) -> RawRunRecord:
    sample = _required(record, "sample", "raw_run_record"); patient = str(_required(record, "subject_id", "raw_run_record")); run = str(_required(record, "run_id", patient))
    channels = list(map(str, _required(record, "channel_names_norm", run))); signal = np.asarray(_required(sample, "raw_waveform", run), dtype=np.float32)
    if signal.ndim != 2: raise COPSchemaError(f"MISSING_REQUIRED_FIELD: {run}.raw_waveform must be [C,T]")
    if signal.shape[0] != len(channels) and signal.shape[1] == len(channels): signal = signal.T
    if signal.shape[0] != len(channels): raise COPSchemaError(f"MISSING_REQUIRED_FIELD: {run} raw channel axis mismatch")
    sfreq = float(_required(sample, "raw_temporal_sfreq", run)); start = float(_required(sample, "start_sec", run)); onset = float(_required(sample, "seizure_onset_sec", run))
    onset_sample = int(round((onset-start)*sfreq));
    if not (0 <= onset_sample < signal.shape[1]): raise COPSchemaError(f"MISSING_REQUIRED_FIELD: {run} onset outside raw waveform")
    valid = np.isfinite(signal).all(axis=1); target = _target_for(patient, channels, target_lookup)
    center = str(record.get("center") or record.get("source_center") or patient.split(":", 1)[0])
    return RawRunRecord(patient, center, run, signal, sfreq, onset_sample, channels, valid, target)


def inspect_cache_schema(cache: Mapping[str, Any], kind: str) -> dict[str, Any]:
    runs = cache.get("run_records"); patients = cache.get("patient_index")
    if not isinstance(runs, list): raise COPSchemaError("MISSING_REQUIRED_FIELD: payload.run_records")
    if not isinstance(patients, dict): raise COPSchemaError("MISSING_REQUIRED_FIELD: payload.patient_index")
    first = runs[0] if runs else {}; sample = first.get("sample", {}) if isinstance(first, Mapping) else {}
    array_shapes = {key:list(np.shape(value)) for key,value in sample.items() if isinstance(value,np.ndarray)}
    return {"cache_kind":kind,"n_runs":len(runs),"n_patients":len(patients),"payload_keys":sorted(cache.keys()),"run_record_keys":sorted(first.keys()),"sample_keys":sorted(sample.keys()),"sample_array_shapes_first_record":array_shapes,"has_raw_waveform":bool("raw_waveform" in sample),"status":"OK"}


def align_feature_raw(feature: Sequence[FeatureRunRecord], raw: Sequence[RawRunRecord], strict: bool=True) -> list[dict[str, Any]]:
    raw_map={(r.patient_key,r.seizure_id):r for r in raw}; rows=[]
    for value in feature:
        other=raw_map.get((value.patient_key,value.seizure_id))
        f={normalize_channel_name(x) for x in value.channel_names}; r={normalize_channel_name(x) for x in other.channel_names} if other else set(); matched=f&r; rate=len(matched)/max(len(f),1)
        rows.append({"patient_key":value.patient_key,"seizure_id":value.seizure_id,"n_feature_channels":len(f),"n_raw_channels":len(r),"n_matched_channels":len(matched),"match_rate":rate,"missing_feature_channels":";".join(sorted(r-f)),"missing_raw_channels":";".join(sorted(f-r))})
        if strict and rate < .90: raise COPSchemaError(f"RAW_FEATURE_ALIGNMENT_BELOW_0P90: {value.seizure_id}={rate:.3f}")
    if strict:
        by_patient={p:[row["match_rate"] for row in rows if row["patient_key"]==p] for p in {row["patient_key"] for row in rows}}
        bad={p:float(np.mean(v)) for p,v in by_patient.items() if np.mean(v)<.95}
        if bad: raise COPSchemaError(f"RAW_FEATURE_PATIENT_ALIGNMENT_BELOW_0P95: {list(bad.items())[:5]}")
    return rows


__all__=["COPSchemaError","FeatureRunRecord","RawRunRecord","adapt_feature_record","adapt_raw_record","align_feature_raw","inspect_cache_schema"]
