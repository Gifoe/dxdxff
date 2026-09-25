from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from ..functional_graph import normalize_channel_name
from ..cop.schema import adapt_feature_record as _adapt_feature
from ..cop.schema import adapt_raw_record as _adapt_raw
from ..cop.schema import inspect_cache_schema


class NGBRSchemaError(ValueError):
    pass


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
    start_sec: float
    seizure_onset_sec: float


@dataclass
class FeatureRunRecord:
    patient_key: str
    center: str
    seizure_id: str
    feature_values: np.ndarray
    feature_names: list[str]
    window_center_sec: np.ndarray
    channel_names: list[str]
    valid_mask: np.ndarray
    clinical_target_mask: np.ndarray


@dataclass
class P2NEZRecord:
    patient_key: str
    center: str
    channel_names: list[str]
    final_nez_probability: np.ndarray
    direct_nez_probability: np.ndarray | None
    valid_channel_mask: np.ndarray
    clinical_target_mask: np.ndarray


@dataclass
class BiomarkerRunMaps:
    patient_key: str
    center: str
    seizure_id: str
    channel_names: list[str]
    fragility: np.ndarray
    ei: np.ndarray
    propagation: np.ndarray
    low_entropy: np.ndarray
    hfo: np.ndarray
    biomarker_consensus: np.ndarray
    biomarker_only_residual: np.ndarray
    nez_only_residual: np.ndarray
    joint_residual: np.ndarray
    biomarker_valid_count: np.ndarray


def adapt_raw_record(record: Mapping[str, Any], target_lookup: Mapping[str, Mapping[str, Any]]) -> tuple[RawRunRecord, dict[str, Any]]:
    try:
        base = _adapt_raw(record, target_lookup)
    except Exception as exc:
        raise NGBRSchemaError(str(exc)) from exc
    sample = record.get("sample", {})
    start = float(sample.get("start_sec"))
    onset = float(sample.get("seizure_onset_sec"))
    value = RawRunRecord(
        base.patient_key, base.center, base.seizure_id, base.signal,
        base.sampling_rate, base.onset_sample,
        [normalize_channel_name(x) for x in base.channel_names],
        base.valid_channel_mask, base.clinical_target_mask, start, onset,
    )
    audit = {
        "patient_key": value.patient_key, "seizure_id": value.seizure_id,
        "input_shape": list(np.shape(sample.get("raw_waveform"))),
        "canonical_shape": list(value.signal.shape), "axis_transform": "C,T",
        "sampling_rate": value.sampling_rate, "onset_sample": value.onset_sample,
        "start_sec": start, "seizure_onset_sec": onset,
    }
    return value, audit


def adapt_feature_record(record: Mapping[str, Any], target_lookup: Mapping[str, Mapping[str, Any]]) -> tuple[FeatureRunRecord, dict[str, Any]]:
    try:
        base, audit = _adapt_feature(record, target_lookup)
    except Exception as exc:
        raise NGBRSchemaError(str(exc)) from exc
    value = FeatureRunRecord(
        base.patient_key, base.center, base.seizure_id, base.feature_values,
        base.feature_names, (base.window_start_sec + base.window_end_sec) / 2.0,
        [normalize_channel_name(x) for x in base.channel_names], base.valid_mask,
        base.clinical_target_mask,
    )
    return value, audit


def validate_target(record: RawRunRecord | FeatureRunRecord) -> None:
    valid = record.valid_channel_mask if isinstance(record, RawRunRecord) else record.valid_mask.any(axis=0)
    target = np.asarray(record.clinical_target_mask, dtype=bool)[valid]
    if target.size == 0 or not target.any() or target.all():
        raise NGBRSchemaError(
            f"INVALID_CLINICAL_TARGET: patient={record.patient_key}, seizure={record.seizure_id}, "
            f"target must contain both 0 and 1 among valid channels"
        )


__all__ = [
    "BiomarkerRunMaps", "FeatureRunRecord", "NGBRSchemaError", "P2NEZRecord",
    "RawRunRecord", "adapt_feature_record", "adapt_raw_record",
    "inspect_cache_schema", "validate_target",
]
