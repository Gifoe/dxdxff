"""Strict feature/raw cache alignment for the N6 dual-view model.

Raw arrays are stored per record as onset-centred ``[channel,time]`` signals.
This module indexes records once, then derives every raw window from the
feature window's explicit relative centre and duration.  It never aligns by
array position across caches.
"""

from __future__ import annotations

import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .raw_brainbert_data import normalize_channel_name


def _sample(record: Mapping[str, Any]) -> Mapping[str, Any]:
    value = record.get("sample", {})
    return value if isinstance(value, Mapping) else {}


def _field(record: Mapping[str, Any], name: str, default: Any = None) -> Any:
    return record[name] if name in record else _sample(record).get(name, default)


def _record_key(record: Mapping[str, Any]) -> tuple[str, str, str, str]:
    sample = _sample(record)
    subject = str(record.get("subject_id", ""))
    run = str(record.get("run_id", ""))
    sample_id = str(sample.get("sample_id", record.get("sample_id", run)))
    onset = _field(record, "seizure_onset_sec", None)
    source_seizure = _field(record, "source_seizure_id", None)
    start = _field(record, "start_sec", None)
    end = _field(record, "end_sec", None)
    identity = str(source_seizure) if source_seizure not in (None, "") else f"onset={onset}|start={start}|end={end}"
    if not subject or not run or not sample_id or onset is None and source_seizure in (None, ""):
        raise ValueError("Feature/raw record is missing subject_id, run_id, or sample_id.")
    return subject, run, sample_id, identity


def _channel_names(record: Mapping[str, Any]) -> list[str]:
    names = list(_field(record, "channel_names_norm", []) or [])
    normalized = [normalize_channel_name(name) for name in names]
    duplicates = sorted({name for name in normalized if normalized.count(name) > 1})
    if duplicates:
        raise ValueError(f"Ambiguous normalized channel names in record {_record_key(record)}: {duplicates[:5]}")
    return normalized


def _window_centers(record: Mapping[str, Any]) -> np.ndarray:
    centers = np.asarray(_field(record, "window_relative_centers_sec", []), dtype=np.float64)
    if centers.ndim != 1 or not centers.size or not np.isfinite(centers).all():
        raise ValueError(f"Record {_record_key(record)} has invalid window_relative_centers_sec.")
    rounded = np.round(centers, 8)
    if np.unique(rounded).size != rounded.size:
        raise ValueError(f"Duplicate/ambiguous feature window centres in record {_record_key(record)}.")
    return centers


def _load_payload(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict) or not isinstance(payload.get("run_records"), list):
        raise ValueError(f"Cache has no run_records list: {path}")
    return payload


class RawAlignmentStore:
    """One-time raw record index plus on-demand aligned raw-window extraction."""

    def __init__(
        self,
        feature_records: Iterable[Mapping[str, Any]],
        *,
        feature_cache_path: str | Path,
        raw_cache_path: str | Path,
        raw_target_samples: int,
        raw_target_sampling_rate: float,
        output_audit_path: str | Path | None = None,
    ) -> None:
        self.feature_cache_path = str(feature_cache_path)
        self.raw_cache_path = str(raw_cache_path)
        self.raw_target_samples = int(raw_target_samples)
        self.raw_target_sampling_rate = float(raw_target_sampling_rate)
        if self.raw_target_samples < 1 or self.raw_target_sampling_rate <= 0:
            raise ValueError("raw_target_samples must be >=1 and raw_target_sampling_rate must be positive.")
        self.feature_records = list(feature_records)
        raw_payload = _load_payload(raw_cache_path)
        self.raw_records: dict[tuple[str, str, str, str], Mapping[str, Any]] = {}
        duplicate_raw_keys: list[str] = []
        for record in raw_payload["run_records"]:
            key = _record_key(record)
            if key in self.raw_records:
                duplicate_raw_keys.append("::".join(key))
            self.raw_records[key] = record
        if duplicate_raw_keys:
            raise ValueError(f"Duplicate raw compound record keys: {duplicate_raw_keys[:5]}")
        self.feature_by_key: dict[tuple[str, str, str, str], Mapping[str, Any]] = {}
        duplicate_feature_keys: list[str] = []
        for record in self.feature_records:
            key = _record_key(record)
            if key in self.feature_by_key:
                duplicate_feature_keys.append("::".join(key))
            self.feature_by_key[key] = record
        if duplicate_feature_keys:
            raise ValueError(f"Duplicate feature compound record keys: {duplicate_feature_keys[:5]}")
        self.audit = self._audit(duplicate_feature_keys, duplicate_raw_keys)
        if output_audit_path is not None:
            destination = Path(output_audit_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(self.audit, indent=2, sort_keys=True), encoding="utf-8")

    def _audit(self, duplicate_feature_keys: list[str], duplicate_raw_keys: list[str]) -> dict[str, Any]:
        feature_subjects = {key[0] for key in self.feature_by_key}
        raw_subjects = {key[0] for key in self.raw_records}
        matched_keys = sorted(set(self.feature_by_key) & set(self.raw_records))
        matched_subjects = {key[0] for key in matched_keys}
        feature_channels = raw_channels = matched_channels = 0
        feature_windows = raw_windows = matched_windows = 0
        missing_by_patient: dict[str, int] = defaultdict(int)
        missing_by_center: dict[str, int] = defaultdict(int)
        raw_shapes: Counter[str] = Counter()
        raw_dtypes: Counter[str] = Counter()
        raw_lengths: Counter[int] = Counter()
        nonfinite = 0
        for key, feature_record in self.feature_by_key.items():
            feature_names = _channel_names(feature_record)
            feature_centres = _window_centers(feature_record)
            feature_channels += len(feature_names)
            feature_windows += len(feature_names) * len(feature_centres)
            raw_record = self.raw_records.get(key)
            if raw_record is None:
                missing_by_patient[key[0]] += len(feature_names)
                center = str(feature_record.get("source_center", feature_record.get("center", "unknown"))).lower()
                missing_by_center[center] += len(feature_names)
                continue
            raw = np.asarray(_sample(raw_record).get("raw_waveform"))
            raw_shapes[str(tuple(raw.shape))] += 1
            raw_dtypes[str(raw.dtype)] += 1
            if raw.ndim != 2:
                raise ValueError(f"raw_waveform must be [channel,time] for {key}, got {raw.shape}")
            raw_lengths[int(raw.shape[-1])] += 1
            nonfinite += int(raw.size - np.isfinite(raw).sum())
            raw_names = _channel_names(raw_record)
            raw_channels += len(raw_names)
            raw_windows += len(raw_names) * len(feature_centres)
            overlap = set(feature_names) & set(raw_names)
            matched_channels += len(overlap)
            matched_windows += len(overlap) * len(feature_centres)
            missing_channel_count = len(feature_names) - len(overlap)
            if missing_channel_count:
                missing_by_patient[key[0]] += missing_channel_count
                center = str(feature_record.get("source_center", feature_record.get("center", "unknown"))).lower()
                missing_by_center[center] += missing_channel_count
        matched_records = len(matched_keys)
        patient_rate = len(matched_subjects) / max(len(feature_subjects), 1)
        record_rate = matched_records / max(len(self.feature_by_key), 1)
        channel_rate = matched_channels / max(feature_channels, 1)
        window_rate = matched_windows / max(feature_windows, 1)
        return {
            "feature_cache_path": self.feature_cache_path,
            "raw_cache_path": self.raw_cache_path,
            "n_feature_patients": len(feature_subjects), "n_raw_patients": len(raw_subjects), "n_matched_patients": len(matched_subjects),
            "n_feature_records": len(self.feature_by_key), "n_raw_records": len(self.raw_records), "n_matched_records": matched_records,
            "n_feature_channels": feature_channels, "n_raw_channels": raw_channels, "n_matched_channels": matched_channels,
            "n_feature_windows": feature_windows, "n_raw_windows": raw_windows, "n_matched_windows": matched_windows,
            "patient_match_rate": patient_rate, "record_match_rate": record_rate, "channel_match_rate": channel_rate, "window_match_rate": window_rate,
            "duplicate_feature_keys": duplicate_feature_keys, "duplicate_raw_keys": duplicate_raw_keys,
            "missing_raw_by_center": dict(sorted(missing_by_center.items())), "missing_raw_by_patient": dict(sorted(missing_by_patient.items())),
            "raw_shapes": dict(raw_shapes), "raw_dtypes": dict(raw_dtypes), "raw_sample_lengths": dict(raw_lengths), "raw_nonfinite_count": nonfinite,
            "raw_target_samples": self.raw_target_samples, "raw_target_sampling_rate": self.raw_target_sampling_rate,
            "raw_patient_coverage": patient_rate,
            # Formal acceptance is decided by the configured channel/window
            # thresholds. Individual missing windows are maskable, not an
            # automatic structural audit failure.
            "status": "passed" if nonfinite == 0 else "failed",
        }

    def assert_formal_coverage(self, *, min_channel_match_rate: float, min_window_match_rate: float, expected_patients: int | None = None) -> None:
        if expected_patients is not None and int(self.audit["n_matched_patients"]) != int(expected_patients):
            raise ValueError(f"N6 raw alignment requires {expected_patients} matched patients, got {self.audit['n_matched_patients']}")
        if float(self.audit["channel_match_rate"]) < float(min_channel_match_rate):
            raise ValueError(f"Raw channel_match_rate={self.audit['channel_match_rate']:.4f} below required {min_channel_match_rate:.4f}")
        if float(self.audit["window_match_rate"]) < float(min_window_match_rate):
            raise ValueError(f"Raw window_match_rate={self.audit['window_match_rate']:.4f} below required {min_window_match_rate:.4f}")
        if int(self.audit["raw_nonfinite_count"]) != 0:
            raise ValueError("Raw cache contains non-finite values.")

    def aligned_windows(self, feature_sample: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``[W,C,T]`` raw data, mask, and per-channel availability."""
        key = _record_key(feature_sample)
        raw_record = self.raw_records.get(key)
        feature_names = _channel_names(feature_sample)
        centres = _window_centers(feature_sample)
        output = np.zeros((len(centres), len(feature_names), self.raw_target_samples), dtype=np.float32)
        mask = np.zeros((len(centres), len(feature_names)), dtype=bool)
        if raw_record is None:
            return output, mask, np.zeros((len(feature_names),), dtype=bool)
        raw_sample = _sample(raw_record)
        raw = np.asarray(raw_sample.get("raw_waveform"), dtype=np.float32)
        sfreq = float(raw_sample.get("raw_temporal_sfreq", 0.0) or 0.0)
        duration = float(raw_sample.get("raw_temporal_duration_sec", 0.0) or 0.0)
        if raw.ndim != 2 or sfreq != self.raw_target_sampling_rate or duration <= 0:
            raise ValueError(f"Raw record {key} violates fixed raw sampling-rate/duration contract.")
        if not np.isfinite(raw).all():
            raise ValueError(f"Raw record {key} contains non-finite values.")
        raw_names = _channel_names(raw_record)
        raw_index = {name: index for index, name in enumerate(raw_names)}
        scale = np.asarray(_field(feature_sample, "feature_scale_used_secs", []), dtype=np.float64)
        if scale.size != len(centres):
            scale = np.full(len(centres), self.raw_target_samples / sfreq, dtype=np.float64)
        for window_idx, (center, length_sec) in enumerate(zip(centres, scale)):
            if not np.isfinite(length_sec) or length_sec <= 0:
                continue
            source_length = int(round(float(length_sec) * sfreq))
            if source_length != self.raw_target_samples:
                raise ValueError("Raw target samples must match aligned feature window duration at the configured sampling rate.")
            center_sample = int(round((duration / 2.0 + float(center)) * sfreq))
            start = center_sample - source_length // 2
            end = start + source_length
            if start < 0 or end > raw.shape[1]:
                continue
            for channel_idx, name in enumerate(feature_names):
                raw_idx = raw_index.get(name)
                if raw_idx is None:
                    continue
                output[window_idx, channel_idx] = raw[raw_idx, start:end]
                mask[window_idx, channel_idx] = True
        return output, mask, mask.any(axis=0)


def build_raw_alignment_store(
    feature_records: Iterable[Mapping[str, Any]],
    args: Any,
    *,
    output_audit_path: str | Path | None = None,
) -> RawAlignmentStore:
    return RawAlignmentStore(
        feature_records,
        feature_cache_path=str(getattr(args, "window_cache_path")),
        raw_cache_path=str(getattr(args, "raw_window_cache_path")),
        raw_target_samples=int(getattr(args, "raw_target_samples", 500)),
        raw_target_sampling_rate=float(getattr(args, "raw_target_sampling_rate", 250.0)),
        output_audit_path=output_audit_path,
    )


__all__ = ["RawAlignmentStore", "build_raw_alignment_store"]
