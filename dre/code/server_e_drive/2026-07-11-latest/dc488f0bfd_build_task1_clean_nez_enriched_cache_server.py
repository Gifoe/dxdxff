"""Build the 32-dimensional Task 1 Clean-NEZ cache from feature + raw caches."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ez_features import (
    BURSTNESS_FEATURE_NAMES,
    CLINICAL_ONSET_CORE_FEATURE_NAMES,
    HFO_LITE_FEATURE_NAMES,
    WINDOW_NODE_FEATURE_NAMES,
    _burstness_features,
    _clinical_onset_core_features,
    _hfo_lite_window_features,
)


TASK1_NAMES = (
    "early_high_gamma_slope",
    "early_line_length_slope",
    "onset_latency_high_gamma",
    "onset_latency_line_length",
    "onset_rank_high_gamma",
    "onset_rank_line_length",
    "high_gamma_top20pct_mean",
    "line_length_top20pct_mean",
    "hfo80_150_event_rate",
    "hfo80_150_duration_fraction",
    "hfo80_150_mean_envelope_z",
    "hfo80_150_max_envelope_z",
)


def _load(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Cache does not exist: {path}")
    with path.open("rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Cache top-level object must be dict: {path}")
    if not isinstance(obj.get("run_records"), list):
        raise ValueError(f"Cache missing run_records list: {path}")
    if not isinstance(obj.get("patient_index"), dict):
        raise ValueError(f"Cache missing patient_index dict: {path}")
    return obj


def _read_ledger(path: Path) -> set[str]:
    if not path.exists():
        raise FileNotFoundError(f"Ledger does not exist: {path}")
    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Ledger is empty: {path}")
    key = "subject_id" if "subject_id" in rows[0] else "patient_id" if "patient_id" in rows[0] else None
    if key is None:
        raise ValueError("Ledger must contain subject_id or patient_id")
    result = {str(row.get(key, "")).strip() for row in rows}
    result.discard("")
    return result


def _sample(record: dict[str, Any]) -> dict[str, Any]:
    sample = record.get("sample")
    return sample if isinstance(sample, dict) else {}


def _value(record: dict[str, Any], key: str, default: Any = None) -> Any:
    sample = _sample(record)
    if key in sample:
        return sample[key]
    return record.get(key, default)


def _record_key(record: dict[str, Any]) -> tuple[str, str]:
    return str(record.get("subject_id")), str(record.get("run_id"))


def _channels(record: dict[str, Any]) -> list[str]:
    return list(map(str, record.get("channel_names_norm", [])))


def _centers(record: dict[str, Any]) -> np.ndarray | None:
    value = _value(record, "window_relative_centers_sec")
    if value is None:
        return None
    return np.asarray(value, dtype=np.float32).reshape(-1)


def _compatible(feature: dict[str, Any], raw: dict[str, Any]) -> bool:
    feature_sample = _sample(feature)
    feature_tensor = np.asarray(feature_sample.get("window_features"))
    raw_waveform = _value(raw, "raw_waveform")

    if feature_tensor.ndim != 3 or raw_waveform is None:
        return False

    raw_waveform = np.asarray(raw_waveform)
    if raw_waveform.ndim != 2:
        return False

    if _channels(feature) != _channels(raw):
        return False

    feature_labels = np.asarray(feature.get("labels"))
    raw_labels = np.asarray(raw.get("labels"))
    if feature_labels.shape != raw_labels.shape or not np.array_equal(feature_labels, raw_labels):
        return False

    feature_centers = _centers(feature)
    raw_centers = _centers(raw)
    if feature_centers is not None and raw_centers is not None:
        if feature_centers.shape != raw_centers.shape:
            return False
        if not np.allclose(feature_centers, raw_centers, atol=1e-4, rtol=0.0):
            return False

    if raw_waveform.shape[0] != feature_tensor.shape[1]:
        return False

    return True


def _same_scalar(a: dict[str, Any], b: dict[str, Any], key: str) -> bool:
    av = _value(a, key)
    bv = _value(b, key)
    if av is None and bv is None:
        return True
    return av == bv


def _equivalent_raw_records(candidates: list[dict[str, Any]]) -> bool:
    first = candidates[0]
    first_centers = _centers(first)
    first_wave = np.asarray(_value(first, "raw_waveform"))

    for record in candidates[1:]:
        if _channels(first) != _channels(record):
            return False

        if not np.array_equal(np.asarray(first.get("labels")), np.asarray(record.get("labels"))):
            return False

        centers = _centers(record)
        if first_centers is None or centers is None:
            if not (first_centers is None and centers is None):
                return False
        else:
            if first_centers.shape != centers.shape:
                return False
            if not np.allclose(first_centers, centers, atol=1e-4, rtol=0.0):
                return False

        for key in (
            "raw_temporal_sfreq",
            "raw_temporal_duration_sec",
            "raw_valid_samples",
            "raw_valid_start_sample",
        ):
            if not _same_scalar(first, record, key):
                return False

        wave = np.asarray(_value(record, "raw_waveform"))
        if first_wave.shape != wave.shape or first_wave.dtype != wave.dtype:
            return False
        if not np.array_equal(first_wave, wave):
            return False

    return True


def _candidate_info(record: dict[str, Any]) -> dict[str, Any]:
    waveform = np.asarray(_value(record, "raw_waveform"))
    centers = _centers(record)
    return {
        "sample_id": _value(record, "sample_id"),
        "source_seizure_id": _value(record, "source_seizure_id"),
        "seizure_onset_sec": _value(record, "seizure_onset_sec"),
        "n_channels": len(_channels(record)),
        "raw_shape": list(waveform.shape),
        "raw_sfreq": _value(record, "raw_temporal_sfreq"),
        "window_center_shape": None if centers is None else list(centers.shape),
    }


def resolve_raw_record(
    feature: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> tuple[dict[str, Any], bool]:
    compatible = [record for record in candidates if _compatible(feature, record)]
    if not compatible:
        raise ValueError(f"No compatible raw record for {_record_key(feature)}")

    for key in (
        "sample_id",
        "source_seizure_id",
        "seizure_onset_sec",
        "raw_temporal_duration_sec",
        "raw_temporal_sfreq",
        "raw_valid_samples",
        "raw_valid_start_sample",
    ):
        feature_value = _value(feature, key)
        if feature_value is None:
            continue
        same = [record for record in compatible if _value(record, key) == feature_value]
        if len(same) == 1:
            return same[0], False
        if same:
            compatible = same

    if len(compatible) == 1:
        return compatible[0], False

    if _equivalent_raw_records(compatible):
        return compatible[0], True

    raise ValueError(
        "Ambiguous non-equivalent raw records for "
        f"subject_id={feature.get('subject_id')}, run_id={feature.get('run_id')}: "
        f"{[_candidate_info(record) for record in compatible]}"
    )


def _get_feature_names(
    feature_payload: dict[str, Any],
    feature_record: dict[str, Any],
) -> list[str]:
    names = _value(feature_record, "window_feature_names")
    if names is None:
        names = feature_payload.get("window_feature_names")
    if names is None:
        raise ValueError(f"Missing window_feature_names for {_record_key(feature_record)}")
    return list(map(str, names))


def _window_durations(
    sample: dict[str, Any],
    n_windows: int,
    fallback: float,
) -> tuple[np.ndarray, str]:
    values = sample.get("feature_scale_used_secs")
    if values is not None:
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
        if arr.size == n_windows and np.isfinite(arr).all() and np.all(arr > 0):
            return arr, "feature_scale_used_secs"

    values = sample.get("feature_scales_sec")
    if values is not None:
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
        if arr.size > 0 and np.isfinite(arr[0]) and arr[0] > 0:
            return np.full(n_windows, float(arr[0]), dtype=np.float64), "feature_scales_sec"

    if not np.isfinite(fallback) or fallback <= 0:
        raise ValueError(f"Invalid fallback window duration: {fallback}")

    return np.full(n_windows, fallback, dtype=np.float64), "fallback"


def _extract_raw_window(
    raw_waveform: np.ndarray,
    raw_sample: dict[str, Any],
    center_sec: float,
    duration_sec: float,
    sfreq: float,
) -> np.ndarray:
    n_total = int(raw_waveform.shape[-1])
    raw_duration = float(raw_sample.get("raw_temporal_duration_sec", n_total / sfreq))
    onset_sample = 0.5 * raw_duration * sfreq

    window_samples = max(4, int(round(duration_sec * sfreq)))
    center_sample = onset_sample + center_sec * sfreq
    start = int(round(center_sample - 0.5 * window_samples))
    end = start + window_samples

    valid_start = int(raw_sample.get("raw_valid_start_sample", 0))
    valid_samples = int(raw_sample.get("raw_valid_samples", n_total))
    valid_end = min(n_total, valid_start + valid_samples)

    tolerance = 3
    if start < valid_start - tolerance or end > valid_end + tolerance:
        raise ValueError(
            "Requested raw window falls outside valid raw samples: "
            f"center={center_sec}, duration={duration_sec}, "
            f"requested=[{start},{end}), valid=[{valid_start},{valid_end})"
        )

    start = max(start, valid_start)
    end = min(end, valid_end)
    if end - start < 4:
        raise ValueError(f"Raw window too short after clipping: [{start},{end})")

    return np.asarray(raw_waveform[:, start:end], dtype=np.float32)


def _compute_hfo_per_window(
    raw_record: dict[str, Any],
    centers: np.ndarray,
    durations: np.ndarray,
) -> tuple[np.ndarray, float]:
    raw_sample = _sample(raw_record)
    raw_waveform = np.asarray(_value(raw_record, "raw_waveform"), dtype=np.float32)
    sfreq = float(_value(raw_record, "raw_temporal_sfreq", 0.0))

    if raw_waveform.ndim != 2:
        raise ValueError(f"raw_waveform must be [C,N], got {raw_waveform.shape}")
    if not np.isfinite(sfreq) or sfreq <= 0:
        raise ValueError(f"Invalid raw_temporal_sfreq={sfreq}")

    hfo_names = list(HFO_LITE_FEATURE_NAMES)
    hfo_indices = [hfo_names.index(name) for name in TASK1_NAMES[8:12]]
    rows: list[np.ndarray] = []

    for center, duration in zip(centers.tolist(), durations.tolist()):
        raw_window = _extract_raw_window(
            raw_waveform=raw_waveform,
            raw_sample=raw_sample,
            center_sec=float(center),
            duration_sec=float(duration),
            sfreq=sfreq,
        )
        hfo_full = _hfo_lite_window_features(raw_window, sfreq)
        rows.append(hfo_full[:, hfo_indices])

    return np.stack(rows, axis=0).astype(np.float32, copy=False), sfreq


def enrich_record(
    feature_payload: dict[str, Any],
    feature_record: dict[str, Any],
    raw_record: dict[str, Any],
    window_sec_fallback: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    feature_sample = _sample(feature_record)
    feature_names = _get_feature_names(feature_payload, feature_record)
    base = np.asarray(feature_sample.get("window_features"), dtype=np.float32)
    centers = np.asarray(
        feature_sample.get("window_relative_centers_sec"),
        dtype=np.float32,
    ).reshape(-1)

    if base.ndim != 3:
        raise ValueError(f"window_features must be [T,C,F], got {base.shape}")
    n_windows, n_channels, n_features = base.shape

    if centers.shape != (n_windows,):
        raise ValueError(
            f"Window-center mismatch for {_record_key(feature_record)}: "
            f"centers={centers.shape}, windows={n_windows}"
        )
    if len(feature_names) != n_features:
        raise ValueError(
            f"Feature-name mismatch for {_record_key(feature_record)}: "
            f"names={len(feature_names)}, dim={n_features}"
        )

    existing = sorted(set(feature_names).intersection(TASK1_NAMES))
    if existing:
        raise ValueError(
            f"Input cache already contains Task 1 features for "
            f"{_record_key(feature_record)}: {existing}"
        )

    missing_base = [name for name in WINDOW_NODE_FEATURE_NAMES if name not in feature_names]
    if missing_base:
        raise ValueError(
            f"Missing canonical base features for {_record_key(feature_record)}: "
            f"{missing_base}"
        )

    canonical_indices = [feature_names.index(name) for name in WINDOW_NODE_FEATURE_NAMES]
    canonical_base = base[:, :, canonical_indices]

    clinical = _clinical_onset_core_features(canonical_base, centers)
    burstness = _burstness_features(canonical_base, centers)

    clinical_names = list(CLINICAL_ONSET_CORE_FEATURE_NAMES)
    burstness_names = list(BURSTNESS_FEATURE_NAMES)

    clinical_pick = clinical[
        :,
        :,
        [clinical_names.index(name) for name in TASK1_NAMES[:6]],
    ]
    burstness_pick = burstness[
        :,
        :,
        [burstness_names.index(name) for name in TASK1_NAMES[6:8]],
    ]

    durations, duration_source = _window_durations(
        feature_sample,
        n_windows,
        window_sec_fallback,
    )
    hfo_pick, sfreq = _compute_hfo_per_window(raw_record, centers, durations)

    extra = np.concatenate(
        [clinical_pick, burstness_pick, hfo_pick],
        axis=-1,
    ).astype(np.float32, copy=False)

    expected = (n_windows, n_channels, 12)
    if extra.shape != expected:
        raise RuntimeError(f"Expected Task 1 feature shape {expected}, got {extra.shape}")
    if not np.isfinite(extra).all():
        raise ValueError(f"Non-finite Task 1 features for {_record_key(feature_record)}")

    enriched_features = np.concatenate([base, extra], axis=-1).astype(np.float32, copy=False)
    enriched_names = feature_names + list(TASK1_NAMES)

    output_record = copy.deepcopy(feature_record)
    output_sample = dict(feature_sample)
    output_sample["window_features"] = enriched_features
    output_sample["window_feature_names"] = enriched_names
    output_sample["task1_clean_nez_enriched"] = True
    output_record["sample"] = output_sample

    audit = {
        "subject_id": str(feature_record.get("subject_id")),
        "run_id": str(feature_record.get("run_id")),
        "n_windows": n_windows,
        "n_channels": n_channels,
        "raw_sfreq": sfreq,
        "window_duration_source": duration_source,
        "hfo_nonzero_fraction": float(
            np.count_nonzero(hfo_pick) / max(hfo_pick.size, 1)
        ),
    }
    return output_record, audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-cache-path", type=Path, required=True)
    parser.add_argument("--raw-cache-path", type=Path, required=True)
    parser.add_argument("--allowed-subjects-ledger", type=Path, required=True)
    parser.add_argument("--require-n-patients", type=int, default=90)
    parser.add_argument("--output-cache-path", type=Path, required=True)
    parser.add_argument("--audit-path", type=Path, required=True)
    parser.add_argument("--window-sec-fallback", type=float, default=2.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_path = args.output_cache_path
    audit_path = args.audit_path
    if (output_path.exists() or audit_path.exists()) and not args.overwrite:
        raise FileExistsError("Output exists; pass --overwrite")

    allowed = _read_ledger(args.allowed_subjects_ledger)
    if len(allowed) != args.require_n_patients:
        raise ValueError(
            f"Ledger contains {len(allowed)} unique subjects; "
            f"expected {args.require_n_patients}"
        )

    feature_payload = _load(args.feature_cache_path)
    raw_payload = _load(args.raw_cache_path)

    feature_records = [
        record
        for record in feature_payload["run_records"]
        if str(record.get("subject_id")) in allowed
    ]
    patient_index = {
        str(subject_id): copy.deepcopy(metadata)
        for subject_id, metadata in feature_payload["patient_index"].items()
        if str(subject_id) in allowed
    }

    if set(patient_index) != allowed:
        raise ValueError(
            "Ledger/patient_index mismatch: "
            f"missing={sorted(allowed - set(patient_index))}, "
            f"extra={sorted(set(patient_index) - allowed)}"
        )

    raw_index: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in raw_payload["run_records"]:
        if str(record.get("subject_id")) not in allowed:
            continue
        raw_index.setdefault(_record_key(record), []).append(record)

    n_duplicate_raw_keys = sum(len(candidates) > 1 for candidates in raw_index.values())

    enriched_records: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    n_equivalent_resolved = 0

    total = len(feature_records)
    for i, feature_record in enumerate(feature_records, start=1):
        key = _record_key(feature_record)
        raw_record, equivalent_resolved = resolve_raw_record(
            feature_record,
            raw_index.get(key, []),
        )
        n_equivalent_resolved += int(equivalent_resolved)

        enriched_record, row = enrich_record(
            feature_payload,
            feature_record,
            raw_record,
            args.window_sec_fallback,
        )
        row["equivalent_raw_duplicate_resolved"] = bool(equivalent_resolved)
        enriched_records.append(enriched_record)
        audit_rows.append(row)

        if i == 1 or i % 10 == 0 or i == total:
            print(f"[Task1 cache] processed {i}/{total} records", flush=True)

    first_names = list(enriched_records[0]["sample"]["window_feature_names"])
    for record in enriched_records:
        sample = record["sample"]
        tensor = np.asarray(sample["window_features"])
        names = list(sample["window_feature_names"])
        if names != first_names:
            raise ValueError("Inconsistent window_feature_names across records")
        if tensor.ndim != 3 or tensor.shape[-1] != 32:
            raise RuntimeError(
                f"Invalid enriched shape for {_record_key(record)}: {tensor.shape}"
            )
        if tensor.shape[-1] != len(names):
            raise RuntimeError(
                f"Feature-name mismatch for {_record_key(record)}: "
                f"dim={tensor.shape[-1]}, names={len(names)}"
            )
        if not np.isfinite(tensor).all():
            raise RuntimeError(f"Non-finite enriched features for {_record_key(record)}")

    output_payload = dict(feature_payload)
    output_payload["cache_version"] = "task1_clean_nez_v1"
    output_payload["source_feature_cache"] = str(args.feature_cache_path)
    output_payload["source_raw_cache"] = str(args.raw_cache_path)
    output_payload["run_records"] = enriched_records
    output_payload["patient_index"] = patient_index
    output_payload["window_feature_names"] = first_names

    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("wb") as f:
        pickle.dump(output_payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    report = {
        "n_patients": len(patient_index),
        "n_run_records": len(enriched_records),
        "input_feature_dim": 20,
        "task1_feature_dim": 12,
        "output_feature_dim": 32,
        "n_duplicate_raw_keys": n_duplicate_raw_keys,
        "n_records_with_equivalent_raw_duplicate_resolved": n_equivalent_resolved,
        "task1_feature_names": list(TASK1_NAMES),
        "all_finite": True,
        "records": audit_rows,
    }
    audit_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print()
    print("Task 1 enriched cache complete.")
    print(f"Patients: {report['n_patients']}")
    print(f"Run records: {report['n_run_records']}")
    print(f"Input feature dimension: {report['input_feature_dim']}")
    print(f"Task 1 feature dimension: {report['task1_feature_dim']}")
    print(f"Output feature dimension: {report['output_feature_dim']}")
    print(f"Duplicate raw keys: {report['n_duplicate_raw_keys']}")
    print(
        "Equivalent duplicate resolutions: "
        f"{report['n_records_with_equivalent_raw_duplicate_resolved']}"
    )
    print(f"Output cache: {output_path}")
    print(f"Audit: {audit_path}")


if __name__ == "__main__":
    main()