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


TASK1_PHYSICS_FEATURE_NAMES = (
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the Task 1 Clean-NEZ enriched cache by combining the "
            "success/failure feature cache with aligned raw waveforms."
        )
    )
    parser.add_argument("--feature-cache-path", type=Path, required=True)
    parser.add_argument("--raw-cache-path", type=Path, required=True)
    parser.add_argument("--output-cache-path", type=Path, required=True)
    parser.add_argument("--audit-path", type=Path, default=None)
    parser.add_argument("--allowed-subjects-ledger", type=Path, default=None)
    parser.add_argument("--require-n-patients", type=int, default=90)
    parser.add_argument("--window-sec-fallback", type=float, default=2.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def load_cache(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Cache does not exist: {path}")

    with path.open("rb") as fin:
        payload = pickle.load(fin)

    if not isinstance(payload, dict):
        raise ValueError(f"Cache top-level object must be dict: {path}")
    if not isinstance(payload.get("run_records"), list):
        raise ValueError(f"Cache missing run_records list: {path}")
    if not isinstance(payload.get("patient_index"), dict):
        raise ValueError(f"Cache missing patient_index dict: {path}")

    return payload


def read_allowed_subjects(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"Allowed-subject ledger does not exist: {path}")

    with path.open("r", newline="", encoding="utf-8-sig") as fin:
        rows = list(csv.DictReader(fin))

    if not rows:
        raise ValueError(f"Allowed-subject ledger is empty: {path}")

    if "subject_id" in rows[0]:
        key = "subject_id"
    elif "patient_id" in rows[0]:
        key = "patient_id"
    else:
        raise ValueError(
            "Allowed-subject ledger must contain subject_id or patient_id."
        )

    subjects = {
        str(row.get(key, "")).strip()
        for row in rows
        if str(row.get(key, "")).strip()
    }
    if not subjects:
        raise ValueError("Allowed-subject ledger contains no valid subjects.")

    return subjects


def record_key(record: dict[str, Any]) -> tuple[str, str]:
    return str(record.get("subject_id")), str(record.get("run_id"))


def get_feature_names(
    payload: dict[str, Any],
    record: dict[str, Any],
) -> list[str]:
    sample = record.get("sample", {})
    names = sample.get("window_feature_names")
    if names is None:
        names = record.get("window_feature_names")
    if names is None:
        names = payload.get("window_feature_names")
    if names is None:
        raise ValueError(
            f"Missing window_feature_names for record={record_key(record)}"
        )
    return list(map(str, names))


def get_window_durations(
    sample: dict[str, Any],
    n_windows: int,
    fallback: float,
) -> tuple[np.ndarray, str]:
    used = sample.get("feature_scale_used_secs")
    if used is not None:
        values = np.asarray(used, dtype=np.float64).reshape(-1)
        if (
            values.size == n_windows
            and np.isfinite(values).all()
            and np.all(values > 0)
        ):
            return values, "feature_scale_used_secs"

    scales = sample.get("feature_scales_sec")
    if scales is not None:
        values = np.asarray(scales, dtype=np.float64).reshape(-1)
        if values.size and np.isfinite(values[0]) and values[0] > 0:
            return (
                np.full(n_windows, float(values[0]), dtype=np.float64),
                "feature_scales_sec",
            )

    if not np.isfinite(fallback) or fallback <= 0:
        raise ValueError(f"Invalid window-sec fallback: {fallback}")

    return (
        np.full(n_windows, float(fallback), dtype=np.float64),
        "fallback",
    )


def validate_alignment(
    feature_record: dict[str, Any],
    raw_record: dict[str, Any],
) -> None:
    feature_channels = list(
        map(str, feature_record.get("channel_names_norm", []))
    )
    raw_channels = list(
        map(str, raw_record.get("channel_names_norm", []))
    )

    if feature_channels != raw_channels:
        raise ValueError(
            f"Feature/raw channel mismatch for {record_key(feature_record)}.\n"
            f"feature={feature_channels}\n"
            f"raw={raw_channels}"
        )

    feature_labels = np.asarray(
        feature_record.get("labels"),
        dtype=np.float32,
    )
    raw_labels = np.asarray(
        raw_record.get("labels"),
        dtype=np.float32,
    )

    if (
        feature_labels.shape != raw_labels.shape
        or not np.array_equal(feature_labels, raw_labels)
    ):
        raise ValueError(
            f"Feature/raw label mismatch for {record_key(feature_record)}"
        )


def _sample_id(record: dict[str, Any]) -> str:
    sample = record.get("sample", {})
    return str(sample.get("sample_id", "")).strip()


def _source_seizure_id(record: dict[str, Any]) -> str:
    metadata = record.get("metadata", {})
    return str(metadata.get("source_seizure_id", "")).strip()


def _raw_candidate_matches(
    feature_record: dict[str, Any],
    raw_record: dict[str, Any],
) -> bool:
    try:
        validate_alignment(feature_record, raw_record)
    except ValueError:
        return False

    raw_sample = raw_record.get("sample", {})
    if raw_sample.get("raw_waveform") is None:
        return False

    feature_sample = feature_record.get("sample", {})
    feature_centers = feature_sample.get(
        "window_relative_centers_sec"
    )
    raw_centers = raw_sample.get(
        "window_relative_centers_sec"
    )
    if feature_centers is not None and raw_centers is not None:
        feature_centers = np.asarray(
            feature_centers,
            dtype=np.float32,
        )
        raw_centers = np.asarray(
            raw_centers,
            dtype=np.float32,
        )
        if (
            feature_centers.shape != raw_centers.shape
            or not np.allclose(
                feature_centers,
                raw_centers,
                atol=1e-4,
                rtol=0.0,
            )
        ):
            return False

    return True


def _raw_records_equivalent(
    first: dict[str, Any],
    second: dict[str, Any],
) -> bool:
    if list(map(str, first.get("channel_names_norm", []))) != list(
        map(str, second.get("channel_names_norm", []))
    ):
        return False

    first_labels = np.asarray(
        first.get("labels"),
        dtype=np.float32,
    )
    second_labels = np.asarray(
        second.get("labels"),
        dtype=np.float32,
    )
    if (
        first_labels.shape != second_labels.shape
        or not np.array_equal(first_labels, second_labels)
    ):
        return False

    first_sample = first.get("sample", {})
    second_sample = second.get("sample", {})

    scalar_keys = (
        "raw_temporal_sfreq",
        "raw_temporal_duration_sec",
        "raw_valid_samples",
        "raw_valid_start_sample",
    )
    for key in scalar_keys:
        if first_sample.get(key) != second_sample.get(key):
            return False

    first_centers = first_sample.get(
        "window_relative_centers_sec"
    )
    second_centers = second_sample.get(
        "window_relative_centers_sec"
    )
    if first_centers is None and second_centers is not None:
        return False
    if first_centers is not None and second_centers is None:
        return False
    if first_centers is not None:
        first_centers = np.asarray(
            first_centers,
            dtype=np.float32,
        )
        second_centers = np.asarray(
            second_centers,
            dtype=np.float32,
        )
        if (
            first_centers.shape != second_centers.shape
            or not np.allclose(
                first_centers,
                second_centers,
                atol=1e-4,
                rtol=0.0,
            )
        ):
            return False

    first_waveform = first_sample.get("raw_waveform")
    second_waveform = second_sample.get("raw_waveform")
    if first_waveform is None or second_waveform is None:
        return first_waveform is None and second_waveform is None

    first_waveform = np.asarray(
        first_waveform,
        dtype=np.float32,
    )
    second_waveform = np.asarray(
        second_waveform,
        dtype=np.float32,
    )
    return (
        first_waveform.shape == second_waveform.shape
        and np.array_equal(
            first_waveform,
            second_waveform,
        )
    )


def resolve_raw_record(
    feature_record: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> tuple[dict[str, Any], bool]:
    if not candidates:
        raise ValueError(
            "No matching raw record for feature record: "
            f"{record_key(feature_record)}"
        )

    compatible = [
        candidate
        for candidate in candidates
        if _raw_candidate_matches(
            feature_record,
            candidate,
        )
    ]
    if not compatible:
        raise ValueError(
            "Raw records exist for the key, but none align "
            "with feature channels/labels/window centers: "
            f"{record_key(feature_record)}"
        )

    feature_sample_id = _sample_id(feature_record)
    if feature_sample_id:
        exact_sample = [
            candidate
            for candidate in compatible
            if _sample_id(candidate) == feature_sample_id
        ]
        if exact_sample:
            compatible = exact_sample

    feature_source_seizure = _source_seizure_id(
        feature_record
    )
    if feature_source_seizure:
        exact_source = [
            candidate
            for candidate in compatible
            if _source_seizure_id(candidate)
            == feature_source_seizure
        ]
        if exact_source:
            compatible = exact_source

    if len(compatible) == 1:
        return compatible[0], False

    first = compatible[0]
    if all(
        _raw_records_equivalent(first, other)
        for other in compatible[1:]
    ):
        return first, True

    details = [
        {
            "sample_id": _sample_id(candidate),
            "source_seizure_id": _source_seizure_id(
                candidate
            ),
            "raw_shape": list(
                np.asarray(
                    candidate.get("sample", {}).get(
                        "raw_waveform"
                    )
                ).shape
            ),
        }
        for candidate in compatible
    ]
    raise ValueError(
        "Ambiguous non-equivalent raw duplicates for "
        f"{record_key(feature_record)}: {details}"
    )


def reconstruct_raw_window(
    raw_waveform: np.ndarray,
    raw_sample: dict[str, Any],
    relative_center_sec: float,
    window_duration_sec: float,
) -> np.ndarray:
    sfreq = float(raw_sample.get("raw_temporal_sfreq", 0.0))
    if not np.isfinite(sfreq) or sfreq <= 0:
        raise ValueError(f"Invalid raw_temporal_sfreq={sfreq}")

    total_samples = int(raw_waveform.shape[-1])
    duration_sec = float(
        raw_sample.get(
            "raw_temporal_duration_sec",
            total_samples / sfreq,
        )
    )

    # _build_centered_raw_waveform places seizure onset at the midpoint
    # of the fixed-duration raw waveform.
    onset_sample = 0.5 * duration_sec * sfreq

    window_samples = max(
        4,
        int(round(float(window_duration_sec) * sfreq)),
    )
    center_sample = onset_sample + float(relative_center_sec) * sfreq
    start = int(round(center_sample - 0.5 * window_samples))
    end = start + window_samples

    valid_start = int(raw_sample.get("raw_valid_start_sample", 0))
    valid_samples = int(
        raw_sample.get("raw_valid_samples", total_samples)
    )
    valid_end = min(total_samples, valid_start + valid_samples)

    # Permit a very small rounding discrepancy introduced by resampling.
    tolerance = 3
    if start < valid_start - tolerance or end > valid_end + tolerance:
        raise ValueError(
            "Requested raw window falls outside valid raw samples: "
            f"center={relative_center_sec}, "
            f"window_sec={window_duration_sec}, "
            f"requested=[{start},{end}), "
            f"valid=[{valid_start},{valid_end})"
        )

    start = max(start, valid_start)
    end = min(end, valid_end)

    if end - start < 4:
        raise ValueError(
            f"Raw window is too short after clipping: [{start},{end})"
        )

    return np.asarray(
        raw_waveform[:, start:end],
        dtype=np.float32,
    )


def enrich_record(
    feature_payload: dict[str, Any],
    feature_record: dict[str, Any],
    raw_record: dict[str, Any],
    window_sec_fallback: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    validate_alignment(feature_record, raw_record)

    feature_sample = feature_record.get("sample", {})
    raw_sample = raw_record.get("sample", {})

    feature_names = get_feature_names(
        feature_payload,
        feature_record,
    )
    feature_tensor = np.asarray(
        feature_sample.get("window_features"),
        dtype=np.float32,
    )
    centers = np.asarray(
        feature_sample.get("window_relative_centers_sec"),
        dtype=np.float32,
    )

    if feature_tensor.ndim != 3:
        raise ValueError(
            "window_features must have shape [T,C,F], "
            f"got {feature_tensor.shape} "
            f"for {record_key(feature_record)}"
        )

    n_windows, n_channels, n_features = feature_tensor.shape

    if centers.shape != (n_windows,):
        raise ValueError(
            f"Window-center mismatch for {record_key(feature_record)}: "
            f"centers={centers.shape}, windows={n_windows}"
        )

    if len(feature_names) != n_features:
        raise ValueError(
            f"Feature-name mismatch for {record_key(feature_record)}: "
            f"names={len(feature_names)}, dim={n_features}"
        )

    existing = sorted(
        set(feature_names).intersection(
            TASK1_PHYSICS_FEATURE_NAMES
        )
    )
    if existing:
        raise ValueError(
            "Input feature cache already contains Task 1 physics "
            f"features for {record_key(feature_record)}: {existing}"
        )

    missing_base = [
        name
        for name in WINDOW_NODE_FEATURE_NAMES
        if name not in feature_names
    ]
    if missing_base:
        raise ValueError(
            f"Missing canonical 20-D base features for "
            f"{record_key(feature_record)}: {missing_base}"
        )

    canonical_indices = [
        feature_names.index(name)
        for name in WINDOW_NODE_FEATURE_NAMES
    ]
    canonical_base = feature_tensor[:, :, canonical_indices]

    # These functions derive seizure-level early dynamics and burstness
    # from the existing aligned 20-D window feature sequence.
    clinical_full = _clinical_onset_core_features(
        canonical_base,
        centers,
    )
    burstness_full = _burstness_features(
        canonical_base,
        centers,
    )

    clinical_names = list(CLINICAL_ONSET_CORE_FEATURE_NAMES)
    burstness_names = list(BURSTNESS_FEATURE_NAMES)
    hfo_names = list(HFO_LITE_FEATURE_NAMES)

    clinical_selected_names = list(
        TASK1_PHYSICS_FEATURE_NAMES[:6]
    )
    burstness_selected_names = list(
        TASK1_PHYSICS_FEATURE_NAMES[6:8]
    )
    hfo_selected_names = list(
        TASK1_PHYSICS_FEATURE_NAMES[8:12]
    )

    clinical_selected = clinical_full[
        :,
        :,
        [
            clinical_names.index(name)
            for name in clinical_selected_names
        ],
    ]
    burstness_selected = burstness_full[
        :,
        :,
        [
            burstness_names.index(name)
            for name in burstness_selected_names
        ],
    ]

    raw_waveform = raw_sample.get("raw_waveform")
    if raw_waveform is None:
        raise ValueError(
            f"raw_waveform is missing for {record_key(raw_record)}"
        )

    raw_waveform = np.asarray(
        raw_waveform,
        dtype=np.float32,
    )

    if raw_waveform.ndim != 2:
        raise ValueError(
            "raw_waveform must have shape [C,N], "
            f"got {raw_waveform.shape} "
            f"for {record_key(raw_record)}"
        )

    if raw_waveform.shape[0] != n_channels:
        raise ValueError(
            f"Feature/raw channel-count mismatch for "
            f"{record_key(feature_record)}: "
            f"feature={n_channels}, raw={raw_waveform.shape[0]}"
        )

    sfreq = float(raw_sample.get("raw_temporal_sfreq", 0.0))
    if not np.isfinite(sfreq) or sfreq <= 0:
        raise ValueError(
            f"Invalid raw_temporal_sfreq for "
            f"{record_key(raw_record)}: {sfreq}"
        )

    raw_centers = raw_sample.get(
        "window_relative_centers_sec"
    )
    if raw_centers is not None:
        raw_centers = np.asarray(
            raw_centers,
            dtype=np.float32,
        )
        if (
            raw_centers.shape != centers.shape
            or not np.allclose(
                raw_centers,
                centers,
                atol=1e-4,
                rtol=0.0,
            )
        ):
            raise ValueError(
                f"Feature/raw center mismatch for "
                f"{record_key(feature_record)}"
            )

    durations, duration_source = get_window_durations(
        feature_sample,
        n_windows,
        window_sec_fallback,
    )

    hfo_selected_indices = [
        hfo_names.index(name)
        for name in hfo_selected_names
    ]
    hfo_rows: list[np.ndarray] = []

    for window_idx in range(n_windows):
        raw_window = reconstruct_raw_window(
            raw_waveform=raw_waveform,
            raw_sample=raw_sample,
            relative_center_sec=float(centers[window_idx]),
            window_duration_sec=float(durations[window_idx]),
        )

        hfo_full = _hfo_lite_window_features(
            raw_window,
            sfreq=sfreq,
        )
        hfo_rows.append(
            hfo_full[:, hfo_selected_indices]
        )

    hfo_selected = np.stack(
        hfo_rows,
        axis=0,
    ).astype(np.float32, copy=False)

    task1_tensor = np.concatenate(
        [
            clinical_selected,
            burstness_selected,
            hfo_selected,
        ],
        axis=-1,
    ).astype(np.float32, copy=False)

    expected_shape = (n_windows, n_channels, 12)
    if task1_tensor.shape != expected_shape:
        raise RuntimeError(
            f"Expected Task 1 feature shape {expected_shape}, "
            f"got {task1_tensor.shape}"
        )

    if not np.isfinite(task1_tensor).all():
        raise ValueError(
            f"Non-finite Task 1 features for "
            f"{record_key(feature_record)}"
        )

    enriched_tensor = np.concatenate(
        [feature_tensor, task1_tensor],
        axis=-1,
    ).astype(np.float32, copy=False)
    enriched_names = (
        feature_names
        + list(TASK1_PHYSICS_FEATURE_NAMES)
    )

    output_record = copy.deepcopy(feature_record)
    output_sample = dict(output_record.get("sample", {}))
    output_sample["window_features"] = enriched_tensor
    output_sample["window_feature_names"] = enriched_names
    output_sample["task1_clean_nez_feature_source"] = {
        "base_feature_cache": "success_failure_feature_v1",
        "raw_cache": "success_failure_raw_v1",
        "early_onset_features": (
            "derived_from_existing_window_features"
        ),
        "burstness_features": (
            "derived_from_existing_window_features"
        ),
        "hfo_lite_features": (
            "derived_from_aligned_raw_waveform"
        ),
        "physics_feature_names": list(
            TASK1_PHYSICS_FEATURE_NAMES
        ),
    }
    output_record["sample"] = output_sample

    hfo_fraction = float(
        np.count_nonzero(hfo_selected)
        / max(hfo_selected.size, 1)
    )

    audit = {
        "subject_id": str(
            feature_record.get("subject_id")
        ),
        "run_id": str(feature_record.get("run_id")),
        "n_windows": int(n_windows),
        "n_channels": int(n_channels),
        "input_feature_dim": int(n_features),
        "task1_feature_dim": 12,
        "output_feature_dim": int(
            enriched_tensor.shape[-1]
        ),
        "raw_sfreq": float(sfreq),
        "window_duration_source": duration_source,
        "hfo_nonzero_fraction": hfo_fraction,
    }

    return output_record, audit


def main() -> None:
    args = build_parser().parse_args()

    if (
        args.output_cache_path.exists()
        and not args.overwrite
    ):
        raise FileExistsError(
            "Output cache exists; pass --overwrite to "
            f"replace it: {args.output_cache_path}"
        )

    feature_payload = load_cache(
        args.feature_cache_path
    )
    raw_payload = load_cache(args.raw_cache_path)
    allowed_subjects = read_allowed_subjects(
        args.allowed_subjects_ledger
    )

    raw_index: dict[
        tuple[str, str],
        list[dict[str, Any]],
    ] = {}
    for raw_record in raw_payload["run_records"]:
        raw_subject_id = str(
            raw_record.get("subject_id")
        )
        if (
            allowed_subjects is not None
            and raw_subject_id not in allowed_subjects
        ):
            continue
        key = record_key(raw_record)
        raw_index.setdefault(key, []).append(
            raw_record
        )

    duplicate_raw_keys = {
        key: len(candidates)
        for key, candidates in raw_index.items()
        if len(candidates) > 1
    }
    if duplicate_raw_keys:
        print(
            "[Task1 cache] duplicate raw keys detected; "
            "equivalent duplicates will be resolved safely. "
            f"n_duplicate_keys={len(duplicate_raw_keys)}",
            flush=True,
        )

    selected_feature_records: list[
        dict[str, Any]
    ] = []
    for record in feature_payload["run_records"]:
        subject_id = str(record.get("subject_id"))
        if (
            allowed_subjects is not None
            and subject_id not in allowed_subjects
        ):
            continue
        selected_feature_records.append(record)

    if not selected_feature_records:
        raise ValueError(
            "No feature records were selected."
        )

    selected_subjects = {
        str(record["subject_id"])
        for record in selected_feature_records
    }

    if allowed_subjects is not None:
        missing_subjects = sorted(
            allowed_subjects - selected_subjects
        )
        if missing_subjects:
            raise ValueError(
                "Allowed subjects missing from feature "
                f"cache: {missing_subjects}"
            )

    if (
        args.require_n_patients > 0
        and len(selected_subjects)
        != args.require_n_patients
    ):
        raise ValueError(
            f"require_n_patients="
            f"{args.require_n_patients}, "
            f"but selected {len(selected_subjects)}"
        )

    enriched_records: list[
        dict[str, Any]
    ] = []
    audit_rows: list[dict[str, Any]] = []

    total_records = len(selected_feature_records)

    for index, feature_record in enumerate(
        selected_feature_records,
        start=1,
    ):
        key = record_key(feature_record)
        raw_record, duplicate_resolved = (
            resolve_raw_record(
                feature_record,
                raw_index.get(key, []),
            )
        )

        enriched_record, audit = enrich_record(
            feature_payload=feature_payload,
            feature_record=feature_record,
            raw_record=raw_record,
            window_sec_fallback=(
                args.window_sec_fallback
            ),
        )
        enriched_records.append(enriched_record)
        audit["equivalent_raw_duplicate_resolved"] = bool(
            duplicate_resolved
        )
        audit_rows.append(audit)

        if (
            index == 1
            or index % 10 == 0
            or index == total_records
        ):
            print(
                "[Task1 cache] processed "
                f"{index}/{total_records} records",
                flush=True,
            )

    selected_patient_index = {
        str(subject_id): copy.deepcopy(metadata)
        for subject_id, metadata
        in feature_payload["patient_index"].items()
        if str(subject_id) in selected_subjects
    }

    if set(selected_patient_index) != selected_subjects:
        missing_index = sorted(
            selected_subjects
            - set(selected_patient_index)
        )
        extra_index = sorted(
            set(selected_patient_index)
            - selected_subjects
        )
        raise ValueError(
            "patient_index mismatch after filtering: "
            f"missing={missing_index}, extra={extra_index}"
        )

    first_names = list(
        enriched_records[0]["sample"][
            "window_feature_names"
        ]
    )

    for record in enriched_records:
        sample = record["sample"]
        names = list(
            sample["window_feature_names"]
        )
        tensor = np.asarray(
            sample["window_features"]
        )

        if names != first_names:
            raise ValueError(
                "Inconsistent window_feature_names "
                "across enriched records."
            )

        if tensor.ndim != 3:
            raise ValueError(
                f"Invalid enriched tensor shape for "
                f"{record_key(record)}: {tensor.shape}"
            )

        if tensor.shape[-1] != len(names):
            raise ValueError(
                "Feature dimension/name mismatch for "
                f"{record_key(record)}: "
                f"dim={tensor.shape[-1]}, "
                f"names={len(names)}"
            )

    output_payload = dict(feature_payload)
    output_payload["cache_version"] = (
        "task1_clean_nez_enriched_feature_v1"
    )
    output_payload["source_feature_cache"] = str(
        args.feature_cache_path
    )
    output_payload["source_raw_cache"] = str(
        args.raw_cache_path
    )
    output_payload["window_feature_names"] = (
        first_names
    )
    output_payload["window_feature_groups"] = {
        "base": True,
        "task1_clean_nez": list(
            TASK1_PHYSICS_FEATURE_NAMES
        ),
    }
    output_payload["run_records"] = (
        enriched_records
    )
    output_payload["patient_index"] = (
        selected_patient_index
    )

    args.output_cache_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    with args.output_cache_path.open("wb") as fout:
        pickle.dump(
            output_payload,
            fout,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    audit_path = (
        args.audit_path
        if args.audit_path is not None
        else args.output_cache_path.with_suffix(
            ".audit.json"
        )
    )
    audit_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary = {
        "feature_cache_path": str(
            args.feature_cache_path
        ),
        "raw_cache_path": str(
            args.raw_cache_path
        ),
        "output_cache_path": str(
            args.output_cache_path
        ),
        "allowed_subjects_ledger": (
            str(args.allowed_subjects_ledger)
            if args.allowed_subjects_ledger
            is not None
            else None
        ),
        "n_patients": len(
            selected_patient_index
        ),
        "n_run_records": len(enriched_records),
        "input_feature_dim": (
            len(first_names)
            - len(TASK1_PHYSICS_FEATURE_NAMES)
        ),
        "task1_feature_dim": len(
            TASK1_PHYSICS_FEATURE_NAMES
        ),
        "output_feature_dim": len(first_names),
        "task1_feature_names": list(
            TASK1_PHYSICS_FEATURE_NAMES
        ),
        "all_finite": True,
        "mean_hfo_nonzero_fraction": float(
            np.mean(
                [
                    row["hfo_nonzero_fraction"]
                    for row in audit_rows
                ]
            )
        ),
        "n_duplicate_raw_keys": len(
            duplicate_raw_keys
        ),
        "n_records_with_equivalent_raw_duplicate_resolved": int(
            sum(
                bool(
                    row.get(
                        "equivalent_raw_duplicate_resolved",
                        False,
                    )
                )
                for row in audit_rows
            )
        ),
        "records": audit_rows,
    }

    with audit_path.open(
        "w",
        encoding="utf-8",
    ) as fout:
        json.dump(
            summary,
            fout,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("Task 1 enriched cache complete.")
    print(
        f"Patients: "
        f"{len(selected_patient_index)}"
    )
    print(
        f"Run records: {len(enriched_records)}"
    )
    print(
        "Input feature dimension: "
        f"{summary['input_feature_dim']}"
    )
    print(
        "Task 1 feature dimension: "
        f"{summary['task1_feature_dim']}"
    )
    print(
        "Output feature dimension: "
        f"{summary['output_feature_dim']}"
    )
    print(
        f"Output cache: "
        f"{args.output_cache_path}"
    )
    print(f"Audit: {audit_path}")


if __name__ == "__main__":
    main()
