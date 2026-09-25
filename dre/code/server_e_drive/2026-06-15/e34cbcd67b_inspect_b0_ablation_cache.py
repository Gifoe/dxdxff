from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np


try:
    from ez_features import BASE_SPECTRAL_FEATURE_NAMES, WINDOW_NODE_FEATURE_NAMES
except Exception:
    BASE_SPECTRAL_FEATURE_NAMES = (
        "log_bp_delta",
        "log_bp_theta",
        "log_bp_alpha",
        "log_bp_beta",
        "log_bp_low_gamma",
        "log_bp_high_gamma",
        "log_total_power",
        "rms",
        "variance",
        "line_length_per_sec",
        "spectral_entropy",
        "hjorth_mobility",
        "hjorth_complexity",
    )
    WINDOW_NODE_FEATURE_NAMES = tuple(BASE_SPECTRAL_FEATURE_NAMES) + (
        "degree_norm",
        "strength_norm",
        "clustering_coeff",
        "eigenvector_centrality",
        "pagerank",
        "kcore_norm",
        "local_efficiency",
    )


SAMPLE_REQUIRED_FOR_FLATTEN = (
    "sample_id",
    "analysis_phase",
    "start_sec",
    "end_sec",
    "seizure_onset_sec",
    "seizure_offset_sec",
    "ictal_duration_total_sec",
    "ictal_duration_used_sec",
    "feature_scales_sec",
    "feature_scale_used_secs",
    "raw_temporal_duration_sec",
    "raw_temporal_sfreq",
    "raw_valid_samples",
    "spectral_features",
    "graph_features",
    "raw_waveform",
)

RUN_REQUIRED_FOR_FLATTEN = (
    "subject_id",
    "run_id",
    "task",
    "phase_group",
    "channel_names_norm",
    "labels",
    "sample",
)


def _shape(value: Any) -> list[int] | None:
    try:
        arr = np.asarray(value)
    except Exception:
        return None
    return [int(dim) for dim in arr.shape]


def _is_nonzero_adjacency(adjacency: np.ndarray) -> bool:
    if adjacency.ndim != 3 or adjacency.shape[1] != adjacency.shape[2] or adjacency.shape[0] == 0:
        return False
    if not np.isfinite(adjacency).all():
        return False
    c = int(adjacency.shape[1])
    if c <= 1:
        return False
    offdiag = adjacency.copy()
    eye = np.eye(c, dtype=bool)[None, :, :]
    offdiag[eye.repeat(adjacency.shape[0], axis=0)] = 0.0
    return bool(np.nanmax(np.abs(offdiag)) > 0.0)


def _first_existing_sample(run_records: list[dict[str, Any]]) -> dict[str, Any] | None:
    for record in run_records:
        sample = record.get("sample") if isinstance(record, dict) else None
        if isinstance(sample, dict):
            return sample
    return None


def inspect_cache(path: Path, max_records: int = 200) -> dict[str, Any]:
    with path.open("rb") as fin:
        payload = pickle.load(fin)

    report: dict[str, Any] = {
        "path": str(path),
        "top_level_type": type(payload).__name__,
        "usable_window_cache": False,
        "errors": [],
        "warnings": [],
        "counts": {},
        "shapes": {},
        "ablation_support": {},
    }

    if not isinstance(payload, dict):
        report["errors"].append("Top-level object is not a dict. This is probably patient_records.pkl, not a window cache.")
        return report

    run_records = payload.get("run_records")
    patient_index = payload.get("patient_index")
    if not isinstance(run_records, list) or not isinstance(patient_index, dict):
        report["errors"].append("Missing top-level run_records list and patient_index dict.")
        return report
    if not run_records:
        report["errors"].append("run_records is empty.")
    if not patient_index:
        report["errors"].append("patient_index is empty.")

    report["counts"]["run_records"] = len(run_records)
    report["counts"]["patients"] = len(patient_index)

    inspected = [record for record in run_records[: max(1, int(max_records))] if isinstance(record, dict)]
    subjects = {str(record.get("subject_id")) for record in inspected if "subject_id" in record}
    report["counts"]["inspected_records"] = len(inspected)
    report["counts"]["inspected_subjects"] = len(subjects)

    missing_run_keys: dict[str, int] = {key: 0 for key in RUN_REQUIRED_FOR_FLATTEN}
    missing_sample_keys: dict[str, int] = {key: 0 for key in SAMPLE_REQUIRED_FOR_FLATTEN}
    feature_shapes: list[tuple[int, int, int]] = []
    adjacency_shapes: list[tuple[int, int, int]] = []
    center_lengths: list[int] = []
    raw_shapes: list[tuple[int, ...]] = []
    has_negative_centers = 0
    has_post_centers = 0
    nonzero_adjacency = 0
    interictal_attached = 0
    channel_mismatch = 0
    finite_feature_failures = 0

    for record in inspected:
        for key in RUN_REQUIRED_FOR_FLATTEN:
            if key not in record:
                missing_run_keys[key] += 1
        sample = record.get("sample")
        if not isinstance(sample, dict):
            continue
        for key in SAMPLE_REQUIRED_FOR_FLATTEN:
            if key not in sample:
                missing_sample_keys[key] += 1

        labels = np.asarray(record.get("labels", []))
        channels = list(record.get("channel_names_norm", []))
        features = np.asarray(sample.get("window_features", np.zeros((0, 0, 0))), dtype=np.float32)
        adjacency = np.asarray(sample.get("window_adjacency", np.zeros((0, 0, 0))), dtype=np.float32)
        centers = np.asarray(sample.get("window_relative_centers_sec", np.zeros((0,))), dtype=np.float32)
        raw = np.asarray(sample.get("raw_waveform", np.zeros((0, 0))), dtype=np.float32)

        if features.ndim == 3:
            feature_shapes.append(tuple(int(dim) for dim in features.shape))
            if not np.isfinite(features).all():
                finite_feature_failures += 1
        if adjacency.ndim == 3:
            adjacency_shapes.append(tuple(int(dim) for dim in adjacency.shape))
            if _is_nonzero_adjacency(adjacency):
                nonzero_adjacency += 1
        if centers.ndim == 1:
            center_lengths.append(int(centers.shape[0]))
            if np.any(centers < 0.0):
                has_negative_centers += 1
            if np.any(centers >= 0.0):
                has_post_centers += 1
        if raw.ndim >= 1:
            raw_shapes.append(tuple(int(dim) for dim in raw.shape))
        if features.ndim == 3 and (features.shape[1] != len(channels) or features.shape[1] != labels.shape[0]):
            channel_mismatch += 1
        if any(key in sample for key in ("interictal_window_features", "interictal_baseline_features", "baseline_window_features")):
            interictal_attached += 1

    first_sample = _first_existing_sample(run_records)
    first_feature = None
    first_adj = None
    first_centers = None
    if first_sample is not None:
        first_feature = np.asarray(first_sample.get("window_features", np.zeros((0, 0, 0))), dtype=np.float32)
        first_adj = np.asarray(first_sample.get("window_adjacency", np.zeros((0, 0, 0))), dtype=np.float32)
        first_centers = np.asarray(first_sample.get("window_relative_centers_sec", np.zeros((0,))), dtype=np.float32)
        report["shapes"]["first_window_features"] = _shape(first_sample.get("window_features"))
        report["shapes"]["first_window_adjacency"] = _shape(first_sample.get("window_adjacency"))
        report["shapes"]["first_window_relative_centers_sec"] = _shape(first_sample.get("window_relative_centers_sec"))
        report["shapes"]["first_raw_waveform"] = _shape(first_sample.get("raw_waveform"))

    base_dim = len(WINDOW_NODE_FEATURE_NAMES)
    spectral_dim = len(BASE_SPECTRAL_FEATURE_NAMES)
    feature_dim = int(first_feature.shape[-1]) if isinstance(first_feature, np.ndarray) and first_feature.ndim == 3 else 0
    graph_dim = max(0, feature_dim - spectral_dim)

    report["counts"]["records_with_negative_centers"] = has_negative_centers
    report["counts"]["records_with_post_centers"] = has_post_centers
    report["counts"]["records_with_nonzero_adjacency"] = nonzero_adjacency
    report["counts"]["records_with_interictal_baseline_fields"] = interictal_attached
    report["counts"]["channel_shape_mismatches"] = channel_mismatch
    report["counts"]["records_with_nonfinite_window_features"] = finite_feature_failures
    report["feature_names"] = {
        "expected_base_dim": base_dim,
        "expected_spectral_classical_dim": spectral_dim,
        "observed_first_feature_dim": feature_dim,
        "observed_graph_dim_if_standard_order": graph_dim,
        "standard_order": list(WINDOW_NODE_FEATURE_NAMES),
    }

    missing_run_keys = {key: value for key, value in missing_run_keys.items() if value}
    missing_sample_keys = {key: value for key, value in missing_sample_keys.items() if value}
    if missing_run_keys:
        report["errors"].append(f"Missing run-level keys required by flatten_window_samples: {missing_run_keys}")
    if missing_sample_keys:
        report["errors"].append(f"Missing sample-level keys required by flatten_window_samples: {missing_sample_keys}")
    if channel_mismatch:
        report["errors"].append(f"{channel_mismatch} inspected records have channel/label/window feature shape mismatches.")
    if finite_feature_failures:
        report["errors"].append(f"{finite_feature_failures} inspected records contain non-finite window_features.")

    if feature_dim < base_dim:
        report["warnings"].append(
            f"window_features dim={feature_dim}, smaller than expected standard feature dim={base_dim}; feature-group ablations may be invalid."
        )
    if has_negative_centers == 0:
        report["warnings"].append("No inspected records have negative pre-onset centers; self-reference zdelta/ratio will use all windows as baseline.")
    if nonzero_adjacency == 0:
        report["warnings"].append("No inspected records have nonzero off-diagonal adjacency; real/random/identity adjacency ablations are not meaningful.")
    if interictal_attached == 0:
        report["warnings"].append("No interictal baseline fields detected. This is acceptable for B0/C0, but not for C1/C2/C3 interictal claims.")

    enough_flatten = not missing_run_keys and not missing_sample_keys and channel_mismatch == 0 and finite_feature_failures == 0
    b0_feature_parts = enough_flatten and feature_dim > 0 and has_negative_centers > 0
    feature_group = b0_feature_parts and feature_dim >= base_dim
    adjacency_real = enough_flatten and nonzero_adjacency > 0
    temporal = enough_flatten and bool(feature_shapes) and max(shape[0] for shape in feature_shapes) > 1
    current_code_relative_z = False

    report["usable_window_cache"] = enough_flatten and len(run_records) > 0 and len(patient_index) > 0
    report["ablation_support"] = {
        "F_self_reference_parts_abs_delta_zdelta_ratio": {
            "supported_by_cache": b0_feature_parts,
            "code_note": "Supported by run_neuroez_v2.py. Not honored by neuroez_c/evidence_views.py C0 path in current code.",
        },
        "G_feature_groups": {
            "supported_by_cache": feature_group,
            "code_note": "Cache can support this only if standard feature order is unchanged. Current parser has no --b0_feature_groups switch.",
        },
        "S_real_adjacency_message_passing": {
            "supported_by_cache": adjacency_real,
            "code_note": "no-message and no-channel-attention are supported; random/identity/zero adjacency require new code.",
        },
        "T_temporal_and_seizure_pooling": {
            "supported_by_cache": temporal,
            "code_note": "Supported by current temporal_encoder and seizure_pooling args if there is more than one window/seizure.",
        },
        "R_rank_loss_bce_only": {
            "supported_by_cache": report["usable_window_cache"],
            "code_note": "rank_loss_weight=0 is supported.",
        },
        "R_no_patient_relative_z": {
            "supported_by_cache": report["usable_window_cache"],
            "code_note": "Not supported in current PatientChannelRanker; _patient_relative_zscore is unconditional.",
        },
        "R_fixed_threshold": {
            "supported_by_cache": report["usable_window_cache"],
            "code_note": "Supported with --tune_decision_rule false --decision_threshold 0.5.",
        },
    }

    if not current_code_relative_z:
        report["warnings"].append("PatientChannelRanker relative z-score is unconditional in current code; R1 requires code changes.")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect NeuroEZ window cache support for B0/C0 localization ablations.")
    parser.add_argument("caches", nargs="+", type=Path, help="One or more *_window_cache.pkl files.")
    parser.add_argument("--max-records", type=int, default=200, help="Maximum records per cache to inspect.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    args = parser.parse_args()

    reports = [inspect_cache(path, max_records=args.max_records) for path in args.caches]
    if args.json:
        print(json.dumps(reports, indent=2, ensure_ascii=False))
        return

    for report in reports:
        print("=" * 88)
        print(report["path"])
        print(f"usable_window_cache: {report['usable_window_cache']}")
        print(f"counts: {json.dumps(report['counts'], ensure_ascii=False, sort_keys=True)}")
        print(f"shapes: {json.dumps(report['shapes'], ensure_ascii=False, sort_keys=True)}")
        print(f"feature_names: observed_dim={report['feature_names']['observed_first_feature_dim']} expected_base_dim={report['feature_names']['expected_base_dim']}")
        if report["errors"]:
            print("errors:")
            for item in report["errors"]:
                print(f"  - {item}")
        if report["warnings"]:
            print("warnings:")
            for item in report["warnings"]:
                print(f"  - {item}")
        print("ablation_support:")
        for name, item in report["ablation_support"].items():
            print(f"  - {name}: cache={item['supported_by_cache']} | {item['code_note']}")


if __name__ == "__main__":
    main()
