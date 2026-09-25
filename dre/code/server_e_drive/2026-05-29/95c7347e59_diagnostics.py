from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from utils import write_csv


def _checksum(array: np.ndarray) -> str:
    values = np.asarray(array, dtype=np.float32)
    if values.size == 0:
        return ""
    digest = hashlib.sha1(values.tobytes()).hexdigest()
    return digest[:16]


def _update_digest(digest: Any, array: np.ndarray) -> None:
    values = np.asarray(array, dtype=np.float32)
    if values.size:
        digest.update(values.tobytes())


def _sample_has_interictal(sample: dict[str, Any]) -> bool:
    for key in ("interictal_window_features", "interictal_baseline_features", "baseline_window_features"):
        value = sample.get(key)
        if value is None:
            continue
        arr = np.asarray(value)
        if arr.ndim == 3 and arr.shape[0] > 0:
            return True
    return False


def diagnose_cache(cache_path: Path, experiment: str) -> dict[str, Any]:
    with Path(cache_path).open("rb") as fin:
        payload = pickle.load(fin)
    run_records = payload.get("run_records", [])
    patient_index = payload.get("patient_index", {})

    ictal_records = [r for r in run_records if str(r.get("phase_group", "")).lower() == "ictal"]
    interictal_records = [r for r in run_records if str(r.get("phase_group", "")).lower() == "interictal"]
    patients_with_baseline = set()
    interictal_windows_total = 0
    feature_dims = set()
    feature_chunks = []
    adjacency_digest = hashlib.sha1()
    adjacency_abs_sum = 0.0
    adjacency_count = 0
    samples_with_baseline = 0

    for record in run_records:
        sample = record.get("sample", {})
        features = np.asarray(sample.get("window_features", np.zeros((0, 0, 0))), dtype=np.float32)
        adjacency = np.asarray(sample.get("window_adjacency", np.zeros((0, 0, 0))), dtype=np.float32)
        if features.ndim == 3 and features.shape[0] > 0:
            feature_dims.add(int(features.shape[-1]))
            feature_chunks.append(features.reshape(-1, features.shape[-1]))
        if adjacency.ndim == 3 and adjacency.shape[0] > 0:
            adjacency_abs_sum += float(np.sum(np.abs(adjacency)))
            adjacency_count += int(adjacency.size)
            _update_digest(adjacency_digest, adjacency)
        if _sample_has_interictal(sample):
            samples_with_baseline += 1
            patients_with_baseline.add(str(record.get("subject_id", "")))
            interictal = np.asarray(sample.get("interictal_window_features", np.zeros((0, 0, 0))))
            if interictal.ndim == 3:
                interictal_windows_total += int(interictal.shape[0])

    all_features = np.concatenate(feature_chunks, axis=0) if feature_chunks else np.zeros((0, 0), dtype=np.float32)
    return {
        "experiment": experiment,
        "cache_path": str(cache_path),
        "num_patients": len(patient_index),
        "num_run_records": len(run_records),
        "num_ictal_runs": len(ictal_records),
        "num_interictal_runs": len(interictal_records),
        "num_samples_with_valid_interictal_baseline": samples_with_baseline,
        "num_patients_with_valid_interictal_baseline": len(patients_with_baseline),
        "num_interictal_windows_attached": interictal_windows_total,
        "feature_dim": ";".join(str(v) for v in sorted(feature_dims)),
        "mean_abs_features": float(np.mean(np.abs(all_features))) if all_features.size else "",
        "std_features": float(np.std(all_features)) if all_features.size else "",
        "mean_abs_adjacency": float(adjacency_abs_sum / adjacency_count) if adjacency_count else "",
        "adjacency_checksum": adjacency_digest.hexdigest()[:16] if adjacency_count else "",
    }


def write_cache_diagnostics(output_path: Path, caches: dict[str, Path]) -> list[dict[str, Any]]:
    rows = [diagnose_cache(path, experiment) for experiment, path in caches.items()]
    write_csv(output_path, rows)
    return rows
