"""Read-only real-cache parity check for the PaReSet historical B0 adapter."""
from __future__ import annotations

import argparse
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--adapted", type=Path, required=True)
    parser.add_argument("--production-root", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.production_root.resolve()))
    from neuroez_c.evidence_views import (
        BASE_SPECTRAL_FEATURE_NAMES,
        PRUNED_SPECTRAL_FEATURE_NAMES,
        b0_self_reference_features,
    )

    with args.cache.open("rb") as stream:
        cache = pickle.load(stream)
    with args.adapted.open("rb") as stream:
        adapted = pickle.load(stream)
    source = defaultdict(list)
    for item in cache["run_records"]:
        source[str(item["subject_id"])].append(item)
    source_names = list(cache["window_feature_names"])
    historical_args = SimpleNamespace(window_feature_names=source_names,
                                      b0_feature_groups="spectral_classical",
                                      b0_feature_parts="abs,delta,zdelta,ratio", self_compare_eps=1e-5)
    adapter_args = SimpleNamespace(window_feature_names=list(BASE_SPECTRAL_FEATURE_NAMES),
                                   b0_feature_groups="spectral_classical",
                                   b0_feature_parts="abs,delta,zdelta,ratio", self_compare_eps=1e-5)
    selected = [BASE_SPECTRAL_FEATURE_NAMES.index(name) for name in PRUNED_SPECTRAL_FEATURE_NAMES]
    max_absolute = 0.0
    compared = 0
    for patient in adapted[:3]:
        patient_id = patient["patient_id"]
        descriptors = np.asarray(patient["descriptors"], dtype=np.float32)
        records = sorted(source[patient_id], key=lambda record: (
            str(record.get("run_id")), str((record.get("sample") or record).get("sample_id"))
        ))
        if len(records) != descriptors.shape[0]:
            raise RuntimeError("Seizure count differs")
        canonical = {name: index for index, name in enumerate(patient["channel_names"])}
        for seizure, record in enumerate(records):
            sample = record.get("sample") or record
            raw = np.asarray(sample["window_features"], dtype=np.float32)
            times = np.asarray(sample["window_relative_centers_sec"], dtype=np.float32)
            local_names = list(map(str, record["channel_names_norm"]))
            local_indices = [canonical[name] for name in local_names]
            original = b0_self_reference_features(raw, times, historical_args)
            adapter_raw = np.zeros((len(times), len(patient["channel_names"]), len(BASE_SPECTRAL_FEATURE_NAMES)), dtype=np.float32)
            adapter_raw[..., selected] = descriptors[seizure, :len(times)]
            rebuilt = b0_self_reference_features(adapter_raw, times, adapter_args)[:, local_indices]
            delta = float(np.max(np.abs(original - rebuilt)))
            max_absolute = max(max_absolute, delta)
            compared += int(original.size)
    if max_absolute != 0.0:
        raise RuntimeError(f"Historical B0 parity failed: max_absolute={max_absolute}")
    print({"status": "PASS", "patients": min(3, len(adapted)), "values_compared": compared,
           "maximum_absolute_difference": max_absolute})


if __name__ == "__main__":
    main()
