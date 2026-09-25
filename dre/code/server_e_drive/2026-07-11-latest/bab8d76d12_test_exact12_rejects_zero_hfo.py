import pickle

import numpy as np
import pandas as pd

from scripts.check_cache_compatibility import S5_12, inspect


def test_exact12_zero_hfo_values_are_not_compatible(tmp_path) -> None:
    names = list(S5_12)
    patient_index = {}
    records = []
    subjects = []
    for patient_idx in range(90):
        sid = f"p{patient_idx:03d}"
        subjects.append(sid)
        patient_index[sid] = {"canonical_channels": ["a"], "labels": np.array([patient_idx % 2], dtype=np.float32)}
        for run_idx in range(4 if patient_idx < 11 else 3):
            records.append({"subject_id": sid, "run_id": f"{sid}_{run_idx}", "channel_names_norm": ["a"], "sample": {"window_features": np.zeros((2, 1, len(names)), dtype=np.float32), "raw_temporal_sfreq": 512.0}})
    payload = {"window_feature_names": names, "run_records": records, "patient_index": patient_index}
    cache = tmp_path / "cache.pkl"
    with cache.open("wb") as handle:
        pickle.dump(payload, handle)
    subjects_path = tmp_path / "subjects.csv"
    pd.DataFrame({"subject_id": subjects}).to_csv(subjects_path, index=False)
    report = inspect(cache, subjects_path)
    assert report["exact_s5_12_schema_compatible"] is True
    assert report["exact_s5_12_value_compatible"] is False
    assert report["exact_s5_12_compatible"] is False
