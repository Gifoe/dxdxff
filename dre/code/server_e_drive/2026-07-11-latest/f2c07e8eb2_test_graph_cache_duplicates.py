import numpy as np
import pytest

from neuroez_c.task2.graph_cache import build_graph_cache


def _record(start):
    return {"subject_id": "x", "run_id": "r", "channel_names_norm": ["A1", "A2", "A3", "A4"], "sample": {"sample_id": "s", "start_sec": start, "seizure_onset_sec": start + 1, "raw_temporal_sfreq": 250.0, "raw_waveform": np.zeros((4, 500), dtype=np.float32)}}


def test_strict_graph_cache_rejects_ambiguous_duplicate_keys(tmp_path):
    cache = {"run_records": [_record(0), _record(10)], "patient_index": {"x": {}}}
    with pytest.raises(ValueError, match="Ambiguous duplicate"):
        build_graph_cache(cache, cache, tmp_path, raw_source_path=tmp_path / "raw.pkl", strict=True)
