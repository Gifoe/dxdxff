from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pytest
import torch

from outcome_hifos.cache_schema import load_cache_contract
from outcome_hifos.fm.base import AdapterSpec, FrozenFMAdapter
from outcome_hifos.fm.embedding_cache import FMEmbeddingError, _continuous_raw_windows, extract_embedding_cache, load_fm_embedding_as_feature_cache


class TrackingAdapter(FrozenFMAdapter):
    def __init__(self) -> None:
        super().__init__(AdapterSpec("tracking", 10.0, 1.0, "single", "none", "mean", 2, "v1"))
        self.model = torch.nn.Linear(10, 2, bias=False)
        self.max_batch_seen = 0
        self.audit_info["checkpoint_sha256"] = "checkpoint-hash"
        self.freeze()

    def encode_batch(self, waveforms: np.ndarray) -> np.ndarray:
        self.max_batch_seen = max(self.max_batch_seen, len(waveforms))
        return np.column_stack([waveforms.mean(axis=1), waveforms.max(axis=1)]).astype(np.float32)


def _payload(windows: int = 40) -> dict:
    waveform = np.arange(windows * 2 * 10, dtype=np.float32).reshape(windows, 2, 10)
    return {
        "run_records": [
            {
                "subject_id": "p1",
                "run_id": "r1",
                "channel_names_norm": ["A1", "A2"],
                "sample": {
                    "sample_id": "s1",
                    "raw_window_waveforms": waveform,
                    "window_relative_centers_sec": np.arange(windows, dtype=np.float32),
                    "raw_temporal_sfreq": 10.0,
                },
            }
        ],
        "patient_index": {"p1": {"canonical_channels": ["A1", "A2"], "outcome_group": "success", "source_center": "c1"}},
    }


def test_streaming_embedding_never_buffers_more_than_encoder_batch(tmp_path: Path) -> None:
    adapter = TrackingAdapter()
    manifest = extract_embedding_cache(load_cache_contract(_payload()), adapter, tmp_path, batch_size=7)
    assert len(manifest) == 80
    assert adapter.max_batch_seen <= 7
    assert manifest["chunk_file"].nunique() > 1
    metadata = json.loads((tmp_path / "audit.json").read_text(encoding="utf-8"))
    assert metadata["max_buffered_waveforms"] <= 7
    assert metadata["checkpoint_sha256"] == "checkpoint-hash"
    assert metadata["version"] == "v1"


def test_streaming_loader_reconstructs_channel_window_embeddings(tmp_path: Path) -> None:
    extract_embedding_cache(load_cache_contract(_payload(windows=3)), TrackingAdapter(), tmp_path, batch_size=2)
    cache = load_fm_embedding_as_feature_cache(tmp_path)
    tensor = cache.run_records[0]["sample"]["window_features"]
    assert tensor.shape == (3, 2, 2)
    assert np.isfinite(tensor).all()


def test_unknown_continuous_raw_time_origin_fails_closed(tmp_path: Path) -> None:
    payload = _payload(windows=3)
    sample = payload["run_records"][0]["sample"]
    del sample["raw_window_waveforms"]
    sample["raw_waveform"] = np.ones((2, 100), dtype=np.float32)
    sample["raw_valid_start_sample"] = 0
    sample["raw_valid_samples"] = 100
    with pytest.raises(FMEmbeddingError, match="time origin"):
        extract_embedding_cache(load_cache_contract(payload), TrackingAdapter(), tmp_path, raw_time_origin="unknown")


def test_continuous_raw_zero_is_fixed_target_midpoint_not_valid_interval_midpoint() -> None:
    sample = {
        "raw_waveform": np.tile(np.arange(30, dtype=np.float32), (2, 1)),
        "raw_valid_start_sample": 2,
        "raw_valid_samples": 20,
        "window_relative_centers_sec": np.array([0.0], dtype=np.float32),
    }
    window = _continuous_raw_windows(sample, TrackingAdapter(), 10.0)
    np.testing.assert_array_equal(window[0, 0], np.arange(10, 20, dtype=np.float32))


def test_fm_trace_duplicate_fails_fast(tmp_path: Path) -> None:
    extract_embedding_cache(load_cache_contract(_payload(windows=2)), TrackingAdapter(), tmp_path, batch_size=2)
    metadata_path = tmp_path / "all_windows_embeddings.pkl"
    with metadata_path.open("rb") as handle:
        payload = pickle.load(handle)
    payload["manifest"].append(dict(payload["manifest"][0]))
    with metadata_path.open("wb") as handle:
        pickle.dump(payload, handle)
    with pytest.raises(FMEmbeddingError, match="duplicate trace key"):
        load_fm_embedding_as_feature_cache(tmp_path)
