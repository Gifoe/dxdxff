from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from outcome_hifos.cache_schema import load_cache_contract
from outcome_hifos.fm.base import AdapterSpec, FrozenFMAdapter
from outcome_hifos.fm.embedding_cache import FMEmbeddingError, extract_embedding_cache
from outcome_hifos.fm.registry import build_fm_adapter


class FakeAdapter(FrozenFMAdapter):
    def __init__(self) -> None:
        super().__init__(
            AdapterSpec(
                name="fake",
                expected_sampling_rate=100.0,
                input_duration_sec=1.0,
                channel_handling="single_channel",
                normalization="none",
                embedding_layer="linear",
                output_dim=4,
                version="test-v1",
            )
        )
        self.model = torch.nn.Linear(100, 4, bias=False)
        self.freeze()

    def encode_batch(self, waveforms: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return self.model(torch.as_tensor(waveforms, dtype=torch.float32)).cpu().numpy()


def _raw_payload() -> dict:
    return {
        "run_records": [
            {
                "subject_id": "p1",
                "run_id": "r1",
                "channel_names_norm": ["A1", "A2"],
                "sample": {
                    "sample_id": "s1",
                    "raw_window_waveforms": np.ones((3, 2, 100), dtype=np.float32),
                    "window_relative_centers_sec": np.asarray([-1.0, 0.0, 1.0], dtype=np.float32),
                    "raw_temporal_sfreq": 100.0,
                },
            }
        ],
        "patient_index": {"p1": {"canonical_channels": ["A1", "A2"]}},
    }


def test_embedding_cache_preserves_channel_window_granularity_and_frozen_state(tmp_path: Path) -> None:
    adapter = FakeAdapter()
    manifest = extract_embedding_cache(load_cache_contract(_raw_payload()), adapter, tmp_path)
    assert adapter.model.training is False
    assert all(not parameter.requires_grad for parameter in adapter.model.parameters())
    assert manifest.shape[0] == 6
    assert {"subject_id", "run_id", "sample_id", "window_idx", "channel_name", "embedding_index"} <= set(manifest.columns)
    assert (tmp_path / "all_windows_embeddings.pkl").exists()
    assert (tmp_path / "manifest.csv").exists()
    assert (tmp_path / "audit.json").exists()
    with (tmp_path / "all_windows_embeddings.pkl").open("rb") as handle:
        payload = pickle.load(handle)
    assert payload["storage_format"] == "chunked_npy_v1"
    assert payload["embedding_dim"] == 4
    assert "embeddings" not in payload


def test_missing_explicit_raw_window_key_fails_instead_of_guessing(tmp_path: Path) -> None:
    payload = _raw_payload()
    del payload["run_records"][0]["sample"]["raw_window_waveforms"]
    payload["run_records"][0]["sample"]["raw_waveform"] = np.ones((2, 300), dtype=np.float32)
    with pytest.raises(FMEmbeddingError, match="raw_valid_start_sample"):
        extract_embedding_cache(
            load_cache_contract(payload),
            FakeAdapter(),
            tmp_path,
            raw_time_origin="seizure_onset_at_raw_target_midpoint",
        )


def test_continuous_raw_uses_audited_seizure_onset_midpoint_and_window_centers(tmp_path: Path) -> None:
    payload = _raw_payload()
    sample = payload["run_records"][0]["sample"]
    del sample["raw_window_waveforms"]
    sample["raw_waveform"] = np.tile(np.arange(300, dtype=np.float32), (2, 1))
    sample["raw_valid_start_sample"] = 0
    sample["raw_valid_samples"] = 300
    manifest = extract_embedding_cache(
        load_cache_contract(payload),
        FakeAdapter(),
        tmp_path,
        raw_time_origin="seizure_onset_at_raw_target_midpoint",
    )
    assert manifest.shape[0] == 6
    assert set(manifest["raw_window_source"]) == {"continuous_raw_seizure_onset_midpoint"}


def test_registry_exposes_real_adapters_and_rejects_unknown() -> None:
    for name in ("biot", "cbramod", "labram", "brainbert"):
        adapter = build_fm_adapter(name, {"target_sfreq": 200, "window_sec": 4.0})
        assert adapter.spec.name == name
    with pytest.raises(ValueError, match="Unsupported frozen FM"):
        build_fm_adapter("not-a-model", {})
