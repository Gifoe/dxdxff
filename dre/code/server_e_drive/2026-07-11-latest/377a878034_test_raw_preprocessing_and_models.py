from __future__ import annotations

import numpy as np
import pytest
import torch

from task1_baselines.raw_preprocessing import RawPreprocessingConfig, extract_and_preprocess_windows
from task1_baselines.models.raw_backbones import build_raw_token_encoder


def test_continuous_raw_windows_use_onset_midpoint_and_centers() -> None:
    sample = {
        "raw_waveform": np.tile(np.arange(1000, dtype=np.float32), (2, 1)),
        "raw_valid_start_sample": 0,
        "raw_valid_samples": 1000,
        "window_relative_centers_sec": np.asarray([-1.0, 1.0]),
    }
    config = RawPreprocessingConfig(target_sfreq=100.0, window_sec=1.0, bandpass=None, notch=None)
    windows, audit = extract_and_preprocess_windows(sample, original_sfreq=100.0, config=config)
    assert windows.shape == (2, 2, 100)
    assert audit["time_origin"] == "seizure_onset_at_raw_target_midpoint"
    assert audit["valid_windows"] == 2
    assert np.isfinite(windows).all()


def test_raw_preprocessing_rejects_unverified_valid_interval() -> None:
    sample = {"raw_waveform": np.ones((2, 100)), "window_relative_centers_sec": [0.0]}
    with pytest.raises(ValueError, match="raw_valid"):
        extract_and_preprocess_windows(sample, original_sfreq=100.0, config=RawPreprocessingConfig())


def test_braindecode_builder_fails_clearly_when_dependency_missing() -> None:
    with pytest.raises(RuntimeError, match="braindecode"):
        build_raw_token_encoder("eegnet", n_times=200, sfreq=100.0, embedding_dim=16)

