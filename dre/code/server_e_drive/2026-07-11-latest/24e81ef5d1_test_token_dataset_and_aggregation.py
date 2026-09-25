from __future__ import annotations

import numpy as np
import pandas as pd

from outcome_hifos.cache_schema import load_cache_contract
from task1_baselines.raw_preprocessing import RawPreprocessingConfig
from task1_baselines.token_data import build_task1_raw_tokens
from task1_baselines.prediction_aggregation import aggregate_task1_window_probabilities


def _payload() -> dict:
    return {
        "run_records": [
            {
                "subject_id": "p1", "run_id": "r1", "channel_names_norm": ["A1", "A2"],
                "sample": {
                    "raw_window_waveforms": np.ones((2, 2, 20), dtype=np.float32),
                    "window_relative_centers_sec": [-1.0, 1.0], "raw_temporal_sfreq": 10.0,
                },
            }
        ],
        "patient_index": {"p1": {"canonical_channels": ["A1", "A2"], "labels": [1, 0], "source_center": "c1"}},
    }


def test_raw_token_builder_preserves_patient_seizure_channel_window_ids() -> None:
    tokens = build_task1_raw_tokens(load_cache_contract(_payload()), {"p1"}, RawPreprocessingConfig(target_sfreq=10, window_sec=2, bandpass=None, notch=None))
    assert tokens.values.shape == (4, 20)
    assert tokens.rows[["subject_id", "seizure_id", "channel_name", "window_id"]].duplicated().sum() == 0
    assert set(tokens.rows["label_nez"]) == {0, 1}


def test_window_probability_aggregation_is_median_within_then_across_seizures() -> None:
    rows = pd.DataFrame(
        {
            "subject_id": ["p1"] * 6, "center": ["c1"] * 6,
            "seizure_id": ["s1"] * 3 + ["s2"] * 3, "channel_name": ["A1"] * 6,
            "window_id": list(range(6)), "label_nez": [1] * 6,
            "score_nez_probability": [0.1, 0.5, 0.9, 0.2, 0.4, 0.8],
        }
    )
    channel = aggregate_task1_window_probabilities(rows)
    assert len(channel) == 1
    assert channel.iloc[0]["score_nez_probability"] == 0.45
    assert channel.iloc[0]["valid_seizure_count"] == 2
    assert channel.iloc[0]["valid_window_count"] == 6
