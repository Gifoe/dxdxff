from __future__ import annotations

import numpy as np

from outcome_hifos.cache_schema import load_cache_contract
from outcome_hifos.dataset import OutcomePatientExample
from outcome_hifos.baselines.token_data import build_raw_patient_examples, build_embedding_patient_examples
from task1_baselines.raw_preprocessing import RawPreprocessingConfig


def _reference() -> OutcomePatientExample:
    return OutcomePatientExample(
        subject_id="p1", center="c1", target=1.0, canonical_channels=("A1", "A2"),
        model_input={"feature_runs": [np.ones((1, 2, 1))], "window_centers": [np.asarray([0.0])], "seizure_channel_mask": [np.asarray([True, True])], "canonical_index": np.arange(2)},
        side_metadata={"run_ids": ("r1",), "sample_ids": ("s1",)},
    )


def _payload(values: np.ndarray, key: str) -> dict:
    return {
        "run_records": [{"subject_id": "p1", "run_id": "r1", "channel_names_norm": ["A2", "A1"], "sample": {"sample_id": "s1", key: values, "window_relative_centers_sec": [0.0], "raw_temporal_sfreq": 10.0}}],
        "patient_index": {"p1": {"canonical_channels": ["A1", "A2"]}},
    }


def test_raw_patient_examples_align_channels_without_task1_labels() -> None:
    examples, audit = build_raw_patient_examples(load_cache_contract(_payload(np.ones((1, 2, 20), dtype=np.float32), "raw_window_waveforms")), [_reference()], RawPreprocessingConfig(target_sfreq=10, window_sec=2, bandpass=None, notch=None))
    assert len(examples) == 1
    assert examples[0].model_input["feature_runs"][0].shape == (1, 2, 20)
    assert not any("label" in key.lower() or "ez" in key.lower() for key in examples[0].model_input)
    assert audit[0]["channel_alignment_ratio"] == 1.0


def test_embedding_patient_examples_preserve_window_channel_hierarchy() -> None:
    examples, _ = build_embedding_patient_examples(load_cache_contract(_payload(np.ones((1, 2, 4), dtype=np.float32), "window_features")), [_reference()])
    assert examples[0].model_input["feature_runs"][0].shape == (1, 2, 4)
