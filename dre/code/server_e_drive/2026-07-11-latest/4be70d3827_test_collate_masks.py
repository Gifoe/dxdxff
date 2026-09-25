from __future__ import annotations

import numpy as np

from outcome_hifos.cache_schema import load_cache_contract
from outcome_hifos.collate import collate_outcome_patients
from outcome_hifos.dataset import build_outcome_patient_examples
from outcome_hifos.leakage_guard import assert_label_blind_tree


def _payload(subject: str, channels: list[str], windows: int, outcome: str) -> dict:
    return {
        "run_records": [
            {
                "subject_id": subject,
                "run_id": f"{subject}-r1",
                "channel_names_norm": channels,
                "sample": {
                    "sample_id": f"{subject}-s1",
                    "window_features": np.ones((windows, len(channels), 2), dtype=np.float32),
                    "window_relative_centers_sec": np.arange(windows, dtype=np.float32),
                },
            }
        ],
        "patient_index": {
            subject: {
                "canonical_channels": channels,
                "outcome_group": outcome,
                "source_center": "c1",
            }
        },
    }


def test_collate_shapes_masks_and_target_separation() -> None:
    first = build_outcome_patient_examples(load_cache_contract(_payload("p1", ["A1", "A2"], 2, "success")))[0]
    second = build_outcome_patient_examples(load_cache_contract(_payload("p2", ["B1"], 1, "failure")))[0]
    batch = collate_outcome_patients([first, second], padding_value=99999.0)
    model_input = batch["model_input"]
    assert model_input["feature_x"].shape == (2, 1, 2, 2, 2)
    assert batch["outcome"].tolist() == [1.0, 0.0]
    assert model_input["window_channel_mask"][1, 0, 1, 1].item() is False
    assert model_input["feature_x"][1, 0, 1, 1, 0].item() == 99999.0
    assert_label_blind_tree(model_input, stage="collate")
