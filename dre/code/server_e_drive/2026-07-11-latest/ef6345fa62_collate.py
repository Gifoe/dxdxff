from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch

from .dataset import OutcomePatientExample
from .leakage_guard import assert_label_blind_tree


def collate_outcome_patients(
    examples: Sequence[OutcomePatientExample],
    *,
    padding_value: float = 0.0,
) -> dict[str, Any]:
    if not examples:
        raise ValueError("collate_outcome_patients received an empty batch.")
    batch_size = len(examples)
    max_seizures = max(len(example.model_input["feature_runs"]) for example in examples)
    max_windows = max(run.shape[0] for example in examples for run in example.model_input["feature_runs"])
    max_channels = max(len(example.canonical_channels) for example in examples)
    feature_dim = max(run.shape[-1] for example in examples for run in example.model_input["feature_runs"])

    feature_x = torch.full(
        (batch_size, max_seizures, max_windows, max_channels, feature_dim),
        fill_value=float(padding_value),
        dtype=torch.float32,
    )
    seizure_mask = torch.zeros((batch_size, max_seizures), dtype=torch.bool)
    window_mask = torch.zeros((batch_size, max_seizures, max_windows), dtype=torch.bool)
    channel_mask = torch.zeros((batch_size, max_channels), dtype=torch.bool)
    seizure_channel_mask = torch.zeros((batch_size, max_seizures, max_channels), dtype=torch.bool)
    window_centers = torch.full((batch_size, max_seizures, max_windows), float("nan"), dtype=torch.float32)
    canonical_index = torch.full((batch_size, max_channels), -1, dtype=torch.long)

    for batch_index, example in enumerate(examples):
        channel_count = len(example.canonical_channels)
        channel_mask[batch_index, :channel_count] = True
        canonical_index[batch_index, :channel_count] = torch.arange(channel_count, dtype=torch.long)
        for seizure_index, run in enumerate(example.model_input["feature_runs"]):
            run_array = np.asarray(run, dtype=np.float32)
            window_count, current_channels, current_dim = run_array.shape
            feature_x[batch_index, seizure_index, :window_count, :current_channels, :current_dim] = torch.from_numpy(run_array)
            seizure_mask[batch_index, seizure_index] = True
            window_mask[batch_index, seizure_index, :window_count] = True
            seizure_channel_mask[batch_index, seizure_index, :current_channels] = torch.as_tensor(
                example.model_input["seizure_channel_mask"][seizure_index], dtype=torch.bool
            )
            window_centers[batch_index, seizure_index, :window_count] = torch.as_tensor(
                example.model_input["window_centers"][seizure_index], dtype=torch.float32
            )

    window_channel_mask = (
        seizure_mask[:, :, None, None]
        & window_mask[:, :, :, None]
        & seizure_channel_mask[:, :, None, :]
        & channel_mask[:, None, None, :]
    )
    model_input = {
        "feature_x": feature_x,
        "seizure_mask": seizure_mask,
        "window_mask": window_mask,
        "channel_mask": channel_mask,
        "seizure_channel_mask": seizure_channel_mask,
        "window_channel_mask": window_channel_mask,
        "window_centers": window_centers,
        "canonical_index": canonical_index,
    }
    assert_label_blind_tree(model_input, stage="collated_model_input")
    return {
        "model_input": model_input,
        "outcome": torch.tensor([example.target for example in examples], dtype=torch.float32),
        "subject_id": [example.subject_id for example in examples],
        "center": [example.center for example in examples],
        "canonical_channels": [list(example.canonical_channels) for example in examples],
        "side_metadata": [example.side_metadata for example in examples],
    }


__all__ = ["collate_outcome_patients"]
