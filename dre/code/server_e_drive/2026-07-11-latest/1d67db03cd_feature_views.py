from __future__ import annotations

import numpy as np
from dataclasses import replace

from .dataset import OutcomePatientExample
from .normalization import FoldNormalizer


def build_feature_views(array: np.ndarray, valid_mask: np.ndarray, normalizer: FoldNormalizer) -> np.ndarray:
    """Return train-fitted absolute and patient-relative robust views."""

    return normalizer.transform_with_relative(array, valid_mask)


def fit_normalizer_for_examples(examples: list[OutcomePatientExample]) -> FoldNormalizer:
    arrays = []
    masks = []
    subject_ids = []
    for example in examples:
        for run, channel_mask in zip(example.model_input["feature_runs"], example.model_input["seizure_channel_mask"]):
            value = np.asarray(run, dtype=np.float32)
            arrays.append(value)
            masks.append(np.broadcast_to(np.asarray(channel_mask, dtype=bool), value.shape[:-1]))
            subject_ids.append(example.subject_id)
    return FoldNormalizer().fit(arrays, valid_masks=masks, subject_ids=subject_ids)


def transform_examples(
    examples: list[OutcomePatientExample],
    normalizer: FoldNormalizer,
) -> list[OutcomePatientExample]:
    output = []
    for example in examples:
        model_input = dict(example.model_input)
        runs = [np.asarray(run, dtype=np.float32) for run in example.model_input["feature_runs"]]
        masks = [
            np.broadcast_to(np.asarray(channel_mask, dtype=bool), run.shape[:-1])
            for run, channel_mask in zip(runs, example.model_input["seizure_channel_mask"])
        ]
        relative = normalizer.patient_relative_many(runs, masks)
        model_input["feature_runs"] = [
            normalizer.transform_with_relative(run, mask, relative_view)
            for run, mask, relative_view in zip(runs, masks, relative)
        ]
        output.append(replace(example, model_input=model_input))
    return output


__all__ = ["build_feature_views", "fit_normalizer_for_examples", "transform_examples"]
