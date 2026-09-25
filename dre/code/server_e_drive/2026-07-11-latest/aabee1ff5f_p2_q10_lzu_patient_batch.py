"""Deterministic complete-patient mini-batches for the LZU adapter."""
from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Iterator, Sequence

import pandas as pd
import torch

from .p2_q10_lzu_adapter_protocol import build_frozen_adapter_features, canonicalize_channel_frame


@dataclass(frozen=True)
class PatientAdapterExample:
    subject_id: str
    center: str
    outer_fold: int
    channel_names: tuple[str, ...]
    features: torch.Tensor
    base_nez_logit: torch.Tensor
    label_nez: torch.Tensor


class PatientAdapterDataset(Sequence[PatientAdapterExample]):
    def __init__(self, frame: pd.DataFrame) -> None:
        data = canonicalize_channel_frame(frame)
        self.examples: list[PatientAdapterExample] = []
        for subject, group in data.groupby("subject_id", sort=True):
            features, _ = build_frozen_adapter_features(group)
            self.examples.append(PatientAdapterExample(
                subject_id=str(subject), center=str(group.center.iloc[0]), outer_fold=int(group.outer_fold.iloc[0]),
                channel_names=tuple(group.channel_name.astype(str)), features=features,
                base_nez_logit=torch.tensor(group.base_nez_logit.to_numpy(float), dtype=torch.float32),
                label_nez=torch.tensor(group.label_nez.to_numpy(float), dtype=torch.float32),
            ))
        if not self.examples:
            raise ValueError("PatientAdapterDataset cannot be empty")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> PatientAdapterExample:
        return self.examples[index]


class PatientBatchSampler:
    def __init__(self, dataset: PatientAdapterDataset, patient_batch_size: int, *, global_seed: int, outer_fold: int, epoch: int) -> None:
        if patient_batch_size < 1:
            raise ValueError("patient_batch_size must be positive")
        self.dataset = dataset; self.patient_batch_size = int(patient_batch_size)
        self.seed = int(global_seed) + int(outer_fold) * 10000 + int(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        order = list(range(len(self.dataset)))
        # Start from stable subject ordering, then perform the prescribed deterministic shuffle.
        order.sort(key=lambda index: self.dataset[index].subject_id)
        random.Random(self.seed).shuffle(order)
        for start in range(0, len(order), self.patient_batch_size):
            yield order[start:start+self.patient_batch_size]

    def __len__(self) -> int:
        return math.ceil(len(self.dataset)/self.patient_batch_size)


def collate_complete_patients(examples: Sequence[PatientAdapterExample]) -> dict[str, object]:
    if not examples:
        raise ValueError("Cannot collate an empty patient batch")
    features=[]; logits=[]; labels=[]; patient_index=[]; channel_names=[]
    for index, example in enumerate(examples):
        features.append(example.features); logits.append(example.base_nez_logit); labels.append(example.label_nez)
        patient_index.append(torch.full((len(example.channel_names),), index, dtype=torch.long))
        channel_names.extend(example.channel_names)
    return {
        "subject_ids": [example.subject_id for example in examples],
        "features": torch.cat(features), "base_nez_logit": torch.cat(logits), "label_nez": torch.cat(labels),
        "patient_index": torch.cat(patient_index), "channel_names": channel_names,
        "n_batch_patients": len(examples), "n_batch_channels": sum(len(example.channel_names) for example in examples),
    }


__all__ = ["PatientAdapterExample", "PatientAdapterDataset", "PatientBatchSampler", "collate_complete_patients"]
