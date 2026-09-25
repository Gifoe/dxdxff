"""Patient-balanced staged ranking objective for CANE-PATH-CP."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterator, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Sampler

from .cane_set_loss import (
    clean_nez_prototype_loss,
    high_confidence_ez_rank_loss,
    patient_soft_macro_f1_loss,
    select_high_confidence_ez,
)


class CenterBalancedPatientBatchSampler(Sampler[list[int]]):
    """Draw equal distinct-patient quotas per center in every batch."""

    def __init__(
        self,
        centers: Sequence[str],
        seed: int = 42,
        shuffle: bool = True,
        batch_size: int | None = None,
    ) -> None:
        self.centers = [str(value).lower() for value in centers]
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.epoch = 0
        grouped: dict[str, list[int]] = defaultdict(list)
        for index, center in enumerate(self.centers):
            grouped[center].append(index)
        self.grouped = dict(grouped)
        if len(self.grouped) < 2:
            raise ValueError("Center-balanced batches require at least two centers")
        n_centers = len(self.grouped)
        requested = n_centers if batch_size is None else int(batch_size)
        if requested < n_centers or requested % n_centers != 0:
            raise ValueError(
                f"Center-balanced patient_batch_size must be a multiple of {n_centers}, got {requested}"
            )
        self.per_center = requested // n_centers
        smallest_center = min(len(values) for values in self.grouped.values())
        if self.per_center > smallest_center:
            raise ValueError(
                "Center-balanced patient_batch_size requests more distinct patients per center "
                f"than available: requested={self.per_center}, smallest_center={smallest_center}"
            )
        self.batch_size = self.per_center * n_centers
        self.n_batches = math.ceil(max(len(values) for values in self.grouped.values()) / self.per_center)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.n_batches

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        pools: dict[str, np.ndarray] = {}
        for center, values in sorted(self.grouped.items()):
            array = np.asarray(values, dtype=np.int64)
            if self.shuffle:
                array = rng.permutation(array)
            pools[center] = array
        for batch_index in range(self.n_batches):
            batch: list[int] = []
            for _, values in sorted(pools.items()):
                start = (batch_index * self.per_center) % len(values)
                selected = [int(values[(start + offset) % len(values)]) for offset in range(self.per_center)]
                batch.extend(selected)
            if len(batch) != len(set(batch)):
                raise RuntimeError("Center-balanced sampler repeated a patient inside one batch")
            if len(batch) != self.batch_size:
                raise RuntimeError("Center-balanced sampler emitted an incomplete batch")
            yield batch


def patient_balanced_bce_rows(
    logits: torch.Tensor, labels_nez: torch.Tensor, channel_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rows: list[torch.Tensor] = []
    clean_rows: list[torch.Tensor] = []
    observed_rows: list[torch.Tensor] = []
    zero = logits.sum() * 0.0
    for patient_idx in range(logits.shape[0]):
        valid = channel_mask[patient_idx].bool() & (labels_nez[patient_idx] >= 0)
        clean = valid & (labels_nez[patient_idx] > 0.5)
        observed = valid & (labels_nez[patient_idx] <= 0.5)
        parts = []
        if torch.any(clean):
            value = F.binary_cross_entropy_with_logits(logits[patient_idx, clean], torch.ones_like(logits[patient_idx, clean]))
            clean_rows.append(value)
            parts.append(value)
        if torch.any(observed):
            value = F.binary_cross_entropy_with_logits(logits[patient_idx, observed], torch.zeros_like(logits[patient_idx, observed]))
            observed_rows.append(value)
            parts.append(value)
        rows.append(torch.stack(parts).mean() if parts else zero)
    per_patient = torch.stack(rows)
    clean_mean = torch.stack(clean_rows).mean() if clean_rows else zero
    observed_mean = torch.stack(observed_rows).mean() if observed_rows else zero
    return per_patient.mean(), per_patient, clean_mean, observed_mean


def soft_worst_center_loss(
    patient_losses: torch.Tensor,
    center_id: torch.Tensor,
    temperature: float = 0.20,
) -> tuple[torch.Tensor, int]:
    centers = torch.unique(center_id)
    means = [patient_losses[center_id == center].mean() for center in centers if torch.any(center_id == center)]
    if len(means) < 2:
        return patient_losses.mean(), len(means)
    values = torch.stack(means)
    tau = max(float(temperature), 1e-6)
    return tau * (torch.logsumexp(values / tau, dim=0) - math.log(len(means))), len(means)


def compute_cane_path_cp_ranking_loss(
    outputs: dict[str, torch.Tensor], batch: dict[str, Any], args: Any, epoch: int
) -> tuple[torch.Tensor, dict[str, float | str]]:
    labels_ez = batch["labels_ez"].to(outputs["logits"].device, outputs["logits"].dtype)
    labels_nez = torch.where(labels_ez >= 0, 1.0 - labels_ez, torch.full_like(labels_ez, -1.0))
    mask = batch["channel_mask"].to(outputs["logits"].device).bool()
    pbce, patient_rows, clean_bce, observed_bce = patient_balanced_bce_rows(outputs["final_nez_logit"], labels_nez, mask)
    center_id = batch["center_id"].to(outputs["logits"].device)
    worst, present_centers = soft_worst_center_loss(patient_rows, center_id, float(getattr(args, "cane_soft_worst_temperature", 0.20)))
    worst_weight = float(getattr(args, "cane_soft_worst_weight", 0.20)) if present_centers >= 2 else 0.0
    group_pbce = (1.0 - worst_weight) * pbce + worst_weight * worst
    soft_f1, soft_nez, soft_ez = patient_soft_macro_f1_loss(outputs["final_nez_logit"], labels_nez, mask)
    anchor, compactness, diversity = clean_nez_prototype_loss(
        outputs["anchor_distance"], labels_nez, mask, outputs["normalized_prototypes"],
        float(getattr(args, "cane_prototype_similarity_margin", 0.50)),
    )
    hc = select_high_confidence_ez(
        labels_nez, mask, outputs["direct_score_nez"], outputs["anchor_nez_evidence"],
        outputs["seizure_nez_probability_mean"], float(getattr(args, "cane_hc_ez_fraction", 0.25)),
    )
    rank, valid_rank_patients = high_confidence_ez_rank_loss(
        outputs["final_nez_logit"], labels_nez, mask, hc, float(getattr(args, "cane_rank_margin", 0.10))
    )
    if torch.any(mask):
        residual_l2 = (
            outputs["anchor_residual"][mask].square() + outputs["seizure_residual"][mask].square()
            + outputs["causal_residual"][mask].square()
        ).mean()
    else:
        residual_l2 = outputs["logits"].sum() * 0.0
    after_stage1 = int(epoch) > int(getattr(args, "cane_stage1_end_epoch", 5))
    rank_active = int(epoch) > int(getattr(args, "cane_rank_start_epoch", 15))
    anchor_weight = float(getattr(args, "cane_anchor_weight", 0.03)) if after_stage1 else 0.0
    rank_weight = float(getattr(args, "cane_hc_rank_weight", 0.015)) if rank_active else 0.0
    residual_weight = float(getattr(args, "cane_residual_l2_weight", 0.005)) if after_stage1 else 0.0
    total = (
        group_pbce + float(getattr(args, "cane_soft_f1_weight", 0.15)) * soft_f1
        + anchor_weight * anchor + rank_weight * rank + residual_weight * residual_l2
    )
    valid_count = mask.sum().clamp_min(1).to(outputs["logits"].dtype)
    utilization = outputs["prototype_utilization"]
    parts: dict[str, float | str] = {
        "cane_path_total_loss": float(total.detach().cpu()),
        "cane_path_group_pbce": float(group_pbce.detach().cpu()),
        "cane_path_patient_balanced_bce": float(pbce.detach().cpu()),
        "cane_path_clean_nez_bce": float(clean_bce.detach().cpu()),
        "cane_path_observed_ez_bce": float(observed_bce.detach().cpu()),
        "cane_path_soft_worst_loss": float(worst.detach().cpu()),
        "cane_path_present_center_count": float(present_centers),
        "cane_path_single_center_batch": float(present_centers < 2),
        "cane_path_soft_macro_f1_loss": float(soft_f1.detach().cpu()),
        "cane_path_soft_nez_f1": float(soft_nez.detach().cpu()),
        "cane_path_soft_ez_f1": float(soft_ez.detach().cpu()),
        "cane_path_anchor_loss": float(anchor.detach().cpu()) if after_stage1 else 0.0,
        "cane_path_anchor_compactness": float(compactness.detach().cpu()),
        "cane_path_prototype_diversity_loss": float(diversity.detach().cpu()),
        "cane_path_hc_rank_loss": float(rank.detach().cpu()) if rank_active else 0.0,
        "cane_path_valid_rank_patient_count": float(valid_rank_patients if rank_active else 0),
        "cane_path_residual_l2": float(residual_l2.detach().cpu()),
        "cane_path_mean_anchor_residual": float(outputs["anchor_residual"][mask].mean().detach().cpu()) if torch.any(mask) else 0.0,
        "cane_path_mean_abs_anchor_residual": float(outputs["anchor_residual"][mask].abs().mean().detach().cpu()) if torch.any(mask) else 0.0,
        "cane_path_mean_seizure_residual": float(outputs["seizure_residual"][mask].mean().detach().cpu()) if torch.any(mask) else 0.0,
        "cane_path_mean_abs_seizure_residual": float(outputs["seizure_residual"][mask].abs().mean().detach().cpu()) if torch.any(mask) else 0.0,
        "cane_path_mean_causal_residual": float(outputs["causal_residual"][mask].mean().detach().cpu()) if torch.any(mask) else 0.0,
        "cane_path_mean_abs_causal_residual": float(outputs["causal_residual"][mask].abs().mean().detach().cpu()) if torch.any(mask) else 0.0,
        "cane_path_causal_residual_saturation_rate": float(outputs["causal_residual_saturation_rate"].detach().cpu()),
        "cane_path_stage": "stage1" if not after_stage1 else "stage2" if not rank_active else "stage3",
        "bce": float(group_pbce.detach().cpu()),
    }
    for index in range(int(utilization.numel())):
        parts[f"cane_path_prototype_utilization_{index}"] = float(utilization[index].detach().cpu())
    return total, parts


__all__ = [
    "CenterBalancedPatientBatchSampler", "compute_cane_path_cp_ranking_loss",
    "patient_balanced_bce_rows", "soft_worst_center_loss",
]
