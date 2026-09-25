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


def _patient_percentile(values: torch.Tensor) -> torch.Tensor:
    """Detached within-patient ascending percentile ranks in [0, 1]."""
    if values.numel() <= 1:
        return torch.zeros_like(values)
    order = torch.argsort(values.detach())
    ranks = torch.empty_like(values)
    ranks[order] = torch.arange(values.numel(), device=values.device, dtype=values.dtype)
    return ranks / float(values.numel() - 1)


def rtc_tail_rank_losses(
    outputs: dict[str, torch.Tensor], labels_nez: torch.Tensor, mask: torch.Tensor, epoch: int, args: Any,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Tail ranking that selects observed-EZ only from detached direct/anchor evidence."""
    zero = outputs["final_nez_logit"].sum() * 0.0
    if not bool(getattr(args, "use_p2_rtc_shift", False)) or not bool(getattr(args, "rtc_enable_tail_rank", False)):
        return zero, zero, {"tail_rank_active_weight": 0.0, "tail_rank_valid_patient_count": 0.0, "tail_rank_pair_count": 0.0, "mean_clean_nez_robust_tail": 0.0, "mean_hc_ez_robust_tail": 0.0, "mean_tail_margin": 0.0, "hc_ez_selected_fraction": 0.0}
    robust = outputs["seizure_nez_robust_tail_logit"]
    direct = outputs["direct_score_nez"].detach()
    anchor = outputs["anchor_nez_evidence"].detach()
    start = int(getattr(args, "rtc_tail_rank_start_epoch", 8))
    ramp_epochs = max(1, int(getattr(args, "rtc_tail_rank_ramp_epochs", 5)))
    maximum_weight = float(getattr(args, "rtc_tail_rank_weight", 0.03))
    progress = min(max((int(epoch) - start + 1) / float(ramp_epochs), 0.0), 1.0)
    active_weight = maximum_weight * progress
    rank_values, consistency_values, clean_means, hc_means, margins, selected_fracs = [], [], [], [], [], []
    pair_count = valid_patients = 0
    for patient_idx in range(mask.shape[0]):
        valid = mask[patient_idx].bool() & (labels_nez[patient_idx] >= 0)
        clean = valid & (labels_nez[patient_idx] > 0.5)
        observed = valid & (labels_nez[patient_idx] <= 0.5)
        if torch.any(clean):
            consistency_values.append(F.softplus(0.05 + outputs["seizure_nez_logit_mean"][patient_idx, clean] - robust[patient_idx, clean]).mean())
            clean_means.append(robust[patient_idx, clean].mean())
        if not (torch.any(clean) and torch.any(observed)):
            continue
        observed_indices = torch.nonzero(observed, as_tuple=False).squeeze(1)
        reliability = 0.5 * (_patient_percentile(direct[patient_idx, observed]) + _patient_percentile(anchor[patient_idx, observed]))
        count = max(1, int(math.ceil(float(getattr(args, "rtc_hc_ez_fraction", 0.20)) * observed_indices.numel())))
        selected_local = torch.topk(reliability, k=count, largest=False).indices
        hc_indices = observed_indices[selected_local]
        clean_tail = robust[patient_idx, clean]
        hc_tail = robust[patient_idx, hc_indices]
        margins.append((clean_tail[:, None] - hc_tail[None, :]).mean())
        rank_values.append(F.softplus(float(getattr(args, "rtc_tail_rank_margin", 0.10)) - clean_tail[:, None] + hc_tail[None, :]).mean())
        hc_means.append(hc_tail.mean()); selected_fracs.append(float(count / observed_indices.numel()))
        pair_count += int(clean_tail.numel() * hc_tail.numel()); valid_patients += 1
    rank = torch.stack(rank_values).mean() if rank_values else zero
    consistency = torch.stack(consistency_values).mean() if consistency_values else zero
    diagnostics = {
        "tail_rank_active_weight": active_weight,
        "tail_rank_valid_patient_count": float(valid_patients),
        "tail_rank_pair_count": float(pair_count),
        "mean_clean_nez_robust_tail": float(torch.stack(clean_means).mean().detach().cpu()) if clean_means else 0.0,
        "mean_hc_ez_robust_tail": float(torch.stack(hc_means).mean().detach().cpu()) if hc_means else 0.0,
        "mean_tail_margin": float(torch.stack(margins).mean().detach().cpu()) if margins else 0.0,
        "hc_ez_selected_fraction": float(np.mean(selected_fracs)) if selected_fracs else 0.0,
    }
    return active_weight * rank, float(getattr(args, "rtc_clean_tail_consistency_weight", 0.01)) * consistency, diagnostics


def atc_tail_losses(
    outputs: dict[str, torch.Tensor], labels_nez: torch.Tensor, mask: torch.Tensor, epoch: int, args: Any,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Patient-balanced ATC tail losses; trusted-EZ selection never reads tail/final scores."""
    zero = outputs["final_nez_logit"].sum() * 0.0
    if not bool(getattr(args, "use_p2_atc", False)):
        return zero, zero, {
            "clean_nez_tail_loss": 0.0, "clean_nez_tail_active_weight": 0.0,
            "clean_nez_tail_valid_patient_count": 0.0, "clean_nez_tail_valid_channel_count": 0.0,
            "trusted_ez_tail_pair_loss": 0.0, "trusted_ez_tail_pair_active_weight": 0.0,
            "trusted_ez_valid_patient_count": 0.0, "trusted_ez_selected_channel_count": 0.0,
            "tail_pair_count": 0.0, "mean_clean_nez_robust_tail_logit": 0.0,
            "mean_trusted_ez_robust_tail_logit": 0.0, "mean_clean_minus_trusted_tail_margin": 0.0,
        }
    from .p2_atc_profiles import get_atc_profile

    profile = get_atc_profile(getattr(args, "p2_atc_profile", "A0"))
    if not (profile.clean_nez_tail_loss or profile.trusted_ez_tail_loss):
        return atc_tail_losses({**outputs, "final_nez_logit": outputs["final_nez_logit"]}, labels_nez, mask, epoch, type("Off", (), {"use_p2_atc": False})())

    start = int(getattr(args, "p2_atc_loss_start_epoch", 8))
    ramp_epochs = max(1, int(getattr(args, "p2_atc_loss_ramp_epochs", 5)))
    ramp = min(max((int(epoch) - start + 1) / float(ramp_epochs), 0.0), 1.0)
    robust = outputs["seizure_nez_robust_tail_logit"]
    tail_valid = outputs["tail_valid"].bool()
    valid_count = outputs["valid_seizure_count"]
    eligible = mask.bool() & (labels_nez >= 0) & tail_valid & (valid_count >= 2)
    direct = outputs["direct_score_nez"].detach()
    anchor = outputs["anchor_nez_evidence"].detach()
    floor_rows: list[torch.Tensor] = []
    pair_rows: list[torch.Tensor] = []
    clean_values: list[torch.Tensor] = []
    trusted_values: list[torch.Tensor] = []
    margin_values: list[torch.Tensor] = []
    clean_patients = trusted_patients = clean_channels = trusted_channels = pair_count = 0
    for patient_idx in range(mask.shape[0]):
        clean = eligible[patient_idx] & (labels_nez[patient_idx] > 0.5)
        observed = eligible[patient_idx] & (labels_nez[patient_idx] <= 0.5)
        if torch.any(clean):
            clean_tail = robust[patient_idx, clean]
            clean_values.append(clean_tail.mean())
            if profile.clean_nez_tail_loss:
                floor_rows.append(F.softplus(float(getattr(args, "p2_atc_clean_tail_margin", 0.10)) - clean_tail).mean())
                clean_patients += 1; clean_channels += int(clean_tail.numel())
        if not (profile.trusted_ez_tail_loss and torch.any(clean) and torch.any(observed)):
            continue
        observed_indices = torch.nonzero(observed, as_tuple=False).squeeze(1)
        # The two detached ranks are the only trusted-EZ selection evidence.
        trusted_score = 0.5 * (
            _patient_percentile(direct[patient_idx, observed])
            + _patient_percentile(anchor[patient_idx, observed])
        )
        selected_count = max(1, int(math.ceil(float(getattr(args, "p2_atc_trusted_ez_fraction", 0.20)) * observed_indices.numel())))
        trusted_indices = observed_indices[torch.topk(trusted_score, k=selected_count, largest=False).indices]
        clean_tail = robust[patient_idx, clean]
        trusted_tail = robust[patient_idx, trusted_indices]
        # Deterministic evenly-spaced subsampling caps pathological channel grids.
        left = clean_tail.repeat_interleave(trusted_tail.numel())
        right = trusted_tail.repeat(clean_tail.numel())
        max_pairs = int(getattr(args, "p2_atc_max_pairs_per_patient", 256))
        if left.numel() > max_pairs:
            sampled = torch.linspace(0, left.numel() - 1, max_pairs, device=left.device).round().long()
            left, right = left[sampled], right[sampled]
        pair_rows.append(F.softplus(float(getattr(args, "p2_atc_pair_margin", 0.15)) - left + right).mean())
        trusted_values.append(trusted_tail.mean()); margin_values.append((left - right).mean())
        trusted_patients += 1; trusted_channels += int(trusted_tail.numel()); pair_count += int(left.numel())
    floor = torch.stack(floor_rows).mean() if floor_rows else zero
    pair = torch.stack(pair_rows).mean() if pair_rows else zero
    floor_weight = ramp * float(getattr(args, "p2_atc_clean_tail_weight", 0.020)) if profile.clean_nez_tail_loss else 0.0
    pair_weight = ramp * float(getattr(args, "p2_atc_pair_weight", 0.020)) if profile.trusted_ez_tail_loss else 0.0
    diagnostics = {
        "clean_nez_tail_loss": float(floor.detach().cpu()), "clean_nez_tail_active_weight": float(floor_weight),
        "clean_nez_tail_valid_patient_count": float(clean_patients), "clean_nez_tail_valid_channel_count": float(clean_channels),
        "trusted_ez_tail_pair_loss": float(pair.detach().cpu()), "trusted_ez_tail_pair_active_weight": float(pair_weight),
        "trusted_ez_valid_patient_count": float(trusted_patients), "trusted_ez_selected_channel_count": float(trusted_channels),
        "tail_pair_count": float(pair_count),
        "mean_clean_nez_robust_tail_logit": float(torch.stack(clean_values).mean().detach().cpu()) if clean_values else 0.0,
        "mean_trusted_ez_robust_tail_logit": float(torch.stack(trusted_values).mean().detach().cpu()) if trusted_values else 0.0,
        "mean_clean_minus_trusted_tail_margin": float(torch.stack(margin_values).mean().detach().cpu()) if margin_values else 0.0,
    }
    return floor_weight * floor, pair_weight * pair, diagnostics


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
    base_p2_total = total
    tail_rank, clean_tail, tail_parts = rtc_tail_rank_losses(outputs, labels_nez, mask, epoch, args)
    atc_floor, atc_pair, atc_parts = atc_tail_losses(outputs, labels_nez, mask, epoch, args)
    total = total + tail_rank + clean_tail + atc_floor + atc_pair
    valid_count = mask.sum().clamp_min(1).to(outputs["logits"].dtype)
    utilization = outputs["prototype_utilization"]
    parts: dict[str, float | str] = {
        "cane_path_total_loss": float(total.detach().cpu()),
        "base_p2_total_loss": float(base_p2_total.detach().cpu()),
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
        "tail_rank_loss": float(tail_rank.detach().cpu()),
        "clean_tail_consistency_loss": float(clean_tail.detach().cpu()),
        **tail_parts, **atc_parts,
    }
    for index in range(int(utilization.numel())):
        parts[f"cane_path_prototype_utilization_{index}"] = float(utilization[index].detach().cpu())
    return total, parts


__all__ = [
    "CenterBalancedPatientBatchSampler", "compute_cane_path_cp_ranking_loss",
    "patient_balanced_bce_rows", "soft_worst_center_loss", "rtc_tail_rank_losses", "atc_tail_losses",
]
