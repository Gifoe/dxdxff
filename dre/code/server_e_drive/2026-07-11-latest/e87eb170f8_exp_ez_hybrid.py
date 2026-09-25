from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from data_factory import data_provider, split_train_val_subjects
from ez_dataset import flatten_window_samples
from neuroez_c.dual_view_data import build_raw_alignment_store
from neuroez_c.n6_dualview_ema_loss import compute_n6_total_loss
from neuroez_c.quality import normalize_quality_label, resolve_quality_label


def _read_train_dropout_candidates(path_value: str) -> list[str]:
    """Read a unique patient list for train-only ablations without touching test folds."""
    path = Path(path_value)
    if not path.is_file():
        raise ValueError(f"train_subject_dropout_file does not exist: {path}")
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or "subject_id" not in {str(name).strip().lower() for name in reader.fieldnames}:
                raise ValueError("train_subject_dropout_file CSV must contain a subject_id column")
            subject_column = next(name for name in reader.fieldnames if str(name).strip().lower() == "subject_id")
            candidates = [str(row.get(subject_column, "")).strip() for row in reader]
    else:
        candidates = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines()]
    candidates = [subject_id for subject_id in candidates if subject_id and not subject_id.startswith("#")]
    if not candidates:
        raise ValueError("train_subject_dropout_file contains no patient IDs")
    duplicates = sorted({subject_id for subject_id in candidates if candidates.count(subject_id) > 1})
    if duplicates:
        raise ValueError(f"train_subject_dropout_file contains duplicate patient IDs: {duplicates[:5]}")
    return sorted(candidates)


def _resolve_runtime(args: Any) -> Dict[str, Any]:
    from neuroez_c.dataset import (
        PatientNeuroEZCDataset,
        build_patient_examples as build_neuroez_c_examples,
        collate_patient_ez_batch as collate_neuroez_c_batch,
        fit_window_tensor_normalizer as fit_neuroez_c_normalizer,
    )
    from neuroez_c.model import NeuroEZCModel

    return {
        "name": "A9v5-TwoExpert-NAREZ" if bool(getattr(args, "use_two_expert_router", False)) else "B0-Pruned-EZBackbone",
        "dataset_cls": PatientNeuroEZCDataset,
        "build_patient_examples": build_neuroez_c_examples,
        "collate_fn": collate_neuroez_c_batch,
        "fit_normalizer": fit_neuroez_c_normalizer,
        "model_cls": NeuroEZCModel,
    }


def _set_random_seed(seed: int) -> None:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _clone_ema_teacher(student: torch.nn.Module) -> torch.nn.Module:
    teacher = copy.deepcopy(student)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher


@torch.no_grad()
def _update_ema_teacher(teacher: torch.nn.Module, student: torch.nn.Module, decay: float) -> None:
    for teacher_value, student_value in zip(teacher.parameters(), student.parameters()):
        teacher_value.mul_(float(decay)).add_(student_value.detach(), alpha=1.0 - float(decay))
    for teacher_buffer, student_buffer in zip(teacher.buffers(), student.buffers()):
        if torch.is_floating_point(teacher_buffer):
            teacher_buffer.mul_(float(decay)).add_(student_buffer.detach(), alpha=1.0 - float(decay))
        else:
            teacher_buffer.copy_(student_buffer)
    teacher.eval()


def _acquire_device(args: Any) -> torch.device:
    preferred = str(getattr(args, "device", "auto")).lower()
    if preferred == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if preferred.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(preferred)


def _move_tensors_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    non_blocking = device.type == "cuda"
    return {
        key: value.to(device, non_blocking=non_blocking) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _masked_bce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    labels_ez: torch.Tensor,
    mask: torch.Tensor,
    *,
    class_weight_mode: str,
    ez_negative_weight: torch.Tensor,
) -> torch.Tensor:
    valid = mask & (labels >= 0.0)
    if not torch.any(valid):
        return logits.sum() * 0.0
    losses = F.binary_cross_entropy_with_logits(logits[valid], labels[valid], reduction="none")
    if str(class_weight_mode).lower() == "ez_negative":
        weights = torch.ones_like(losses)
        weights = torch.where(labels_ez[valid] > 0.5, ez_negative_weight.to(logits.device).expand_as(weights), weights)
        return (losses * weights).sum() / weights.sum().clamp_min(1e-6)
    return losses.mean()


def _patient_weight(batch: Dict[str, Any], patient_idx: int, args: Any, logits: torch.Tensor) -> torch.Tensor:
    mode = str(getattr(args, "patient_loss_weighting", "uniform")).lower()
    if mode == "uniform":
        return torch.ones((), dtype=logits.dtype, device=logits.device)
    if mode == "inverse_ez_fraction":
        ez_fraction = batch.get("ez_fraction")
        if torch.is_tensor(ez_fraction) and ez_fraction.numel() > patient_idx:
            fraction = ez_fraction.to(logits.device, dtype=logits.dtype)[patient_idx].clamp(1e-3, 1.0)
        else:
            labels_ez = batch["labels_ez"][patient_idx]
            mask = batch["channel_mask"][patient_idx] & (labels_ez >= 0.0)
            fraction = ((labels_ez[mask] > 0.5).float().mean() if torch.any(mask) else torch.ones((), device=logits.device)).clamp(1e-3, 1.0)
        return (1.0 / fraction).clamp(max=20.0)
    if mode == "inverse_sqrt_channels":
        counts = batch.get("valid_channel_count")
        if torch.is_tensor(counts) and counts.numel() > patient_idx:
            count = counts.to(logits.device, dtype=logits.dtype)[patient_idx].clamp_min(1.0)
        else:
            count = batch["channel_mask"][patient_idx].float().sum().to(logits.device).clamp_min(1.0)
        return torch.rsqrt(count)
    raise ValueError(f"Unsupported patient_loss_weighting={mode!r}.")


def _quality_patient_weight(batch: Dict[str, Any], patient_idx: int, args: Any, logits: torch.Tensor) -> torch.Tensor:
    if not bool(getattr(args, "quality_weight_loss", True)):
        return torch.ones((), dtype=logits.dtype, device=logits.device)
    record_weights = batch.get("record_quality_weight_active", batch.get("record_quality_weight"))
    seizure_mask = batch.get("seizure_mask")
    if not torch.is_tensor(record_weights) or not torch.is_tensor(seizure_mask):
        return torch.ones((), dtype=logits.dtype, device=logits.device)
    if patient_idx >= record_weights.shape[0]:
        return torch.ones((), dtype=logits.dtype, device=logits.device)
    valid = seizure_mask[patient_idx].to(device=logits.device).bool()
    if not torch.any(valid):
        return torch.ones((), dtype=logits.dtype, device=logits.device)
    weights = record_weights.to(device=logits.device, dtype=logits.dtype)[patient_idx][valid].clamp_min(0.0)
    return weights.mean() if weights.numel() else torch.ones((), dtype=logits.dtype, device=logits.device)


def _quality_patient_weights_vector(batch: Dict[str, Any], args: Any, logits: torch.Tensor) -> torch.Tensor | None:
    if not bool(getattr(args, "quality_weight_loss", True)):
        return None
    record_weights = batch.get("record_quality_weight_active", batch.get("record_quality_weight"))
    seizure_mask = batch.get("seizure_mask")
    if not torch.is_tensor(record_weights) or not torch.is_tensor(seizure_mask):
        return None
    weights = [
        _quality_patient_weight(batch, patient_idx, args, logits)
        for patient_idx in range(int(logits.shape[0]))
    ]
    return torch.stack(weights).to(device=logits.device, dtype=logits.dtype) if weights else None


def _patient_balanced_bce_loss(
    logits: torch.Tensor,
    batch: Dict[str, Any],
    args: Any,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    labels = batch["labels"]
    labels_ez = batch["labels_ez"]
    channel_mask = batch["channel_mask"]
    losses: List[torch.Tensor] = []
    weights: List[torch.Tensor] = []
    single_class_count = 0
    zero_quality_count = 0
    for patient_idx in range(logits.shape[0]):
        valid = channel_mask[patient_idx] & (labels[patient_idx] >= 0.0) & (labels_ez[patient_idx] >= 0.0)
        if not torch.any(valid):
            continue
        ez_mask = valid & (labels_ez[patient_idx] > 0.5)
        nez_mask = valid & (labels_ez[patient_idx] <= 0.5)
        class_losses: List[torch.Tensor] = []
        if torch.any(ez_mask):
            class_losses.append(F.binary_cross_entropy_with_logits(logits[patient_idx][ez_mask], labels[patient_idx][ez_mask]))
        if torch.any(nez_mask):
            class_losses.append(F.binary_cross_entropy_with_logits(logits[patient_idx][nez_mask], labels[patient_idx][nez_mask]))
        if len(class_losses) == 1:
            single_class_count += 1
            patient_loss = class_losses[0]
        else:
            patient_loss = 0.5 * class_losses[0] + 0.5 * class_losses[1]
        losses.append(patient_loss)
        patient_weight = _patient_weight(batch, patient_idx, args, logits) * _quality_patient_weight(batch, patient_idx, args, logits)
        if float(patient_weight.detach().cpu()) <= 0.0:
            zero_quality_count += 1
        weights.append(patient_weight)
    if not losses:
        zero = logits.sum() * 0.0
        return zero, {"patient_balanced_single_class_count": 0.0, "quality_zero_weight_patient_count": 0.0}
    stacked = torch.stack(losses)
    patient_weights = torch.stack(weights).to(logits.device, dtype=logits.dtype)
    if float(patient_weights.sum().detach().cpu()) <= 0.0:
        return logits.sum() * 0.0, {
            "patient_balanced_single_class_count": float(single_class_count),
            "quality_zero_weight_patient_count": float(zero_quality_count),
            "quality_effective_patient_weight_mean": 0.0,
        }
    loss = (stacked * patient_weights).sum() / patient_weights.sum().clamp_min(1e-6)
    return loss, {
        "patient_balanced_single_class_count": float(single_class_count),
        "quality_zero_weight_patient_count": float(zero_quality_count),
        "quality_effective_patient_weight_mean": float(patient_weights.mean().detach().cpu()),
    }


def _prevalence_weighted_bce_loss(logits: torch.Tensor, batch: Dict[str, Any]) -> torch.Tensor:
    labels = batch["labels"]
    labels_ez = batch["labels_ez"]
    valid = batch["channel_mask"] & (labels >= 0.0) & (labels_ez >= 0.0)
    if not torch.any(valid):
        return logits.sum() * 0.0
    losses = F.binary_cross_entropy_with_logits(logits[valid], labels[valid], reduction="none")
    weights = torch.ones_like(losses)
    ez_valid = labels_ez[valid] > 0.5
    n_ez = ez_valid.float().sum().clamp_min(1.0)
    n_nez = (~ez_valid).float().sum().clamp_min(1.0)
    weights = torch.where(ez_valid, 0.5 / n_ez, 0.5 / n_nez)
    return (losses * weights).sum()


def _ez_semantic_scores(logits: torch.Tensor, positive_label: str = "ez") -> torch.Tensor:
    return logits if str(positive_label).lower() == "ez" else -logits


def _a9v11_patient_balanced_bce_vector(
    logits: torch.Tensor,
    batch: Dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    labels = batch["labels"]
    labels_ez = batch["labels_ez"]
    channel_mask = batch["channel_mask"]
    patient_losses: list[torch.Tensor] = []
    patient_valid: list[torch.Tensor] = []
    for patient_idx in range(logits.shape[0]):
        valid = channel_mask[patient_idx] & (labels[patient_idx] >= 0.0) & (labels_ez[patient_idx] >= 0.0)
        if not torch.any(valid):
            patient_losses.append(logits[patient_idx].sum() * 0.0)
            patient_valid.append(torch.zeros((), dtype=torch.bool, device=logits.device))
            continue
        raw = F.binary_cross_entropy_with_logits(logits[patient_idx][valid], labels[patient_idx][valid], reduction="none")
        ez_mask = labels_ez[patient_idx][valid] > 0.5
        nez_mask = ~ez_mask
        if torch.any(ez_mask) and torch.any(nez_mask):
            patient_loss = 0.5 * raw[ez_mask].mean() + 0.5 * raw[nez_mask].mean()
        else:
            patient_loss = raw.mean()
        patient_losses.append(patient_loss)
        patient_valid.append(torch.ones((), dtype=torch.bool, device=logits.device))
    return torch.stack(patient_losses), torch.stack(patient_valid)


def _a9v11_soft_topk_coverage_loss(
    scores_ez: torch.Tensor,
    labels_ez: torch.Tensor,
    channel_mask: torch.Tensor,
    *,
    soft_rank_tau: float = 0.1,
    soft_topk_tau: float = 0.25,
) -> torch.Tensor:
    tau_rank = max(float(soft_rank_tau), 1e-6)
    tau_topk = max(float(soft_topk_tau), 1e-6)
    patient_losses: list[torch.Tensor] = []
    for patient_idx in range(scores_ez.shape[0]):
        valid = channel_mask[patient_idx] & (labels_ez[patient_idx] >= 0.0)
        y = labels_ez[patient_idx][valid]
        if not torch.any(valid):
            continue
        k_pos = int((y > 0.5).sum().detach().cpu())
        if k_pos <= 0:
            continue
        scores = scores_ez[patient_idx][valid]
        diff = (scores.view(1, -1) - scores.view(-1, 1)) / tau_rank
        pair = torch.sigmoid(diff)
        eye = torch.eye(scores.numel(), dtype=torch.bool, device=scores.device)
        pair = pair.masked_fill(eye, 0.0)
        soft_rank = 1.0 + pair.sum(dim=1)
        soft_topk = torch.sigmoid((float(k_pos) + 0.5 - soft_rank) / tau_topk)
        coverage = (y * soft_topk).sum() / max(float(k_pos), 1.0)
        patient_losses.append(1.0 - coverage)
    if not patient_losses:
        return scores_ez.sum() * 0.0
    return torch.nan_to_num(torch.stack(patient_losses).mean(), nan=0.0, posinf=0.0, neginf=0.0)


def _a9v11_hard_pairwise_loss(
    scores_ez: torch.Tensor,
    labels_ez: torch.Tensor,
    channel_mask: torch.Tensor,
    *,
    margin: float = 0.1,
    top_negatives: int = 16,
) -> torch.Tensor:
    patient_losses: list[torch.Tensor] = []
    top_negatives = max(1, int(top_negatives))
    for patient_idx in range(scores_ez.shape[0]):
        valid = channel_mask[patient_idx] & (labels_ez[patient_idx] >= 0.0)
        pos = valid & (labels_ez[patient_idx] > 0.5)
        neg = valid & (labels_ez[patient_idx] <= 0.5)
        if not torch.any(pos) or not torch.any(neg):
            continue
        neg_scores = scores_ez[patient_idx][neg]
        hard_k = min(top_negatives, int(neg_scores.numel()))
        hard_neg = neg_scores.topk(hard_k, largest=True).values
        pos_scores = scores_ez[patient_idx][pos]
        patient_losses.append(F.softplus(float(margin) - pos_scores.view(-1, 1) + hard_neg.view(1, -1)).mean())
    if not patient_losses:
        return scores_ez.sum() * 0.0
    return torch.stack(patient_losses).mean()


def _a9v11_first_positive_rank_loss(
    scores_ez: torch.Tensor,
    labels_ez: torch.Tensor,
    channel_mask: torch.Tensor,
    *,
    margin: float = 0.1,
    tau: float = 0.1,
    top_negatives: int = 16,
) -> torch.Tensor:
    patient_losses: list[torch.Tensor] = []
    tau = max(float(tau), 1e-6)
    top_negatives = max(1, int(top_negatives))
    for patient_idx in range(scores_ez.shape[0]):
        valid = channel_mask[patient_idx] & (labels_ez[patient_idx] >= 0.0)
        pos = valid & (labels_ez[patient_idx] > 0.5)
        neg = valid & (labels_ez[patient_idx] <= 0.5)
        if not torch.any(pos) or not torch.any(neg):
            continue
        max_pos = tau * torch.logsumexp(scores_ez[patient_idx][pos] / tau, dim=0)
        neg_scores = scores_ez[patient_idx][neg]
        hard_k = min(top_negatives, int(neg_scores.numel()))
        max_neg = neg_scores.topk(hard_k, largest=True).values.max()
        patient_losses.append(F.softplus(float(margin) - max_pos + max_neg))
    if not patient_losses:
        return scores_ez.sum() * 0.0
    return torch.stack(patient_losses).mean()


def _a9v11_multi_positive_diversity_loss(
    scores_ez: torch.Tensor,
    labels_ez: torch.Tensor,
    channel_mask: torch.Tensor,
) -> torch.Tensor:
    patient_losses: list[torch.Tensor] = []
    for patient_idx in range(scores_ez.shape[0]):
        valid = channel_mask[patient_idx] & (labels_ez[patient_idx] > 0.5)
        if int(valid.sum().detach().cpu()) < 2:
            continue
        pos_scores = scores_ez[patient_idx][valid]
        patient_losses.append(F.relu(pos_scores.mean() - pos_scores).mean())
    if not patient_losses:
        return scores_ez.sum() * 0.0
    return torch.stack(patient_losses).mean()


def _a9v11_group_reduce(
    patient_losses: torch.Tensor,
    patient_valid: torch.Tensor,
    batch: Dict[str, Any],
    args: Any,
    *,
    group_weights: torch.Tensor | None = None,
    update_group_dro: bool = False,
) -> tuple[torch.Tensor, Dict[str, Any]]:
    mode = str(getattr(args, "group_robust_mode", "none")).lower()
    if mode == "none":
        return patient_losses[patient_valid].mean() if torch.any(patient_valid) else patient_losses.sum() * 0.0, {
            "group_robust_mode": "none",
            "group_robust_mode_effective": "none",
            "group_names": "",
            "group_weights_final": "",
        }
    center_id = batch.get("center_id")
    if center_id is None:
        return patient_losses[patient_valid].mean() if torch.any(patient_valid) else patient_losses.sum() * 0.0, {
            "group_robust_mode": mode,
            "group_robust_mode_effective": "no_center_id_uniform_patient_mean",
            "group_names": "unknown",
            "group_weights_final": "1.0",
        }
    center_id = center_id.to(patient_losses.device).long()
    min_count = max(1, int(getattr(args, "group_dro_min_count", 1)))
    group_losses: list[torch.Tensor] = []
    group_ids: list[int] = []
    group_names: list[str] = []
    for group_id, group_name in ((0, "hup"), (1, "lzu"), (2, "multicenter"), (3, "pediatric"), (4, "unknown")):
        group_mask = patient_valid & (center_id == group_id)
        if int(group_mask.sum().detach().cpu()) >= min_count:
            group_losses.append(patient_losses[group_mask].mean())
            group_ids.append(group_id)
            group_names.append(group_name)
    if not group_losses:
        return patient_losses[patient_valid].mean() if torch.any(patient_valid) else patient_losses.sum() * 0.0, {
            "group_robust_mode": mode,
            "group_robust_mode_effective": "no_available_groups_uniform_patient_mean",
            "group_names": "",
            "group_weights_final": "",
        }
    stacked = torch.stack(group_losses)
    if mode == "center_balanced":
        weights = torch.full_like(stacked, 1.0 / float(stacked.numel()))
        full_weights = torch.zeros((5,), dtype=stacked.dtype, device=stacked.device)
        for local_idx, group_id in enumerate(group_ids):
            full_weights[group_id] = weights[local_idx]
        effective = "center_balanced"
    elif mode == "center_group_dro":
        eta = float(getattr(args, "group_dro_eta", 0.05))
        if group_weights is None:
            full_weights = torch.full((5,), 0.2, dtype=stacked.dtype, device=stacked.device)
            effective = "uniform_center_group_dro_no_persistent_state"
        else:
            full_weights = group_weights.to(device=stacked.device, dtype=stacked.dtype).clone()
            effective = "persistent_center_group_dro"
            if update_group_dro:
                for local_idx, group_id in enumerate(group_ids):
                    full_weights[group_id] = full_weights[group_id] * torch.exp(eta * stacked[local_idx].detach())
                full_weights = full_weights / full_weights.sum().clamp_min(1e-12)
                group_weights.copy_(full_weights.to(device=group_weights.device, dtype=group_weights.dtype))
        local_weights = torch.stack([full_weights[group_id] for group_id in group_ids])
        weights = local_weights / local_weights.sum().clamp_min(1e-12)
    else:
        raise ValueError(f"Unsupported group_robust_mode={mode!r}.")
    center_names = ("hup", "lzu", "multicenter", "pediatric", "unknown")
    weight_parts = {f"group_weight_{name}": float(full_weights[idx].detach().cpu()) for idx, name in enumerate(center_names)}
    return (weights * stacked).sum(), {
        "group_robust_mode": mode,
        "group_robust_mode_effective": effective,
        "group_dro_eta": float(getattr(args, "group_dro_eta", 0.05)),
        "group_names": ",".join(group_names),
        "group_weights_final": ",".join(f"{float(w.detach().cpu()):.6f}" for w in full_weights),
        **weight_parts,
    }


def _a9v11_dynamic_listwise_loss(
    logits: torch.Tensor,
    batch: Dict[str, Any],
    args: Any,
    *,
    group_weights: torch.Tensor | None = None,
    update_group_dro: bool = False,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    bce_per_patient, patient_valid = _a9v11_patient_balanced_bce_vector(logits, batch)
    labels_ez = batch["labels_ez"]
    channel_mask = batch["channel_mask"]
    scores_ez = _ez_semantic_scores(logits, str(getattr(args, "positive_label", "ez")))
    hard_pair = _a9v11_hard_pairwise_loss(
        scores_ez,
        labels_ez,
        channel_mask,
        margin=float(getattr(args, "hard_pair_margin", 0.1)),
        top_negatives=int(getattr(args, "hard_pair_top_negatives", 16)),
    )
    soft_topk = _a9v11_soft_topk_coverage_loss(
        scores_ez,
        labels_ez,
        channel_mask,
        soft_rank_tau=float(getattr(args, "soft_rank_tau", 0.1)),
        soft_topk_tau=float(getattr(args, "soft_topk_tau", 0.25)),
    )
    first_rank = _a9v11_first_positive_rank_loss(
        scores_ez,
        labels_ez,
        channel_mask,
        margin=float(getattr(args, "hard_pair_margin", 0.1)),
        tau=float(getattr(args, "soft_rank_tau", 0.1)),
        top_negatives=int(getattr(args, "hard_pair_top_negatives", 16)),
    )
    diversity = _a9v11_multi_positive_diversity_loss(scores_ez, labels_ez, channel_mask)
    per_patient = (
        bce_per_patient
        + float(getattr(args, "hard_pairwise_weight", 0.0)) * hard_pair
        + float(getattr(args, "soft_topk_weight", 0.0)) * soft_topk
        + float(getattr(args, "first_rank_weight", 0.0)) * first_rank
        + float(getattr(args, "diversity_weight", 0.0)) * diversity
    )
    loss, group_parts = _a9v11_group_reduce(
        per_patient,
        patient_valid,
        batch,
        args,
        group_weights=group_weights,
        update_group_dro=update_group_dro,
    )
    parts: Dict[str, Any] = {
        "loss_mode": "patient_dynamic_listwise",
        "bce": float(bce_per_patient[patient_valid].mean().detach().cpu()) if torch.any(patient_valid) else 0.0,
        "hard_pairwise_loss": float(hard_pair.detach().cpu()),
        "soft_topk_coverage_loss": float(soft_topk.detach().cpu()),
        "first_positive_rank_loss": float(first_rank.detach().cpu()),
        "multi_positive_diversity_loss": float(diversity.detach().cpu()),
        "hard_pairwise_weight": float(getattr(args, "hard_pairwise_weight", 0.0)),
        "soft_topk_weight": float(getattr(args, "soft_topk_weight", 0.0)),
        "first_rank_weight": float(getattr(args, "first_rank_weight", 0.0)),
        "diversity_weight": float(getattr(args, "diversity_weight", 0.0)),
        "soft_rank_tau": float(getattr(args, "soft_rank_tau", 0.1)),
        "soft_topk_tau": float(getattr(args, "soft_topk_tau", 0.25)),
        "hard_pair_margin": float(getattr(args, "hard_pair_margin", 0.1)),
        "hard_pair_top_negatives": int(getattr(args, "hard_pair_top_negatives", 16)),
        "no_center_features": True,
    }
    parts.update(group_parts)
    return loss, parts


def _parse_broad_ez_centers(spec: str) -> set[int]:
    mapping = {
        "hup": 0,
        "lzu": 1,
        "multicenter": 2,
        "pediatric": 3,
    }
    out: set[int] = set()
    for token in str(spec).split(","):
        token = token.strip().lower()
        if not token:
            continue
        if token.isdigit():
            out.add(int(token))
        elif token in mapping:
            out.add(mapping[token])
        else:
            raise ValueError(f"Unsupported broad EZ center token: {token!r}")
    return out


def _broad_patient_mask(batch: Dict[str, Any], broad_centers: set[int], logits: torch.Tensor) -> torch.Tensor:
    center_id = batch.get("center_id")
    if center_id is None:
        return torch.zeros((logits.shape[0],), dtype=torch.bool, device=logits.device)
    center_id = center_id.to(logits.device).long()
    broad_mask = torch.zeros_like(center_id, dtype=torch.bool)
    for center in broad_centers:
        broad_mask = broad_mask | (center_id == int(center))
    return broad_mask


def _broad_ez_mil_ranking_loss(
    logits: torch.Tensor,
    labels_ez: torch.Tensor,
    channel_mask: torch.Tensor,
    center_id: torch.Tensor | None,
    ez_fraction: torch.Tensor | None,
    *,
    positive_label: str = "ez",
    broad_centers: set[int] | None = None,
    min_ez_fraction: float = 0.35,
    core_frac: float = 0.30,
    margin: float = 0.05,
    hard_neg_multiplier: float = 1.0,
) -> torch.Tensor:
    """Weak-positive ranking loss for centers with broad clinical EZ labels."""
    broad_centers = set(broad_centers or set())
    core_frac = min(1.0, max(1e-6, float(core_frac)))
    hard_neg_multiplier = max(1.0, float(hard_neg_multiplier))
    positive = str(positive_label).lower()
    score_positive = torch.sigmoid(logits)
    if positive == "ez":
        score_ez = score_positive
    elif positive == "nez":
        score_ez = 1.0 - score_positive
    else:
        raise ValueError(f"Unsupported positive_label={positive!r}; expected 'ez' or 'nez'.")

    if center_id is None:
        center_id_device = torch.full((logits.shape[0],), -1, dtype=torch.long, device=logits.device)
    else:
        center_id_device = center_id.to(logits.device).long()
    if ez_fraction is None:
        ez_fraction_device = torch.zeros((logits.shape[0],), dtype=logits.dtype, device=logits.device)
    else:
        ez_fraction_device = ez_fraction.to(logits.device, dtype=logits.dtype)

    patient_losses: list[torch.Tensor] = []
    valid_label = labels_ez >= 0.0
    for patient_idx in range(logits.shape[0]):
        valid = channel_mask[patient_idx] & valid_label[patient_idx]
        ez = (labels_ez[patient_idx] > 0.5) & valid
        nez = (labels_ez[patient_idx] <= 0.5) & valid
        is_broad_center = int(center_id_device[patient_idx].item()) in broad_centers
        is_high_ez_fraction = bool(ez_fraction_device[patient_idx] >= float(min_ez_fraction))
        if not (is_broad_center or is_high_ez_fraction):
            continue
        if not torch.any(ez) or not torch.any(nez):
            continue

        ez_scores = score_ez[patient_idx][ez]
        nez_scores = score_ez[patient_idx][nez]
        n_core = max(1, int(np.ceil(core_frac * int(ez_scores.numel()))))
        n_core = min(n_core, int(ez_scores.numel()))
        core_pos = ez_scores.topk(n_core, largest=True).values
        n_hard = max(1, int(np.ceil(hard_neg_multiplier * n_core)))
        n_hard = min(n_hard, int(nez_scores.numel()))
        hard_neg = nez_scores.topk(n_hard, largest=True).values
        patient_losses.append(F.softplus(float(margin) + hard_neg.mean() - core_pos.mean()))

    if not patient_losses:
        return logits.sum() * 0.0
    return torch.stack(patient_losses).mean()


def _compute_broad_aware_supervised_loss(
    logits: torch.Tensor,
    batch: Dict[str, Any],
    args: Any,
    ez_negative_weight: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    if not bool(getattr(args, "use_broad_ez_mil_loss", False)):
        return _compute_supervised_loss(logits, batch, args, ez_negative_weight)

    broad_centers = _parse_broad_ez_centers(str(getattr(args, "broad_ez_centers", "lzu,pediatric")))
    positive_scale = max(0.0, float(getattr(args, "broad_ez_positive_bce_scale", 0.5)))
    broad_patient = _broad_patient_mask(batch, broad_centers, logits)
    labels = batch["labels"]
    labels_ez = batch["labels_ez"]
    channel_mask = batch["channel_mask"]
    broad_ez_positive = broad_patient[:, None] & (labels_ez > 0.5) & channel_mask
    loss_mode = str(getattr(args, "loss_mode", "masked_bce")).lower()

    if loss_mode == "masked_bce":
        valid = channel_mask & (labels >= 0.0)
        if not torch.any(valid):
            return logits.sum() * 0.0, {"loss_mode": "masked_bce", "broad_ez_positive_bce_scale": positive_scale}
        losses = F.binary_cross_entropy_with_logits(logits[valid], labels[valid], reduction="none")
        weights = torch.ones_like(losses)
        if str(getattr(args, "class_weight_mode", "ez_negative")).lower() == "ez_negative":
            weights = torch.where(labels_ez[valid] > 0.5, ez_negative_weight.to(logits.device).expand_as(weights), weights)
        weights = torch.where(broad_ez_positive[valid], weights * positive_scale, weights)
        loss = (losses * weights).sum() / weights.sum().clamp_min(1e-6)
        return loss, {"loss_mode": "masked_bce", "broad_ez_positive_bce_scale": positive_scale}

    if loss_mode == "patient_balanced_bce":
        losses: List[torch.Tensor] = []
        weights: List[torch.Tensor] = []
        single_class_count = 0
        for patient_idx in range(logits.shape[0]):
            valid = channel_mask[patient_idx] & (labels[patient_idx] >= 0.0) & (labels_ez[patient_idx] >= 0.0)
            if not torch.any(valid):
                continue
            ez_mask = valid & (labels_ez[patient_idx] > 0.5)
            nez_mask = valid & (labels_ez[patient_idx] <= 0.5)
            class_losses: List[torch.Tensor] = []
            if torch.any(ez_mask):
                ez_loss = F.binary_cross_entropy_with_logits(logits[patient_idx][ez_mask], labels[patient_idx][ez_mask])
                if bool(broad_patient[patient_idx]):
                    ez_loss = ez_loss * positive_scale
                class_losses.append(ez_loss)
            if torch.any(nez_mask):
                class_losses.append(F.binary_cross_entropy_with_logits(logits[patient_idx][nez_mask], labels[patient_idx][nez_mask]))
            if len(class_losses) == 1:
                single_class_count += 1
                patient_loss = class_losses[0]
            else:
                patient_loss = 0.5 * class_losses[0] + 0.5 * class_losses[1]
            losses.append(patient_loss)
            weights.append(_patient_weight(batch, patient_idx, args, logits))
        if not losses:
            zero = logits.sum() * 0.0
            return zero, {"loss_mode": "patient_balanced_bce", "patient_balanced_single_class_count": 0.0, "broad_ez_positive_bce_scale": positive_scale}
        stacked = torch.stack(losses)
        patient_weights = torch.stack(weights).to(logits.device, dtype=logits.dtype)
        loss = (stacked * patient_weights).sum() / patient_weights.sum().clamp_min(1e-6)
        return loss, {
            "loss_mode": "patient_balanced_bce",
            "patient_balanced_single_class_count": float(single_class_count),
            "broad_ez_positive_bce_scale": positive_scale,
        }

    if loss_mode == "prevalence_weighted_bce":
        valid = channel_mask & (labels >= 0.0) & (labels_ez >= 0.0)
        if not torch.any(valid):
            return logits.sum() * 0.0, {"loss_mode": "prevalence_weighted_bce", "broad_ez_positive_bce_scale": positive_scale}
        losses = F.binary_cross_entropy_with_logits(logits[valid], labels[valid], reduction="none")
        weights = torch.ones_like(losses)
        ez_valid = labels_ez[valid] > 0.5
        n_ez = ez_valid.float().sum().clamp_min(1.0)
        n_nez = (~ez_valid).float().sum().clamp_min(1.0)
        weights = torch.where(ez_valid, 0.5 / n_ez, 0.5 / n_nez)
        weights = torch.where(broad_ez_positive[valid], weights * positive_scale, weights)
        return (losses * weights).sum(), {"loss_mode": "prevalence_weighted_bce", "broad_ez_positive_bce_scale": positive_scale}

    raise ValueError(f"Unsupported loss_mode={loss_mode!r}.")


def _compute_supervised_loss(
    logits: torch.Tensor,
    batch: Dict[str, Any],
    args: Any,
    ez_negative_weight: torch.Tensor,
    *,
    group_weights: torch.Tensor | None = None,
    update_group_dro: bool = False,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    loss_mode = str(getattr(args, "loss_mode", "masked_bce")).lower()
    if loss_mode == "masked_bce":
        loss = _masked_bce_loss(
            logits,
            batch["labels"],
            batch["labels_ez"],
            batch["channel_mask"],
            class_weight_mode=str(getattr(args, "class_weight_mode", "ez_negative")),
            ez_negative_weight=ez_negative_weight,
        )
        return loss, {"loss_mode": "masked_bce"}
    if loss_mode == "patient_balanced_bce":
        loss, parts = _patient_balanced_bce_loss(logits, batch, args)
        parts["loss_mode"] = "patient_balanced_bce"
        return loss, parts
    if loss_mode == "patient_dynamic_listwise":
        return _a9v11_dynamic_listwise_loss(
            logits,
            batch,
            args,
            group_weights=group_weights,
            update_group_dro=update_group_dro,
        )
    if loss_mode == "prevalence_weighted_bce":
        return _prevalence_weighted_bce_loss(logits, batch), {"loss_mode": "prevalence_weighted_bce"}
    raise ValueError(f"Unsupported loss_mode={loss_mode!r}.")


def _ez_pairwise_ranking_loss(
    logits: torch.Tensor,
    labels_ez: torch.Tensor,
    channel_mask: torch.Tensor,
    margin: float = 0.10,
    positive_label: str = "nez",
    patient_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    valid = channel_mask & (labels_ez >= 0.0)
    score_positive = torch.sigmoid(logits)
    score_ez = score_positive if str(positive_label).lower() == "ez" else 1.0 - score_positive
    patient_losses: List[torch.Tensor] = []

    ranking_weights: list[torch.Tensor] = []
    for patient_idx in range(logits.shape[0]):
        patient_valid = valid[patient_idx]
        ez_mask = patient_valid & (labels_ez[patient_idx] > 0.5)
        nez_mask = patient_valid & (labels_ez[patient_idx] <= 0.5)
        if not torch.any(ez_mask) or not torch.any(nez_mask):
            continue
        score_ez_ez = score_ez[patient_idx][ez_mask].view(-1, 1)
        score_ez_nez = score_ez[patient_idx][nez_mask].view(1, -1)
        pair_loss = F.relu(float(margin) - score_ez_ez + score_ez_nez)
        patient_losses.append(pair_loss.mean())
        if patient_weights is not None and patient_idx < patient_weights.numel():
            ranking_weights.append(patient_weights.to(device=logits.device, dtype=logits.dtype)[patient_idx].clamp_min(0.0))
        else:
            ranking_weights.append(torch.ones((), dtype=logits.dtype, device=logits.device))

    if not patient_losses:
        return logits.sum() * 0.0
    stacked = torch.stack(patient_losses)
    weights = torch.stack(ranking_weights).to(device=logits.device, dtype=logits.dtype)
    if float(weights.sum().detach().cpu()) <= 0.0:
        return logits.sum() * 0.0
    return (stacked * weights).sum() / weights.sum().clamp_min(1e-6)


def _hard_topk_retrieval_loss(
    logits: torch.Tensor,
    labels_ez: torch.Tensor,
    channel_mask: torch.Tensor,
    *,
    positive_label: str,
    margin: float = 0.05,
    topk_multiplier: float = 1.0,
) -> torch.Tensor:
    """Hard top-k retrieval loss: penalises when mean(EZ score) does not
    exceed mean(top-k hardest NEZ scores) by at least *margin*."""
    score_positive = torch.sigmoid(logits)
    if str(positive_label).lower() == "ez":
        score_ez = score_positive
    else:
        score_ez = 1.0 - score_positive

    patient_losses: list[torch.Tensor] = []
    for patient_idx in range(logits.shape[0]):
        valid = channel_mask[patient_idx] & (labels_ez[patient_idx] >= 0.0)
        if not valid.any():
            continue
        ez_mask = valid & (labels_ez[patient_idx] > 0.5)
        nez_mask = valid & (labels_ez[patient_idx] <= 0.5)
        if not torch.any(ez_mask) or not torch.any(nez_mask):
            continue
        k_ez = int(ez_mask.float().sum().item())
        num_nez = int(nez_mask.float().sum().item())
        hard_k = min(num_nez, max(1, int(np.ceil(k_ez * float(topk_multiplier)))))
        scores_nez = score_ez[patient_idx][nez_mask]
        hard_neg_mean = scores_nez.topk(hard_k, largest=True).values.mean()
        pos_mean = score_ez[patient_idx][ez_mask].mean()
        patient_losses.append(F.softplus(float(margin) + hard_neg_mean - pos_mean))

    if not patient_losses:
        return logits.sum() * 0.0
    return torch.stack(patient_losses).mean()


def _lcbo_norm_id(value: object) -> str:
    return str(value).strip().lower()


def _lcbo_norm_channel(value: object) -> str:
    return str(value).strip().lower().replace(" ", "").replace("-", "_").replace(".", "_")


def load_lcbo_fold_targets(target_dir: str | Path, *, fold_id: int) -> pd.DataFrame:
    path = Path(target_dir) / f"latent_core_targets_fold{int(fold_id)}_train.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing LCBO target file: {path}")
    df = pd.read_csv(path)
    required = {"fold_id", "channel_name", "label_ez", "pseudo_core_q", "a9v3_oof_score"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"LCBO target file {path} is missing columns: {missing}")
    if "patient_id" not in df.columns and "subject_id" not in df.columns:
        raise ValueError(f"LCBO target file {path} must include patient_id or subject_id.")
    if bool((df["fold_id"].astype(int) == int(fold_id)).any()):
        raise ValueError(f"LCBO target file {path} contains heldout fold {int(fold_id)} rows.")
    return df


def build_lcbo_target_lookup(targets: pd.DataFrame) -> dict[tuple[str, str], dict[str, float]]:
    lookup: dict[tuple[str, str], dict[str, float]] = {}
    for _, row in targets.iterrows():
        channel = _lcbo_norm_channel(row["channel_name"])
        payload = {
            "label_ez": float(row["label_ez"]),
            "pseudo_core_q": float(row["pseudo_core_q"]),
            "a9v3_oof_score": float(row["a9v3_oof_score"]),
        }
        for id_col in ("patient_id", "subject_id"):
            if id_col in targets.columns and pd.notna(row.get(id_col)):
                lookup[(_lcbo_norm_id(row[id_col]), channel)] = payload
    return lookup


def _lcbo_target_tensors(
    batch: Dict[str, Any],
    lookup: dict[tuple[str, str], dict[str, float]],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    labels_ez = batch["labels_ez"].to(device=device, dtype=dtype)
    core_target = torch.zeros_like(labels_ez)
    teacher_score = torch.zeros_like(labels_ez)
    target_mask = torch.zeros_like(batch["channel_mask"], dtype=torch.bool, device=device)
    subject_ids = list(batch.get("subject_id", []))
    canonical_channels = list(batch.get("canonical_channels", []))
    for pi, sid in enumerate(subject_ids):
        if pi >= len(canonical_channels):
            continue
        sid_norm = _lcbo_norm_id(sid)
        for ci, channel in enumerate(canonical_channels[pi]):
            if pi >= core_target.shape[0] or ci >= core_target.shape[1]:
                continue
            item = lookup.get((sid_norm, _lcbo_norm_channel(channel)))
            if item is None:
                continue
            core_target[pi, ci] = float(item["pseudo_core_q"])
            teacher_score[pi, ci] = float(item["a9v3_oof_score"])
            target_mask[pi, ci] = True
    return labels_ez, core_target, teacher_score, target_mask


def compute_lcbo_target_match_stats(
    batch: Dict[str, Any],
    lookup: dict[tuple[str, str], dict[str, float]],
) -> Dict[str, float]:
    device = batch["channel_mask"].device if torch.is_tensor(batch.get("channel_mask")) else torch.device("cpu")
    dtype = torch.float32
    labels_ez, core_target, _, target_mask = _lcbo_target_tensors(batch, lookup, device, dtype)
    del labels_ez
    channel_mask = batch["channel_mask"].to(device)
    valid = channel_mask.bool()
    matched = valid & target_mask
    total_valid = float(valid.sum().detach().cpu())
    matched_channels = float(matched.sum().detach().cpu())
    q_matched = core_target[matched] if torch.any(matched) else core_target.new_zeros((0,))
    sum_q = float(q_matched.sum().detach().cpu()) if q_matched.numel() else 0.0
    n_core_pos = float((q_matched > 0.0).sum().detach().cpu()) if q_matched.numel() else 0.0
    return {
        "target_match_rate": float(matched_channels / max(total_valid, 1.0)),
        "matched_channels": matched_channels,
        "total_valid_channels": total_valid,
        "sum_pseudo_core_q": sum_q,
        "n_core_positive_channels": n_core_pos,
        "mean_pseudo_core_q": float(sum_q / max(matched_channels, 1.0)),
    }


def _lcbo_patient_balanced_bce(logits: torch.Tensor, labels_ez: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    losses: List[torch.Tensor] = []
    for patient_idx in range(logits.shape[0]):
        valid = mask[patient_idx] & (labels_ez[patient_idx] >= 0.0)
        if not torch.any(valid):
            continue
        raw = F.binary_cross_entropy_with_logits(logits[patient_idx][valid], labels_ez[patient_idx][valid], reduction="none")
        pos = labels_ez[patient_idx][valid] > 0.5
        neg = ~pos
        if torch.any(pos) and torch.any(neg):
            losses.append(0.5 * raw[pos].mean() + 0.5 * raw[neg].mean())
        else:
            losses.append(raw.mean())
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


def _lcbo_core_rank_loss(
    logits_core: torch.Tensor,
    labels_ez: torch.Tensor,
    core_target: torch.Tensor,
    channel_mask: torch.Tensor,
    target_mask: torch.Tensor,
    *,
    margin: float,
    hard_neg_top_frac: float,
    min_core_mass: float,
) -> torch.Tensor:
    losses: List[torch.Tensor] = []
    top_frac = float(np.clip(float(hard_neg_top_frac), 0.0, 1.0))
    for patient_idx in range(logits_core.shape[0]):
        valid = channel_mask[patient_idx]
        pos = valid & target_mask[patient_idx] & (core_target[patient_idx] > 0.0)
        neg = valid & (labels_ez[patient_idx] <= 0.5)
        if not torch.any(pos) or not torch.any(neg):
            continue
        pos_mass = core_target[patient_idx][pos].sum()
        if float(pos_mass.detach().cpu()) < float(min_core_mass):
            continue
        neg_logits = logits_core[patient_idx][neg]
        if top_frac > 0.0 and top_frac < 1.0 and neg_logits.numel() > 1:
            k = max(1, int(np.ceil(float(neg_logits.numel()) * top_frac)))
            neg_logits = neg_logits.topk(k, largest=True).values
        pair = F.softplus(float(margin) + neg_logits.view(1, -1) - logits_core[patient_idx][pos].view(-1, 1))
        weights = core_target[patient_idx][pos].view(-1, 1).clamp_min(0.0)
        losses.append((pair * weights).sum() / weights.sum().clamp_min(1e-6))
    if not losses:
        return logits_core.sum() * 0.0
    return torch.stack(losses).mean()


def _lcbo_soft_mrr_loss(
    logits_eval: torch.Tensor,
    core_target: torch.Tensor,
    channel_mask: torch.Tensor,
    target_mask: torch.Tensor,
    *,
    tau: float,
    min_core_mass: float,
) -> torch.Tensor:
    eps = 1e-8
    losses: List[torch.Tensor] = []
    tau_t = max(float(tau), 1e-6)
    for patient_idx in range(logits_eval.shape[0]):
        valid = channel_mask[patient_idx] & target_mask[patient_idx]
        pos = valid & (core_target[patient_idx] > 0.0)
        if not torch.any(pos):
            continue
        if float(core_target[patient_idx][pos].sum().detach().cpu()) < float(min_core_mass):
            continue
        scores = logits_eval[patient_idx][valid]
        q = core_target[patient_idx][valid].clamp_min(0.0)
        rank = 1.0 + torch.sigmoid((scores.view(1, -1) - scores.view(-1, 1)) / tau_t).sum(dim=1)
        pos_q = q[q > 0.0]
        pos_rank = rank[q > 0.0]
        soft_mrr = (pos_q / pos_rank).sum() / pos_q.sum().clamp_min(eps)
        losses.append(-torch.log(soft_mrr.clamp_min(eps)))
    if not losses:
        return logits_eval.sum() * 0.0
    return torch.stack(losses).mean()


def _lcbo_core_distill_loss(
    logits_core: torch.Tensor,
    teacher_score: torch.Tensor,
    channel_mask: torch.Tensor,
    target_mask: torch.Tensor,
) -> torch.Tensor:
    losses: List[torch.Tensor] = []
    for patient_idx in range(logits_core.shape[0]):
        valid = channel_mask[patient_idx] & target_mask[patient_idx]
        if int(valid.sum().item()) < 2:
            continue
        core = logits_core[patient_idx][valid]
        teacher = teacher_score[patient_idx][valid]
        core_z = (core - core.mean()) / core.std(unbiased=False).clamp_min(1e-6)
        teacher_z = (teacher - teacher.mean()) / teacher.std(unbiased=False).clamp_min(1e-6)
        losses.append(F.mse_loss(core_z, teacher_z))
    if not losses:
        return logits_core.sum() * 0.0
    return torch.stack(losses).mean()


def _torch_patient_zscore(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(values)
    if int(valid.sum().item()) < 2:
        return out
    selected = values[valid]
    std = selected.std(unbiased=False)
    if not torch.isfinite(std) or float(std.detach().cpu()) < 1e-6:
        return out
    out[valid] = (selected - selected.mean()) / std.clamp_min(1e-6)
    return out


def _safe_logit_tensor(probs: torch.Tensor) -> torch.Tensor:
    clipped = probs.clamp(1e-7, 1.0 - 1e-7)
    return torch.log(clipped / (1.0 - clipped))


def apply_teacher_anchor_eval_scores(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, Any],
    target_lookup: dict[tuple[str, str], dict[str, float]],
    args: Any,
) -> Dict[str, torch.Tensor]:
    if not bool(getattr(args, "use_teacher_anchor_eval", False)):
        return outputs
    if not target_lookup and bool(getattr(args, "teacher_anchor_require_oof_score", True)):
        raise RuntimeError("A9v8 teacher anchor eval requires a non-empty LCBO OOF target lookup.")

    logits_eval = outputs["logits_eval"]
    _, _, teacher_score, target_mask = _lcbo_target_tensors(
        batch,
        target_lookup,
        logits_eval.device,
        logits_eval.dtype,
    )
    channel_mask = batch["channel_mask"].to(logits_eval.device).bool()
    valid_mask = channel_mask & target_mask
    if bool(getattr(args, "teacher_anchor_require_oof_score", True)) and torch.any(channel_mask & ~target_mask):
        matched = int((channel_mask & target_mask).sum().detach().cpu())
        total = int(channel_mask.sum().detach().cpu())
        raise RuntimeError(
            "A9v8 teacher anchor eval target match below 1.0: "
            f"matched_channels={matched}, total_valid_channels={total}"
        )

    alpha = float(getattr(args, "teacher_anchor_alpha", 0.10))
    beta = float(getattr(args, "teacher_anchor_beta", 0.00))
    temperature = max(float(getattr(args, "teacher_anchor_temperature", 1.0)), 1e-6)
    space = str(getattr(args, "teacher_anchor_space", "patient_zscore_probability"))
    anchored = torch.full_like(logits_eval, -1e9)
    for patient_idx in range(logits_eval.shape[0]):
        valid = valid_mask[patient_idx]
        if space == "patient_zscore_logit":
            t = _torch_patient_zscore(_safe_logit_tensor(teacher_score[patient_idx]), valid)
            b = _torch_patient_zscore(outputs["logits_broad"][patient_idx], valid)
            c = _torch_patient_zscore(outputs["logits_core"][patient_idx], valid)
        else:
            t = _torch_patient_zscore(teacher_score[patient_idx], valid)
            b = _torch_patient_zscore(outputs["score_broad"][patient_idx], valid)
            c = _torch_patient_zscore(outputs["score_core"][patient_idx], valid)
        anchored[patient_idx] = t + alpha * b + beta * c
        anchored[patient_idx] = anchored[patient_idx].masked_fill(~channel_mask[patient_idx], -1e9)

    outputs["logits_eval_original"] = outputs["logits_eval"]
    outputs["score_eval_original"] = outputs["score_eval"]
    outputs["logits_eval"] = anchored
    outputs["score_eval"] = torch.sigmoid(anchored / temperature)
    outputs["logits"] = outputs["logits_eval"]
    outputs["scores"] = outputs["score_eval"]
    outputs["score_ez"] = outputs["score_eval"]
    outputs["score_nez"] = 1.0 - outputs["score_eval"]
    outputs["teacher_anchor_alpha"] = torch.tensor(alpha, device=logits_eval.device, dtype=logits_eval.dtype)
    outputs["teacher_anchor_beta"] = torch.tensor(beta, device=logits_eval.device, dtype=logits_eval.dtype)
    return outputs


def _lcbo_corr(score_core: torch.Tensor, score_broad: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask & torch.isfinite(score_core) & torch.isfinite(score_broad)
    if int(valid.sum().item()) < 2:
        return score_core.sum() * 0.0
    x = score_core[valid]
    y = score_broad[valid]
    x = x - x.mean()
    y = y - y.mean()
    return (x * y).mean() / (x.square().mean().sqrt() * y.square().mean().sqrt()).clamp_min(1e-6)


def compute_a9v8_lcbo_loss(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, Any],
    target_lookup: dict[tuple[str, str], dict[str, float]],
    args: Any,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    logits_broad = outputs["logits_broad"]
    logits_core = outputs["logits_core"]
    logits_eval = outputs["logits_eval"]
    score_broad = outputs["score_broad"]
    score_core = outputs["score_core"]
    channel_mask = batch["channel_mask"].to(logits_broad.device)
    labels_ez, core_target, teacher_score, target_mask = _lcbo_target_tensors(
        batch,
        target_lookup,
        logits_broad.device,
        logits_broad.dtype,
    )
    target_stats = compute_lcbo_target_match_stats(batch, target_lookup)
    broad_bce = _lcbo_patient_balanced_bce(logits_broad, labels_ez, channel_mask)
    core_rank = _lcbo_core_rank_loss(
        logits_core,
        labels_ez,
        core_target,
        channel_mask,
        target_mask,
        margin=float(getattr(args, "core_rank_margin", 0.05)),
        hard_neg_top_frac=float(getattr(args, "lcbo_hard_neg_top_frac", 0.30)),
        min_core_mass=float(getattr(args, "lcbo_min_core_mass", 1.0)),
    )
    soft_mrr = _lcbo_soft_mrr_loss(
        logits_eval,
        core_target,
        channel_mask,
        target_mask,
        tau=float(getattr(args, "soft_mrr_tau", 0.10)),
        min_core_mass=float(getattr(args, "lcbo_min_core_mass", 1.0)),
    )
    subset_eps = float(getattr(args, "subset_eps", 0.0))
    subset_margin = F.relu(score_core - score_broad - subset_eps)
    subset = subset_margin[channel_mask].mean() if torch.any(channel_mask) else score_core.sum() * 0.0
    core_distill = _lcbo_core_distill_loss(logits_core, teacher_score, channel_mask, target_mask)
    loss = (
        broad_bce
        + float(getattr(args, "lambda_core_rank", 0.20)) * core_rank
        + float(getattr(args, "lambda_soft_mrr", 0.05)) * soft_mrr
        + float(getattr(args, "lambda_subset", 0.02)) * subset
        + float(getattr(args, "lambda_core_distill", 0.05)) * core_distill
    )
    valid = channel_mask & torch.isfinite(score_core) & torch.isfinite(score_broad)
    subset_violation = (subset_margin[valid] > 0.0).float().mean() if torch.any(valid) else score_core.sum() * 0.0
    corr = _lcbo_corr(score_core, score_broad, valid)
    parts = {
        "mean_broad_bce_loss": float(broad_bce.detach().cpu()),
        "mean_core_rank_loss": float(core_rank.detach().cpu()),
        "mean_soft_mrr_loss": float(soft_mrr.detach().cpu()),
        "mean_subset_loss": float(subset.detach().cpu()),
        "mean_core_distill_loss": float(core_distill.detach().cpu()),
        "score_core_broad_corr": float(corr.detach().cpu()),
        "subset_violation_rate": float(subset_violation.detach().cpu()),
        **target_stats,
    }
    return loss, parts


def _estimate_ez_negative_weight(dataset: Any, cap: float = 20.0) -> float:
    ez = 0.0
    nez = 0.0
    for item in dataset.patient_examples:
        labels_ez = np.asarray(item["labels_ez"], dtype=np.float32)
        mask = np.asarray(item["channel_mask"], dtype=bool)
        ez += float(((labels_ez == 1.0) & mask).sum())
        nez += float(((labels_ez == 0.0) & mask).sum())
    if ez <= 0.0:
        return 1.0
    return float(np.clip(nez / max(ez, 1.0), 1.0, float(cap)))


def _select_topk(scores: np.ndarray, k: int, valid_mask: np.ndarray, *, descending: bool = True) -> np.ndarray:
    pred = np.zeros(scores.shape[0], dtype=bool)
    valid_idx = np.where(valid_mask)[0]
    if valid_idx.size == 0:
        return pred
    k = max(1, min(int(k), int(valid_idx.size)))
    order = valid_idx[np.argsort(scores[valid_idx])]
    if descending:
        order = order[::-1]
    pred[order[:k]] = True
    return pred


def _reciprocal_rank(y_true: np.ndarray, scores: np.ndarray) -> float:
    if y_true.size == 0 or int((y_true == 1).sum()) == 0:
        return 0.0
    order = np.argsort(scores)[::-1]
    positive_ranks = np.where(y_true[order] == 1)[0]
    return float(1.0 / float(positive_ranks[0] + 1)) if positive_ranks.size else 0.0


def _recall_at_true_count(y_true: np.ndarray, scores: np.ndarray) -> float:
    true_count = int((y_true == 1).sum())
    if y_true.size == 0 or true_count <= 0:
        return 0.0
    pred = _select_topk(scores, true_count, np.ones_like(y_true, dtype=bool), descending=True)
    return float(((y_true == 1) & pred).sum() / max(true_count, 1))


def _safe_mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _quality_counts(labels: Sequence[str]) -> dict[str, int]:
    counts = {"good": 0, "review": 0, "poor": 0}
    for label in labels:
        normalized = normalize_quality_label(label)
        counts[normalized] = counts.get(normalized, 0) + 1
    return counts


def _quality_labels_for_samples(samples: Sequence[dict[str, Any]], field: str) -> list[str]:
    return [resolve_quality_label(sample, field) for sample in samples]


def _quality_audit_frames(
    run_records: Sequence[dict[str, Any]],
    patient_index: dict[str, dict[str, Any]],
    outer_splits: Sequence[dict[str, Any]],
    args: Any,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    field = str(getattr(args, "edf_quality_field", "")).strip()
    if not field:
        raise ValueError("--edf_quality_field is required when --use_edf_quality_weighting is enabled.")
    samples = flatten_window_samples(run_records)
    by_subject: dict[str, list[dict[str, Any]]] = {}
    for sample in samples:
        by_subject.setdefault(str(sample["subject_id"]), []).append(sample)

    center_rows: dict[str, dict[str, Any]] = {}
    patient_rows: list[dict[str, Any]] = []
    patients_only_review_no_good = 0
    for subject_id, patient_samples in sorted(by_subject.items()):
        labels = _quality_labels_for_samples(patient_samples, field)
        counts = _quality_counts(labels)
        patient_meta = patient_index.get(subject_id, {})
        center = str(patient_meta.get("center", patient_meta.get("source_center", patient_samples[0].get("center", "unknown")))).lower()
        only_review_no_good = counts["review"] > 0 and counts["good"] == 0
        if only_review_no_good:
            patients_only_review_no_good += 1
        patient_rows.append(
            {
                "subject_id": subject_id,
                "center": center,
                "n_records": int(len(patient_samples)),
                "good": counts["good"],
                "review": counts["review"],
                "poor": counts["poor"],
                "only_review_no_good": bool(only_review_no_good),
            }
        )
        row = center_rows.setdefault(center, {"center": center, "n_patients": 0, "n_records": 0, "good": 0, "review": 0, "poor": 0})
        row["n_patients"] += 1
        row["n_records"] += len(patient_samples)
        row["good"] += counts["good"]
        row["review"] += counts["review"]
        row["poor"] += counts["poor"]

    fold_rows: list[dict[str, Any]] = []
    base_seed = int(getattr(args, "random_seed", 42))
    for split in outer_splits:
        fold_idx = int(split["fold_idx"])
        fit_subjects, val_subjects = split_train_val_subjects(
            list(split["train_subjects"]),
            val_ratio=float(getattr(args, "val_ratio", 0.2)),
            random_seed=base_seed,
            fold_idx=fold_idx,
        )
        for split_name, subjects in (
            ("train", set(fit_subjects)),
            ("val", set(val_subjects)),
            ("test", set(split["test_subjects"])),
        ):
            split_labels: list[str] = []
            for subject_id in subjects:
                split_labels.extend(_quality_labels_for_samples(by_subject.get(str(subject_id), []), field))
            counts = _quality_counts(split_labels)
            fold_rows.append(
                {
                    "fold_idx": fold_idx,
                    "split": split_name,
                    "n_records": int(len(split_labels)),
                    "good": counts["good"],
                    "review": counts["review"],
                    "poor": counts["poor"],
                }
            )
    summary = {
        "edf_quality_field": field,
        "n_records": int(len(samples)),
        "n_patients": int(len(patient_rows)),
        "patients_only_review_no_good": int(patients_only_review_no_good),
    }
    return (
        pd.DataFrame(sorted(center_rows.values(), key=lambda row: str(row["center"]))),
        pd.DataFrame(patient_rows),
        pd.DataFrame(fold_rows),
        summary,
    )


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def _rank_robust_composite_score(summary: Dict[str, float]) -> float:
    return float(
        _finite_float(summary.get("patient_macro_f1", 0.0))
        + 0.5 * _finite_float(summary.get("patient_macro_ez_f1", 0.0))
        + 0.25 * _finite_float(summary.get("patient_macro_auprc_ez", 0.0))
        + 0.10 * _finite_float(summary.get("patient_macro_ez_mrr", 0.0))
        - 0.15 * _finite_float(summary.get("center_gap_f1", 0.0))
    )


def _add_center_robustness_fields(summary: Dict[str, float], enriched: Sequence[Dict[str, Any]]) -> None:
    center_values: Dict[str, List[float]] = {}
    center_nez_values: Dict[str, List[float]] = {}
    for record in enriched:
        center = str(record.get("center", "unknown")).strip().lower()
        value = _finite_float(record.get("patient_macro_f1", 0.0), default=float("nan"))
        if np.isfinite(value):
            center_values.setdefault(center, []).append(value)
        nez_value = _finite_float(record.get("patient_nez_f1", 0.0), default=float("nan"))
        if np.isfinite(nez_value):
            center_nez_values.setdefault(center, []).append(nez_value)

    center_means = {
        center: float(np.mean(values))
        for center, values in center_values.items()
        if values
    }
    if center_means:
        worst = float(min(center_means.values()))
        best = float(max(center_means.values()))
    else:
        worst = 0.0
        best = 0.0
    summary["worst_center_f1"] = worst
    summary["best_center_f1"] = best
    summary["center_gap_f1"] = float(best - worst) if len(center_means) >= 2 else 0.0

    for center in ("hup", "lzu", "multicenter", "pediatric"):
        summary[f"center_{center}_patient_macro_f1"] = float(center_means.get(center, 0.0))

    center_nez_means = {
        center: float(np.mean(values))
        for center, values in center_nez_values.items()
        if values
    }
    if center_nez_means:
        worst_nez = float(min(center_nez_means.values()))
        best_nez = float(max(center_nez_means.values()))
    else:
        worst_nez = 0.0
        best_nez = 0.0
    summary["worst_center_nez_f1"] = worst_nez
    summary["best_center_nez_f1"] = best_nez
    summary["center_gap_nez_f1"] = float(best_nez - worst_nez) if len(center_nez_means) >= 2 else 0.0
    for center in ("hup", "lzu", "multicenter", "pediatric"):
        summary[f"center_{center}_nez_f1"] = float(center_nez_means.get(center, 0.0))


def _summarize_prediction_records(
    records: Sequence[Dict[str, Any]],
    classification_threshold: float | None = None,
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    patient_metrics: Dict[str, List[float]] = {
        "accuracy": [],
        "balanced_accuracy": [],
        "macro_f1": [],
        "weighted_f1": [],
        "nez_precision": [],
        "nez_recall": [],
        "nez_f1": [],
        "ez_precision": [],
        "ez_recall": [],
        "ez_f1": [],
        "auroc_nez": [],
        "auprc_nez": [],
        "auroc_ez": [],
        "auprc_ez": [],
        "ez_recall_at_true_count": [],
        "ez_mrr": [],
        "top1_is_ez": [],
    }
    pooled_y_nez: List[np.ndarray] = []
    pooled_pred_nez: List[np.ndarray] = []
    pooled_score_nez: List[np.ndarray] = []
    pooled_score_ez: List[np.ndarray] = []
    enriched: List[Dict[str, Any]] = []

    for record in records:
        labels_nez_source = record["labels_nez"] if "labels_nez" in record else record["labels"]
        labels_nez = np.asarray(labels_nez_source, dtype=np.float32)
        labels_ez = np.asarray(record.get("labels_ez", 1.0 - labels_nez), dtype=np.float32)
        score_nez_source = record["score_nez"] if "score_nez" in record else record["scores"]
        score_nez = np.asarray(score_nez_source, dtype=np.float32)
        score_ez = np.asarray(record.get("score_ez", 1.0 - score_nez), dtype=np.float32)
        valid_mask = np.asarray(record["channel_mask"], dtype=bool)
        if not np.any(valid_mask):
            continue

        y_nez = labels_nez[valid_mask].astype(int)
        y_ez = labels_ez[valid_mask].astype(int)
        score_nez_valid = score_nez[valid_mask]
        score_ez_valid = score_ez[valid_mask]
        positive_label = str(record.get("positive_label", "nez")).lower()
        threshold = float(
            classification_threshold
            if classification_threshold is not None
            else record.get("classification_threshold", 0.5)
        )
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"classification threshold must be between 0 and 1, got {threshold}")
        if positive_label == "ez":
            pred_ez_mask = (score_ez >= threshold) & valid_mask
            pred_nez_mask = (~pred_ez_mask) & valid_mask
        else:
            pred_nez_mask = (score_nez >= threshold) & valid_mask
            pred_ez_mask = (~pred_nez_mask) & valid_mask
        pred_nez = pred_nez_mask[valid_mask].astype(int)

        precision, recall, f1, _ = precision_recall_fscore_support(y_nez, pred_nez, labels=[1, 0], zero_division=0)
        patient_accuracy = float(accuracy_score(y_nez, pred_nez))
        patient_balanced_accuracy = float(balanced_accuracy_score(y_nez, pred_nez))
        patient_macro_f1 = float(f1_score(y_nez, pred_nez, average="macro", zero_division=0))
        patient_weighted_f1 = float(f1_score(y_nez, pred_nez, average="weighted", zero_division=0))
        patient_metrics["accuracy"].append(patient_accuracy)
        patient_metrics["balanced_accuracy"].append(patient_balanced_accuracy)
        patient_metrics["macro_f1"].append(patient_macro_f1)
        patient_metrics["weighted_f1"].append(patient_weighted_f1)
        patient_metrics["nez_precision"].append(float(precision[0]))
        patient_metrics["nez_recall"].append(float(recall[0]))
        patient_metrics["nez_f1"].append(float(f1[0]))
        patient_metrics["ez_precision"].append(float(precision[1]))
        patient_metrics["ez_recall"].append(float(recall[1]))
        patient_metrics["ez_f1"].append(float(f1[1]))
        patient_metrics["ez_recall_at_true_count"].append(_recall_at_true_count(y_ez, score_ez_valid))
        patient_metrics["ez_mrr"].append(_reciprocal_rank(y_ez, score_ez_valid))
        top1_valid_idx = np.where(valid_mask)[0][int(np.argmax(score_ez_valid))]
        top1_is_ez = float(labels_ez[top1_valid_idx] == 1.0)
        patient_metrics["top1_is_ez"].append(top1_is_ez)

        if np.unique(y_nez).size > 1:
            patient_metrics["auroc_nez"].append(float(roc_auc_score(y_nez, score_nez_valid)))
            patient_metrics["auprc_nez"].append(float(average_precision_score(y_nez, score_nez_valid)))
            patient_metrics["auroc_ez"].append(float(roc_auc_score(y_ez, score_ez_valid)))
            patient_metrics["auprc_ez"].append(float(average_precision_score(y_ez, score_ez_valid)))

        pooled_y_nez.append(y_nez)
        pooled_pred_nez.append(pred_nez)
        pooled_score_nez.append(score_nez_valid)
        pooled_score_ez.append(score_ez_valid)

        channel_names = list(record["canonical_channels"])
        enriched_record = dict(record)
        center = str(record.get("center", "unknown"))
        enriched_record["center"] = center
        enriched_record["center_id"] = int(record.get("center_id", 4))
        enriched_record["classification_threshold"] = threshold
        enriched_record["predicted_mask"] = pred_ez_mask.astype(int).tolist()
        enriched_record["predicted_ez_mask"] = pred_ez_mask.astype(int).tolist()
        enriched_record["predicted_nez_mask"] = pred_nez_mask.astype(int).tolist()
        enriched_record["predicted_ez_channels"] = [channel_names[idx] for idx, flag in enumerate(pred_ez_mask) if flag]
        enriched_record["predicted_nez_channels"] = [channel_names[idx] for idx, flag in enumerate(pred_nez_mask) if flag]
        enriched_record["predicted_channels"] = enriched_record["predicted_ez_channels"]
        enriched_record["true_ez_channels"] = [channel_names[idx] for idx, flag in enumerate(labels_ez == 1.0) if flag]
        enriched_record["true_nez_channels"] = [channel_names[idx] for idx, flag in enumerate(labels_nez == 1.0) if flag]
        enriched_record["patient_accuracy"] = patient_accuracy
        enriched_record["patient_balanced_accuracy"] = patient_balanced_accuracy
        enriched_record["patient_macro_f1"] = patient_macro_f1
        enriched_record["patient_weighted_f1"] = patient_weighted_f1
        enriched_record["patient_nez_precision"] = float(precision[0])
        enriched_record["patient_nez_recall"] = float(recall[0])
        enriched_record["patient_nez_f1"] = float(f1[0])
        enriched_record["patient_ez_precision"] = float(precision[1])
        enriched_record["patient_ez_recall"] = float(recall[1])
        enriched_record["patient_ez_f1"] = float(f1[1])
        enriched_record["ez_recall_at_true_count"] = patient_metrics["ez_recall_at_true_count"][-1]
        enriched_record["ez_mrr"] = patient_metrics["ez_mrr"][-1]
        enriched_record["top1_is_ez"] = top1_is_ez
        enriched_record["true_nez_count"] = float((labels_nez[valid_mask] == 1.0).sum())
        enriched_record["true_ez_count"] = float((labels_ez[valid_mask] == 1.0).sum())
        enriched_record["n_channels"] = int(valid_mask.sum())
        enriched_record["ez_fraction"] = float(enriched_record["true_ez_count"] / max(int(valid_mask.sum()), 1))
        enriched_record["predicted_nez_count"] = float(pred_nez_mask.sum())
        enriched_record["predicted_ez_count"] = float(pred_ez_mask.sum())
        enriched.append(enriched_record)

    summary = {
        "classification_threshold": float(
            classification_threshold
            if classification_threshold is not None
            else np.mean([float(record.get("classification_threshold", 0.5)) for record in enriched])
            if enriched
            else 0.5
        ),
        "patient_macro_accuracy": _safe_mean(patient_metrics["accuracy"]),
        "patient_macro_balanced_accuracy": _safe_mean(patient_metrics["balanced_accuracy"]),
        "patient_macro_f1": _safe_mean(patient_metrics["macro_f1"]),
        "patient_weighted_f1": _safe_mean(patient_metrics["weighted_f1"]),
        "patient_macro_nez_precision": _safe_mean(patient_metrics["nez_precision"]),
        "patient_macro_nez_recall": _safe_mean(patient_metrics["nez_recall"]),
        "patient_macro_nez_f1": _safe_mean(patient_metrics["nez_f1"]),
        "patient_macro_ez_precision": _safe_mean(patient_metrics["ez_precision"]),
        "patient_macro_ez_recall": _safe_mean(patient_metrics["ez_recall"]),
        "patient_macro_ez_f1": _safe_mean(patient_metrics["ez_f1"]),
        "patient_macro_auroc_nez": _safe_mean(patient_metrics["auroc_nez"]),
        "patient_macro_auprc_nez": _safe_mean(patient_metrics["auprc_nez"]),
        "patient_macro_auroc_ez": _safe_mean(patient_metrics["auroc_ez"]),
        "patient_macro_auprc_ez": _safe_mean(patient_metrics["auprc_ez"]),
        "patient_macro_ez_recall_at_true_count": _safe_mean(patient_metrics["ez_recall_at_true_count"]),
        "patient_macro_ez_mrr": _safe_mean(patient_metrics["ez_mrr"]),
        "top1_is_ez_rate": _safe_mean(patient_metrics["top1_is_ez"]),
    }
    summary["macro_topk_recall"] = summary["patient_macro_ez_recall_at_true_count"]
    summary["ez_recall_at_true_count"] = summary["patient_macro_ez_recall_at_true_count"]
    summary["n_patient_rows"] = float(len(enriched))
    summary["n_unique_subjects"] = float(len({str(record.get("subject_id")) for record in enriched}))
    center_counts: Dict[str, int] = {}
    for record in enriched:
        center = str(record.get("center", "unknown"))
        center_counts[center] = center_counts.get(center, 0) + 1
    summary["centers_count_string"] = ",".join(f"{key}={center_counts[key]}" for key in sorted(center_counts))
    _add_center_robustness_fields(summary, enriched)
    for diag_key in (
        "negative_anchor_gate",
        "negative_anchor_gate_hup",
        "negative_anchor_gate_lzu",
        "negative_anchor_gate_multicenter",
        "negative_anchor_gate_pediatric",
        "negative_anchor_loss",
        "negative_anchor_mean_d_ez",
        "negative_anchor_mean_d_nez",
        "negative_anchor_separation",
        "negative_anchor_alpha",
        "negative_anchor_beta",
        "hard_topk_loss",
        "broad_ez_mil_loss",
        "broad_ez_mil_loss_weight",
        "broad_ez_core_frac",
        "broad_ez_min_fraction",
        "broad_ez_positive_bce_scale",
        "broad_ez_margin",
        "two_expert_lambda_mean",
        "two_expert_lambda_hup",
        "two_expert_lambda_lzu",
        "two_expert_lambda_multicenter",
        "two_expert_lambda_pediatric",
        "two_expert_pediatric_preserve_loss",
        "two_expert_gate_l2_loss",
        "two_expert_entropy_loss",
        "two_expert_aux_loss",
        "two_expert_anchor_ranking_loss",
        "two_expert_final_ranking_loss",
        "score_core_broad_corr",
        "subset_violation_rate",
        "mean_broad_bce_loss",
        "mean_core_rank_loss",
        "mean_soft_mrr_loss",
        "mean_subset_loss",
        "mean_core_distill_loss",
        "train_mean_core_rank_loss",
        "train_mean_soft_mrr_loss",
        "train_mean_core_distill_loss",
        "eval_target_match_rate",
        "eval_core_rank_loss_for_diagnostic",
        "eval_soft_mrr_loss_for_diagnostic",
        "eval_core_distill_loss_for_diagnostic",
        "target_match_rate",
        "matched_channels",
        "total_valid_channels",
        "sum_pseudo_core_q",
        "n_core_positive_channels",
        "mean_pseudo_core_q",
        "expert_s_physics_gate_mean",
        "expert_a_physics_gate_mean",
        "diffusion_score_gate",
        "mean_abs_graph_delta_lzu",
        "mean_abs_graph_delta_non_lzu",
    ):
        values = [float(record[diag_key]) for record in enriched if diag_key in record and np.isfinite(float(record[diag_key]))]
        if values:
            summary[diag_key] = _safe_mean(values)
    matched_values = [float(record["matched_channels"]) for record in enriched if "matched_channels" in record]
    total_values = [float(record["total_valid_channels"]) for record in enriched if "total_valid_channels" in record]
    if matched_values and total_values:
        matched_sum = float(np.sum(matched_values))
        total_sum = float(np.sum(total_values))
        summary["matched_channels"] = matched_sum
        summary["total_valid_channels"] = total_sum
        summary["target_match_rate"] = matched_sum / max(total_sum, 1.0)
        if "eval_target_match_rate" in summary:
            summary["eval_target_match_rate"] = summary["target_match_rate"]
    q_values = [float(record["sum_pseudo_core_q"]) for record in enriched if "sum_pseudo_core_q" in record]
    core_values = [float(record["n_core_positive_channels"]) for record in enriched if "n_core_positive_channels" in record]
    if q_values:
        summary["sum_pseudo_core_q"] = float(np.sum(q_values))
        summary["mean_pseudo_core_q"] = summary["sum_pseudo_core_q"] / max(float(summary.get("matched_channels", 0.0)), 1.0)
    if core_values:
        summary["n_core_positive_channels"] = float(np.sum(core_values))

    if pooled_y_nez:
        y_nez_all = np.concatenate(pooled_y_nez)
        pred_nez_all = np.concatenate(pooled_pred_nez)
        score_nez_all = np.concatenate(pooled_score_nez)
        score_ez_all = np.concatenate(pooled_score_ez)
        y_ez_all = 1 - y_nez_all
        summary.update(
            {
                "pooled_accuracy": float(accuracy_score(y_nez_all, pred_nez_all)),
                "pooled_balanced_accuracy": float(balanced_accuracy_score(y_nez_all, pred_nez_all)),
                "pooled_macro_f1": float(f1_score(y_nez_all, pred_nez_all, average="macro", zero_division=0)),
                "pooled_weighted_f1": float(f1_score(y_nez_all, pred_nez_all, average="weighted", zero_division=0)),
                "pooled_auroc_nez": float(roc_auc_score(y_nez_all, score_nez_all)) if np.unique(y_nez_all).size > 1 else 0.0,
                "pooled_auprc_nez": float(average_precision_score(y_nez_all, score_nez_all)) if np.unique(y_nez_all).size > 1 else 0.0,
                "pooled_auroc_ez": float(roc_auc_score(y_ez_all, score_ez_all)) if np.unique(y_ez_all).size > 1 else 0.0,
                "pooled_auprc_ez": float(average_precision_score(y_ez_all, score_ez_all)) if np.unique(y_ez_all).size > 1 else 0.0,
            }
        )
    else:
        summary.update(
            {
                "pooled_accuracy": 0.0,
                "pooled_balanced_accuracy": 0.0,
                "pooled_macro_f1": 0.0,
                "pooled_weighted_f1": 0.0,
                "pooled_auroc_nez": 0.0,
                "pooled_auprc_nez": 0.0,
                "pooled_auroc_ez": 0.0,
                "pooled_auprc_ez": 0.0,
            }
        )
    return summary, enriched


def _summary_score(summary: Dict[str, float], args: Any, *, val_loss: float = 0.0) -> float:
    metric = str(getattr(args, "early_stop_metric", "patient_macro_f1")).lower()
    if metric in {"loss", "val_loss"}:
        return -float(val_loss)
    if metric in {"rank_robust_composite", "a10_rank_robust_composite"}:
        return _rank_robust_composite_score(summary)
    if metric == "balanced_patient_auprc_hmean":
        nez = float(summary.get("patient_macro_auprc_nez", 0.0))
        ez = float(summary.get("patient_macro_auprc_ez", 0.0))
        return 2.0 * nez * ez / (nez + ez + 1e-8)
    aliases = {
        "macro_topk_recall": "ez_recall_at_true_count",
        "patient_topk_recall": "ez_recall_at_true_count",
        "patient_macro_f1": "patient_macro_f1",
        "patient_balanced_accuracy": "patient_macro_balanced_accuracy",
        "pooled_macro_f1": "pooled_macro_f1",
        "pooled_balanced_accuracy": "pooled_balanced_accuracy",
    }
    key = aliases.get(metric, metric)
    return float(summary.get(key, summary.get("patient_macro_f1", 0.0)))


def select_best_decision_rule(records: Sequence[Dict[str, Any]], args: Any | None = None):
    threshold = _select_validation_classification_threshold(
        records,
        positive_label=str(getattr(args, "positive_label", "nez")) if args is not None else "nez",
    )
    summary, enriched = _summarize_prediction_records(records, classification_threshold=threshold)
    return {"strategy": "validation_selected_threshold", "classification_threshold": threshold}, summary, enriched


def _select_validation_classification_threshold(
    records: Sequence[Dict[str, Any]],
    *,
    positive_label: str = "nez",
) -> float:
    """Select a fixed threshold from validation scores only; never use true channel counts."""
    del positive_label  # Records carry the score semantics explicitly.
    candidates = np.linspace(0.05, 0.95, 19, dtype=np.float32)
    best_threshold = 0.5
    best_score = -float("inf")
    for threshold in candidates:
        summary, _ = _summarize_prediction_records(records, classification_threshold=float(threshold))
        score = float(summary.get("patient_macro_f1", 0.0))
        if score > best_score + 1e-12 or (
            abs(score - best_score) <= 1e-12 and abs(float(threshold) - 0.5) < abs(best_threshold - 0.5)
        ):
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def _select_n6_validation_threshold(
    validation_records: Sequence[Dict[str, Any]],
) -> Tuple[float, Dict[str, float], List[Dict[str, Any]]]:
    """Select one global N6 threshold from final EMA validation scores only."""
    finite_scores: list[np.ndarray] = []
    for record in validation_records:
        score_nez = np.asarray(record.get("score_nez", record.get("scores", [])), dtype=np.float64)
        valid = np.asarray(record.get("channel_mask", np.ones(score_nez.shape, dtype=bool)), dtype=bool)
        selected = score_nez[valid & np.isfinite(score_nez)]
        if selected.size:
            finite_scores.append(selected)
    candidates = np.linspace(0.01, 0.99, 199, dtype=np.float64)
    if finite_scores:
        pooled = np.concatenate(finite_scores)
        quantiles = np.quantile(pooled, np.linspace(0.0, 1.0, min(199, max(2, pooled.size))))
        candidates = np.concatenate([candidates, quantiles])
    candidates = np.unique(np.clip(candidates[np.isfinite(candidates)], 0.01, 0.99))
    best_threshold = 0.5
    best_summary: Dict[str, float] = {}
    best_records: List[Dict[str, Any]] = []
    best_key = (-float("inf"), -float("inf"), -float("inf"), -float("inf"), -float("inf"))
    for threshold in candidates:
        summary, enriched = _summarize_prediction_records(
            validation_records,
            classification_threshold=float(threshold),
        )
        key = (
            float(summary.get("patient_macro_f1", 0.0)),
            float(summary.get("patient_macro_balanced_accuracy", 0.0)),
            min(
                float(summary.get("patient_macro_nez_f1", 0.0)),
                float(summary.get("patient_macro_ez_f1", 0.0)),
            ),
            float(summary.get("patient_macro_ez_f1", 0.0)),
            -abs(float(threshold) - 0.5),
        )
        if key > best_key:
            best_key = key
            best_threshold = float(threshold)
            best_summary = summary
            best_records = enriched
    return best_threshold, best_summary, best_records


class Exp_EZHybridLocalization:
    """BCE-only patient-level B0-Pruned backbone for EZ localization."""

    @staticmethod
    def _log(message: str) -> None:
        print(f"[B0-Pruned][Train] {message}", flush=True)

    def __init__(self, args: Any) -> None:
        self.args = args
        self.runtime = _resolve_runtime(args)
        self.device = _acquire_device(args)
        self.run_records, self.patient_index, self.outer_splits = data_provider(args)
        self.raw_alignment_store = None
        self.dual_view_audit: dict[str, Any] = {}
        self.use_n6_dual_view_ema = bool(getattr(args, "use_n6_dual_view_ema", False))
        if bool(getattr(self, "use_n6_dual_view_ema", False)):
            audit_path = Path(getattr(args, "dual_view_cache_audit_path", "") or (Path(getattr(args, "output_dir", "outputs")) / "dual_view_cache_audit.json"))
            self.raw_alignment_store = build_raw_alignment_store(self.run_records, args, output_audit_path=audit_path)
            self.raw_alignment_store.assert_formal_coverage(
                min_channel_match_rate=float(getattr(args, "raw_min_channel_match_rate", 0.95)),
                min_window_match_rate=float(getattr(args, "raw_min_window_match_rate", 0.90)),
                expected_patients=90 if bool(getattr(args, "enforce_fixed_all90_protocol", False)) else None,
            )
            self.dual_view_audit = dict(self.raw_alignment_store.audit)
        classification_threshold = float(getattr(args, "classification_threshold", 0.5))
        if not 0.0 <= classification_threshold <= 1.0:
            raise ValueError("classification_threshold must be between 0 and 1")
        self.train_subject_dropout_count = int(getattr(args, "train_subject_dropout_count", 0) or 0)
        self.train_subject_dropout_seed = int(getattr(args, "train_subject_dropout_seed", -1) or -1)
        self.train_subject_dropout_candidates: tuple[str, ...] = ()
        if self.train_subject_dropout_count < 0:
            raise ValueError("train_subject_dropout_count must be non-negative")
        dropout_file = str(getattr(args, "train_subject_dropout_file", "") or "").strip()
        if self.train_subject_dropout_count > 0:
            if not dropout_file:
                raise ValueError("--train_subject_dropout_file is required when --train_subject_dropout_count is positive")
            candidates = _read_train_dropout_candidates(dropout_file)
            missing_candidates = sorted(set(candidates) - set(map(str, self.patient_index)))
            if missing_candidates:
                raise ValueError(
                    "train_subject_dropout_file contains patients outside the active cohort: "
                    f"{missing_candidates}"
                )
            if self.train_subject_dropout_count > len(candidates):
                raise ValueError(
                    "train_subject_dropout_count exceeds the number of configured candidates: "
                    f"{self.train_subject_dropout_count} > {len(candidates)}"
                )
            self.train_subject_dropout_candidates = tuple(candidates)
        if bool(getattr(args, "enforce_fixed_all90_protocol", False)):
            from neuroez_c.protocol import assert_fixed_all90_protocol

            cache_feature_names = None
            if self.run_records:
                first_record = self.run_records[0]
                first_sample = first_record.get("sample", first_record)
                if isinstance(first_sample, dict):
                    cache_feature_names = first_sample.get("window_feature_names")
            audit = assert_fixed_all90_protocol(
                args,
                patient_index=self.patient_index,
                outer_splits=self.outer_splits,
                cache_feature_names=cache_feature_names,
            )
            output_dir = Path(getattr(args, "output_dir", "outputs"))
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(output_dir / "fixed_all90_protocol_audit.json", "w", encoding="utf-8") as fout:
                json.dump(audit, fout, indent=2, ensure_ascii=False, sort_keys=True)
        self.current_epoch = 0
        self.pediatric_preserve = bool(getattr(args, "two_expert_static_preserve_pediatric", False))
        self.gate_l2 = float(getattr(args, "two_expert_gate_l2", 0.0))
        self.entropy_reg = float(getattr(args, "two_expert_entropy_reg", 0.0))
        self.feature_sep = bool(getattr(args, "use_feature_separated_two_expert", False))
        self.aux_loss_weight = float(getattr(args, "two_expert_aux_loss_weight", 0.0))
        self.anchor_rank_weight = float(getattr(args, "two_expert_anchor_ranking_loss_weight", 0.0))
        self.final_rank_weight = float(getattr(args, "two_expert_final_ranking_loss_weight", 0.0))
        self.use_broad_ez_mil_loss = bool(getattr(args, "use_broad_ez_mil_loss", False))
        self.broad_ez_mil_loss_weight = float(getattr(args, "broad_ez_mil_loss_weight", 0.05))
        self.broad_ez_core_frac = float(getattr(args, "broad_ez_core_frac", 0.30))
        self.broad_ez_min_fraction = float(getattr(args, "broad_ez_min_fraction", 0.35))
        self.broad_ez_centers = _parse_broad_ez_centers(getattr(args, "broad_ez_centers", "lzu,pediatric"))
        self.broad_ez_positive_bce_scale = float(getattr(args, "broad_ez_positive_bce_scale", 0.5))
        self.broad_ez_margin = float(getattr(args, "broad_ez_margin", 0.05))
        self.broad_ez_hard_neg_multiplier = float(getattr(args, "broad_ez_hard_neg_multiplier", 1.0))
        self.use_a9v8_lcbo = bool(getattr(args, "use_a9v8_lcbo", False))
        self.a9v11_group_weights = torch.full((5,), 0.2, dtype=torch.float32)
        self.current_lcbo_target_lookup: dict[tuple[str, str], dict[str, float]] = {}
        self.current_lcbo_eval_lookup: dict[tuple[str, str], dict[str, float]] = {}
        if self.use_a9v8_lcbo and str(getattr(args, "positive_label", "nez")).lower() != "ez":
            raise ValueError("A9v8 LCBO requires --positive_label ez.")
        if self.use_n6_dual_view_ema:
            if str(getattr(args, "positive_label", "nez")).lower() != "nez":
                raise ValueError("N6 requires --positive_label nez.")
            if self.use_a9v8_lcbo or self.use_broad_ez_mil_loss:
                raise ValueError("N6 cannot be combined with A9v8 LCBO or broad-EZ MIL.")
            if self.train_subject_dropout_count != 0:
                raise ValueError("N6 fixed-All90 does not permit train-subject dropout.")
        if bool(getattr(args, "use_edf_quality_weighting", False)) and not str(getattr(args, "edf_quality_field", "")).strip():
            raise ValueError("A9v3-QC requires --edf_quality_field when --use_edf_quality_weighting is enabled.")
        self._log(
            f"Experiment ready with {len(self.run_records)} ictal records, "
            f"{len(self.patient_index)} patients, {len(self.outer_splits)} outer split(s), "
            f"runtime={self.runtime['name']}, device={self.device}."
        )

    def _sample_train_subject_dropout(self, fit_subjects: Sequence[str], fold_idx: int) -> tuple[list[str], list[str]]:
        """Remove only fit-fold patients; validation and held-out test stay fixed for comparable metrics."""
        fit_subjects = list(map(str, fit_subjects))
        if self.train_subject_dropout_count == 0:
            return fit_subjects, []
        eligible = sorted(set(fit_subjects) & set(self.train_subject_dropout_candidates))
        if len(eligible) < self.train_subject_dropout_count:
            raise RuntimeError(
                f"Fold {fold_idx} has only {len(eligible)} eligible fit patients for "
                f"train_subject_dropout_count={self.train_subject_dropout_count}; eligible={eligible}"
            )
        seed_base = self.train_subject_dropout_seed
        if seed_base < 0:
            seed_base = int(getattr(self.args, "random_seed", 42))
        rng = np.random.default_rng(seed_base + 1009 * int(fold_idx))
        dropped = sorted(rng.choice(np.asarray(eligible, dtype=object), size=self.train_subject_dropout_count, replace=False).tolist())
        dropped_set = set(dropped)
        retained = [subject_id for subject_id in fit_subjects if subject_id not in dropped_set]
        if not retained:
            raise RuntimeError(f"Fold {fold_idx} train-subject dropout removed every fit patient")
        return retained, dropped

    def _write_quality_audit_outputs(self, outer_splits: Sequence[dict[str, Any]]) -> dict[str, Any]:
        output_dir = Path(getattr(self.args, "output_dir", "outputs"))
        output_dir.mkdir(parents=True, exist_ok=True)
        by_center, by_patient, by_fold, summary = _quality_audit_frames(
            self.run_records,
            self.patient_index,
            outer_splits,
            self.args,
        )
        by_center.to_csv(output_dir / "quality_by_center_summary.csv", index=False)
        by_patient.to_csv(output_dir / "quality_by_patient_summary.csv", index=False)
        by_fold.to_csv(output_dir / "quality_by_fold_split_summary.csv", index=False)
        with (output_dir / "quality_audit_summary.json").open("w", encoding="utf-8") as fout:
            json.dump(summary, fout, indent=2, ensure_ascii=False, sort_keys=True)
        self._log(
            "A9v3-QC quality audit | "
            f"field={summary['edf_quality_field']} | records={summary['n_records']} | "
            f"patients_only_review_no_good={summary['patients_only_review_no_good']}"
        )
        return summary

    def _load_lcbo_targets_for_fold(self, fold_idx: int) -> None:
        target_dir = str(getattr(self.args, "latent_core_target_dir", "")).strip()
        if not target_dir:
            raise ValueError("A9v8 LCBO requires --latent_core_target_dir.")
        target_df = load_lcbo_fold_targets(target_dir, fold_id=int(fold_idx))
        self.current_lcbo_target_lookup = build_lcbo_target_lookup(target_df)
        all_oof_path = Path(target_dir) / "latent_core_targets_all_oof.csv"
        self.current_lcbo_eval_lookup = (
            build_lcbo_target_lookup(pd.read_csv(all_oof_path)) if all_oof_path.exists() else dict(self.current_lcbo_target_lookup)
        )
        self._log(
            f"Fold {fold_idx}: loaded LCBO train targets from {target_dir} | "
            f"rows={len(target_df)} | keyed_channels={len(self.current_lcbo_target_lookup)}"
        )

    def _make_loader(self, dataset: Any, *, shuffle: bool, batch_size: int) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=int(getattr(self.args, "num_workers", 0)),
            collate_fn=self.runtime["collate_fn"],
            pin_memory=self.device.type == "cuda",
        )

    def _prepare_quality_batch(self, batch: Dict[str, Any], *, split_name: str) -> Dict[str, Any]:
        if not bool(getattr(self.args, "use_edf_quality_weighting", False)):
            return batch
        base = batch.get("record_quality_weight")
        seizure_mask = batch.get("seizure_mask")
        if not torch.is_tensor(base) or not torch.is_tensor(seizure_mask):
            return batch
        mode = str(getattr(self.args, "quality_weight_record_aggregation", "weighted_mean_std")).lower()
        active = base.clone()
        labels = batch.get("record_quality_label", [])
        good_only = torch.zeros_like(active)
        for patient_idx, patient_labels in enumerate(labels):
            for seizure_idx, label in enumerate(patient_labels):
                if patient_idx < good_only.shape[0] and seizure_idx < good_only.shape[1]:
                    good_only[patient_idx, seizure_idx] = 1.0 if str(label).lower() == "good" else 0.0
        is_train = str(split_name).lower() == "train"
        curriculum_epochs = max(0, int(getattr(self.args, "quality_curriculum_epochs", 0)))
        if is_train and mode == "train_good_only_all_eval":
            active = good_only
        elif is_train and curriculum_epochs > 0 and int(getattr(self, "current_epoch", 0)) <= curriculum_epochs:
            active = good_only
        elif (not is_train) and mode in {"train_only_weighted_mean_std", "train_good_only_all_eval"}:
            active = seizure_mask.to(dtype=active.dtype)
        batch["record_quality_weight_active"] = active.to(device=base.device, dtype=base.dtype)
        return batch

    def _build_datasets(
        self,
        fit_subjects: Sequence[str],
        val_subjects: Sequence[str],
        test_subjects: Sequence[str],
    ) -> Tuple[Any, Any, Any, Any]:
        fit_samples = flatten_window_samples(self.run_records, subject_ids=fit_subjects)
        val_samples = flatten_window_samples(self.run_records, subject_ids=val_subjects)
        test_samples = flatten_window_samples(self.run_records, subject_ids=test_subjects)
        normalizer = self.runtime["fit_normalizer"](fit_samples, args=self.args)
        build_kwargs = {"raw_alignment_store": self.raw_alignment_store} if self.use_n6_dual_view_ema else {}
        fit_examples = self.runtime["build_patient_examples"](fit_samples, self.patient_index, normalizer=normalizer, subject_ids=fit_subjects, args=self.args, **build_kwargs)
        val_examples = self.runtime["build_patient_examples"](val_samples, self.patient_index, normalizer=normalizer, subject_ids=val_subjects, args=self.args, **build_kwargs)
        test_examples = self.runtime["build_patient_examples"](test_samples, self.patient_index, normalizer=normalizer, subject_ids=test_subjects, args=self.args, **build_kwargs)
        if not val_examples:
            val_examples = fit_examples
        dataset_cls = self.runtime["dataset_cls"]
        return dataset_cls(fit_examples), dataset_cls(val_examples), dataset_cls(test_examples), normalizer

    def _compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, Any],
        ez_negative_weight: torch.Tensor,
        *,
        split_name: str = "train",
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if bool(getattr(self, "use_n6_dual_view_ema", False)):
            teacher_outputs = outputs.pop("_n6_teacher_outputs", None)
            if teacher_outputs is None:
                teacher_outputs = outputs
            loss, parts = compute_n6_total_loss(outputs, teacher_outputs, batch, self.args, epoch=int(self.current_epoch))
            parts["bce"] = float(parts.get("n6_classification_loss", 0.0))
            return loss, parts
        use_lcbo = bool(getattr(self, "use_a9v8_lcbo", bool(getattr(self.args, "use_a9v8_lcbo", False))))
        if use_lcbo:
            is_train = str(split_name).lower() == "train"
            train_lookup = getattr(self, "current_lcbo_target_lookup", {})
            eval_lookup = getattr(self, "current_lcbo_eval_lookup", {}) or train_lookup
            target_lookup = train_lookup if is_train else eval_lookup
            if not target_lookup:
                raise RuntimeError(
                    "A9v8 LCBO target_match_rate below 0.99: target_match_rate=0.000000, "
                    "matched_channels=0, total_valid_channels=0; target lookup is empty. "
                    "Did fold target loading run?"
                )
            if is_train:
                loss, parts = compute_a9v8_lcbo_loss(outputs, batch, target_lookup, self.args)
                parts["bce"] = parts["mean_broad_bce_loss"]
                parts["train_mean_core_rank_loss"] = parts["mean_core_rank_loss"]
                parts["train_mean_soft_mrr_loss"] = parts["mean_soft_mrr_loss"]
                parts["train_mean_core_distill_loss"] = parts["mean_core_distill_loss"]
                if float(parts.get("target_match_rate", 0.0)) < 0.99:
                    raise RuntimeError(
                        "A9v8 LCBO train target_match_rate below 0.99: "
                        f"target_match_rate={parts.get('target_match_rate', 0.0):.6f}, "
                        f"matched_channels={parts.get('matched_channels', 0.0):.0f}, "
                        f"total_valid_channels={parts.get('total_valid_channels', 0.0):.0f}"
                    )
                return loss, parts

            broad_bce = _lcbo_patient_balanced_bce(
                outputs["logits_broad"],
                batch["labels_ez"].to(device=outputs["logits_broad"].device, dtype=outputs["logits_broad"].dtype),
                batch["channel_mask"].to(outputs["logits_broad"].device),
            )
            _, diag_parts = compute_a9v8_lcbo_loss(outputs, batch, target_lookup, self.args)
            parts = {
                "bce": float(broad_bce.detach().cpu()),
                "mean_broad_bce_loss": float(broad_bce.detach().cpu()),
                "eval_target_match_rate": float(diag_parts.get("target_match_rate", 0.0)),
                "eval_core_rank_loss_for_diagnostic": float(diag_parts.get("mean_core_rank_loss", 0.0)),
                "eval_soft_mrr_loss_for_diagnostic": float(diag_parts.get("mean_soft_mrr_loss", 0.0)),
                "eval_core_distill_loss_for_diagnostic": float(diag_parts.get("mean_core_distill_loss", 0.0)),
                "target_match_rate": float(diag_parts.get("target_match_rate", 0.0)),
                "matched_channels": float(diag_parts.get("matched_channels", 0.0)),
                "total_valid_channels": float(diag_parts.get("total_valid_channels", 0.0)),
                "sum_pseudo_core_q": float(diag_parts.get("sum_pseudo_core_q", 0.0)),
                "n_core_positive_channels": float(diag_parts.get("n_core_positive_channels", 0.0)),
                "mean_pseudo_core_q": float(diag_parts.get("mean_pseudo_core_q", 0.0)),
            }
            return broad_bce, parts
        if bool(getattr(self, "use_broad_ez_mil_loss", False)):
            bce, supervised_parts = _compute_broad_aware_supervised_loss(
                outputs["logits"],
                batch,
                self.args,
                ez_negative_weight,
            )
        else:
            bce, supervised_parts = _compute_supervised_loss(
                outputs["logits"],
                batch,
                self.args,
                ez_negative_weight,
                group_weights=getattr(self, "a9v11_group_weights", None),
                update_group_dro=str(split_name).lower() == "train",
            )
        loss = bce
        zero = bce.sum() * 0.0
        physics_next_state = outputs.get("physics_next_state_loss", zero)
        physics_source_sparse = outputs.get("physics_source_sparse_loss", zero)
        physics_velocity_l2 = outputs.get("physics_velocity_l2_loss", zero)
        diffusion_source_sparse = outputs.get("diffusion_source_sparse_loss", zero)
        diffusion_residual_l2 = outputs.get("diffusion_residual_l2_loss", zero)
        negative_anchor = outputs.get("negative_anchor_loss", zero)
        view_gate_l1 = outputs.get("view_gate_l1_loss", zero)
        physics_enabled = bool(getattr(self.args, "use_physics_dynamics", False))
        diffusion_enabled = bool(getattr(self.args, "use_diffusion_residual", False))
        negative_anchor_enabled = bool(getattr(self.args, "use_negative_anchor_head", False))
        view_gate_enabled = bool(getattr(self.args, "use_view_gated_fusion", False))
        if physics_enabled:
            loss = loss + float(getattr(self.args, "physics_loss_weight", 0.0)) * physics_next_state
            loss = loss + float(getattr(self.args, "physics_source_sparse_weight", 0.0)) * physics_source_sparse
            loss = loss + float(getattr(self.args, "physics_velocity_l2_weight", 0.0)) * physics_velocity_l2
        if diffusion_enabled:
            loss = loss + float(getattr(self.args, "diffusion_source_sparse_weight", 0.0)) * diffusion_source_sparse
            loss = loss + float(getattr(self.args, "diffusion_residual_l2_weight", 0.0)) * diffusion_residual_l2
        if negative_anchor_enabled:
            loss = loss + float(getattr(self.args, "negative_anchor_loss_weight", 0.03)) * negative_anchor
        if view_gate_enabled:
            loss = loss + float(getattr(self.args, "view_gate_l1", 0.0)) * view_gate_l1
        ranking_enabled = bool(getattr(self.args, "use_ez_ranking_loss", False))
        ranking_weight = float(getattr(self.args, "ez_ranking_loss_weight", 0.0))
        ranking_margin = float(getattr(self.args, "ez_ranking_margin", 0.10))
        if ranking_enabled and ranking_weight > 0.0:
            quality_patient_weights = _quality_patient_weights_vector(batch, self.args, outputs["logits"])
            ez_ranking = _ez_pairwise_ranking_loss(
                outputs["logits"],
                batch["labels_ez"],
                batch["channel_mask"],
                margin=ranking_margin,
                positive_label=str(getattr(self.args, "positive_label", "nez")),
                patient_weights=quality_patient_weights,
            )
            loss = loss + ranking_weight * ez_ranking
        else:
            ez_ranking = zero
        hard_topk_enabled = bool(getattr(self.args, "use_hard_topk_loss", False))
        hard_topk_weight = float(getattr(self.args, "hard_topk_loss_weight", 0.02))
        hard_topk_margin = float(getattr(self.args, "hard_topk_margin", 0.05))
        hard_topk_multiplier = float(getattr(self.args, "hard_topk_multiplier", 1.0))
        if hard_topk_enabled and hard_topk_weight > 0.0:
            hard_topk = _hard_topk_retrieval_loss(
                outputs["logits"],
                batch["labels_ez"],
                batch["channel_mask"],
                positive_label=str(getattr(self.args, "positive_label", "nez")),
                margin=hard_topk_margin,
                topk_multiplier=hard_topk_multiplier,
            )
            loss = loss + hard_topk_weight * hard_topk
        else:
            hard_topk = zero
        if bool(getattr(self, "use_broad_ez_mil_loss", False)) and float(getattr(self, "broad_ez_mil_loss_weight", 0.0)) > 0.0:
            broad_mil = _broad_ez_mil_ranking_loss(
                outputs["logits"],
                batch["labels_ez"],
                batch["channel_mask"],
                batch.get("center_id"),
                batch.get("ez_fraction"),
                positive_label=str(getattr(self.args, "positive_label", "ez")),
                broad_centers=self.broad_ez_centers,
                min_ez_fraction=self.broad_ez_min_fraction,
                core_frac=self.broad_ez_core_frac,
                margin=self.broad_ez_margin,
                hard_neg_multiplier=self.broad_ez_hard_neg_multiplier,
            )
            loss = loss + float(getattr(self, "broad_ez_mil_loss_weight", 0.0)) * broad_mil
        else:
            broad_mil = zero
        parts = {"bce": float(bce.detach().cpu())}
        parts.update({key: float(value) for key, value in supervised_parts.items() if isinstance(value, (int, float))})
        n_invalid = outputs.get("n_channels_all_windows_invalid")
        if torch.is_tensor(n_invalid):
            parts["n_channels_all_windows_invalid"] = float(n_invalid.detach().cpu())
        if physics_enabled or "physics_next_state_loss" in outputs:
            parts.update(
                {
                    "physics_next_state": float(physics_next_state.detach().cpu()),
                    "physics_source_sparse": float(physics_source_sparse.detach().cpu()),
                    "physics_velocity_l2": float(physics_velocity_l2.detach().cpu()),
                    "physics_gate_mean": float(outputs.get("physics_gate_mean", zero).detach().cpu()),
                }
            )
        if diffusion_enabled or "diffusion_source_sparse_loss" in outputs:
            parts.update(
                {
                    "diffusion_source_sparse": float(diffusion_source_sparse.detach().cpu()),
                    "diffusion_residual_l2": float(diffusion_residual_l2.detach().cpu()),
                    "diffusion_gate_mean": float(outputs.get("diffusion_gate_mean", zero).detach().cpu()),
                    "diffusion_score_gate": float(outputs.get("diffusion_score_gate", zero).detach().cpu()),
                    "mean_abs_graph_delta_lzu": float(outputs.get("mean_abs_graph_delta_lzu", zero).detach().cpu()),
                    "mean_abs_graph_delta_non_lzu": float(outputs.get("mean_abs_graph_delta_non_lzu", zero).detach().cpu()),
                    "diffusion_beta": float(outputs.get("diffusion_beta", zero).detach().cpu()),
                    "diffusion_graph_density": float(outputs.get("diffusion_graph_density", zero).detach().cpu()),
                }
            )
        if negative_anchor_enabled or "negative_anchor_loss" in outputs:
            parts.update(
                {
                    "negative_anchor_loss": float(negative_anchor.detach().cpu()),
                    "negative_anchor_gate": float(outputs.get("negative_anchor_gate", zero).detach().cpu()),
                    "negative_anchor_gate_hup": float(outputs.get("negative_anchor_gate_hup", zero).detach().cpu()),
                    "negative_anchor_gate_lzu": float(outputs.get("negative_anchor_gate_lzu", zero).detach().cpu()),
                    "negative_anchor_gate_multicenter": float(outputs.get("negative_anchor_gate_multicenter", zero).detach().cpu()),
                    "negative_anchor_gate_pediatric": float(outputs.get("negative_anchor_gate_pediatric", zero).detach().cpu()),
                    "negative_anchor_mean_d_ez": float(outputs.get("negative_anchor_mean_d_ez", zero).detach().cpu()),
                    "negative_anchor_mean_d_nez": float(outputs.get("negative_anchor_mean_d_nez", zero).detach().cpu()),
                    "negative_anchor_separation": float(outputs.get("negative_anchor_separation", zero).detach().cpu()),
                    "negative_anchor_alpha": float(outputs.get("negative_anchor_alpha", zero).detach().cpu()),
                    "negative_anchor_beta": float(outputs.get("negative_anchor_beta", zero).detach().cpu()),
                }
            )
        if view_gate_enabled or "view_gate_l1_loss" in outputs:
            parts["view_gate_l1_loss"] = float(view_gate_l1.detach().cpu())
            for key, value in outputs.items():
                if key.startswith("view_gate_") and key != "view_gate_l1_loss" and torch.is_tensor(value) and value.ndim == 0:
                    parts[key] = float(value.detach().cpu())
        if ranking_enabled:
            parts["ez_ranking_loss"] = float(ez_ranking.detach().cpu())
            parts["ez_ranking_weight"] = ranking_weight
            parts["ez_ranking_margin"] = ranking_margin
        if hard_topk_enabled:
            parts["hard_topk_loss"] = float(hard_topk.detach().cpu())
            parts["hard_topk_loss_weight"] = hard_topk_weight
            parts["hard_topk_margin"] = hard_topk_margin
            parts["hard_topk_multiplier"] = hard_topk_multiplier
        if bool(getattr(self, "use_broad_ez_mil_loss", False)):
            parts["broad_ez_mil_loss"] = float(broad_mil.detach().cpu())
            parts["broad_ez_mil_loss_weight"] = self.broad_ez_mil_loss_weight
            parts["broad_ez_core_frac"] = self.broad_ez_core_frac
            parts["broad_ez_min_fraction"] = self.broad_ez_min_fraction
            parts["broad_ez_positive_bce_scale"] = self.broad_ez_positive_bce_scale
            parts["broad_ez_margin"] = self.broad_ez_margin
        two_expert = bool(getattr(self.args, "use_two_expert_router", False))
        if two_expert:
            preserve_loss = outputs.get("two_expert_pediatric_preserve_loss", zero)
            gate_l2_loss = outputs.get("two_expert_gate_l2_loss", zero)
            entropy_loss = outputs.get("two_expert_entropy_loss", zero)
            if self.pediatric_preserve:
                loss = loss + float(getattr(self.args, "two_expert_pediatric_preserve_weight", 0.05)) * preserve_loss
            if self.gate_l2 > 0.0:
                loss = loss + self.gate_l2 * gate_l2_loss
            if self.entropy_reg > 0.0:
                loss = loss + self.entropy_reg * entropy_loss
            parts.update({
                "two_expert_pediatric_preserve_loss": float(preserve_loss.detach().cpu()) if torch.is_tensor(preserve_loss) else 0.0,
                "two_expert_gate_l2_loss": float(gate_l2_loss.detach().cpu()) if torch.is_tensor(gate_l2_loss) else 0.0,
                "two_expert_entropy_loss": float(entropy_loss.detach().cpu()) if torch.is_tensor(entropy_loss) else 0.0,
            })
            for lam_key in ("two_expert_lambda_mean", "two_expert_lambda_hup", "two_expert_lambda_lzu",
                            "two_expert_lambda_multicenter", "two_expert_lambda_pediatric"):
                val = outputs.get(lam_key)
                if val is not None and torch.is_tensor(val) and val.ndim == 0:
                    parts[lam_key] = float(val.detach().cpu())
            for prefix in ("expert_s_", "expert_a_"):
                for key, value in outputs.items():
                    if key.startswith(prefix) and torch.is_tensor(value) and value.ndim == 0:
                        parts[key] = float(value.detach().cpu())
        # ---- A9v6 feature-separated aux loss + ranking ----
        if bool(getattr(self, "feature_sep", False)):
            logits_static = outputs.get("logits_static")
            logits_anchor = outputs.get("logits_anchor")
            aux_loss_val = zero
            if self.aux_loss_weight > 0.0 and logits_static is not None and logits_anchor is not None:
                bce_s, _ = _compute_supervised_loss(logits_static, batch, self.args, ez_negative_weight)
                bce_a, _ = _compute_supervised_loss(logits_anchor, batch, self.args, ez_negative_weight)
                aux_loss_val = 0.5 * (bce_s + bce_a)
                loss = loss + self.aux_loss_weight * aux_loss_val
                parts["two_expert_aux_loss"] = float(aux_loss_val.detach().cpu())
                parts["two_expert_static_bce"] = float(bce_s.detach().cpu())
                parts["two_expert_anchor_bce"] = float(bce_a.detach().cpu())
            anchor_rank_loss = zero
            if self.anchor_rank_weight > 0.0 and logits_anchor is not None:
                anchor_rank_loss = _ez_pairwise_ranking_loss(
                    logits_anchor, batch["labels_ez"], batch["channel_mask"],
                    margin=0.05, positive_label=str(getattr(self.args, "positive_label", "ez")),
                )
                loss = loss + self.anchor_rank_weight * anchor_rank_loss
                parts["two_expert_anchor_ranking_loss"] = float(anchor_rank_loss.detach().cpu())
            final_rank_loss = zero
            if self.final_rank_weight > 0.0:
                final_rank_loss = _ez_pairwise_ranking_loss(
                    outputs["logits"], batch["labels_ez"], batch["channel_mask"],
                    margin=0.05, positive_label=str(getattr(self.args, "positive_label", "ez")),
                )
                loss = loss + self.final_rank_weight * final_rank_loss
                parts["two_expert_final_ranking_loss"] = float(final_rank_loss.detach().cpu())
        return loss, parts

    def _dry_initialize_lazy_layers(self, model: torch.nn.Module, loader: DataLoader) -> None:
        model.eval()
        with torch.no_grad():
            for batch in loader:
                _ = model(_move_tensors_to_device(batch, self.device))
                return

    def _train_one_epoch(
        self,
        model: torch.nn.Module,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        ez_negative_weight: torch.Tensor,
        ema_teacher: torch.nn.Module | None = None,
    ) -> Dict[str, float]:
        model.train()
        losses: List[float] = []
        part_losses: Dict[str, List[float]] = {}
        for batch in loader:
            batch = _move_tensors_to_device(batch, self.device)
            batch = self._prepare_quality_batch(batch, split_name="train")
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch)
            if self.use_n6_dual_view_ema:
                if ema_teacher is None:
                    raise RuntimeError("N6 training requires an EMA teacher.")
                with torch.no_grad():
                    outputs["_n6_teacher_outputs"] = ema_teacher(batch)
            if bool(getattr(self, "use_a9v8_lcbo", False)) and bool(getattr(self.args, "use_teacher_anchor_eval", False)) and bool(getattr(self.args, "teacher_anchor_apply_to_train_loss", False)):
                outputs = apply_teacher_anchor_eval_scores(
                    outputs, batch,
                    getattr(self, "current_lcbo_target_lookup", {}),
                    self.args,
                )
            loss, parts = self._compute_loss(outputs, batch, ez_negative_weight, split_name="train")
            loss.backward()
            grad_clip = float(getattr(self.args, "grad_clip", 1.0))
            if grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            if self.use_n6_dual_view_ema and ema_teacher is not None:
                _update_ema_teacher(ema_teacher, model, float(getattr(self.args, "n6_ema_decay", 0.995)))
            losses.append(float(loss.detach().cpu()))
            for key, value in parts.items():
                part_losses.setdefault(key, []).append(float(value))
        metrics = {"loss": _safe_mean(losses)}
        metrics.update({key: _safe_mean(values) for key, values in part_losses.items()})
        if bool(getattr(self, "use_a9v8_lcbo", False)) and "matched_channels" in part_losses:
            matched = float(np.sum(part_losses.get("matched_channels", [])))
            total = float(np.sum(part_losses.get("total_valid_channels", [])))
            sum_q = float(np.sum(part_losses.get("sum_pseudo_core_q", [])))
            n_core = float(np.sum(part_losses.get("n_core_positive_channels", [])))
            metrics.update(
                {
                    "target_match_rate": matched / max(total, 1.0),
                    "matched_channels": matched,
                    "total_valid_channels": total,
                    "sum_pseudo_core_q": sum_q,
                    "n_core_positive_channels": n_core,
                    "mean_pseudo_core_q": sum_q / max(matched, 1.0),
                }
            )
        return metrics

    def _evaluate(
        self,
        model: torch.nn.Module,
        loader: DataLoader,
        ez_negative_weight: torch.Tensor,
        *,
        split_name: str = "eval",
    ) -> Tuple[float, Dict[str, float], List[Dict[str, Any]]]:
        model.eval()
        losses: List[float] = []
        part_losses: Dict[str, List[float]] = {}
        records: List[Dict[str, Any]] = []
        with torch.no_grad():
            for batch in loader:
                batch_device = _move_tensors_to_device(batch, self.device)
                batch_device = self._prepare_quality_batch(batch_device, split_name=split_name)
                raw_outputs = model(batch_device)
                # Compute loss on raw (unanchored) outputs for clean LCBO diagnostic
                loss, loss_parts = self._compute_loss(raw_outputs, batch_device, ez_negative_weight, split_name=split_name)
                losses.append(float(loss.detach().cpu()))
                for key, value in loss_parts.items():
                    if isinstance(value, (int, float)):
                        part_losses.setdefault(key, []).append(float(value))
                # Apply teacher anchor only for prediction outputs (scores + records)
                if bool(getattr(self, "use_a9v8_lcbo", False)) and bool(getattr(self.args, "use_teacher_anchor_eval", False)):
                    lookup = (
                        getattr(self, "current_lcbo_target_lookup", {})
                        if str(split_name).lower() == "train"
                        else (getattr(self, "current_lcbo_eval_lookup", {}) or getattr(self, "current_lcbo_target_lookup", {}))
                    )
                    pred_outputs = dict(raw_outputs)
                    pred_outputs = apply_teacher_anchor_eval_scores(pred_outputs, batch_device, lookup, self.args)
                else:
                    pred_outputs = raw_outputs
                score_nez = pred_outputs["score_nez"].detach().cpu().numpy()
                score_ez = pred_outputs["score_ez"].detach().cpu().numpy()
                optional_channel_arrays: Dict[str, np.ndarray] = {}
                for key in (
                    "score_ez_base", "score_ez_anchor", "score_ez_final", "score_ez_static",
                    "score_ez_anchor_final", "score_core", "score_broad", "score_clinical", "score_eval",
                    "score_nez_feature", "score_nez_raw", "score_nez_fused", "fusion_gate_feature",
                    "score_eval_original", "logits_core", "logits_broad", "logits_eval",
                    "logits_eval_original",
                    "negative_anchor_distance", "graph_delta_logits",
                ):
                    if key in pred_outputs and torch.is_tensor(pred_outputs[key]) and pred_outputs[key].shape == pred_outputs["score_ez"].shape:
                        optional_channel_arrays[key] = pred_outputs[key].detach().cpu().numpy()
                patient_embedding_np: np.ndarray | None = None
                if "patient_channel_embedding" in pred_outputs and torch.is_tensor(pred_outputs["patient_channel_embedding"]):
                    patient_embedding_np = pred_outputs["patient_channel_embedding"].detach().cpu().numpy()
                seizure_embedding_np: np.ndarray | None = None
                if "seizure_channel_embedding" in pred_outputs and torch.is_tensor(pred_outputs["seizure_channel_embedding"]):
                    seizure_embedding_np = pred_outputs["seizure_channel_embedding"].detach().cpu().numpy()
                if bool(getattr(self, "use_a9v8_lcbo", False)) and getattr(self, "current_lcbo_eval_lookup", None):
                    _, q_diag, teacher_diag, target_diag_mask = _lcbo_target_tensors(
                        batch_device,
                        self.current_lcbo_eval_lookup,
                        pred_outputs["score_ez"].device,
                        pred_outputs["score_ez"].dtype,
                    )
                    q_np = q_diag.detach().cpu().numpy().astype(np.float32)
                    teacher_np = teacher_diag.detach().cpu().numpy().astype(np.float32)
                    target_np = target_diag_mask.detach().cpu().numpy().astype(bool)
                    q_np = np.where(target_np, q_np, np.nan).astype(np.float32)
                    teacher_np = np.where(target_np, teacher_np, np.nan).astype(np.float32)
                    optional_channel_arrays["pseudo_core_q"] = q_np
                    optional_channel_arrays["a9v3_oof_score"] = teacher_np
                # per-patient gate tensor (A9v4 center-gate)
                gate_per_patient_np: np.ndarray | None = None
                if "negative_anchor_gate_per_patient" in pred_outputs and torch.is_tensor(pred_outputs["negative_anchor_gate_per_patient"]):
                    gate_per_patient_np = pred_outputs["negative_anchor_gate_per_patient"].detach().cpu().numpy()
                optional_scalars: Dict[str, float] = {}
                for key in (
                    "negative_anchor_gate",
                    "negative_anchor_gate_hup",
                    "negative_anchor_gate_lzu",
                    "negative_anchor_gate_multicenter",
                    "negative_anchor_gate_pediatric",
                    "negative_anchor_loss",
                    "negative_anchor_mean_d_ez",
                    "negative_anchor_mean_d_nez",
                    "negative_anchor_separation",
                    "negative_anchor_alpha",
                    "negative_anchor_beta",
                    "two_expert_lambda_mean",
                    "two_expert_lambda_hup",
                    "two_expert_lambda_lzu",
                    "two_expert_lambda_multicenter",
                    "two_expert_lambda_pediatric",
                    "expert_s_physics_gate_mean",
                    "expert_a_physics_gate_mean",
                    "two_expert_static_feature_dim",
                    "two_expert_anchor_feature_dim",
                    "diffusion_score_gate",
                    "mean_abs_graph_delta_lzu",
                    "mean_abs_graph_delta_non_lzu",
                    "teacher_anchor_alpha",
                    "teacher_anchor_beta",
                ):
                    if key in pred_outputs and torch.is_tensor(pred_outputs[key]) and pred_outputs[key].ndim == 0:
                        optional_scalars[key] = float(pred_outputs[key].detach().cpu())
                # carry hard_topk diagnostics from loss_parts into heldout summary
                for htk_key in ("hard_topk_loss", "hard_topk_loss_weight", "hard_topk_margin", "hard_topk_multiplier"):
                    if htk_key in loss_parts:
                        optional_scalars[htk_key] = float(loss_parts[htk_key])
                for broad_key in (
                    "broad_ez_mil_loss",
                    "broad_ez_mil_loss_weight",
                    "broad_ez_core_frac",
                    "broad_ez_min_fraction",
                    "broad_ez_positive_bce_scale",
                    "broad_ez_margin",
                ):
                    if broad_key in loss_parts:
                        optional_scalars[broad_key] = float(loss_parts[broad_key])
                # carry two_expert diagnostics from loss_parts
                for tex_key in ("two_expert_pediatric_preserve_loss", "two_expert_gate_l2_loss",
                                "two_expert_entropy_loss", "two_expert_lambda_mean",
                                "two_expert_lambda_hup", "two_expert_lambda_lzu",
                                "two_expert_lambda_multicenter", "two_expert_lambda_pediatric"):
                    if tex_key in loss_parts:
                        optional_scalars[tex_key] = float(loss_parts[tex_key])
                # carry A9v6 feature-separated aux/ranking loss diagnostics
                for fsep_key in ("two_expert_aux_loss", "two_expert_static_bce", "two_expert_anchor_bce",
                                 "two_expert_anchor_ranking_loss", "two_expert_final_ranking_loss"):
                    if fsep_key in loss_parts:
                        optional_scalars[fsep_key] = float(loss_parts[fsep_key])
                for lcbo_key in (
                    "score_core_broad_corr",
                    "subset_violation_rate",
                    "mean_broad_bce_loss",
                    "mean_core_rank_loss",
                    "mean_soft_mrr_loss",
                    "mean_subset_loss",
                    "mean_core_distill_loss",
                    "train_mean_core_rank_loss",
                    "train_mean_soft_mrr_loss",
                    "train_mean_core_distill_loss",
                    "eval_target_match_rate",
                    "eval_core_rank_loss_for_diagnostic",
                    "eval_soft_mrr_loss_for_diagnostic",
                                "eval_core_distill_loss_for_diagnostic",
                                "target_match_rate",
                                "matched_channels",
                                "total_valid_channels",
                                "sum_pseudo_core_q",
                                "n_core_positive_channels",
                                "mean_pseudo_core_q",
                ):
                    if lcbo_key in loss_parts:
                        optional_scalars[lcbo_key] = float(loss_parts[lcbo_key])
                # per-patient two_expert lambda
                lambda_per_patient_np: np.ndarray | None = None
                if "two_expert_lambda_patient" in pred_outputs and torch.is_tensor(pred_outputs["two_expert_lambda_patient"]):
                    lambda_per_patient_np = pred_outputs["two_expert_lambda_patient"].detach().cpu().numpy()
                elif "two_expert_lambda_per_patient" in pred_outputs and torch.is_tensor(pred_outputs["two_expert_lambda_per_patient"]):
                    lambda_per_patient_np = pred_outputs["two_expert_lambda_per_patient"].detach().cpu().numpy()
                labels = batch["labels"].cpu().numpy()
                labels_nez = batch["labels_nez"].cpu().numpy()
                labels_ez = batch["labels_ez"].cpu().numpy()
                channel_mask = batch["channel_mask"].cpu().numpy().astype(bool)
                raw_available = batch.get("raw_channel_available")
                raw_available_np = raw_available.cpu().numpy().astype(bool) if torch.is_tensor(raw_available) else None
                _center_name_map = {0: "hup", 1: "lzu", 2: "multicenter", 3: "pediatric"}
                for idx, subject_id in enumerate(batch["subject_id"]):
                    c = len(batch["canonical_channels"][idx])
                    record = {
                        "subject_id": str(subject_id),
                        "center": str(batch.get("center", ["unknown"] * len(batch["subject_id"]))[idx]),
                        "center_id": int(batch.get("center_id", torch.full((len(batch["subject_id"]),), 4))[idx].cpu().item()),
                        "ez_fraction": float(batch.get("ez_fraction", torch.zeros(len(batch["subject_id"])))[idx].cpu().item()),
                        "valid_channel_count": int(batch.get("valid_channel_count", torch.zeros(len(batch["subject_id"]), dtype=torch.long))[idx].cpu().item()),
                        "ez_channel_count": int(batch.get("ez_channel_count", torch.zeros(len(batch["subject_id"]), dtype=torch.long))[idx].cpu().item()),
                        "canonical_channels": list(batch["canonical_channels"][idx]),
                        "channel_meta": list(batch.get("channel_meta", [[]])[idx]),
                        "labels": labels[idx, :c].astype(np.float32),
                        "labels_nez": labels_nez[idx, :c].astype(np.float32),
                        "labels_ez": labels_ez[idx, :c].astype(np.float32),
                        "scores": score_nez[idx, :c].astype(np.float32),
                        "score_nez": score_nez[idx, :c].astype(np.float32),
                        "score_ez": score_ez[idx, :c].astype(np.float32),
                        "positive_label": str(getattr(self.args, "positive_label", "nez")),
                        "classification_threshold": float(getattr(self.args, "classification_threshold", 0.5)),
                        "channel_mask": channel_mask[idx, :c],
                        "run_ids": list(batch.get("run_ids", [[]])[idx]),
                        "sample_ids": list(batch.get("sample_ids", [[]])[idx]),
                        "record_quality_labels": list(batch.get("record_quality_label", [[]])[idx]),
                        "record_quality_weights": (
                            batch.get("record_quality_weight", torch.ones((len(batch["subject_id"]), 0)))[idx].detach().cpu().numpy().astype(np.float32).tolist()
                            if torch.is_tensor(batch.get("record_quality_weight"))
                            else []
                        ),
                        "record_quality_weights_active": (
                            batch.get("record_quality_weight_active", torch.ones((len(batch["subject_id"]), 0)))[idx].detach().cpu().numpy().astype(np.float32).tolist()
                            if torch.is_tensor(batch.get("record_quality_weight_active"))
                            else []
                        ),
                        **{key: value[idx, :c].astype(np.float32) for key, value in optional_channel_arrays.items()},
                        **optional_scalars,
                    }
                    if patient_embedding_np is not None and idx < patient_embedding_np.shape[0]:
                        record["patient_channel_embedding"] = patient_embedding_np[idx, :c].astype(np.float32)
                    if seizure_embedding_np is not None and idx < seizure_embedding_np.shape[0]:
                        record["seizure_channel_embedding"] = seizure_embedding_np[idx, :, :c].astype(np.float32)
                    if raw_available_np is not None:
                        record["raw_channel_available"] = raw_available_np[idx, :c]
                    # per-patient anchor gate: overrides batch-aggregated center fields
                    if gate_per_patient_np is not None and idx < len(gate_per_patient_np):
                        gate_val = float(gate_per_patient_np[idx])
                        record["negative_anchor_gate_patient"] = gate_val
                        cid = record["center_id"]
                        if cid in _center_name_map:
                            record[f"negative_anchor_gate_{_center_name_map[cid]}"] = gate_val
                    # per-patient two-expert lambda
                    if lambda_per_patient_np is not None and idx < len(lambda_per_patient_np):
                        lam_val = float(lambda_per_patient_np[idx])
                        record["two_expert_lambda_patient"] = lam_val
                        cid = record["center_id"]
                        if cid in _center_name_map:
                            record[f"two_expert_lambda_{_center_name_map[cid]}"] = lam_val
                    records.append(record)
        summary, enriched = _summarize_prediction_records(records)
        if bool(getattr(self, "use_a9v8_lcbo", False)) and "matched_channels" in part_losses:
            matched = float(np.sum(part_losses.get("matched_channels", [])))
            total = float(np.sum(part_losses.get("total_valid_channels", [])))
            sum_q = float(np.sum(part_losses.get("sum_pseudo_core_q", [])))
            n_core = float(np.sum(part_losses.get("n_core_positive_channels", [])))
            summary.update(
                {
                    "target_match_rate": matched / max(total, 1.0),
                    "eval_target_match_rate": matched / max(total, 1.0),
                    "matched_channels": matched,
                    "total_valid_channels": total,
                    "sum_pseudo_core_q": sum_q,
                    "n_core_positive_channels": n_core,
                    "mean_pseudo_core_q": sum_q / max(matched, 1.0),
                    "eval_core_rank_loss_for_diagnostic": _safe_mean(part_losses.get("eval_core_rank_loss_for_diagnostic", [])),
                    "eval_soft_mrr_loss_for_diagnostic": _safe_mean(part_losses.get("eval_soft_mrr_loss_for_diagnostic", [])),
                    "eval_core_distill_loss_for_diagnostic": _safe_mean(part_losses.get("eval_core_distill_loss_for_diagnostic", [])),
                }
            )
        for key in (
            "hard_pairwise_loss",
            "soft_topk_coverage_loss",
            "first_positive_rank_loss",
            "multi_positive_diversity_loss",
            "group_weight_hup",
            "group_weight_lzu",
            "group_weight_multicenter",
            "group_weight_pediatric",
            "group_weight_unknown",
        ):
            if key in part_losses:
                summary[key] = _safe_mean(part_losses.get(key, []))
        for key, values in part_losses.items():
            if key.startswith("n6_"):
                summary[key] = _safe_mean(values)
        summary["selection_score"] = _summary_score(summary, self.args, val_loss=_safe_mean(losses))
        summary["strategy"] = "fixed_probability_threshold"
        return _safe_mean(losses), summary, enriched

    def _save_outputs(self, records: Sequence[Dict[str, Any]], *, fold_idx: int, split_name: str) -> None:
        output_dir = Path(getattr(self.args, "output_dir", "outputs"))
        output_dir.mkdir(parents=True, exist_ok=True)
        patient_rows = []
        channel_rows = []
        for record in records:
            quality_counts = _quality_counts(record.get("record_quality_labels", [])) if record.get("record_quality_labels") else {"good": 0, "review": 0, "poor": 0}
            active_quality_weights = [float(value) for value in record.get("record_quality_weights_active", []) if np.isfinite(float(value))]
            patient_rows.append(
                {
                    "fold_idx": int(fold_idx),
                    "split": split_name,
                    "subject_id": record["subject_id"],
                    "center": str(record.get("center", "unknown")),
                    "center_id": int(record.get("center_id", 4)),
                    "n_channels": int(record.get("n_channels", record.get("valid_channel_count", 0))),
                    "n_ez": int(record.get("true_ez_count", record.get("ez_channel_count", 0))),
                    "ez_fraction": float(record.get("ez_fraction", 0.0)),
                    "true_nez_count": float(record.get("true_nez_count", 0.0)),
                    "true_ez_count": float(record.get("true_ez_count", 0.0)),
                    "predicted_nez_count": float(record.get("predicted_nez_count", 0.0)),
                    "predicted_ez_count": float(record.get("predicted_ez_count", 0.0)),
                    "patient_macro_f1": float(record.get("patient_macro_f1", 0.0)),
                    "patient_balanced_accuracy": float(record.get("patient_balanced_accuracy", 0.0)),
                    "patient_weighted_f1": float(record.get("patient_weighted_f1", 0.0)),
                    "patient_nez_precision": float(record.get("patient_nez_precision", 0.0)),
                    "patient_nez_recall": float(record.get("patient_nez_recall", 0.0)),
                    "patient_nez_f1": float(record.get("patient_nez_f1", 0.0)),
                    "patient_ez_precision": float(record.get("patient_ez_precision", 0.0)),
                    "patient_ez_recall": float(record.get("patient_ez_recall", 0.0)),
                    "patient_ez_f1": float(record.get("patient_ez_f1", 0.0)),
                    "patient_accuracy": float(record.get("patient_accuracy", 0.0)),
                    "ez_mrr": float(record.get("ez_mrr", 0.0)),
                    "ez_recall_at_true_count": float(record.get("ez_recall_at_true_count", 0.0)),
                    "top1_is_ez": float(record.get("top1_is_ez", 0.0)),
                    "predicted_ez_channels": ";".join(record.get("predicted_ez_channels", record.get("predicted_channels", []))),
                    "predicted_nez_channels": ";".join(record.get("predicted_nez_channels", [])),
                    "true_ez_channels": ";".join(record.get("true_ez_channels", [])),
                    "true_nez_channels": ";".join(record.get("true_nez_channels", [])),
                    "n_seizures": len(record.get("run_ids", [])),
                    "quality_good_records": int(quality_counts.get("good", 0)),
                    "quality_review_records": int(quality_counts.get("review", 0)),
                    "quality_poor_records": int(quality_counts.get("poor", 0)),
                    "quality_active_weight_mean": _safe_mean(active_quality_weights),
                    "classification_threshold": float(record.get("classification_threshold", 0.5)),
                }
            )
            labels_nez = np.asarray(record.get("labels_nez", record["labels"]), dtype=np.float32)
            labels_ez = np.asarray(record.get("labels_ez", 1.0 - labels_nez), dtype=np.float32)
            score_nez = np.asarray(record.get("score_nez", record["scores"]), dtype=np.float32)
            score_ez = np.asarray(record.get("score_ez", 1.0 - score_nez), dtype=np.float32)
            optional_channel_arrays = {
                key: np.asarray(record[key], dtype=np.float32)
                for key in (
                    "score_ez_base", "score_ez_anchor", "score_ez_final", "score_ez_static",
                    "score_core", "score_broad", "score_clinical", "score_eval", "score_eval_original",
                    "score_nez_feature", "score_nez_raw", "score_nez_fused", "fusion_gate_feature", "raw_channel_available",
                    "score_nez_uncalibrated", "score_ez_uncalibrated",
                    "logits_core", "logits_broad", "logits_eval", "logits_eval_original",
                    "logits_base",
                    "pseudo_core_q", "a9v3_oof_score",
                    "negative_anchor_distance", "graph_delta_logits",
                )
                if key in record
            }
            pred_ez = np.asarray(record.get("predicted_ez_mask", record.get("predicted_mask", np.zeros_like(labels_nez))), dtype=int)
            pred_nez = np.asarray(record.get("predicted_nez_mask", 1 - pred_ez), dtype=int)
            valid_mask = np.asarray(record["channel_mask"], dtype=bool)
            rank_ez_desc = np.full(score_ez.shape[0], -1, dtype=int)
            valid_idx = np.where(valid_mask)[0]
            if valid_idx.size > 0:
                for rank, idx in enumerate(valid_idx[np.argsort(score_ez[valid_idx])[::-1]], start=1):
                    rank_ez_desc[idx] = rank
            for channel_idx, channel_name in enumerate(record["canonical_channels"]):
                channel_dict = {
                        "fold_idx": int(fold_idx),
                        "split": split_name,
                        "subject_id": record["subject_id"],
                        "center": str(record.get("center", "unknown")),
                        "center_id": int(record.get("center_id", 4)),
                        "channel_id": int(channel_idx),
                        "channel_name": channel_name,
                        "true_nez": float(labels_nez[channel_idx]),
                        "true_ez": float(labels_ez[channel_idx]),
                        "score_nez_probability": float(score_nez[channel_idx]),
                        "score_ez_probability": float(score_ez[channel_idx]),
                        "classification_threshold": float(record.get("classification_threshold", 0.5)),
                        "rank_ez_desc": int(rank_ez_desc[channel_idx]),
                        "predicted_nez": int(pred_nez[channel_idx]),
                        "predicted_ez": int(pred_ez[channel_idx]),
                        **{
                            key: float(value[channel_idx])
                            for key, value in optional_channel_arrays.items()
                            if channel_idx < value.shape[0]
                        },
                    }
                channel_rows.append(channel_dict)
        pd.DataFrame(patient_rows).to_csv(output_dir / f"{split_name}_patient_predictions_neuroez_v2_fold_{fold_idx}.csv", index=False)
        pd.DataFrame(channel_rows).to_csv(output_dir / f"{split_name}_channel_predictions_neuroez_v2_fold_{fold_idx}.csv", index=False)

    def run(self) -> List[Dict[str, Any]]:
        all_test_records: List[Dict[str, Any]] = []
        fold_summary_rows: List[Dict[str, Any]] = []
        train_dropout_rows: List[Dict[str, Any]] = []
        lcbo_target_match_rows: List[Dict[str, Any]] = []
        lcbo_train_loss_rows: List[Dict[str, Any]] = []
        n6_train_loss_rows: List[Dict[str, Any]] = []
        n6_validation_threshold_rows: List[Dict[str, Any]] = []
        base_seed = int(getattr(self.args, "random_seed", 42))
        batch_size = int(getattr(self.args, "patient_batch_size", getattr(self.args, "batch_size", 2)))
        epochs = int(getattr(self.args, "epochs", 50))
        patience = int(getattr(self.args, "patience", 15))
        min_epochs_before_early_stop = max(0, int(getattr(self.args, "min_epochs_before_early_stop", 0)))
        log_interval = int(getattr(self.args, "log_interval", 1))

        outer_splits = list(self.outer_splits)
        max_outer_folds = int(getattr(self.args, "max_outer_folds", 0) or 0)
        if max_outer_folds > 0:
            outer_splits = outer_splits[:max_outer_folds]
            self._log(
                f"Limiting run to first {len(outer_splits)} outer fold(s) because max_outer_folds={max_outer_folds}."
            )
        quality_audit_summary: dict[str, Any] = {}
        if bool(getattr(self.args, "use_edf_quality_weighting", False)):
            quality_audit_summary = self._write_quality_audit_outputs(outer_splits)

        self._log(
            "Starting B0-Pruned cross-validation | "
            f"folds={len(outer_splits)} | epochs={epochs} | batch_size={batch_size} | "
            f"loss={str(getattr(self.args, 'loss_mode', 'masked_bce'))} | "
            f"classification_threshold={float(getattr(self.args, 'classification_threshold', 0.5)):.3f}"
        )

        for split in outer_splits:
            fold_idx = int(split["fold_idx"])
            _set_random_seed(base_seed + fold_idx)
            self.a9v11_group_weights = torch.full(
                (5,),
                0.2,
                dtype=torch.float32,
                device=self.device if hasattr(self, "device") else torch.device("cpu"),
            ).cpu()
            self._log(f"Fold {fold_idx}: reset A9v11 group-DRO weights to uniform.")
            fit_subjects, val_subjects = split_train_val_subjects(
                list(split["train_subjects"]),
                val_ratio=float(getattr(self.args, "val_ratio", 0.2)),
                random_seed=base_seed,
                fold_idx=fold_idx,
            )
            fit_subjects_before_dropout = list(fit_subjects)
            fit_subjects, dropped_train_subjects = self._sample_train_subject_dropout(fit_subjects, fold_idx)
            test_subjects = list(split["test_subjects"])
            dropped_set = set(dropped_train_subjects)
            for subject_id in self.train_subject_dropout_candidates:
                train_dropout_rows.append(
                    {
                        "fold_id": int(fold_idx),
                        "subject_id": subject_id,
                        "status": "dropped" if subject_id in dropped_set else (
                            "eligible_not_dropped" if subject_id in fit_subjects_before_dropout else "not_in_fit"
                        ),
                        "fit_patients_before_dropout": int(len(fit_subjects_before_dropout)),
                        "fit_patients_after_dropout": int(len(fit_subjects)),
                        "dropped_count": int(len(dropped_train_subjects)),
                        "dropout_seed": int(
                            self.train_subject_dropout_seed
                            if self.train_subject_dropout_seed >= 0
                            else base_seed
                        ),
                    }
                )
            if dropped_train_subjects:
                self._log(f"Fold {fold_idx}: train-only dropout removed {len(dropped_train_subjects)} patient(s): {dropped_train_subjects}")
            train_dataset, val_dataset, test_dataset, normalizer = self._build_datasets(fit_subjects, val_subjects, test_subjects)
            if len(train_dataset) == 0 or len(test_dataset) == 0:
                self._log(f"Fold {fold_idx}: skipped because train/test dataset is empty.")
                continue
            if self.use_a9v8_lcbo:
                self._load_lcbo_targets_for_fold(fold_idx)

            train_loader = self._make_loader(train_dataset, shuffle=True, batch_size=batch_size)
            val_loader = self._make_loader(val_dataset, shuffle=False, batch_size=batch_size)
            test_loader = self._make_loader(test_dataset, shuffle=False, batch_size=batch_size)

            model = self.runtime["model_cls"](self.args).to(self.device)
            self._dry_initialize_lazy_layers(model, train_loader)
            ema_teacher = _clone_ema_teacher(model).to(self.device) if self.use_n6_dual_view_ema else None
            fold_dir = Path(getattr(self.args, "output_dir", "outputs")) / f"fold_{fold_idx}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            if bool(getattr(self.args, "pretrain_masked_windows", False)) and int(getattr(self.args, "pretrain_epochs", 0)) > 0:
                from neuroez_c.masked_pretraining import pretrain_masked_windows

                pretrain_summary = pretrain_masked_windows(
                    model,
                    train_loader,
                    self.args,
                    self.device,
                    train_subjects=fit_subjects,
                    forbidden_subjects=list(val_subjects) + list(test_subjects),
                    output_path=fold_dir / f"pretrain_summary_fold_{fold_idx}.json",
                )
                self._log(
                    f"Fold {fold_idx}: masked pretraining complete | "
                    f"epochs={pretrain_summary.get('epochs', 0)} | "
                    f"final_loss={float(pretrain_summary.get('final_loss', 0.0)):.4f}"
                )
            ez_weight_arg = str(getattr(self.args, "ez_negative_weight", "2")).lower()
            ez_negative_weight_value = (
                _estimate_ez_negative_weight(train_dataset, cap=float(getattr(self.args, "ez_negative_weight_cap", 20.0)))
                if ez_weight_arg == "auto"
                else float(ez_weight_arg)
            )
            ez_negative_weight = torch.tensor(ez_negative_weight_value, dtype=torch.float32, device=self.device)
            if self.use_n6_dual_view_ema:
                raw_prefixes = (
                    "raw_window_encoder", "raw_temporal_encoder", "raw_seizure_aggregator",
                    "raw_channel_classifier", "raw_fusion_projection", "raw_fusion_norm", "fusion_gate",
                )
                raw_parameters = [parameter for name, parameter in model.named_parameters() if name.startswith(raw_prefixes)]
                raw_ids = {id(parameter) for parameter in raw_parameters}
                feature_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad and id(parameter) not in raw_ids]
                optimizer = torch.optim.AdamW([
                    {"params": feature_parameters, "lr": float(getattr(self.args, "learning_rate", 1e-4))},
                    {"params": raw_parameters, "lr": float(getattr(self.args, "learning_rate", 1e-4)) * float(getattr(self.args, "raw_lr_multiplier", 2.0))},
                ], weight_decay=float(getattr(self.args, "weight_decay", 1e-3)))
            else:
                optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=float(getattr(self.args, "learning_rate", 1e-4)), weight_decay=float(getattr(self.args, "weight_decay", 1e-3)))

            self._log(
                f"Fold {fold_idx}/{len(self.outer_splits)} ready | "
                f"fit_patients={len(train_dataset)} | val_patients={len(val_dataset)} | test_patients={len(test_dataset)} | "
                f"window_feature_dim={normalizer.feature_dim} | ez_negative_weight={ez_negative_weight_value:.3f}"
            )

            best_state = copy.deepcopy(model.state_dict())
            best_ema_state = copy.deepcopy(ema_teacher.state_dict()) if ema_teacher is not None else None
            best_summary: Dict[str, float] = {"patient_macro_f1": 0.0, "selection_score": -1.0}
            best_classification_threshold = float(getattr(self.args, "classification_threshold", 0.5))
            best_score = -1.0
            best_epoch = 0
            epochs_without_improvement = 0
            best_model_path = fold_dir / "best_b0_pruned_model.pth"

            for epoch in range(1, epochs + 1):
                self.current_epoch = epoch
                train_metrics = self._train_one_epoch(model, train_loader, optimizer, ez_negative_weight, ema_teacher=ema_teacher)
                if self.use_a9v8_lcbo:
                    train_row = {
                        "fold_id": int(fold_idx),
                        "split": "train",
                        "epoch": int(epoch),
                        **{
                            key: float(train_metrics.get(key, 0.0))
                            for key in (
                                "loss",
                                "mean_broad_bce_loss",
                                "mean_core_rank_loss",
                                "mean_soft_mrr_loss",
                                "mean_subset_loss",
                                "mean_core_distill_loss",
                                "train_mean_core_rank_loss",
                                "train_mean_soft_mrr_loss",
                                "train_mean_core_distill_loss",
                                "target_match_rate",
                                "matched_channels",
                                "total_valid_channels",
                                "sum_pseudo_core_q",
                                "n_core_positive_channels",
                                "mean_pseudo_core_q",
                            )
                        },
                    }
                    lcbo_train_loss_rows.append(train_row)
                    lcbo_target_match_rows.append(
                        {
                            key: train_row[key]
                            for key in (
                                "fold_id",
                                "split",
                                "epoch",
                                "target_match_rate",
                                "matched_channels",
                                "total_valid_channels",
                                "sum_pseudo_core_q",
                                "n_core_positive_channels",
                                "mean_pseudo_core_q",
                            )
                        }
                    )
                validation_model = ema_teacher if self.use_n6_dual_view_ema else model
                val_loss, raw_val_summary, raw_val_records = self._evaluate(validation_model, val_loader, ez_negative_weight, split_name="val")
                if self.use_n6_dual_view_ema:
                    # Epoch selection is threshold-free. Threshold selection happens once after loading the best checkpoint.
                    val_classification_threshold = 0.5
                else:
                    val_classification_threshold = _select_validation_classification_threshold(
                        raw_val_records,
                        positive_label=str(getattr(self.args, "positive_label", "nez")),
                    )
                val_summary, val_records = _summarize_prediction_records(
                    raw_val_records,
                    classification_threshold=val_classification_threshold,
                )
                val_summary.update({
                    key: raw_val_summary[key]
                    for key in raw_val_summary
                    if key.startswith(("loss", "mean_", "target_", "matched_", "total_", "sum_", "n_core_", "eval_", "score_", "subset_", "broad_", "two_expert_", "negative_anchor_", "n6_"))
                })
                val_summary["classification_threshold"] = float(val_classification_threshold)
                if self.use_n6_dual_view_ema:
                    n6_train_loss_rows.append({
                        "outer_fold": int(fold_idx),
                        "epoch": int(epoch),
                        "train_total_loss": float(train_metrics.get("n6_total_loss", train_metrics.get("loss", 0.0))),
                        "train_fused_classification_loss": float(train_metrics.get("n6_classification_loss", 0.0)),
                        "train_clean_nez_loss": float(train_metrics.get("n6_clean_nez_loss", 0.0)),
                        "train_observed_ez_loss": float(train_metrics.get("n6_observed_ez_loss", 0.0)),
                        "train_feature_aux_loss": float(train_metrics.get("n6_feature_aux_loss", 0.0)),
                        "train_raw_aux_loss": float(train_metrics.get("n6_raw_aux_loss", 0.0)),
                        "train_rank_loss": float(train_metrics.get("n6_rank_loss", 0.0)),
                        "train_gate_loss": float(train_metrics.get("n6_gate_loss", 0.0)),
                        "ema_mean_reliability": float(train_metrics.get("n6_mean_reliability", 1.0)),
                        "ema_min_reliability": float(train_metrics.get("n6_min_reliability", 1.0)),
                        "ema_reliability_floor_rate": float(train_metrics.get("n6_reliability_floor_rate", 0.0)),
                        "fusion_gate_feature_mean": float(train_metrics.get("n6_fusion_gate_feature_mean", 1.0)),
                        "fusion_gate_feature_std": float(train_metrics.get("n6_fusion_gate_feature_std", 0.0)),
                        "fusion_gate_feature_min": float(train_metrics.get("n6_fusion_gate_feature_min", 1.0)),
                        "fusion_gate_feature_max": float(train_metrics.get("n6_fusion_gate_feature_max", 1.0)),
                        "raw_channel_available_rate": float(train_metrics.get("n6_raw_channel_available_rate", 0.0)),
                        "val_patient_macro_auprc_nez": float(val_summary.get("patient_macro_auprc_nez", 0.0)),
                        "val_patient_macro_auprc_ez": float(val_summary.get("patient_macro_auprc_ez", 0.0)),
                        "val_balanced_auprc_hmean": float(_summary_score(val_summary, self.args, val_loss=val_loss)),
                        "val_patient_macro_f1_at_0_5": float(val_summary.get("patient_macro_f1", 0.0)),
                        "val_patient_macro_nez_f1_at_0_5": float(val_summary.get("patient_macro_nez_f1", 0.0)),
                        "val_patient_macro_ez_f1_at_0_5": float(val_summary.get("patient_macro_ez_f1", 0.0)),
                    })
                if self.use_a9v8_lcbo:
                    lcbo_target_match_rows.append(
                        {
                            "fold_id": int(fold_idx),
                            "split": "val",
                            "epoch": int(epoch),
                            "target_match_rate": float(val_summary.get("target_match_rate", val_summary.get("eval_target_match_rate", 0.0))),
                            "matched_channels": float(val_summary.get("matched_channels", 0.0)),
                            "total_valid_channels": float(val_summary.get("total_valid_channels", 0.0)),
                            "sum_pseudo_core_q": float(val_summary.get("sum_pseudo_core_q", 0.0)),
                            "n_core_positive_channels": float(val_summary.get("n_core_positive_channels", 0.0)),
                            "mean_pseudo_core_q": float(val_summary.get("mean_pseudo_core_q", 0.0)),
                        }
                    )
                val_score = _summary_score(val_summary, self.args, val_loss=val_loss)
                val_summary["selection_score"] = float(val_score)
                improved = val_score > best_score + 1e-6
                if improved:
                    best_score = val_score
                    best_epoch = int(epoch)
                    best_state = copy.deepcopy(model.state_dict())
                    best_ema_state = copy.deepcopy(ema_teacher.state_dict()) if ema_teacher is not None else None
                    best_summary = dict(val_summary)
                    if not self.use_n6_dual_view_ema:
                        best_classification_threshold = float(val_classification_threshold)
                    epochs_without_improvement = 0
                    torch.save(
                        {
                            "epoch": epoch,
                            "student_model_state_dict": best_state,
                            "ema_teacher_state_dict": best_ema_state,
                            "model_state_dict": best_state,
                            "optimizer_state_dict": optimizer.state_dict(),
                            "val_summary": best_summary,
                            "normalizer_mean": normalizer.mean,
                            "normalizer_std": normalizer.std,
                            "normalizer_physics_mean": getattr(getattr(normalizer, "physics", None), "mean", None),
                            "normalizer_physics_std": getattr(getattr(normalizer, "physics", None), "std", None),
                            "label_semantics": "1=EZ,0=NEZ" if str(getattr(self.args, "positive_label", "nez")).lower() == "ez" else "1=NEZ,0=EZ",
                            "score_semantics": str(getattr(self.args, "score_semantics", "nez_probability")),
                            "positive_label": str(getattr(self.args, "positive_label", "nez")),
                            "ez_negative_weight": float(ez_negative_weight_value),
                            "classification_threshold": (
                                None if self.use_n6_dual_view_ema else float(best_classification_threshold)
                            ),
                            "threshold_source": (
                                "best_ema_validation_only" if self.use_n6_dual_view_ema else "per_epoch_validation"
                            ),
                            "raw_audit_metadata": dict(self.dual_view_audit) if self.use_n6_dual_view_ema else {},
                        },
                        best_model_path,
                    )
                    if not self.use_n6_dual_view_ema:
                        self._save_outputs(val_records, fold_idx=fold_idx, split_name="val")
                else:
                    epochs_without_improvement += 1

                if epoch == 1 or epoch % log_interval == 0 or improved:
                    self._log(
                        f"Fold {fold_idx} epoch {epoch}/{epochs} | "
                        f"train_loss={train_metrics['loss']:.4f} | val_loss={val_loss:.4f} | "
                        f"val_patient_macro_f1={float(val_summary.get('patient_macro_f1', 0.0)):.4f} | "
                        f"val_patient_macro_nez_f1={float(val_summary.get('patient_macro_nez_f1', 0.0)):.4f} | "
                        f"selection_score={float(val_score):.4f} | "
                        f"val_threshold={val_classification_threshold:.2f} | "
                        f"val_ez_recall_at_true_count={float(val_summary.get('ez_recall_at_true_count', 0.0)):.4f} | "
                        f"{'improved' if improved else 'no_improve'}"
                    )
                if patience > 0 and epoch > min_epochs_before_early_stop and epochs_without_improvement >= patience:
                    self._log(f"Fold {fold_idx}: early stop at epoch {epoch}.")
                    break

            model.load_state_dict(best_state)
            if self.use_n6_dual_view_ema:
                if ema_teacher is None or best_ema_state is None:
                    raise RuntimeError("N6 best EMA state was not initialized.")
                ema_teacher.load_state_dict(best_ema_state)
                _, final_val_loss_summary, final_val_raw_records = self._evaluate(
                    ema_teacher, val_loader, ez_negative_weight, split_name="val"
                )
                (
                    best_classification_threshold,
                    final_val_summary,
                    final_val_records,
                ) = _select_n6_validation_threshold(final_val_raw_records)
                final_val_summary.update({
                    key: final_val_loss_summary[key]
                    for key in final_val_loss_summary
                    if key.startswith("n6_")
                })
                self._save_outputs(final_val_records, fold_idx=fold_idx, split_name="val")
                n6_validation_threshold_rows.append({
                    "outer_fold": int(fold_idx),
                    "selected_threshold": float(best_classification_threshold),
                    "patient_macro_f1": float(final_val_summary.get("patient_macro_f1", 0.0)),
                    "patient_macro_balanced_accuracy": float(final_val_summary.get("patient_macro_balanced_accuracy", 0.0)),
                    "patient_macro_nez_f1": float(final_val_summary.get("patient_macro_nez_f1", 0.0)),
                    "patient_macro_ez_f1": float(final_val_summary.get("patient_macro_ez_f1", 0.0)),
                    "patient_macro_auprc_nez": float(final_val_summary.get("patient_macro_auprc_nez", 0.0)),
                    "n_validation_patients": int(len(final_val_records)),
                })
            final_model = ema_teacher if self.use_n6_dual_view_ema else model
            test_loss, raw_test_summary, raw_test_records = self._evaluate(final_model, test_loader, ez_negative_weight, split_name="test")
            test_summary, test_records = _summarize_prediction_records(
                raw_test_records,
                classification_threshold=best_classification_threshold,
            )
            test_summary.update({
                key: raw_test_summary[key]
                for key in raw_test_summary
                if key.startswith(("loss", "mean_", "target_", "matched_", "total_", "sum_", "n_core_", "eval_", "score_", "subset_", "broad_", "two_expert_", "negative_anchor_", "n6_"))
            })
            test_summary["classification_threshold"] = float(best_classification_threshold)
            self._save_outputs(test_records, fold_idx=fold_idx, split_name="test")
            all_test_records.extend(test_records)
            if self.use_n6_dual_view_ema:
                fit_set = set(map(str, fit_subjects))
                validation_set = set(map(str, val_subjects))
                test_set = set(map(str, test_subjects))
                n6_fold_audit = {
                    "method": "N6F_NEZ_DualView_EMA_RobustRank",
                    "outer_fold": int(fold_idx),
                    "fit_subjects": sorted(fit_set),
                    "validation_subjects": sorted(validation_set),
                    "test_subjects": sorted(test_set),
                    "fit_validation_overlap": sorted(fit_set & validation_set),
                    "fit_test_overlap": sorted(fit_set & test_set),
                    "validation_test_overlap": sorted(validation_set & test_set),
                    "best_epoch": int(best_epoch),
                    "best_validation_balanced_auprc_hmean": float(best_score),
                    "selected_validation_threshold": float(best_classification_threshold),
                    "threshold_source": "best_ema_validation_only",
                    "student_checkpoint_saved": True,
                    "ema_checkpoint_saved": True,
                    "final_prediction_model": "ema_teacher",
                    "test_evaluations_before_model_selection": 0,
                    "test_labels_used_for_training": False,
                    "test_labels_used_for_threshold": False,
                    "test_scores_used_for_threshold": False,
                    "center_used_as_model_input": False,
                    "true_ez_count_used_for_prediction": False,
                    "raw_patient_coverage": float(self.dual_view_audit.get("raw_patient_coverage", 0.0)),
                    "raw_channel_match_rate": float(self.dual_view_audit.get("channel_match_rate", 0.0)),
                    "raw_window_match_rate": float(self.dual_view_audit.get("window_match_rate", 0.0)),
                    "raw_missing_channels_in_test": int(sum(self.dual_view_audit.get("missing_raw_by_patient", {}).get(subject, 0) for subject in test_set)),
                    "learned_class_prior": False,
                    "propensity_head": False,
                    "nnpu": False,
                    "posthoc_calibration": "none",
                }
                output_dir = Path(getattr(self.args, "output_dir", "outputs"))
                with (output_dir / f"n6_fold_{fold_idx}_audit.json").open("w", encoding="utf-8") as fout:
                    json.dump(n6_fold_audit, fout, indent=2, ensure_ascii=False, sort_keys=True)
            if self.use_a9v8_lcbo:
                lcbo_target_match_rows.append(
                    {
                        "fold_id": int(fold_idx),
                        "split": "test",
                        "epoch": int(getattr(self, "current_epoch", 0)),
                        "target_match_rate": float(test_summary.get("target_match_rate", test_summary.get("eval_target_match_rate", 0.0))),
                        "matched_channels": float(test_summary.get("matched_channels", 0.0)),
                        "total_valid_channels": float(test_summary.get("total_valid_channels", 0.0)),
                        "sum_pseudo_core_q": float(test_summary.get("sum_pseudo_core_q", 0.0)),
                        "n_core_positive_channels": float(test_summary.get("n_core_positive_channels", 0.0)),
                        "mean_pseudo_core_q": float(test_summary.get("mean_pseudo_core_q", 0.0)),
                    }
                )
            fold_summary_rows.append(
                {
                    "fold_id": int(fold_idx),
                    "n_test_patients": int(len(test_records)),
                    "fit_patients_before_dropout": int(len(fit_subjects_before_dropout)),
                    "fit_patients_after_dropout": int(len(fit_subjects)),
                    "train_subject_dropout_count": int(len(dropped_train_subjects)),
                    "train_subject_dropout_subjects": ",".join(dropped_train_subjects),
                    "center_count_string": str(test_summary.get("centers_count_string", "")),
                    "test_loss": float(test_loss),
                    **{
                        key: value
                        for key, value in test_summary.items()
                        if isinstance(value, (int, float, str, bool))
                    },
                }
            )
            self._log(
                f"Fold {fold_idx}/{len(self.outer_splits)} complete | "
                f"test_loss={test_loss:.4f} | test_patient_macro_f1={float(test_summary.get('patient_macro_f1', 0.0)):.4f} | "
                f"test_ez_recall_at_true_count={float(test_summary.get('ez_recall_at_true_count', 0.0)):.4f} | "
                f"best_val_patient_macro_f1={float(best_summary.get('patient_macro_f1', 0.0)):.4f}"
            )

        if all_test_records:
            overall_summary, _ = _summarize_prediction_records(all_test_records)
            output_dir = Path(getattr(self.args, "output_dir", "outputs"))
            output_dir.mkdir(parents=True, exist_ok=True)
            if train_dropout_rows:
                pd.DataFrame(train_dropout_rows).to_csv(output_dir / "train_subject_dropout_audit.csv", index=False)
            if n6_train_loss_rows:
                pd.DataFrame(n6_train_loss_rows).to_csv(output_dir / "n6_train_loss_by_fold_epoch.csv", index=False)
            if n6_validation_threshold_rows:
                pd.DataFrame(n6_validation_threshold_rows).to_csv(output_dir / "n6_validation_thresholds.csv", index=False)
            for key in (
                "positive_label",
                "score_semantics",
                "drop_high_ez_fraction_lzu",
                "loss_mode",
                "patient_loss_weighting",
                "use_negative_anchor_head",
                "negative_anchor_norm",
                "negative_anchor_distance_mode",
                "negative_anchor_center_gate",
                "use_hard_topk_loss",
                "hard_topk_loss_weight",
                "hard_topk_margin",
                "hard_topk_multiplier",
                "use_broad_ez_mil_loss",
                "broad_ez_mil_loss_weight",
                "broad_ez_core_frac",
                "broad_ez_min_fraction",
                "broad_ez_centers",
                "broad_ez_positive_bce_scale",
                "broad_ez_margin",
                "broad_ez_hard_neg_multiplier",
                "use_two_expert_router",
                "two_expert_router_mode",
                "use_feature_separated_two_expert",
                "use_view_gated_fusion",
                "diffusion_center_mode",
                "diffusion_score_residual",
                "pretrain_masked_windows",
                "pretrain_epochs",
                "use_physics_dynamics",
                "physics_state_features",
                "temporal_pooling",
                "channel_pooling_mode",
                "early_pool_frac",
                "lse_pool_tau",
                "temporal_pooling_top_p",
                "temporal_pooling_tau",
                "record_pooling",
                "record_pooling_top_p",
                "record_pooling_alpha",
                "rank_loss_weight",
                "hard_pairwise_weight",
                "soft_topk_weight",
                "first_rank_weight",
                "diversity_weight",
                "soft_rank_tau",
                "soft_topk_tau",
                "hard_pair_margin",
                "hard_pair_top_negatives",
                "group_robust_mode",
                "group_dro_eta",
                "group_dro_min_count",
                "use_a9v8_lcbo",
                "use_n6_dual_view_ema",
                "raw_window_cache_path",
                "raw_target_samples",
                "raw_target_sampling_rate",
                "n6_ema_decay",
                "n6_warmup_epochs",
                "n6_noise_discount",
                "n6_reliability_min",
                "n6_feature_aux_weight",
                "n6_raw_aux_weight",
                "n6_rank_loss_weight",
                "n6_rank_margin",
                "n6_gate_anchor",
                "n6_gate_loss_weight",
                "latent_core_target_dir",
                "eval_score_fusion_gamma",
                "lambda_core_rank",
                "lambda_soft_mrr",
                "lambda_subset",
                "lambda_core_distill",
                "core_rank_margin",
                "soft_mrr_tau",
                "subset_eps",
                "lcbo_hard_neg_top_frac",
                "lcbo_min_core_mass",
                "config_name",
                "use_patient_context_reranker",
                "use_multi_seizure_consistency",
                "use_shaft_local_residual",
                "use_clinical_mixture_head",
                "freeze_a9v3_backbone",
                "base_aux_loss_weight",
                "use_edf_quality_weighting",
                "edf_quality_field",
                "review_weight",
                "poor_weight",
                "quality_weight_loss",
                "quality_weight_record_aggregation",
                "quality_curriculum_epochs",
                "train_subject_dropout_file",
                "train_subject_dropout_count",
                "train_subject_dropout_seed",
            ):
                value = getattr(self.args, key, None)
                if isinstance(value, (str, int, float, bool)) or value is None:
                    overall_summary[key] = value
            overall_summary.update(
                {
                    f"quality_audit_{key}": value
                    for key, value in quality_audit_summary.items()
                    if isinstance(value, (str, int, float, bool))
                }
            )
            if self.use_a9v8_lcbo and lcbo_train_loss_rows:
                for key in (
                    "train_mean_core_rank_loss",
                    "train_mean_soft_mrr_loss",
                    "train_mean_core_distill_loss",
                    "target_match_rate",
                    "matched_channels",
                    "total_valid_channels",
                    "sum_pseudo_core_q",
                    "n_core_positive_channels",
                    "mean_pseudo_core_q",
                ):
                    values = [float(row.get(key, 0.0)) for row in lcbo_train_loss_rows if key in row]
                    if values:
                        overall_summary[key] = _safe_mean(values)
            if self.use_a9v8_lcbo and fold_summary_rows:
                matched = float(np.sum([float(row.get("matched_channels", 0.0)) for row in fold_summary_rows]))
                total = float(np.sum([float(row.get("total_valid_channels", 0.0)) for row in fold_summary_rows]))
                sum_q = float(np.sum([float(row.get("sum_pseudo_core_q", 0.0)) for row in fold_summary_rows]))
                n_core = float(np.sum([float(row.get("n_core_positive_channels", 0.0)) for row in fold_summary_rows]))
                if total > 0.0:
                    overall_summary.update(
                        {
                            "eval_target_match_rate": matched / max(total, 1.0),
                            "matched_channels": matched,
                            "total_valid_channels": total,
                            "sum_pseudo_core_q": sum_q,
                            "n_core_positive_channels": n_core,
                            "mean_pseudo_core_q": sum_q / max(matched, 1.0),
                        }
                    )
                for key in (
                    "eval_core_rank_loss_for_diagnostic",
                    "eval_soft_mrr_loss_for_diagnostic",
                    "eval_core_distill_loss_for_diagnostic",
                ):
                    values = [float(row.get(key, 0.0)) for row in fold_summary_rows if key in row]
                    if values:
                        overall_summary[key] = _safe_mean(values)
            if str(getattr(self.args, "split_strategy", "")).lower() == "5fold" and int(getattr(self.args, "n_splits", 0)) == 5:
                expected = int(getattr(self.args, "expected_heldout_patients", 90))
                if len(all_test_records) != expected:
                    overall_summary["heldout_patient_count_warning"] = (
                        f"Expected {expected} held-out patient rows for all90 5-fold mode, got {len(all_test_records)}."
                    )
            overall_summary["rank_robust_composite"] = _rank_robust_composite_score(overall_summary)
            pd.DataFrame([overall_summary]).to_csv(output_dir / "heldout_summary_neuroez_v3.csv", index=False)
            pd.DataFrame(fold_summary_rows).to_csv(output_dir / "heldout_fold_summary_neuroez_v3.csv", index=False)
            pd.DataFrame([overall_summary]).to_csv(output_dir / "quality_ablation_summary.csv", index=False)
            if self.use_a9v8_lcbo:
                pd.DataFrame(lcbo_target_match_rows).to_csv(output_dir / "lcbo_target_match_summary.csv", index=False)
                pd.DataFrame(lcbo_train_loss_rows).to_csv(output_dir / "lcbo_train_loss_by_fold_epoch.csv", index=False)
            pd.Series(overall_summary).to_json(output_dir / "heldout_summary_neuroez_v3.json", indent=2)
            self._log(
                "Cross-validation held-out mean | "
                f"patients={len(all_test_records)} | "
                f"patient_macro_f1={float(overall_summary.get('patient_macro_f1', 0.0)):.4f} | "
                f"ez_recall_at_true_count={float(overall_summary.get('ez_recall_at_true_count', 0.0)):.4f} | "
                f"pooled_auroc_nez={float(overall_summary.get('pooled_auroc_nez', 0.0)):.4f} | "
                f"pooled_auprc_ez={float(overall_summary.get('pooled_auprc_ez', 0.0)):.4f}"
            )
        self._log(f"Cross-validation finished. Total held-out patient records: {len(all_test_records)}")
        return all_test_records


__all__ = ["Exp_EZHybridLocalization", "select_best_decision_rule"]
