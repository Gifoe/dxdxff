"""Complete profile-gated P2.1 objective."""

from __future__ import annotations

from typing import Any

import torch

from .cane_path_cp_loss import patient_balanced_bce_rows, soft_worst_center_loss
from .cane_set_loss import clean_nez_prototype_loss, patient_soft_macro_f1_loss
from .p21_center_alignment import clean_nez_center_moment_alignment
from .p21_profiles import get_p21_profile
from .p21_ranking_loss import (
    compute_observed_ez_reliability,
    ranking_preservation_loss,
    reliability_weighted_pairwise_rank_loss,
)


def compute_p21_loss(
    outputs: dict[str, torch.Tensor], batch: dict[str, Any], args: Any, epoch: int
) -> tuple[torch.Tensor, dict[str, float | str]]:
    profile = get_p21_profile(getattr(args, "p21_profile", "R5_FULL"))
    if profile.legacy_p2:
        from .cane_path_cp_loss import compute_cane_path_cp_ranking_loss
        return compute_cane_path_cp_ranking_loss(outputs, batch, args, epoch)
    device, dtype = outputs["final_nez_logit"].device, outputs["final_nez_logit"].dtype
    labels_ez = batch["labels_ez"].to(device=device, dtype=dtype)
    labels_nez = torch.where(labels_ez >= 0, 1.0 - labels_ez, torch.full_like(labels_ez, -1.0))
    mask = batch["channel_mask"].to(device).bool()
    final_bce, patient_rows, clean_bce, observed_bce = patient_balanced_bce_rows(
        outputs["final_nez_logit"], labels_nez, mask
    )
    center_id = batch.get("center_id")
    center_id = center_id.to(device) if torch.is_tensor(center_id) else None
    worst, center_count = (
        soft_worst_center_loss(patient_rows, center_id, 0.20)
        if center_id is not None else (final_bce, 0)
    )
    worst_weight = float(getattr(args, "p21_soft_worst_weight", 0.075)) if center_count >= 2 else 0.0
    group_cls = (1.0 - worst_weight) * final_bce + worst_weight * worst
    soft_f1, soft_nez, soft_ez = patient_soft_macro_f1_loss(outputs["final_nez_logit"], labels_nez, mask)
    direct_aux, _, _, _ = patient_balanced_bce_rows(outputs["direct_nez_logit"], labels_nez, mask)
    prototype, compactness, diversity = clean_nez_prototype_loss(
        outputs["anchor_distance"], labels_nez, mask, outputs["normalized_prototypes"],
        float(getattr(args, "cane_prototype_similarity_margin", 0.50)),
    )
    reliability = compute_observed_ez_reliability(
        labels_nez, mask, outputs["direct_score_nez"], outputs["anchor_nez_evidence"],
        outputs["seizure_nez_probability_mean"], outputs["cp_early_source_rank"],
        seizure_available=outputs.get("seizure_evidence_valid"),
        causal_available=outputs.get("causal_feature_valid"),
    )
    outputs.update(reliability)
    rank, rank_diag = reliability_weighted_pairwise_rank_loss(
        outputs["final_nez_logit"], labels_nez, mask, reliability["ez_reliability"],
        margin=float(getattr(args, "p21_rank_margin", 0.10)),
        max_pairs_per_patient=int(getattr(args, "p21_max_pairs_per_patient", 4096)),
        model_seed=int(getattr(args, "model_seed", 42)), epoch=int(epoch),
        subject_ids=batch.get("subject_id"), active_from_epoch=int(getattr(args, "p21_rank_start_epoch", 10)),
    )
    reference = outputs["v3_anchor_standardized"] if profile.v3_anchor else outputs["direct_standardized"]
    preserve, preserve_diag = ranking_preservation_loss(
        outputs["final_nez_logit"], reference, labels_nez, mask, reliability["ez_reliability"],
        safe_margin=float(getattr(args, "p21_preserve_safe_margin", 0.30)),
        beta=float(getattr(args, "p21_preserve_beta", 0.80)),
        max_pairs_per_patient=int(getattr(args, "p21_max_pairs_per_patient", 4096)),
        model_seed=int(getattr(args, "model_seed", 42)), epoch=int(epoch),
        subject_ids=batch.get("subject_id"), active_from_epoch=int(getattr(args, "p21_preserve_start_epoch", 10)),
    )
    center_align, center_diag = clean_nez_center_moment_alignment(
        outputs["contextual_channel_embedding"], labels_nez, mask, center_id
    )
    delta_l2 = outputs["delta"][mask].square().mean() if torch.any(mask) else outputs["delta"].sum() * 0.0
    rank_weight = float(getattr(args, "p21_rank_weight", 0.03)) if profile.weighted_rank else 0.0
    preserve_weight = float(getattr(args, "p21_preserve_weight", 0.02)) if profile.preserve else 0.0
    align_weight = float(getattr(args, "p21_center_align_weight", 0.005)) if profile.center_alignment and int(epoch) > int(getattr(args, "p21_stage1_end", 8)) else 0.0
    total = (
        group_cls
        + 0.15 * soft_f1
        + float(getattr(args, "p21_direct_aux_weight", 0.05)) * direct_aux
        + rank_weight * rank
        + preserve_weight * preserve
        + float(getattr(args, "p21_prototype_weight", 0.03)) * prototype
        + align_weight * center_align
        + float(getattr(args, "p21_delta_l2_weight", 0.005)) * delta_l2
    )
    diagnostics: dict[str, float | str] = {
        "p21_total_loss": float(total.detach().cpu()),
        "p21_group_classification_loss": float(group_cls.detach().cpu()),
        "p21_patient_balanced_bce": float(final_bce.detach().cpu()),
        "p21_clean_nez_bce": float(clean_bce.detach().cpu()),
        "p21_observed_ez_bce": float(observed_bce.detach().cpu()),
        "p21_soft_macro_f1_loss": float(soft_f1.detach().cpu()),
        "p21_soft_nez_f1": float(soft_nez.detach().cpu()),
        "p21_soft_ez_f1": float(soft_ez.detach().cpu()),
        "p21_direct_aux_loss": float(direct_aux.detach().cpu()),
        "p21_weighted_rank_loss": float(rank.detach().cpu()),
        "p21_preserve_loss": float(preserve.detach().cpu()),
        "p21_prototype_loss": float(prototype.detach().cpu()),
        "p21_prototype_compactness": float(compactness.detach().cpu()),
        "p21_prototype_diversity": float(diversity.detach().cpu()),
        "p21_center_alignment_loss": float(center_align.detach().cpu()),
        "p21_delta_l2": float(delta_l2.detach().cpu()),
        "p21_soft_worst_loss": float(worst.detach().cpu()),
        "p21_present_center_count": float(center_count),
        "p21_profile": profile.name,
        "p21_stage": "stage1" if int(epoch) <= int(getattr(args, "p21_stage1_end", 8)) else "stage2" if int(epoch) <= int(getattr(args, "p21_stage2_end", 20)) else "stage3",
        "bce": float(group_cls.detach().cpu()),
    }
    for mapping in (rank_diag, preserve_diag, center_diag):
        diagnostics.update({f"p21_{key}": float(value.detach().cpu()) for key, value in mapping.items()})
    return total, diagnostics


__all__ = ["compute_p21_loss"]
