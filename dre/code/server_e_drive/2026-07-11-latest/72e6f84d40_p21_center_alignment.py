"""Training-only clean-NEZ center moment alignment."""

from __future__ import annotations

import torch


def clean_nez_center_moment_alignment(
    contextual_channel_embedding: torch.Tensor,
    labels_nez: torch.Tensor,
    channel_mask: torch.Tensor,
    center_id: torch.Tensor | None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    zero = contextual_channel_embedding.sum() * 0.0
    if center_id is None:
        return zero, {"center_alignment_patient_count": zero, "center_alignment_center_count": zero}
    patient_means, patient_stds, patient_centers = [], [], []
    for patient_idx in range(contextual_channel_embedding.shape[0]):
        clean = channel_mask[patient_idx].bool() & (labels_nez[patient_idx] > 0.5)
        if not torch.any(clean):
            continue
        selected = contextual_channel_embedding[patient_idx, clean]
        patient_means.append(selected.mean(dim=0))
        patient_stds.append(selected.std(dim=0, unbiased=False))
        patient_centers.append(center_id[patient_idx])
    if not patient_means:
        return zero, {"center_alignment_patient_count": zero, "center_alignment_center_count": zero}
    means = torch.stack(patient_means)
    stds = torch.stack(patient_stds)
    centers = torch.stack(patient_centers)
    global_mean, global_std = means.mean(dim=0), stds.mean(dim=0)
    mean_terms, std_terms = [], []
    for center in torch.unique(centers):
        selected = centers == center
        if torch.any(selected):
            mean_terms.append((means[selected].mean(dim=0) - global_mean).square().mean())
        if int(selected.sum()) >= 2:
            std_terms.append((stds[selected].mean(dim=0) - global_std).square().mean())
    mean_loss = torch.stack(mean_terms).mean() if mean_terms else zero
    std_loss = torch.stack(std_terms).mean() if std_terms else zero
    total = mean_loss + 0.10 * std_loss
    return total, {
        "center_alignment_mean_loss": mean_loss,
        "center_alignment_std_loss": std_loss,
        "center_alignment_patient_count": torch.tensor(float(len(patient_means)), device=zero.device),
        "center_alignment_center_count": torch.tensor(float(len(mean_terms)), device=zero.device),
    }


__all__ = ["clean_nez_center_moment_alignment"]
