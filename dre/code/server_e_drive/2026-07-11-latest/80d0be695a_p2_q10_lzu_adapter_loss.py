"""Patient-equal losses for the frozen P2-Q10 LZU adapter."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _patient_ids(patient_index: torch.Tensor) -> torch.Tensor:
    return torch.unique(patient_index, sorted=True)


def patient_balanced_bce(adapted_nez_logit: torch.Tensor, label_nez: torch.Tensor, patient_index: torch.Tensor) -> torch.Tensor:
    """Equal patient weight, then equal class weight when both classes exist."""
    values: list[torch.Tensor] = []
    for patient in _patient_ids(patient_index):
        mask = patient_index == patient
        logits, labels = adapted_nez_logit[mask], label_nez[mask]
        class_losses = [F.binary_cross_entropy_with_logits(logits[labels == klass], labels[labels == klass]) for klass in (0.0, 1.0) if torch.any(labels == klass)]
        values.append(torch.stack(class_losses).mean())
    return torch.stack(values).mean() if values else adapted_nez_logit.sum() * 0.0


def patient_mean_unweighted_bce(adapted_nez_logit: torch.Tensor, label_nez: torch.Tensor, patient_index: torch.Tensor) -> torch.Tensor:
    values = [F.binary_cross_entropy_with_logits(adapted_nez_logit[patient_index == patient], label_nez[patient_index == patient]) for patient in _patient_ids(patient_index)]
    return torch.stack(values).mean() if values else adapted_nez_logit.sum() * 0.0


def original_ez_pairwise_ranking_loss(adapted_nez_logit: torch.Tensor, label_nez: torch.Tensor, patient_index: torch.Tensor, margin: float = 0.10) -> torch.Tensor:
    """True EZ (label_nez=0) must receive the higher EZ semantic score -logit."""
    values: list[torch.Tensor] = []
    ez_score = -adapted_nez_logit
    for patient in _patient_ids(patient_index):
        score, labels = ez_score[patient_index == patient], label_nez[patient_index == patient]
        true_ez, true_nez = score[labels == 0], score[labels == 1]
        if true_ez.numel() and true_nez.numel():
            values.append(F.relu(float(margin) - true_ez[:, None] + true_nez[None, :]).mean())
    return torch.stack(values).mean() if values else adapted_nez_logit.sum() * 0.0


def adapter_residual_regularization(adapter_delta: torch.Tensor) -> torch.Tensor:
    return adapter_delta.square().mean() if adapter_delta.numel() else adapter_delta.sum() * 0.0


def compute_p2_lzu_adapter_loss(adapted_nez_logit: torch.Tensor, label_nez: torch.Tensor, patient_index: torch.Tensor, adapter_delta: torch.Tensor, *, balanced_bce_weight: float = .60, unweighted_bce_weight: float = .40, pairwise_weight: float = .02, residual_weight: float = .02) -> dict[str, torch.Tensor]:
    balanced = patient_balanced_bce(adapted_nez_logit, label_nez, patient_index)
    unweighted = patient_mean_unweighted_bce(adapted_nez_logit, label_nez, patient_index)
    pairwise = original_ez_pairwise_ranking_loss(adapted_nez_logit, label_nez, patient_index)
    residual = adapter_residual_regularization(adapter_delta)
    total = float(balanced_bce_weight) * balanced + float(unweighted_bce_weight) * unweighted + float(pairwise_weight) * pairwise + float(residual_weight) * residual
    return {"total_loss": total, "patient_balanced_bce": balanced, "patient_mean_unweighted_bce": unweighted, "pairwise_loss": pairwise, "residual_loss": residual}


__all__ = ["patient_balanced_bce", "patient_mean_unweighted_bce", "original_ez_pairwise_ranking_loss", "adapter_residual_regularization", "compute_p2_lzu_adapter_loss"]
