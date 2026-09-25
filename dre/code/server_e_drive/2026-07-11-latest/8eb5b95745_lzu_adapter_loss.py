"""Losses for the frozen-shared-model LZU residual adapter."""
from __future__ import annotations
import torch
import torch.nn.functional as F
from .v3_rcc_loss import patient_mean_unweighted_bce

def patient_balanced_bce(logit: torch.Tensor, label_ez: torch.Tensor, patient_index: torch.Tensor) -> torch.Tensor:
    losses=[]
    for p in torch.unique(patient_index):
        keep=patient_index==p; y=label_ez[keep]; x=logit[keep]; terms=[]
        for klass in (0.0,1.0):
            mask=y==klass
            if torch.any(mask): terms.append(F.binary_cross_entropy_with_logits(x[mask], y[mask]))
        losses.append(sum(terms)/len(terms))
    return torch.stack(losses).mean() if losses else logit.sum()*0.0

def adapter_loss(final_nez_logit: torch.Tensor, labels_ez: torch.Tensor, patient_index: torch.Tensor, delta: torch.Tensor) -> dict[str, torch.Tensor]:
    ez_logit=-final_nez_logit
    balanced=patient_balanced_bce(ez_logit, labels_ez, patient_index)
    # Build padded tensors only for the reusable patient-mean function is unnecessary here.
    mean=torch.stack([F.binary_cross_entropy_with_logits(ez_logit[patient_index==p], labels_ez[patient_index==p]) for p in torch.unique(patient_index)]).mean()
    pair=[]
    for p in torch.unique(patient_index):
        keep=patient_index==p; s=-final_nez_logit[keep]; y=labels_ez[keep]; pos=s[y>0.5]; neg=s[y<=0.5]
        if pos.numel() and neg.numel(): pair.append(F.softplus(0.10 + neg[:,None] - pos[None,:]).mean())
    pairwise=torch.stack(pair).mean() if pair else final_nez_logit.sum()*0.0
    residual=delta.square().mean()
    total=.60*balanced+.40*mean+.03*pairwise+.01*residual
    return {"total_loss":total,"patient_balanced_bce":balanced,"patient_mean_unweighted_bce":mean,"pairwise_loss":pairwise,"residual_loss":residual}

__all__=["adapter_loss","patient_balanced_bce"]
