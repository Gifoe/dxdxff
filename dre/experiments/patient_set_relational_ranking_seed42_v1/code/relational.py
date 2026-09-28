"""Small, frozen-R4 patient-set residual readouts.

An anchor score explicitly uses other channels for R1/R2/R3. R2/R3 do not
factor into independent f(anchor)-f(reference) scores.
"""
from __future__ import annotations

import torch
from torch import nn


class RelationalReadout(nn.Module):
    def __init__(self, kind: str):
        super().__init__()
        if kind not in {"R1_DEEPSET_RESIDUAL", "R2_PAIRWISE_RELRANK", "R3_WEIGHTED_RELRANK"}:
            raise ValueError(kind)
        self.kind = kind
        if kind == "R1_DEEPSET_RESIDUAL":
            self.phi = nn.Sequential(nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 32), nn.GELU())
        else:
            self.psi = nn.Sequential(nn.Linear(193, 64), nn.GELU(), nn.Linear(64, 32), nn.GELU())
            if kind == "R3_WEIGHTED_RELRANK":
                self.reference_weight = nn.Sequential(nn.Linear(32, 16), nn.GELU(), nn.Linear(16, 1))
        self.rho = nn.Sequential(nn.Linear(97, 32), nn.GELU(), nn.Linear(32, 1))
        nn.init.zeros_(self.rho[-1].weight)
        nn.init.zeros_(self.rho[-1].bias)

    def forward(self, anchor_z: torch.Tensor, anchor_m0: torch.Tensor,
                context_z: torch.Tensor | None = None,
                context_m0: torch.Tensor | None = None,
                self_index: torch.Tensor | None = None) -> torch.Tensor:
        if context_z is None: context_z = anchor_z
        if context_m0 is None: context_m0 = anchor_m0
        if (anchor_z.ndim != 2 or context_z.ndim != 2 or
                anchor_z.shape[1] != 64 or context_z.shape[1] != 64 or
                anchor_m0.shape != (len(anchor_z),) or
                context_m0.shape != (len(context_z),) or len(context_z) < 2):
            raise ValueError("Malformed 64D patient-set inputs")
        if self_index is not None:
            if self_index.shape != (len(anchor_z),) or torch.any(self_index >= len(context_z)):
                raise ValueError("Invalid self indices")
        if self.kind == "R1_DEEPSET_RESIDUAL":
            summary = self.phi(context_z).mean(dim=0).expand(len(anchor_z), -1)
        else:
            diff = anchor_z[:, None, :] - context_z[None, :, :]
            product = anchor_z[:, None, :] * context_z[None, :, :]
            margin_diff = (anchor_m0[:, None] - context_m0[None, :]).unsqueeze(-1)
            e = self.psi(torch.cat((diff, diff.abs(), product, margin_diff), dim=-1))
            valid = torch.ones(e.shape[:2], dtype=torch.bool, device=e.device)
            if self_index is not None:
                valid[torch.arange(len(anchor_z), device=e.device), self_index.long()] = False
            if not torch.all(valid.any(dim=1)):
                raise ValueError("No nonself reference for an anchor")
            if self.kind == "R2_PAIRWISE_RELRANK":
                summary = (e * valid.unsqueeze(-1)).sum(dim=1) / valid.sum(dim=1, keepdim=True)
            else:
                logits = self.reference_weight(e).squeeze(-1).masked_fill(~valid, -torch.inf)
                summary = (torch.softmax(logits, dim=1).unsqueeze(-1) * e).sum(dim=1)
        residual = self.rho(torch.cat((anchor_z, summary, anchor_m0[:, None]), dim=1)).squeeze(-1)
        return anchor_m0 + residual


def patient_equal_loss(margin: torch.Tensor, y_ez: torch.Tensor, lambda_rank: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact A1 EZ=2/NEZ=1 weighted BCE plus patient-pair softplus ranking."""
    if margin.shape != y_ez.shape or not torch.all((y_ez == 0) | (y_ez == 1)):
        raise ValueError("Invalid patient labels or scores")
    weights = 1 + y_ez
    classification = (torch.nn.functional.softplus((1 - 2*y_ez)*margin)*weights).sum()/weights.sum()
    pos = margin[y_ez > .5]; neg = margin[y_ez < .5]
    ranking = (torch.nn.functional.softplus(neg[None, :] - pos[:, None]).mean()
               if len(pos) and len(neg) else margin.sum()*0)
    return classification + float(lambda_rank)*ranking, classification, ranking
