"""Zero-init rank-four coordinate adapter on frozen A1 classifier-input R4."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class FrozenR4Adapter(nn.Module):
    def __init__(self, dim: int, variant: str):
        super().__init__()
        if variant not in ("P1", "P2", "P3"):
            raise ValueError(variant)
        self.dim = int(dim)
        self.variant = variant
        self.rank = 4
        self.alpha = 0.25
        generator = torch.Generator(device="cpu").manual_seed(42)
        self.U = nn.Parameter(torch.linalg.qr(torch.randn(dim, 4, generator=generator), mode="reduced").Q)
        self.V = nn.Parameter(torch.linalg.qr(torch.randn(dim, 4, generator=generator), mode="reduced").Q)
        if variant == "P1":
            self.a_global = nn.Parameter(torch.zeros(4))
        else:
            self.phi = nn.Sequential(nn.Linear(dim, 32), nn.GELU(), nn.Linear(32, 16))
            self.rho = nn.Sequential(nn.Linear(32, 16), nn.GELU(), nn.LayerNorm(16))
            self.hyper = nn.Sequential(nn.Linear(16, 16), nn.GELU(), nn.Linear(16, 4))
            nn.init.zeros_(self.hyper[-1].weight)
            nn.init.zeros_(self.hyper[-1].bias)

    def coefficients(self, context_h: torch.Tensor, context_mask: torch.Tensor) -> torch.Tensor:
        if self.variant == "P1":
            return torch.tanh(self.a_global).unsqueeze(0).expand(context_h.shape[0], -1)
        if context_h.ndim != 3 or context_mask.shape != context_h.shape[:2]:
            raise ValueError("Context must be [B,C,D] and mask [B,C]")
        if torch.any(context_mask.sum(dim=1) < 2):
            raise RuntimeError("Patient context has fewer than two channels")
        z = self.phi(context_h)
        valid = context_mask.to(z.dtype).unsqueeze(-1)
        count = valid.sum(dim=1).clamp_min(1.0)
        mu = (z * valid).sum(dim=1) / count
        variance = (((z - mu[:, None, :]) * valid) ** 2).sum(dim=1) / count
        patient = self.rho(torch.cat((mu, torch.sqrt(variance)), dim=-1))
        return torch.tanh(self.hyper(patient))

    def forward(self, target_h: torch.Tensor, context_h: torch.Tensor, context_mask: torch.Tensor,
                frozen_classifier: nn.Module):
        if target_h.shape[0] != context_h.shape[0] or target_h.shape[-1] != self.dim:
            raise ValueError("Target/context batch or R4 dimension mismatch")
        a = self.coefficients(context_h, context_mask)
        u, v = F.normalize(self.U, dim=0), F.normalize(self.V, dim=0)
        projection = target_h @ v
        correction = (projection * a[:, None, :]) @ u.T
        adapted = target_h + self.alpha * correction
        logits = frozen_classifier(adapted).squeeze(-1)
        return logits, a, adapted


def assert_identity(adapter: FrozenR4Adapter, classifier: nn.Module, h: torch.Tensor, mask: torch.Tensor):
    adapter.eval()
    classifier.eval()
    with torch.no_grad():
        logits, a, adapted = adapter(h, h, mask, classifier)
        reference = classifier(h).squeeze(-1)
    error = torch.max(torch.abs(logits[mask] - reference[mask])).item()
    if error > 1e-6 or not torch.equal(h, adapted) or not torch.equal(a, torch.zeros_like(a)):
        raise RuntimeError(f"Zero-init source identity failed: {error}")
    return error
