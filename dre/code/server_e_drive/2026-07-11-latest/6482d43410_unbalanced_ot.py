from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class UOTResult:
    plan: torch.Tensor
    cost: torch.Tensor
    transport_energy: torch.Tensor
    unmatched_source_mass: torch.Tensor
    unmatched_target_mass: torch.Tensor
    iterations: int
    converged: bool


def _generalized_kl(observed: torch.Tensor, reference: torch.Tensor, mask: torch.Tensor, eps: float) -> torch.Tensor:
    observed_safe = observed.clamp_min(eps)
    reference_safe = reference.clamp_min(eps)
    value = observed_safe * (torch.log(observed_safe) - torch.log(reference_safe)) - observed_safe + reference_safe
    return (value * mask.to(value.dtype)).sum(dim=-1)


class UnbalancedSinkhornTransport(nn.Module):
    """Entropic KL-relaxed unbalanced transport with log-domain updates."""

    def __init__(self, epsilon: float = 0.1, tau: float = 1.0, iterations: int = 20, tolerance: float = 1e-5) -> None:
        super().__init__()
        if epsilon <= 0 or tau <= 0:
            raise ValueError("epsilon and tau must be positive.")
        self.epsilon = float(epsilon)
        self.tau = float(tau)
        self.iterations = int(iterations)
        self.tolerance = float(tolerance)

    def forward(
        self,
        source_mass: torch.Tensor,
        target_mass: torch.Tensor,
        cost: torch.Tensor,
        *,
        source_mask: torch.Tensor | None = None,
        target_mask: torch.Tensor | None = None,
    ) -> UOTResult:
        if source_mass.ndim != 2 or target_mass.ndim != 2 or cost.shape != (source_mass.shape[0], source_mass.shape[1], target_mass.shape[1]):
            raise ValueError("UOT expects source [B,K], target [B,L], and cost [B,K,L].")
        source_mask = torch.ones_like(source_mass, dtype=torch.bool) if source_mask is None else source_mask.bool()
        target_mask = torch.ones_like(target_mass, dtype=torch.bool) if target_mask is None else target_mask.bool()
        if not source_mask.any(dim=1).all() or not target_mask.any(dim=1).all():
            raise ValueError("Every UOT batch item needs at least one valid source and target core.")
        eps = max(torch.finfo(cost.dtype).eps, 1e-12)
        source = torch.where(source_mask, source_mass.clamp_min(0.0), torch.zeros_like(source_mass))
        target = torch.where(target_mask, target_mass.clamp_min(0.0), torch.zeros_like(target_mass))
        pair_mask = source_mask.unsqueeze(-1) & target_mask.unsqueeze(-2)
        negative_large = -torch.finfo(cost.dtype).max / 16.0
        log_kernel = torch.where(pair_mask, -cost / self.epsilon, torch.full_like(cost, negative_large))
        log_source = torch.log(source.clamp_min(eps))
        log_target = torch.log(target.clamp_min(eps))
        log_u = torch.zeros_like(source)
        log_v = torch.zeros_like(target)
        relaxation = self.tau / (self.tau + self.epsilon)
        converged = False
        completed = 0
        for iteration in range(self.iterations):
            previous_u = log_u
            source_lse = torch.logsumexp(log_kernel + log_v.unsqueeze(-2), dim=-1)
            log_u = relaxation * (log_source - source_lse)
            log_u = torch.where(source_mask, log_u, torch.zeros_like(log_u))
            target_lse = torch.logsumexp(log_kernel + log_u.unsqueeze(-1), dim=-2)
            log_v = relaxation * (log_target - target_lse)
            log_v = torch.where(target_mask, log_v, torch.zeros_like(log_v))
            completed = iteration + 1
            if self.tolerance > 0 and float((log_u - previous_u).detach().abs().max().cpu()) < self.tolerance:
                converged = True
                break
        log_plan = log_kernel + log_u.unsqueeze(-1) + log_v.unsqueeze(-2)
        plan = torch.exp(log_plan) * pair_mask.to(cost.dtype)
        row_mass = plan.sum(dim=-1)
        column_mass = plan.sum(dim=-2)
        transport_energy = (plan * cost).sum(dim=(-1, -2))
        marginal_penalty = self.tau * (
            _generalized_kl(row_mass, source, source_mask, eps)
            + _generalized_kl(column_mass, target, target_mask, eps)
        )
        reported_cost = transport_energy + marginal_penalty
        unmatched_source = torch.relu(source - row_mass).sum(dim=-1)
        unmatched_target = torch.relu(target - column_mass).sum(dim=-1)
        return UOTResult(plan, reported_cost, transport_energy, unmatched_source, unmatched_target, completed, converged)


__all__ = ["UOTResult", "UnbalancedSinkhornTransport"]
