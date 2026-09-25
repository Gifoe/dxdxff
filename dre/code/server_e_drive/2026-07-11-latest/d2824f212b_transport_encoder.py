from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import nn

from .unbalanced_ot import UnbalancedSinkhornTransport


TRANSPORT_DESCRIPTOR_DIM = 10


@dataclass(frozen=True)
class AdjacentTransportOutput:
    plans: torch.Tensor
    costs: torch.Tensor
    descriptors: torch.Tensor
    pair_mask: torch.Tensor


def _membership_js(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    source_distribution = source.transpose(-1, -2)
    target_distribution = target.transpose(-1, -2)
    source_distribution = source_distribution / source_distribution.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    target_distribution = target_distribution / target_distribution.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    p = source_distribution.unsqueeze(-2)
    q = target_distribution.unsqueeze(-3)
    midpoint = 0.5 * (p + q)
    return 0.5 * (
        (p * (torch.log(p.clamp_min(1e-8)) - torch.log(midpoint.clamp_min(1e-8)))).sum(dim=-1)
        + (q * (torch.log(q.clamp_min(1e-8)) - torch.log(midpoint.clamp_min(1e-8)))).sum(dim=-1)
    )


def transport_descriptors(
    plan: torch.Tensor,
    source_mass: torch.Tensor,
    target_mass: torch.Tensor,
    cost: torch.Tensor,
) -> torch.Tensor:
    transported = plan.sum(dim=(-1, -2))
    energy = (plan * cost).sum(dim=(-1, -2)) / transported.clamp_min(1e-8)
    diagonal = torch.diagonal(plan, dim1=-2, dim2=-1).sum(dim=-1)
    diagonal_ratio = diagonal / transported.clamp_min(1e-8)
    off_diagonal_ratio = (transported - diagonal) / transported.clamp_min(1e-8)
    row = plan.sum(dim=-1)
    column = plan.sum(dim=-2)
    attenuation = torch.relu(source_mass - row).sum(dim=-1)
    emergence = torch.relu(target_mass - column).sum(dim=-1)
    row_distribution = plan / row.unsqueeze(-1).clamp_min(1e-8)
    column_distribution = plan / column.unsqueeze(-2).clamp_min(1e-8)
    split_entropy = -(row_distribution * torch.log(row_distribution.clamp_min(1e-8))).sum(dim=-1).mean(dim=-1) / max(math.log(max(plan.shape[-1], 2)), 1e-8)
    merge_entropy = -(column_distribution * torch.log(column_distribution.clamp_min(1e-8))).sum(dim=-2).mean(dim=-1) / max(math.log(max(plan.shape[-2], 2)), 1e-8)
    source_entropy = -(source_mass / source_mass.sum(dim=-1, keepdim=True).clamp_min(1e-8) * torch.log((source_mass / source_mass.sum(dim=-1, keepdim=True).clamp_min(1e-8)).clamp_min(1e-8))).sum(dim=-1)
    target_entropy = -(target_mass / target_mass.sum(dim=-1, keepdim=True).clamp_min(1e-8) * torch.log((target_mass / target_mass.sum(dim=-1, keepdim=True).clamp_min(1e-8)).clamp_min(1e-8))).sum(dim=-1)
    focality_expansion = target_entropy - source_entropy
    mass_change = target_mass.sum(dim=-1) - source_mass.sum(dim=-1)
    return torch.stack([energy, transported, attenuation, emergence, split_entropy, merge_entropy, diagonal_ratio, off_diagonal_ratio, focality_expansion, mass_change], dim=-1)


def adjacent_core_transport(
    core_states: torch.Tensor,
    core_masses: torch.Tensor,
    responsibilities: torch.Tensor,
    window_mask: torch.Tensor,
    solver: UnbalancedSinkhornTransport,
    *,
    representation_weight: float = 1.0,
    membership_weight: float = 0.5,
    identity_weight: float = 0.1,
    mass_weight: float = 0.05,
) -> AdjacentTransportOutput:
    batch, seizures, windows, cores, dim = core_states.shape
    if windows < 2:
        empty_plan = core_states.new_zeros((batch, seizures, 0, cores, cores))
        return AdjacentTransportOutput(empty_plan, core_states.new_zeros((batch, seizures, 0)), core_states.new_zeros((batch, seizures, 0, TRANSPORT_DESCRIPTOR_DIM)), window_mask[:, :, :0])
    source_state = core_states[:, :, :-1]
    target_state = core_states[:, :, 1:]
    source_norm = functional.normalize(source_state, dim=-1, eps=1e-8)
    target_norm = functional.normalize(target_state, dim=-1, eps=1e-8)
    representation = 1.0 - torch.einsum("bswkd,bswld->bswkl", source_norm, target_norm)
    core_responsibility = responsibilities[..., :-1]
    membership = _membership_js(core_responsibility[:, :, :-1], core_responsibility[:, :, 1:])
    identity = 1.0 - torch.eye(cores, device=core_states.device, dtype=core_states.dtype).view(1, 1, 1, cores, cores)
    source_mass = core_masses[:, :, :-1]
    target_mass = core_masses[:, :, 1:]
    log_mass = torch.abs(torch.log(source_mass.clamp_min(1e-8)).unsqueeze(-1) - torch.log(target_mass.clamp_min(1e-8)).unsqueeze(-2))
    cost = representation_weight * representation + membership_weight * membership + identity_weight * identity + mass_weight * log_mass
    pair_mask = window_mask[:, :, :-1] & window_mask[:, :, 1:]
    flat_count = batch * seizures * (windows - 1)
    flat_source = source_mass.reshape(flat_count, cores)
    flat_target = target_mass.reshape(flat_count, cores)
    flat_cost = cost.reshape(flat_count, cores, cores)
    core_mask = pair_mask.reshape(flat_count, 1).expand(flat_count, cores)
    safe_core_mask = core_mask.clone()
    invalid_pairs = ~pair_mask.reshape(flat_count)
    if invalid_pairs.any():
        safe_core_mask[invalid_pairs, 0] = True
    result = solver(flat_source, flat_target, flat_cost, source_mask=safe_core_mask, target_mask=safe_core_mask)
    plans = result.plan.reshape(batch, seizures, windows - 1, cores, cores) * pair_mask[..., None, None].to(core_states.dtype)
    costs = result.cost.reshape(batch, seizures, windows - 1) * pair_mask.to(core_states.dtype)
    descriptors = transport_descriptors(plans, source_mass, target_mass, cost) * pair_mask.unsqueeze(-1).to(core_states.dtype)
    return AdjacentTransportOutput(plans, costs, descriptors, pair_mask)


class TemporalCoreTransportEncoder(nn.Module):
    def __init__(self, model_dim: int, num_cores: int, focality_dim: int, *, graph_mode: bool) -> None:
        super().__init__()
        self.model_dim = int(model_dim)
        self.num_cores = int(num_cores)
        self.graph_mode = bool(graph_mode)
        input_dim = self.model_dim + self.num_cores + int(focality_dim) + TRANSPORT_DESCRIPTOR_DIM
        self.input_projection = nn.Linear(input_dim, self.model_dim)
        self.temporal_gru = nn.GRU(self.model_dim, self.model_dim, batch_first=True)
        self.graph_gru = nn.GRUCell(self.model_dim, self.model_dim) if self.graph_mode else None

    def forward(
        self,
        core_states: torch.Tensor,
        core_masses: torch.Tensor,
        focality: torch.Tensor,
        transport: AdjacentTransportOutput,
        window_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        states = core_states
        if self.graph_gru is not None and states.shape[2] > 1:
            updated = [states[:, :, 0]]
            for window in range(1, states.shape[2]):
                plan = transport.plans[:, :, window - 1]
                previous = updated[-1]
                incoming = torch.einsum("bskl,bskd->bsld", plan, previous)
                incoming = incoming / plan.sum(dim=-2).unsqueeze(-1).clamp_min(1e-8)
                current = states[:, :, window]
                flat = self.graph_gru(incoming.reshape(-1, self.model_dim), current.reshape(-1, self.model_dim))
                updated.append(flat.reshape_as(current))
            states = torch.stack(updated, dim=2)
        core_summary = (states * core_masses.unsqueeze(-1)).sum(dim=-2) / core_masses.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        transport_at_window = torch.zeros((*core_summary.shape[:-1], TRANSPORT_DESCRIPTOR_DIM), device=core_summary.device, dtype=core_summary.dtype)
        if transport.descriptors.shape[2] > 0:
            transport_at_window[:, :, 1:] = transport.descriptors
        temporal_input = self.input_projection(torch.cat([core_summary, core_masses, focality, transport_at_window], dim=-1))
        batch, seizures, windows, dim = temporal_input.shape
        output, _ = self.temporal_gru(temporal_input.reshape(batch * seizures, windows, dim))
        lengths = window_mask.sum(dim=-1).clamp_min(1).reshape(-1)
        last = output[torch.arange(batch * seizures, device=output.device), lengths - 1]
        seizure_embeddings = last.reshape(batch, seizures, dim) * window_mask.any(dim=-1).unsqueeze(-1).to(output.dtype)
        return seizure_embeddings, states


__all__ = [
    "AdjacentTransportOutput",
    "TRANSPORT_DESCRIPTOR_DIM",
    "TemporalCoreTransportEncoder",
    "adjacent_core_transport",
    "transport_descriptors",
]
