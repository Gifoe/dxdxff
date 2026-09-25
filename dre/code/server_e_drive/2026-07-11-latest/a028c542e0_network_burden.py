from __future__ import annotations

import torch


def compute_network_burden(
    adjacency: torch.Tensor,
    risk_membership: torch.Tensor,
    channel_mask: torch.Tensor,
    phase_mask: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> dict[str, torch.Tensor]:
    """Five phase-wise network burdens using upper-triangular undirected edges."""
    if adjacency.ndim != 5:
        raise ValueError("adjacency must be [B,S,P,C,C]")
    bsz, seizures, phases, channels, _ = adjacency.shape
    valid_node = channel_mask.bool().unsqueeze(2).expand(bsz, seizures, phases, channels).clone()
    valid_node = valid_node & phase_mask.bool()
    upper = torch.triu(torch.ones((channels, channels), dtype=torch.bool, device=adjacency.device), diagonal=1)
    pair = valid_node.unsqueeze(-1) & valid_node.unsqueeze(-2) & upper
    weights = torch.where(pair, torch.nan_to_num(adjacency).clamp_min(0.0), torch.zeros_like(adjacency))
    total_weight = weights.sum(dim=(-2, -1))
    graph_valid = (total_weight > eps) & phase_mask.any(dim=-1)
    risk = risk_membership.unsqueeze(2).expand(bsz, seizures, phases, channels).clamp(0.0, 1.0)
    left, right = risk.unsqueeze(-1), risk.unsqueeze(-2)
    b_edge = (weights * left * right).sum(dim=(-2, -1)) / total_weight.clamp_min(eps)
    cross = left * (1.0 - right) + (1.0 - left) * right
    b_cross = (weights * cross).sum(dim=(-2, -1)) / (2.0 * total_weight).clamp_min(eps)
    e_lap = (weights * (left - right).square()).sum(dim=(-2, -1)) / total_weight.clamp_min(eps)
    masked_risk = risk * valid_node.to(risk.dtype)
    pi = masked_risk / masked_risk.sum(dim=-1, keepdim=True).clamp_min(eps)
    entropy = -(pi * torch.log(pi.clamp_min(eps))).sum(dim=-1)
    n_valid = valid_node.sum(dim=-1).to(risk.dtype)
    entropy = entropy / torch.log(n_valid.clamp_min(2.0))
    strength = weights.sum(dim=-1) + weights.sum(dim=-2)
    b_hub = (strength * risk).sum(dim=-1) / strength.sum(dim=-1).clamp_min(eps)
    features = torch.stack((b_edge, b_cross, e_lap, entropy.clamp(0.0, 1.0), b_hub), dim=-1)
    features = features.masked_fill(~graph_valid.unsqueeze(-1), 0.0)
    return {
        "network_burden": features,
        "B_edge": features[..., 0],
        "B_cross": features[..., 1],
        "E_lap": features[..., 2],
        "H_risk": features[..., 3],
        "B_hub": features[..., 4],
        "graph_valid": graph_valid,
    }


__all__ = ["compute_network_burden"]
