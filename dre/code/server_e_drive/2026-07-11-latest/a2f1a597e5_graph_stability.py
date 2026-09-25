from __future__ import annotations

from itertools import combinations

import torch


def _rank(values: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(values, stable=True)
    ranks = torch.empty_like(values, dtype=torch.float32)
    ranks[order] = torch.arange(values.numel(), device=values.device, dtype=torch.float32)
    return ranks


def _correlation(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    x, y = _rank(left), _rank(right)
    x, y = x - x.mean(), y - y.mean()
    denominator = x.square().sum().sqrt() * y.square().sum().sqrt()
    return (x * y).sum() / denominator.clamp_min(1e-6)


def compute_graph_stability(
    risk_membership: torch.Tensor,
    adjacency: torch.Tensor,
    seizure_mask: torch.Tensor,
    seizure_channel_mask: torch.Tensor,
    graph_valid: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Patient-level risk Spearman and phase edge cosine stability."""
    bsz = risk_membership.shape[0]
    output = risk_membership.new_zeros((bsz, 7))
    for patient_idx in range(bsz):
        seizures = torch.where(seizure_mask[patient_idx].bool())[0].tolist()
        risk_scores: list[torch.Tensor] = []
        edge_scores: list[torch.Tensor] = []
        for left_idx, right_idx in combinations(seizures, 2):
            shared = seizure_channel_mask[patient_idx, left_idx].bool() & seizure_channel_mask[patient_idx, right_idx].bool()
            indices = torch.where(shared)[0]
            if indices.numel() >= 4:
                risk_scores.append(_correlation(risk_membership[patient_idx, left_idx, indices], risk_membership[patient_idx, right_idx, indices]))
            for phase_idx in range(adjacency.shape[2]):
                if not bool(graph_valid[patient_idx, left_idx, phase_idx] and graph_valid[patient_idx, right_idx, phase_idx]) or indices.numel() < 4:
                    continue
                left = adjacency[patient_idx, left_idx, phase_idx][indices][:, indices]
                right = adjacency[patient_idx, right_idx, phase_idx][indices][:, indices]
                upper = torch.triu(torch.ones_like(left, dtype=torch.bool), diagonal=1)
                left, right = left[upper], right[upper]
                denominator = left.norm() * right.norm()
                if float(denominator.detach()) > 1e-8:
                    edge_scores.append(torch.dot(left, right) / denominator)
        if risk_scores:
            values = torch.stack(risk_scores)
            output[patient_idx, 0] = values.mean()
            output[patient_idx, 1] = values.std(unbiased=False) if values.numel() > 1 else 0.0
        if edge_scores:
            values = torch.stack(edge_scores)
            output[patient_idx, 2] = values.mean()
            output[patient_idx, 3] = values.std(unbiased=False) if values.numel() > 1 else 0.0
        output[patient_idx, 4] = torch.log1p(torch.tensor(float(len(risk_scores)), device=output.device))
        output[patient_idx, 5] = torch.log1p(torch.tensor(float(len(edge_scores)), device=output.device))
        output[patient_idx, 6] = float(bool(risk_scores or edge_scores))
    return {
        "graph_stability_vector": output,
        "risk_stability_mean": output[:, 0],
        "risk_stability_std": output[:, 1],
        "edge_stability_mean": output[:, 2],
        "edge_stability_std": output[:, 3],
        "n_valid_risk_pairs_log1p": output[:, 4],
        "n_valid_edge_pairs_log1p": output[:, 5],
        "has_graph_stability": output[:, 6],
    }


__all__ = ["compute_graph_stability"]

