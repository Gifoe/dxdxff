"""Label-blind SCOPE-v2 boundary and cardinality heads."""
from __future__ import annotations

import torch
from torch import nn


def _rank_percentile(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    result = torch.zeros_like(logits)
    for row in range(logits.shape[0]):
        indices = torch.nonzero(mask[row], as_tuple=False).squeeze(1)
        if indices.numel() > 1:
            order = torch.argsort(logits[row, indices].detach())
            result[row, indices[order]] = torch.arange(indices.numel(), device=logits.device, dtype=logits.dtype) / float(indices.numel() - 1)
    return result


class P2ScopeV2BoundaryReranker(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int = 32, max_residual: float = 0.10) -> None:
        super().__init__()
        self.max_residual = float(max_residual)
        self.embedding = nn.Sequential(nn.LayerNorm(embedding_dim), nn.Linear(embedding_dim, hidden_dim), nn.GELU())
        self.scalar = nn.Sequential(nn.LayerNorm(7), nn.Linear(7, 16), nn.GELU())
        self.head = nn.Sequential(nn.Linear(hidden_dim + 16, hidden_dim), nn.GELU(), nn.Dropout(0.10), nn.Linear(hidden_dim, 1), nn.Tanh())
        nn.init.zeros_(self.head[-2].weight); nn.init.zeros_(self.head[-2].bias)

    def forward(self, embedding: torch.Tensor, direct_logit: torch.Tensor, q10: torch.Tensor, temporal: torch.Tensor, onset: torch.Tensor, spread: torch.Tensor, valid_seizures: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        rank = _rank_percentile(direct_logit, mask)
        count = valid_seizures.to(direct_logit.dtype) / valid_seizures.to(direct_logit.dtype).amax(dim=1, keepdim=True).clamp_min(1.0)
        scalars = torch.stack((direct_logit, rank, q10, temporal, onset, spread, count), dim=-1)
        raw = self.head(torch.cat((self.embedding(embedding), self.scalar(scalars)), dim=-1)).squeeze(-1)
        delta = (self.max_residual * raw).masked_fill(~mask, 0.0)
        return {"scope_boundary_delta": delta, "patient_relative_direct_rank": rank}


class P2ScopeV2CardinalityHead(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.mean_projection = nn.Sequential(nn.LayerNorm(embedding_dim), nn.Linear(embedding_dim, 8), nn.GELU())
        self.std_projection = nn.Sequential(nn.LayerNorm(embedding_dim), nn.Linear(embedding_dim, 8), nn.GELU())
        # 16 projected embedding features plus 16 valid-channel scalar features.
        self.net = nn.Sequential(nn.LayerNorm(32), nn.Linear(32, 32), nn.GELU(), nn.Dropout(0.10), nn.Linear(32, 16), nn.GELU())
        self.out = nn.Linear(16, 2)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, embedding: torch.Tensor, logits: torch.Tensor, q10: torch.Tensor, temporal: torch.Tensor, onset: torch.Tensor, spread: torch.Tensor, seizure_mask: torch.Tensor, channel_mask: torch.Tensor, prior_mu: float) -> dict[str, torch.Tensor]:
        rows = []
        for i in range(logits.shape[0]):
            valid = channel_mask[i].bool()
            if not bool(valid.any()):
                raise RuntimeError("P2-SCOPE-v2 cardinality head received an empty patient")
            values = logits[i, valid]
            probability = torch.sigmoid(values).clamp(1e-6, 1 - 1e-6)
            entropy = -(probability * torch.log(probability) + (1 - probability) * torch.log(1 - probability))
            quant = torch.quantile(values, torch.tensor([.1,.25,.5,.75,.9], device=values.device))
            emb = embedding[i, valid]
            emb_mean = self.mean_projection(emb.mean(dim=0))
            emb_std = self.std_projection(emb.std(dim=0, unbiased=False))
            scalar = torch.stack((
                values.mean(), values.std(unbiased=False), *quant, quant[3] - quant[1],
                entropy.mean(), entropy.std(unbiased=False),
                torch.log1p(valid.sum().to(values.dtype)), torch.log1p(seizure_mask[i].sum().to(values.dtype)),
                q10[i, valid].mean(), temporal[i, valid].mean(), onset[i, valid].mean(), spread[i, valid].mean(),
            ))
            rows.append(torch.cat((emb_mean, emb_std, scalar), dim=0))
        raw = self.out(self.net(torch.stack(rows, dim=0)))
        prior = torch.tensor(float(prior_mu), device=logits.device, dtype=logits.dtype).clamp(1e-4, 1-1e-4)
        mu_delta = .50 * torch.tanh(raw[:, 0]); mu = torch.sigmoid(torch.logit(prior) + mu_delta)
        kappa = 2. + 28. * torch.sigmoid(raw[:, 1])
        return {"scope_count_prior_mu0": torch.full_like(mu, prior), "scope_count_mu_delta": mu_delta, "scope_count_predicted_mu": mu, "scope_count_predicted_kappa": kappa, "scope_count_alpha": mu*kappa, "scope_count_beta": (1-mu)*kappa, "scope_expected_ez_fraction": mu, "scope_expected_ez_count": mu * channel_mask.sum(dim=1).to(mu.dtype)}
