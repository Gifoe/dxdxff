"""Single label-free Beta-Binomial decoder used by every SCOPE-v2 path."""
from __future__ import annotations

import torch


PREDICTED_K_DECISION_RULE = "predicted_k_topk"
FORMAL_PREDICTION_SOURCE = "predicted_k_topk"


def beta_binomial_log_prob(
    k: torch.Tensor | int,
    n: torch.Tensor | int,
    alpha: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    """Return log P(K=k | n, alpha, beta) with the exact Beta-Binomial PMF."""
    device, dtype = alpha.device, alpha.dtype
    k_t = torch.as_tensor(k, device=device, dtype=dtype)
    n_t = torch.as_tensor(n, device=device, dtype=dtype)
    if bool((alpha <= 0).any()) or bool((beta <= 0).any()):
        raise ValueError("Beta-Binomial alpha and beta must be positive")
    if bool((n_t < 0).any()) or bool((n_t != torch.round(n_t)).any()):
        raise ValueError("Beta-Binomial n must be a non-negative integer")
    if bool((k_t < 0).any()) or bool((k_t > n_t).any()) or bool((k_t != torch.round(k_t)).any()):
        raise ValueError("Beta-Binomial k must be an integer in [0, n]")
    result = (
        torch.lgamma(n_t + 1)
        - torch.lgamma(k_t + 1)
        - torch.lgamma(n_t - k_t + 1)
        + torch.lgamma(k_t + alpha)
        + torch.lgamma(n_t - k_t + beta)
        - torch.lgamma(n_t + alpha + beta)
        + torch.lgamma(alpha + beta)
        - torch.lgamma(alpha)
        - torch.lgamma(beta)
    )
    if not bool(torch.isfinite(result).all()):
        raise FloatingPointError("Non-finite Beta-Binomial log probability")
    return result


def predict_beta_binomial_mode(
    n: int,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    *,
    min_k: int,
    max_k: int,
) -> int:
    """Deterministic mode with protocol-level candidate bounds and tie breaks."""
    n = int(n)
    if not (0 <= int(min_k) <= int(max_k) <= n):
        raise ValueError("Invalid Beta-Binomial decoder candidate range")
    candidates = torch.arange(int(min_k), int(max_k) + 1, device=alpha.device, dtype=alpha.dtype)
    log_prob = beta_binomial_log_prob(candidates, n, alpha, beta)
    best = torch.nonzero(torch.isclose(log_prob, log_prob.max(), rtol=0.0, atol=1e-7), as_tuple=False).flatten()
    mean_target = torch.round(torch.as_tensor(float(n), device=alpha.device, dtype=alpha.dtype) * alpha / (alpha + beta))
    distance = (candidates[best] - mean_target).abs()
    tied = best[distance == distance.min()]
    return int(candidates[tied].min().item())


def decode_predicted_k_topk(
    *,
    scope_ez_score: torch.Tensor,
    channel_mask: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    min_k: int = 1,
    max_k: int | None = None,
) -> dict[str, torch.Tensor | str | bool]:
    """Formal decoder. It intentionally accepts no labels or true cardinality."""
    if scope_ez_score.ndim != 2 or channel_mask.shape != scope_ez_score.shape:
        raise ValueError("scope_ez_score and channel_mask must have matching [B, C] shape")
    pred_ez = torch.zeros_like(channel_mask, dtype=torch.bool)
    counts: list[int] = []
    modes: list[int] = []
    means: list[torch.Tensor] = []
    for row in range(channel_mask.shape[0]):
        indices = torch.nonzero(channel_mask[row], as_tuple=False).squeeze(1)
        n_valid = int(indices.numel())
        if n_valid == 0:
            raise RuntimeError("SCOPE decoder received an empty patient")
        row_min = min(int(min_k), n_valid)
        row_max = n_valid if max_k is None else min(int(max_k), n_valid)
        if row_min > row_max:
            raise RuntimeError("SCOPE decoder candidate range is empty")
        k_hat = predict_beta_binomial_mode(n_valid, alpha[row], beta[row], min_k=row_min, max_k=row_max)
        if k_hat:
            ranked = torch.argsort(scope_ez_score[row, indices], descending=True, stable=True)
            pred_ez[row, indices[ranked[:k_hat]]] = True
        counts.append(k_hat)
        modes.append(k_hat)
        means.append(alpha[row] / (alpha[row] + beta[row]))
    count_t = torch.tensor(counts, device=scope_ez_score.device, dtype=torch.long)
    mask = channel_mask.bool()
    return {
        "predicted_ez_mask": pred_ez,
        "predicted_nez_mask": mask & ~pred_ez,
        "scope_predicted_k": count_t,
        "scope_predicted_ez_fraction": count_t.to(scope_ez_score.dtype) / mask.sum(1).clamp_min(1).to(scope_ez_score.dtype),
        "scope_count_prior_mode": torch.tensor(modes, device=scope_ez_score.device, dtype=torch.long),
        "scope_count_prior_mean": torch.stack(means),
        "decision_rule": PREDICTED_K_DECISION_RULE,
        "formal_prediction_source": FORMAL_PREDICTION_SOURCE,
        "true_count_used_for_prediction": False,
    }
