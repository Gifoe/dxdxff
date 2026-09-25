"""Cross-seizure q10 evidence for P23-TRN."""

from __future__ import annotations

import torch
from torch import nn


ALLOWED_SEIZURE_TAIL_QUANTILES = (0.05, 0.10, 0.20)


def validate_seizure_tail_quantile(value: float) -> float:
    """Return a protocol-approved tail quantile without fuzzy matching."""
    quantile = float(value)
    if not any(abs(quantile - allowed) <= 1e-12 for allowed in ALLOWED_SEIZURE_TAIL_QUANTILES):
        raise ValueError(
            "p23_seizure_quantile must be one of "
            f"{list(ALLOWED_SEIZURE_TAIL_QUANTILES)}, got {quantile!r}"
        )
    return quantile


def tail_reliability(valid_seizure_count: torch.Tensor, seizure_agreement: torch.Tensor) -> torch.Tensor:
    """Label-free count/agreement gate used by RTC A2 and later profiles."""
    count = ((valid_seizure_count.to(seizure_agreement.dtype) - 1.0) / 2.0).clamp(0.0, 1.0)
    return (count * torch.sqrt(seizure_agreement.clamp(0.0, 1.0) + 1e-6)).clamp(0.0, 1.0)


class P23CrossSeizureTailEvidence(nn.Module):
    """Cross-seizure evidence with an optional logit-space robust tail."""

    def __init__(
        self, model_dim: int, hidden_dim: int = 12, *, robust_tail: bool = False,
        tau: float = 0.25, feature_mode: str = "auto", quantile: float = 0.10,
    ) -> None:
        super().__init__()
        self.robust_tail = bool(robust_tail)
        self.tau = float(tau)
        self.quantile = validate_seizure_tail_quantile(quantile)
        self.feature_mode = "rtc6" if feature_mode == "auto" and self.robust_tail else ("legacy5" if feature_mode == "auto" else str(feature_mode))
        if self.feature_mode not in {"legacy5", "rtc6", "atc7"}:
            raise ValueError(f"Unknown seizure-tail feature mode: {self.feature_mode}")
        scorer_hidden = max(1, model_dim // 2)
        self.scorer = nn.Sequential(nn.LayerNorm(model_dim), nn.Linear(model_dim, scorer_hidden), nn.GELU(), nn.Dropout(0.10), nn.Linear(scorer_hidden, 1))
        evidence_dim = {"legacy5": 5, "rtc6": 6, "atc7": 7}[self.feature_mode]
        self.evidence = nn.Sequential(nn.Linear(evidence_dim, hidden_dim), nn.GELU(), nn.Dropout(0.10), nn.Linear(hidden_dim, 1), nn.Tanh())
        nn.init.zeros_(self.scorer[-1].weight); nn.init.zeros_(self.scorer[-1].bias)
        nn.init.zeros_(self.evidence[-2].weight); nn.init.zeros_(self.evidence[-2].bias)

    def forward(self, seizure_embedding: torch.Tensor, seizure_mask: torch.Tensor, seizure_channel_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        valid = seizure_mask.bool().unsqueeze(-1) & seizure_channel_mask.bool()
        logits = self.scorer(seizure_embedding).squeeze(-1).masked_fill(~valid, 0.0)
        probability = torch.sigmoid(logits).masked_fill(~valid, 0.0)
        count = valid.sum(dim=1)
        has_valid = count > 0
        mean = probability.sum(dim=1) / count.clamp_min(1).to(probability.dtype)
        centered = (probability - mean.unsqueeze(1)).masked_fill(~valid, 0.0)
        variance = centered.square().sum(dim=1) / count.clamp_min(1).to(probability.dtype)
        # A single valid seizure has exactly zero variance. sqrt(0) has an
        # infinite derivative and poisons the optimizer even though the
        # forward value is finite. Keep the same stabilized contract as the
        # established CANE multi-seizure head.
        std = variance.clamp_min(1e-8).sqrt()
        tail_quantile = torch.zeros_like(mean)
        for batch_idx in range(probability.shape[0]):
            for channel_idx in range(probability.shape[2]):
                values = probability[batch_idx, :, channel_idx][valid[batch_idx, :, channel_idx]]
                if values.numel():
                    tail_quantile[batch_idx, channel_idx] = torch.quantile(values, self.quantile)
        agreement = (1.0 - 4.0 * variance).clamp(0.0, 1.0)
        normalized_count = (count.to(probability.dtype) / 3.0).clamp(0.0, 1.0)
        # Compute the soft minimum in float32.  This prevents half-precision
        # overflow while retaining gradients to the seizure-level scorer.
        tau = max(self.tau, 1e-6)
        logit32 = logits.float()
        neg_scaled = (-logit32 / tau).masked_fill(~valid, float("-inf"))
        soft_tail = -tau * (torch.logsumexp(neg_scaled, dim=1) - count.clamp_min(1).float().log())
        soft_tail = torch.where(has_valid, soft_tail, torch.zeros_like(soft_tail))
        mean_logit = (logit32 * valid.to(logit32.dtype)).sum(dim=1) / count.clamp_min(1).to(logit32.dtype)
        alpha = ((count.to(logit32.dtype) - 1.0) / 2.0).clamp(0.0, 1.0)
        robust_logit = alpha * soft_tail + (1.0 - alpha) * mean_logit
        robust_logit = robust_logit.masked_fill(~has_valid, 0.0).to(logits.dtype)
        tail_gap = (mean_logit.to(logits.dtype) - robust_logit).masked_fill(~has_valid, 0.0)
        robust_probability = torch.sigmoid(robust_logit).masked_fill(~has_valid, 0.0)
        bounded_tail_gap = torch.tanh(tail_gap / 2.0).masked_fill(~has_valid, 0.0)
        if self.feature_mode == "atc7":
            features = torch.stack((
                mean, std, tail_quantile, robust_probability, bounded_tail_gap,
                agreement, normalized_count,
            ), dim=-1)
        elif self.feature_mode == "rtc6":
            features = torch.stack((
                torch.sigmoid(mean_logit).to(probability.dtype), std,
                robust_probability, tail_gap, agreement, normalized_count,
            ), dim=-1)
        else:
            features = torch.stack((mean, std, tail_quantile, agreement, normalized_count), dim=-1)
        u_seizure = self.evidence(features).squeeze(-1).masked_fill(~has_valid, 0.0)
        result = {
            "seizure_nez_logit": logits, "seizure_nez_probability": probability,
            "seizure_nez_probability_mean": mean.masked_fill(~has_valid, 0.0),
            "seizure_nez_probability_std": std.masked_fill(~has_valid, 0.0),
            "seizure_nez_probability_tail_quantile": tail_quantile.masked_fill(~has_valid, 0.0),
            "seizure_tail_quantile": torch.full_like(mean, self.quantile).masked_fill(~has_valid, 0.0),
            "seizure_nez_logit_mean": mean_logit.to(logits.dtype).masked_fill(~has_valid, 0.0),
            "seizure_nez_robust_tail_logit": robust_logit,
            "seizure_nez_robust_tail_probability": robust_probability,
            "seizure_nez_tail_gap": tail_gap,
            "seizure_nez_soft_tail_logit": soft_tail.to(logits.dtype).masked_fill(~has_valid, 0.0),
            "seizure_nez_bounded_tail_gap": bounded_tail_gap,
            "tail_shrinkage_alpha": alpha.to(logits.dtype).masked_fill(~has_valid, 0.0),
            "seizure_agreement": agreement.masked_fill(~has_valid, 0.0),
            "valid_seizure_count": count, "normalized_valid_seizure_count": normalized_count.masked_fill(~has_valid, 0.0),
            "tail_valid": has_valid, "u_seizure": u_seizure,
        }
        # Q10 is the frozen formal model.  Keep its historical ledger keys
        # only for Q10 so Q05/Q20 artifacts cannot be misread as Q10 runs.
        if self.quantile == 0.10:
            result["seizure_nez_probability_q10"] = result["seizure_nez_probability_tail_quantile"]
            result["seizure_nez_probability_raw_q10"] = result["seizure_nez_probability_tail_quantile"]
        return result
