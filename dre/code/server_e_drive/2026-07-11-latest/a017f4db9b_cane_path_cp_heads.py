"""Label-blind bounded residual heads for CANE-PATH-CP."""

from __future__ import annotations

from typing import Any, Iterable

import torch
import torch.nn.functional as F
from torch import nn

from .cane_set_heads import (
    CleanNEZPrototypeAnchor as _CleanNEZPrototypeAnchor,
    MultiSeizureNEZEvidenceResidual as _MultiSeizureNEZEvidenceResidual,
    initialize_clean_nez_prototypes as _initialize_clean_nez_prototypes,
)


class CleanNEZPrototypeAnchor(_CleanNEZPrototypeAnchor):
    """CANE-PATH-CP anchor with the formal 0.20 residual bound."""

    def __init__(
        self,
        input_dim: int,
        projection_dim: int = 16,
        num_prototypes: int = 4,
        temperature: float = 0.10,
        max_residual: float = 0.20,
        prototype_similarity_margin: float = 0.50,
        initial_gate_logit: float = -8.0,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            projection_dim=projection_dim,
            num_prototypes=num_prototypes,
            temperature=temperature,
            max_residual=max_residual,
            prototype_similarity_margin=prototype_similarity_margin,
            initial_gate_logit=initial_gate_logit,
        )

    def forward(self, contextual_channel_embedding: torch.Tensor, channel_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        output = super().forward(contextual_channel_embedding, channel_mask)
        valid = channel_mask.bool()
        output["u_anchor"] = torch.tanh(
            F.softplus(self.anchor_scale_raw) * output["anchor_nez_evidence"] + self.anchor_bias
        ).masked_fill(~valid, 0.0)
        return output


class MultiSeizureNEZEvidenceResidual(_MultiSeizureNEZEvidenceResidual):
    """CANE-PATH-CP per-seizure evidence residual with a 0.20 bound."""

    def __init__(
        self, input_dim: int, max_residual: float = 0.20,
        p21_hidden_dim: int = 12, enable_p21_head: bool = True,
    ) -> None:
        super().__init__(input_dim=input_dim, max_residual=max_residual)
        self.p21_evidence_head = None
        if enable_p21_head:
            self.p21_evidence_head = nn.Sequential(
                nn.Linear(5, int(p21_hidden_dim)), nn.GELU(), nn.Dropout(0.10),
                nn.Linear(int(p21_hidden_dim), 1), nn.Tanh()
            )
            nn.init.zeros_(self.p21_evidence_head[-2].weight)
            nn.init.zeros_(self.p21_evidence_head[-2].bias)

    def forward_p21(
        self,
        seizure_channel_embedding: torch.Tensor,
        seizure_mask: torch.Tensor,
        seizure_channel_mask: torch.Tensor,
        *,
        use_q10: bool,
    ) -> dict[str, torch.Tensor]:
        legacy = super().forward(seizure_channel_embedding, seizure_mask, seizure_channel_mask)
        if self.p21_evidence_head is None:
            raise RuntimeError("P2.1 seizure evidence head was not enabled at construction")
        valid = seizure_mask.bool()[:, :, None] & seizure_channel_mask.bool()
        probability = legacy["seizure_nez_probability"]
        count = valid.sum(dim=1)
        has_valid = count > 0
        q10 = torch.zeros_like(legacy["seizure_nez_probability_mean"])
        for patient_idx in range(probability.shape[0]):
            for channel_idx in range(probability.shape[2]):
                selected = probability[patient_idx, :, channel_idx][valid[patient_idx, :, channel_idx]]
                if selected.numel():
                    q10[patient_idx, channel_idx] = torch.quantile(selected, 0.10)
        normalized_count = (count.to(probability.dtype) / 3.0).clamp(0.0, 1.0)
        if use_q10:
            features = torch.stack((
                legacy["seizure_nez_probability_mean"], q10,
                legacy["seizure_nez_probability_std"], legacy["seizure_nez_agreement"], normalized_count,
            ), dim=-1)
            evidence = self.p21_evidence_head(features).squeeze(-1)
        else:
            old_features = torch.stack((
                legacy["seizure_nez_probability_mean"], legacy["seizure_nez_probability_std"],
                legacy["seizure_nez_agreement"],
            ), dim=-1)
            evidence = torch.tanh(self.evidence_residual_mlp(old_features).squeeze(-1))
        legacy.update({
            "seizure_nez_probability_q10": q10.masked_fill(~has_valid, 0.0),
            "normalized_valid_seizure_count": normalized_count.masked_fill(~has_valid, 0.0),
            "u_seizure": evidence.masked_fill(~has_valid, 0.0),
            "seizure_evidence_valid": has_valid,
        })
        return legacy


class CausalPropagationResidual(nn.Module):
    """Bounded residual from six offline ridge-VAR propagation proxies."""

    def __init__(self, max_residual: float = 0.15, enable_p21_head: bool = True) -> None:
        super().__init__()
        if not 0.0 <= float(max_residual) <= 1.0:
            raise ValueError("causal max_residual must be in [0,1]")
        self.max_residual = float(max_residual)
        self.normalizer = nn.LayerNorm(6)
        self.network = nn.Sequential(nn.Linear(6, 8), nn.GELU(), nn.Dropout(0.10), nn.Linear(8, 1))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)
        self.p21_evidence_network = None
        if enable_p21_head:
            self.p21_evidence_network = nn.Sequential(
                nn.LayerNorm(6), nn.Linear(6, 8), nn.GELU(), nn.Dropout(0.10), nn.Linear(8, 1), nn.Tanh()
            )
            nn.init.zeros_(self.p21_evidence_network[-2].weight)
            nn.init.zeros_(self.p21_evidence_network[-2].bias)

    def forward(
        self,
        cp_features: torch.Tensor,
        cp_feature_valid: torch.Tensor,
        channel_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if cp_features.ndim != 3 or cp_features.shape[-1] != 6:
            raise ValueError(f"causal propagation features must be [B,C,6], got {tuple(cp_features.shape)}")
        valid = cp_feature_valid.bool() & channel_mask.bool()
        normalized = self.normalizer(cp_features)
        raw = self.network(normalized).squeeze(-1)
        residual = self.max_residual * torch.tanh(raw)
        residual = residual.masked_fill(~valid, 0.0)
        raw = raw.masked_fill(~valid, 0.0)
        normalized = normalized * valid.to(normalized.dtype).unsqueeze(-1)
        saturation = (
            (residual[valid].abs() >= 0.95 * self.max_residual).to(residual.dtype).mean()
            if torch.any(valid) and self.max_residual > 0.0
            else residual.sum() * 0.0
        )
        return {
            "causal_residual": residual,
            "causal_raw_output": raw,
            "causal_feature_valid": valid,
            "causal_feature_norm": normalized,
            "causal_residual_saturation_rate": saturation,
        }

    def forward_p21(
        self,
        cp_features: torch.Tensor,
        cp_feature_valid: torch.Tensor,
        channel_mask: torch.Tensor,
        cp_valid_window_fraction: torch.Tensor,
        cp_valid_seizure_count: torch.Tensor,
        cp_mean_var_stability: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if self.p21_evidence_network is None:
            raise RuntimeError("P2.1 causal evidence head was not enabled at construction")
        valid = cp_feature_valid.bool() & channel_mask.bool()
        quality = compute_causal_quality_gate(
            cp_valid_window_fraction, cp_valid_seizure_count, cp_mean_var_stability, valid
        )
        raw = self.p21_evidence_network(cp_features).squeeze(-1).masked_fill(~valid, 0.0)
        evidence = (quality["q_cp"] * raw).masked_fill(~valid, 0.0)
        return {
            "raw_u_causal": raw,
            "u_causal": evidence,
            "causal_feature_valid": valid,
            "causal_feature_norm": self.normalizer(cp_features) * valid.to(cp_features.dtype).unsqueeze(-1),
            **quality,
        }


def initialize_clean_nez_prototypes(
    model: nn.Module,
    fit_loader: Iterable[dict[str, Any]],
    device: torch.device,
    model_seed: int,
    max_samples: int = 20_000,
) -> dict[str, Any]:
    """Fit-only deterministic KMeans initialization with explicit source audit."""
    dataset = getattr(fit_loader, "dataset", None)
    examples = getattr(dataset, "patient_examples", [])
    fit_subjects = sorted({str(item.get("subject_id")) for item in examples if item.get("subject_id") is not None})
    audit = _initialize_clean_nez_prototypes(model, fit_loader, device, model_seed, max_samples=max_samples)
    audit.update({
        "n_clean_nez_embeddings": int(audit["n_clean_nez_embeddings_before_sampling"]),
        "n_sampled": int(audit["n_clean_nez_embeddings_used"]),
        "n_clusters": int(audit["n_prototypes"]),
        "fit_subjects": fit_subjects,
        "validation_used": False,
        "test_used": False,
    })
    return audit


__all__ = [
    "CausalPropagationResidual", "CleanNEZPrototypeAnchor",
    "MultiSeizureNEZEvidenceResidual", "initialize_clean_nez_prototypes",
]
