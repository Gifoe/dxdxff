"""Label-blind heads for CANE-Set-NEZ count-free localization."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch import nn


def _masked_mean(values: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    weights = mask.to(dtype=values.dtype)
    while weights.ndim < values.ndim:
        weights = weights.unsqueeze(-1)
    count = weights.sum(dim=dim).clamp_min(1.0)
    return (values * weights).sum(dim=dim) / count


def _masked_mean_std(
    values: torch.Tensor,
    mask: torch.Tensor,
    dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    mean = _masked_mean(values, mask, dim)
    expanded_mean = mean.unsqueeze(dim)
    weights = mask.to(dtype=values.dtype)
    while weights.ndim < values.ndim:
        weights = weights.unsqueeze(-1)
    count = weights.sum(dim=dim).clamp_min(1.0)
    variance = ((values - expanded_mean).square() * weights).sum(dim=dim) / count
    return mean, torch.sqrt(variance.clamp_min(1e-8))


class CleanNEZPrototypeAnchor(nn.Module):
    """Monotonic bounded residual from distance to clean-NEZ prototypes."""

    def __init__(
        self,
        input_dim: int,
        projection_dim: int = 16,
        num_prototypes: int = 4,
        temperature: float = 0.10,
        max_residual: float = 0.30,
        prototype_similarity_margin: float = 0.50,
        initial_gate_logit: float = -8.0,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or projection_dim <= 0 or num_prototypes <= 0:
            raise ValueError("CANE prototype dimensions and count must be positive")
        if temperature <= 0.0:
            raise ValueError("CANE prototype temperature must be positive")
        if not 0.0 <= max_residual <= 1.0:
            raise ValueError("CANE anchor max_residual must be in [0, 1]")
        self.temperature = float(temperature)
        self.max_residual = float(max_residual)
        self.prototype_similarity_margin = float(prototype_similarity_margin)
        self.projector = nn.Sequential(
            nn.Linear(int(input_dim), int(projection_dim)),
            nn.GELU(),
            nn.LayerNorm(int(projection_dim)),
        )
        self.prototypes = nn.Parameter(torch.randn(int(num_prototypes), int(projection_dim)))
        self.anchor_gate_logit = nn.Parameter(torch.tensor(float(initial_gate_logit)))
        self.anchor_scale_raw = nn.Parameter(torch.tensor(float(np.log(np.expm1(1.0)))))
        self.anchor_bias = nn.Parameter(torch.zeros(()))

    @property
    def normalized_prototypes(self) -> torch.Tensor:
        return F.normalize(self.prototypes, dim=-1, eps=1e-8)

    @property
    def anchor_gate(self) -> torch.Tensor:
        return self.max_residual * torch.sigmoid(self.anchor_gate_logit)

    def project(self, contextual_channel_embedding: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.projector(contextual_channel_embedding), dim=-1, eps=1e-8)

    def forward(
        self,
        contextual_channel_embedding: torch.Tensor,
        channel_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        valid = channel_mask.to(device=contextual_channel_embedding.device, dtype=torch.bool)
        projected = self.project(contextual_channel_embedding)
        prototypes = self.normalized_prototypes
        similarity = torch.einsum("bcd,md->bcm", projected, prototypes)
        assignment = torch.softmax(similarity / self.temperature, dim=-1)
        soft_similarity = (assignment * similarity).sum(dim=-1)
        distance = 1.0 - soft_similarity

        weights = valid.to(dtype=distance.dtype)
        count = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean = (distance * weights).sum(dim=1, keepdim=True) / count
        variance = ((distance - mean).square() * weights).sum(dim=1, keepdim=True) / count
        distance_z = (distance - mean) / torch.sqrt(variance + 1e-5)
        distance_z = distance_z.masked_fill(~valid, 0.0)
        evidence = -distance_z
        scale = F.softplus(self.anchor_scale_raw)
        residual = self.anchor_gate * torch.tanh(scale * evidence + self.anchor_bias)
        residual = residual.masked_fill(~valid, 0.0)

        valid_assignment = assignment * valid.to(assignment.dtype).unsqueeze(-1)
        utilization = valid_assignment.sum(dim=(0, 1)) / valid.sum().to(assignment.dtype).clamp_min(1.0)
        return {
            "anchor_residual": residual,
            "anchor_distance": distance.masked_fill(~valid, 0.0),
            "anchor_distance_z": distance_z,
            "anchor_nez_evidence": evidence,
            "prototype_assignment": valid_assignment,
            "prototype_utilization": utilization,
            "normalized_projected_embedding": projected * valid.to(projected.dtype).unsqueeze(-1),
            "normalized_prototypes": prototypes,
            "anchor_gate": self.anchor_gate,
        }


class MultiSeizureNEZEvidenceResidual(nn.Module):
    """Bounded residual from differentiable per-seizure NEZ evidence."""

    def __init__(self, input_dim: int, max_residual: float = 0.25) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("CANE seizure evidence input_dim must be positive")
        if not 0.0 <= max_residual <= 1.0:
            raise ValueError("CANE seizure max_residual must be in [0, 1]")
        hidden = max(int(input_dim) // 2, 8)
        self.max_residual = float(max_residual)
        self.seizure_nez_head = nn.Sequential(
            nn.Linear(int(input_dim), hidden),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(hidden, 1),
        )
        self.evidence_residual_mlp = nn.Sequential(
            nn.Linear(3, 16),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(16, 1),
        )
        nn.init.zeros_(self.evidence_residual_mlp[-1].weight)
        nn.init.zeros_(self.evidence_residual_mlp[-1].bias)

    def forward(
        self,
        seizure_channel_embedding: torch.Tensor,
        seizure_mask: torch.Tensor,
        seizure_channel_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        valid = seizure_mask.to(dtype=torch.bool)[:, :, None] & seizure_channel_mask.to(dtype=torch.bool)
        logits = self.seizure_nez_head(seizure_channel_embedding).squeeze(-1)
        probabilities = torch.sigmoid(logits)
        weights = valid.to(dtype=probabilities.dtype)
        count = weights.sum(dim=1).clamp_min(1.0)
        mean = (probabilities * weights).sum(dim=1) / count
        variance = ((probabilities - mean[:, None, :]).square() * weights).sum(dim=1) / count
        # A single valid seizure has exactly zero variance.  sqrt(0) has an
        # infinite derivative, which can turn a finite zero upstream gradient
        # into NaNs during backpropagation.
        std = torch.sqrt(variance.clamp_min(1e-8))
        p = probabilities.clamp(1e-7, 1.0 - 1e-7)
        entropy = -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p)) / np.log(2.0)
        agreement = 1.0 - (entropy * weights).sum(dim=1) / count
        has_valid = valid.any(dim=1)
        mean = mean.masked_fill(~has_valid, 0.0)
        std = std.masked_fill(~has_valid, 0.0)
        agreement = agreement.clamp(0.0, 1.0).masked_fill(~has_valid, 0.0)
        features = torch.stack([mean, std, agreement], dim=-1)
        residual = self.max_residual * torch.tanh(self.evidence_residual_mlp(features).squeeze(-1))
        residual = residual.masked_fill(~has_valid, 0.0)
        return {
            "seizure_residual": residual,
            "seizure_nez_logit": logits.masked_fill(~valid, 0.0),
            "seizure_nez_probability": probabilities.masked_fill(~valid, 0.0),
            "seizure_nez_probability_mean": mean,
            "seizure_nez_probability_std": std,
            "seizure_nez_agreement": agreement,
            "valid_seizure_count_per_channel": valid.sum(dim=1),
        }


class PatientNEZCardinalityHead(nn.Module):
    """Predict a beta-binomial NEZ fraction without reading labels or true counts."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("CANE cardinality input_dim must be positive")
        summary_dim = 2 * int(input_dim) + 9
        self.cardinality_mlp = nn.Sequential(
            nn.Linear(summary_dim, 64),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 2),
        )

    @staticmethod
    def _masked_logit_statistics(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        rows: list[torch.Tensor] = []
        zero = logits.sum() * 0.0
        for patient_idx in range(logits.shape[0]):
            values = logits[patient_idx][mask[patient_idx]]
            if values.numel() == 0:
                rows.append(torch.stack([zero] * 7))
                continue
            quantiles = torch.quantile(values, torch.tensor([0.25, 0.50, 0.75], device=values.device, dtype=values.dtype))
            rows.append(torch.stack([
                values.mean(),
                values.std(unbiased=False),
                quantiles[0],
                quantiles[1],
                quantiles[2],
                values.min(),
                values.max(),
            ]))
        return torch.stack(rows, dim=0)

    def forward(
        self,
        contextual_channel_embedding: torch.Tensor,
        final_nez_logits: torch.Tensor,
        channel_mask: torch.Tensor,
        seizure_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        valid = channel_mask.to(dtype=torch.bool)
        embedding_mean, embedding_std = _masked_mean_std(contextual_channel_embedding, valid, dim=1)
        logit_stats = self._masked_logit_statistics(final_nez_logits, valid)
        valid_channel_count = valid.sum(dim=1).to(dtype=final_nez_logits.dtype)
        valid_seizure_count = seizure_mask.to(dtype=torch.bool).sum(dim=1).to(dtype=final_nez_logits.dtype)
        summary = torch.cat([
            embedding_mean,
            embedding_std,
            logit_stats,
            torch.log1p(valid_channel_count).unsqueeze(-1),
            torch.log1p(valid_seizure_count).unsqueeze(-1),
        ], dim=-1)
        raw = self.cardinality_mlp(summary)
        alpha = (F.softplus(raw[:, 0]) + 1.0).clamp(max=100.0)
        beta = (F.softplus(raw[:, 1]) + 1.0).clamp(max=100.0)
        fraction = alpha / (alpha + beta).clamp_min(1e-8)
        expected = valid_channel_count * fraction
        predicted = torch.round(expected).to(dtype=torch.long)
        predicted = torch.minimum(torch.maximum(predicted, torch.zeros_like(predicted)), valid_channel_count.to(torch.long))
        return {
            "cardinality_alpha": alpha,
            "cardinality_beta": beta,
            "predicted_nez_fraction": fraction,
            "expected_nez_count": expected,
            "predicted_nez_count": predicted,
            "cardinality_concentration": alpha + beta,
            "patient_cardinality_summary": summary,
        }


def deterministic_topk_mask(
    scores: torch.Tensor,
    channel_mask: torch.Tensor,
    predicted_count: torch.Tensor,
) -> torch.Tensor:
    """Select score-descending channels with stable original-index tie breaks."""
    selected = torch.zeros_like(channel_mask, dtype=torch.bool)
    for patient_idx in range(scores.shape[0]):
        valid_indices = torch.nonzero(channel_mask[patient_idx], as_tuple=False).flatten()
        k = int(predicted_count[patient_idx].detach().item())
        k = max(0, min(k, int(valid_indices.numel())))
        if k == 0:
            continue
        valid_scores = scores[patient_idx, valid_indices]
        order = torch.argsort(valid_scores, descending=True, stable=True)
        selected[patient_idx, valid_indices[order[:k]]] = True
    return selected


@torch.no_grad()
def initialize_clean_nez_prototypes(
    model: nn.Module,
    fit_loader: Iterable[dict[str, Any]],
    device: torch.device,
    model_seed: int,
    max_samples: int = 20_000,
) -> dict[str, Any]:
    """Initialize prototypes with deterministic KMeans on fit-only clean NEZ embeddings."""
    anchor = getattr(model, "clean_nez_anchor", None)
    if not isinstance(anchor, CleanNEZPrototypeAnchor):
        raise ValueError("Model does not expose a CANE clean_nez_anchor")
    was_training = model.training
    model.eval()
    collected: list[np.ndarray] = []
    n_fit_batches = 0
    for batch in fit_loader:
        n_fit_batches += 1
        device_batch = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        outputs = model(device_batch)
        labels_ez = device_batch["labels_ez"]
        labels_nez = torch.where(labels_ez >= 0.0, 1.0 - labels_ez, torch.full_like(labels_ez, -1.0))
        valid = device_batch["channel_mask"].bool() & (labels_nez > 0.5)
        projected = outputs["normalized_projected_embedding"][valid]
        if projected.numel():
            collected.append(projected.detach().cpu().numpy().astype(np.float32, copy=False))
    if was_training:
        model.train()
    if not collected:
        raise RuntimeError("CANE prototype initialization found no clean NEZ fit embeddings")
    embeddings = np.concatenate(collected, axis=0)
    rng = np.random.default_rng(int(model_seed))
    if embeddings.shape[0] > int(max_samples):
        indices = np.sort(rng.choice(embeddings.shape[0], size=int(max_samples), replace=False))
        embeddings = embeddings[indices]
    n_prototypes = int(anchor.prototypes.shape[0])
    original_count = int(embeddings.shape[0])
    degraded = original_count < n_prototypes
    if degraded:
        repeat_indices = np.arange(n_prototypes, dtype=np.int64) % max(original_count, 1)
        embeddings = embeddings[repeat_indices]
    kmeans = KMeans(n_clusters=n_prototypes, random_state=int(model_seed), n_init=10)
    centers = kmeans.fit(embeddings).cluster_centers_
    center_tensor = torch.as_tensor(centers, device=anchor.prototypes.device, dtype=anchor.prototypes.dtype)
    anchor.prototypes.copy_(F.normalize(center_tensor, dim=-1, eps=1e-8))
    return {
        "prototype_initialization_source": "fit_clean_nez_only",
        "model_seed": int(model_seed),
        "n_fit_batches": int(n_fit_batches),
        "n_clean_nez_embeddings_before_sampling": int(sum(item.shape[0] for item in collected)),
        "n_clean_nez_embeddings_used": int(original_count),
        "max_samples": int(max_samples),
        "n_prototypes": int(n_prototypes),
        "cluster_counts": np.bincount(kmeans.labels_, minlength=n_prototypes).astype(int).tolist(),
        "degraded_initialization": bool(degraded),
        "used_validation": False,
        "used_test": False,
    }


__all__ = [
    "CleanNEZPrototypeAnchor",
    "MultiSeizureNEZEvidenceResidual",
    "PatientNEZCardinalityHead",
    "deterministic_topk_mask",
    "initialize_clean_nez_prototypes",
]
