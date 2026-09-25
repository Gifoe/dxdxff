"""A9v14 candidate residual modules for patient-wise EZ ranking.

The modules in this file are intentionally residual or diagnostic additions on
top of the existing A9v3 patient-channel backbone.  They never consume center
ids, labels, true EZ counts, or other label-derived reporting fields as model
input features.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Sequence

import torch
from torch import nn


RANK_FEATURE_NAMES = (
    "rank_percentile",
    "score_percentile",
    "gap_to_top1",
    "gap_to_top5_mean",
    "valid_channel_count_normalized",
    "score_z_within_patient",
    "score_centered_by_patient_mean",
    "score_minus_patient_median",
)

CONSISTENCY_FEATURE_NAMES = (
    "n_records_for_patient",
    "n_valid_records_for_channel",
    "mean_score_across_seizures",
    "max_score_across_seizures",
    "min_score_across_seizures",
    "std_score_across_seizures",
    "score_iqr_across_seizures",
    "top1_frequency",
    "top3_frequency",
    "top5_frequency",
    "top10pct_frequency",
    "mean_rank_percentile",
    "best_rank_percentile",
    "worst_rank_percentile",
    "rank_variance",
    "score_slope_if_ordered_available",
    "record_missing_fraction",
)

LOCAL_FEATURE_NAMES = (
    "neighbor_score_mean",
    "neighbor_score_max",
    "neighbor_score_min",
    "neighbor_score_std",
    "neighbor_count",
    "score_minus_neighbor_mean",
    "score_minus_neighbor_max",
    "left_neighbor_score",
    "right_neighbor_score",
    "has_left_neighbor",
    "has_right_neighbor",
)

A9V14_MODEL_INPUT_FEATURE_NAMES = (
    "patient_channel_embedding",
    "base_score_a9v3",
    *RANK_FEATURE_NAMES,
    *CONSISTENCY_FEATURE_NAMES,
    *LOCAL_FEATURE_NAMES,
)

A9V14_LEDGER_REQUIRED_COLUMNS = (
    "config_name",
    "fold",
    "patient_id",
    "record_id",
    "channel_id",
    "channel_name",
    "center",
    "y_true",
    "base_logit_a9v3",
    "base_score_a9v3",
    "final_logit",
    "final_score",
    "base_rank_within_patient",
    "final_rank_within_patient",
    "rank_changed",
    "top1_before",
    "top1_after",
    "reranker_delta",
    "reranker_gate",
    "consistency_delta",
    "consistency_gate",
    "local_delta",
    "local_gate",
    "score_core",
    "score_broad",
    "score_clinical",
    "shaft_id",
    "contact_index",
    "shaft_parse_success",
    "neighbor_count",
    "n_records_for_patient",
    "n_valid_records_for_channel",
)


@dataclass(frozen=True)
class ChannelTopology:
    """Parsed shaft/contact information for a SEEG channel name."""

    shaft_id: str
    contact_index: float
    parse_success: bool


@dataclass(frozen=True)
class ConsistencyFeatureResult:
    """Tensor features and diagnostics for multi-seizure consistency."""

    features: torch.Tensor
    feature_names: tuple[str, ...]
    diagnostics: dict[str, float]


@dataclass(frozen=True)
class LocalFeatureResult:
    """Tensor features and parsed topology metadata for shaft-local context."""

    features: torch.Tensor
    feature_names: tuple[str, ...]
    shaft_ids: list[list[str]]
    contact_indices: list[list[float]]
    parse_success: list[list[bool]]


def _zero_like_valid(scores: torch.Tensor) -> torch.Tensor:
    return torch.zeros((*scores.shape, 0), dtype=scores.dtype, device=scores.device)


def compute_patient_rank_features(base_score: torch.Tensor, channel_mask: torch.Tensor) -> torch.Tensor:
    """Compute label-free within-patient rank features from A9v3 base scores."""

    if base_score.ndim != 2 or channel_mask.shape != base_score.shape:
        raise ValueError("base_score and channel_mask must both have shape [batch, channels].")
    batch, channels = base_score.shape
    out = torch.zeros((batch, channels, len(RANK_FEATURE_NAMES)), dtype=base_score.dtype, device=base_score.device)
    for b in range(batch):
        valid = channel_mask[b].bool() & torch.isfinite(base_score[b])
        idx = torch.nonzero(valid, as_tuple=False).flatten()
        n = int(idx.numel())
        if n == 0:
            continue
        vals = base_score[b, idx]
        order = torch.argsort(vals, descending=True)
        ranks = torch.empty(n, dtype=base_score.dtype, device=base_score.device)
        ranks[order] = torch.arange(1, n + 1, dtype=base_score.dtype, device=base_score.device)
        denom = float(max(n - 1, 1))
        rank_pct = 1.0 - (ranks - 1.0) / denom
        min_v = vals.min()
        max_v = vals.max()
        score_range = (max_v - min_v).abs()
        score_pct = torch.full_like(vals, 0.5) if float(score_range.detach().cpu()) < 1e-8 else (vals - min_v) / score_range.clamp_min(1e-8)
        top1 = max_v
        topk_mean = vals.topk(min(5, n), largest=True).values.mean()
        mean = vals.mean()
        std = vals.std(unbiased=False)
        median = vals.median()
        features = torch.stack(
            [
                rank_pct,
                score_pct,
                top1 - vals,
                topk_mean - vals,
                torch.full_like(vals, float(n) / 100.0),
                (vals - mean) / std.clamp_min(1e-6),
                vals - mean,
                vals - median,
            ],
            dim=-1,
        )
        out[b, idx] = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return out


class PatientContextResidualReranker(nn.Module):
    """Patient-wise DeepSets or SetTransformer residual reranker."""

    def __init__(
        self,
        input_dim: int,
        *,
        reranker_type: str = "deepset",
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.4,
        residual_scale: float = 0.2,
        use_rank_features: bool = False,
        rank_feature_dim: int = len(RANK_FEATURE_NAMES),
        use_patient_gate: bool = False,
    ) -> None:
        super().__init__()
        self.reranker_type = str(reranker_type).strip().lower()
        if self.reranker_type not in {"deepset", "set_transformer"}:
            raise ValueError("reranker_type must be 'deepset' or 'set_transformer'.")
        self.residual_scale = float(residual_scale)
        self.use_rank_features = bool(use_rank_features)
        self.use_patient_gate = bool(use_patient_gate)
        self.rank_feature_dim = int(rank_feature_dim)
        in_dim = int(input_dim) + (self.rank_feature_dim if self.use_rank_features else 0)
        self.input_proj = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Dropout(float(dropout)))
        if self.reranker_type == "set_transformer":
            self.attn_layers = nn.ModuleList(
                [
                    nn.MultiheadAttention(
                        embed_dim=int(hidden_dim),
                        num_heads=1,
                        dropout=float(dropout),
                        batch_first=True,
                    )
                    for _ in range(max(1, int(num_layers)))
                ]
            )
            self.attn_norms = nn.ModuleList([nn.LayerNorm(int(hidden_dim)) for _ in self.attn_layers])
            head_input_dim = int(hidden_dim)
        else:
            head_input_dim = int(hidden_dim) * 3
        self.delta_mlp = nn.Sequential(
            nn.Linear(head_input_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), 1),
        )
        self.gate_mlp = nn.Sequential(
            nn.Linear(head_input_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), 1),
        )

    def _prepare_inputs(
        self,
        embeddings: torch.Tensor,
        channel_mask: torch.Tensor,
        rank_features: torch.Tensor | None,
    ) -> torch.Tensor:
        x = embeddings
        if self.use_rank_features:
            if rank_features is None:
                rank_features = torch.zeros(
                    (*embeddings.shape[:2], self.rank_feature_dim),
                    dtype=embeddings.dtype,
                    device=embeddings.device,
                )
            x = torch.cat([x, rank_features.to(device=embeddings.device, dtype=embeddings.dtype)], dim=-1)
        return x * channel_mask.to(dtype=embeddings.dtype).unsqueeze(-1)

    def forward(
        self,
        embeddings: torch.Tensor,
        base_logits: torch.Tensor,
        channel_mask: torch.Tensor,
        *,
        rank_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        x = self._prepare_inputs(embeddings, channel_mask, rank_features)
        h = self.input_proj(x) * channel_mask.to(dtype=embeddings.dtype).unsqueeze(-1)
        if self.reranker_type == "set_transformer":
            key_padding_mask = ~channel_mask.bool()
            all_invalid = key_padding_mask.all(dim=1)
            if torch.any(all_invalid):
                key_padding_mask = key_padding_mask.clone()
                key_padding_mask[all_invalid] = False
            for attn, norm in zip(self.attn_layers, self.attn_norms):
                context, _ = attn(h, h, h, key_padding_mask=key_padding_mask)
                h = norm(h + context) * channel_mask.to(dtype=h.dtype).unsqueeze(-1)
            head_in = h
        else:
            mask_f = channel_mask.to(dtype=h.dtype).unsqueeze(-1)
            count = mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)
            mean = (h * mask_f).sum(dim=1, keepdim=True) / count
            h_min = torch.finfo(h.dtype).min
            max_pool = h.masked_fill(~channel_mask.bool().unsqueeze(-1), h_min).max(dim=1, keepdim=True).values
            max_pool = torch.where(torch.isfinite(max_pool), max_pool, torch.zeros_like(max_pool))
            context = torch.cat([h, mean.expand_as(h), max_pool.expand_as(h)], dim=-1)
            head_in = context * mask_f
        delta = self.delta_mlp(head_in).squeeze(-1)
        if self.use_patient_gate:
            gate = torch.sigmoid(self.gate_mlp(head_in).squeeze(-1))
        else:
            gate = torch.ones_like(delta)
        delta = delta.masked_fill(~channel_mask.bool(), 0.0)
        gate = gate.masked_fill(~channel_mask.bool(), 0.0)
        residual = float(self.residual_scale) * gate * torch.tanh(delta)
        residual = residual.masked_fill(~channel_mask.bool(), 0.0)
        final_logits = base_logits + residual
        valid_delta = delta[channel_mask.bool()]
        delta_l2 = valid_delta.square().mean() if valid_delta.numel() else delta.sum() * 0.0
        mean_abs = valid_delta.abs().mean() if valid_delta.numel() else delta.sum() * 0.0
        return {
            "delta": delta,
            "gate": gate,
            "residual": residual,
            "final_logits": final_logits,
            "delta_l2_loss": delta_l2,
            "mean_abs_delta": mean_abs,
        }


class FeatureResidualModule(nn.Module):
    """Small gated residual MLP used by consistency and local modules."""

    def __init__(self, feature_dim: int, *, hidden_dim: int, dropout: float, residual_scale: float, prefix: str) -> None:
        super().__init__()
        self.residual_scale = float(residual_scale)
        self.prefix = str(prefix)
        self.delta_mlp = nn.Sequential(
            nn.Linear(int(feature_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), 1),
        )
        self.gate_mlp = nn.Sequential(
            nn.Linear(int(feature_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), 1),
        )

    def forward(self, input_logits: torch.Tensor, features: torch.Tensor, channel_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        clean = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        delta = self.delta_mlp(clean).squeeze(-1).masked_fill(~channel_mask.bool(), 0.0)
        gate = torch.sigmoid(self.gate_mlp(clean).squeeze(-1)).masked_fill(~channel_mask.bool(), 0.0)
        residual = float(self.residual_scale) * gate * torch.tanh(delta)
        residual = residual.masked_fill(~channel_mask.bool(), 0.0)
        final_logits = input_logits + residual
        valid_delta = delta[channel_mask.bool()]
        return {
            "delta": delta,
            "gate": gate,
            "residual": residual,
            "final_logits": final_logits,
            "delta_l2_loss": valid_delta.square().mean() if valid_delta.numel() else delta.sum() * 0.0,
            "mean_abs_delta": valid_delta.abs().mean() if valid_delta.numel() else delta.sum() * 0.0,
        }


class MultiSeizureConsistencyResidualModule(FeatureResidualModule):
    """Gated residual module over record-consistency features."""

    def __init__(self, feature_dim: int = len(CONSISTENCY_FEATURE_NAMES), hidden_dim: int = 64, dropout: float = 0.1, residual_scale: float = 0.15) -> None:
        super().__init__(feature_dim, hidden_dim=hidden_dim, dropout=dropout, residual_scale=residual_scale, prefix="consistency")


class GatedShaftLocalResidualModule(FeatureResidualModule):
    """Gated residual module over local shaft-neighborhood features."""

    def __init__(self, feature_dim: int = len(LOCAL_FEATURE_NAMES), hidden_dim: int = 64, dropout: float = 0.1, residual_scale: float = 0.10) -> None:
        super().__init__(feature_dim, hidden_dim=hidden_dim, dropout=dropout, residual_scale=residual_scale, prefix="local")


class ClinicalPositiveMixtureHead(nn.Module):
    """Core-like plus broad clinical-positive mixture head."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.core_head = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), 1),
        )
        self.broad_head = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), 1),
        )

    def forward(self, embeddings: torch.Tensor, channel_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        core_logit = self.core_head(embeddings).squeeze(-1).masked_fill(~channel_mask.bool(), -1e9)
        broad_logit = self.broad_head(embeddings).squeeze(-1).masked_fill(~channel_mask.bool(), -1e9)
        score_core = torch.sigmoid(core_logit).masked_fill(~channel_mask.bool(), 0.0)
        score_broad = torch.sigmoid(broad_logit).masked_fill(~channel_mask.bool(), 0.0)
        score_clinical = (1.0 - (1.0 - score_core) * (1.0 - score_broad)).masked_fill(~channel_mask.bool(), 0.0)
        clipped = score_clinical.clamp(1e-6, 1.0 - 1e-6)
        clinical_logit = torch.logit(clipped).masked_fill(~channel_mask.bool(), -1e9)
        return {
            "core_logit": core_logit,
            "broad_logit": broad_logit,
            "clinical_logit": clinical_logit,
            "score_core": score_core,
            "score_broad": score_broad,
            "score_clinical": score_clinical,
        }


def build_positive_core_targets(labels_ez: torch.Tensor, base_score_ez: torch.Tensor, channel_mask: torch.Tensor) -> torch.Tensor:
    """Build weak core-like targets without making any clinical positive negative."""

    targets = torch.zeros_like(base_score_ez)
    for b in range(base_score_ez.shape[0]):
        valid_pos = channel_mask[b].bool() & (labels_ez[b] > 0.5)
        idx = torch.nonzero(valid_pos, as_tuple=False).flatten()
        if int(idx.numel()) == 0:
            continue
        vals = base_score_ez[b, idx]
        if int(idx.numel()) == 1 or float((vals.max() - vals.min()).abs().detach().cpu()) < 1e-8:
            targets[b, idx] = 1.0
        else:
            norm = (vals - vals.min()) / (vals.max() - vals.min()).clamp_min(1e-8)
            targets[b, idx] = 0.5 + 0.5 * norm
    return targets.masked_fill(~channel_mask.bool(), 0.0)


def build_consistency_features(
    record_score_ez: torch.Tensor,
    seizure_mask: torch.Tensor,
    seizure_channel_mask: torch.Tensor,
    channel_mask: torch.Tensor,
) -> ConsistencyFeatureResult:
    """Build label-free per-channel consistency features across records."""

    if record_score_ez.ndim != 3:
        raise ValueError("record_score_ez must have shape [batch, records, channels].")
    bsz, n_records, n_channels = record_score_ez.shape
    features = torch.zeros(
        (bsz, n_channels, len(CONSISTENCY_FEATURE_NAMES)),
        dtype=record_score_ez.dtype,
        device=record_score_ez.device,
    )
    rank_pct = torch.zeros_like(record_score_ez)
    top_flags = {1: torch.zeros_like(record_score_ez), 3: torch.zeros_like(record_score_ez), 5: torch.zeros_like(record_score_ez), 10: torch.zeros_like(record_score_ez)}
    for b in range(bsz):
        for s in range(n_records):
            valid = seizure_mask[b, s].bool() & seizure_channel_mask[b, s].bool() & channel_mask[b].bool()
            idx = torch.nonzero(valid, as_tuple=False).flatten()
            n = int(idx.numel())
            if n == 0:
                continue
            vals = record_score_ez[b, s, idx]
            order = torch.argsort(vals, descending=True)
            ranks = torch.empty(n, dtype=record_score_ez.dtype, device=record_score_ez.device)
            ranks[order] = torch.arange(1, n + 1, dtype=record_score_ez.dtype, device=record_score_ez.device)
            pct = 1.0 - (ranks - 1.0) / float(max(n - 1, 1))
            rank_pct[b, s, idx] = pct
            for k in (1, 3, 5):
                top_flags[k][b, s, idx] = (ranks <= min(k, n)).to(record_score_ez.dtype)
            k10 = max(1, int(math.ceil(0.10 * n)))
            top_flags[10][b, s, idx] = (ranks <= k10).to(record_score_ez.dtype)

    for b in range(bsz):
        total_records = int(seizure_mask[b].bool().sum().item())
        for c in range(n_channels):
            if not bool(channel_mask[b, c]):
                continue
            valid_records = seizure_mask[b].bool() & seizure_channel_mask[b, :, c].bool()
            vals = record_score_ez[b, valid_records, c]
            ranks = rank_pct[b, valid_records, c]
            n_valid = int(vals.numel())
            if n_valid == 0:
                features[b, c, 0] = float(total_records)
                features[b, c, -1] = 1.0
                continue
            q75 = torch.quantile(vals, 0.75) if n_valid > 1 else vals[0]
            q25 = torch.quantile(vals, 0.25) if n_valid > 1 else vals[0]
            slope = (vals[-1] - vals[0]) / float(max(n_valid - 1, 1)) if n_valid > 1 else vals[0] * 0.0
            row = torch.stack(
                [
                    vals.new_tensor(float(total_records)),
                    vals.new_tensor(float(n_valid)),
                    vals.mean(),
                    vals.max(),
                    vals.min(),
                    vals.std(unbiased=False) if n_valid > 1 else vals[0] * 0.0,
                    q75 - q25,
                    top_flags[1][b, valid_records, c].mean(),
                    top_flags[3][b, valid_records, c].mean(),
                    top_flags[5][b, valid_records, c].mean(),
                    top_flags[10][b, valid_records, c].mean(),
                    ranks.mean(),
                    ranks.max(),
                    ranks.min(),
                    ranks.var(unbiased=False) if n_valid > 1 else vals[0] * 0.0,
                    slope,
                    vals.new_tensor(1.0 - (float(n_valid) / float(max(total_records, 1)))),
                ]
            )
            features[b, c] = torch.nan_to_num(row, nan=0.0, posinf=0.0, neginf=0.0)
    nan_count = int(torch.isnan(features).sum().detach().cpu())
    return ConsistencyFeatureResult(
        features=torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0),
        feature_names=CONSISTENCY_FEATURE_NAMES,
        diagnostics={"consistency_feature_nan_count": float(nan_count)},
    )


_CONTACT_RE = re.compile(r"^([A-Z]+(?:[-'][A-Z]+|'?)*)(\d+)(?:-(\d+))?'?$")


def parse_channel_topology(channel_name: object) -> ChannelTopology:
    """Parse a channel name into shaft id and contact index."""

    raw = str(channel_name or "").strip().replace(" ", "").upper()
    match = _CONTACT_RE.match(raw)
    if not match:
        return ChannelTopology("UNKNOWN", float("nan"), False)
    shaft = match.group(1).strip("-") or "UNKNOWN"
    first = float(match.group(2))
    second = match.group(3)
    contact = (first + float(second)) / 2.0 if second is not None else first
    return ChannelTopology(shaft, contact, True)


def build_local_context_features(
    score_ez: torch.Tensor,
    channel_names: Sequence[Sequence[object]],
    channel_mask: torch.Tensor,
    *,
    local_window: int = 1,
) -> LocalFeatureResult:
    """Build shaft-local neighbor features without replacing channel scores."""

    if score_ez.ndim != 2:
        raise ValueError("score_ez must have shape [batch, channels].")
    bsz, n_channels = score_ez.shape
    features = torch.zeros((bsz, n_channels, len(LOCAL_FEATURE_NAMES)), dtype=score_ez.dtype, device=score_ez.device)
    shaft_ids: list[list[str]] = []
    contact_indices: list[list[float]] = []
    parse_success: list[list[bool]] = []
    for b in range(bsz):
        names = list(channel_names[b]) if b < len(channel_names) else [f"ch{i}" for i in range(n_channels)]
        parsed = [parse_channel_topology(names[i] if i < len(names) else f"ch{i}") for i in range(n_channels)]
        shaft_ids.append([item.shaft_id for item in parsed])
        contact_indices.append([item.contact_index for item in parsed])
        parse_success.append([item.parse_success for item in parsed])
        by_shaft: dict[str, list[int]] = {}
        for c, item in enumerate(parsed):
            if bool(channel_mask[b, c]) and item.parse_success:
                by_shaft.setdefault(item.shaft_id, []).append(c)
        for c, item in enumerate(parsed):
            if not bool(channel_mask[b, c]) or not item.parse_success:
                continue
            neighbors = [
                other
                for other in by_shaft.get(item.shaft_id, [])
                if other != c and abs(parsed[other].contact_index - item.contact_index) <= max(1, int(local_window))
            ]
            if not neighbors:
                continue
            vals = score_ez[b, torch.as_tensor(neighbors, dtype=torch.long, device=score_ez.device)]
            left_candidates = [other for other in by_shaft[item.shaft_id] if parsed[other].contact_index < item.contact_index]
            right_candidates = [other for other in by_shaft[item.shaft_id] if parsed[other].contact_index > item.contact_index]
            left_idx = max(left_candidates, key=lambda x: parsed[x].contact_index) if left_candidates else None
            right_idx = min(right_candidates, key=lambda x: parsed[x].contact_index) if right_candidates else None
            left_score = score_ez[b, left_idx] if left_idx is not None else score_ez.new_tensor(0.0)
            right_score = score_ez[b, right_idx] if right_idx is not None else score_ez.new_tensor(0.0)
            row = torch.stack(
                [
                    vals.mean(),
                    vals.max(),
                    vals.min(),
                    vals.std(unbiased=False) if vals.numel() > 1 else vals[0] * 0.0,
                    vals.new_tensor(float(vals.numel())),
                    score_ez[b, c] - vals.mean(),
                    score_ez[b, c] - vals.max(),
                    left_score,
                    right_score,
                    vals.new_tensor(1.0 if left_idx is not None else 0.0),
                    vals.new_tensor(1.0 if right_idx is not None else 0.0),
                ]
            )
            features[b, c] = torch.nan_to_num(row, nan=0.0, posinf=0.0, neginf=0.0)
    return LocalFeatureResult(
        features=features,
        feature_names=LOCAL_FEATURE_NAMES,
        shaft_ids=shaft_ids,
        contact_indices=contact_indices,
        parse_success=parse_success,
    )


def compute_rank_tensors(score_ez: torch.Tensor, channel_mask: torch.Tensor) -> torch.Tensor:
    """Return 1-based descending ranks for valid channels and -1 otherwise."""

    ranks = torch.full(score_ez.shape, -1, dtype=torch.long, device=score_ez.device)
    for b in range(score_ez.shape[0]):
        idx = torch.nonzero(channel_mask[b].bool(), as_tuple=False).flatten()
        if int(idx.numel()) == 0:
            continue
        vals = score_ez[b, idx]
        order = torch.argsort(vals, descending=True)
        ranks[b, idx[order]] = torch.arange(1, int(idx.numel()) + 1, dtype=torch.long, device=score_ez.device)
    return ranks


def rank_change_diagnostics(base_score: torch.Tensor, final_score: torch.Tensor, channel_mask: torch.Tensor) -> dict[str, torch.Tensor]:
    """Compute rank-change diagnostics from base and final EZ scores."""

    base_rank = compute_rank_tensors(base_score, channel_mask)
    final_rank = compute_rank_tensors(final_score, channel_mask)
    valid = channel_mask.bool()
    changed = (base_rank != final_rank) & valid
    top1_before = (base_rank == 1) & valid
    top1_after = (final_rank == 1) & valid
    batch_changed = []
    for b in range(base_score.shape[0]):
        batch_changed.append(bool(torch.any(top1_before[b] != top1_after[b]).item()))
    top1_changed = base_score.new_tensor(batch_changed, dtype=base_score.dtype)
    return {
        "base_rank": base_rank,
        "final_rank": final_rank,
        "rank_changed": changed,
        "top1_before": top1_before,
        "top1_after": top1_after,
        "rank_changed_fraction": changed.float().sum() / valid.float().sum().clamp_min(1.0),
        "top1_changed_fraction": top1_changed.mean() if top1_changed.numel() else base_score.sum() * 0.0,
    }


def validate_a9v14_ledger_columns(frame: object) -> None:
    """Fail if a prediction ledger is missing required A9v14 columns."""

    columns = set(getattr(frame, "columns", []))
    missing = sorted(set(A9V14_LEDGER_REQUIRED_COLUMNS) - columns)
    if missing:
        raise ValueError(f"A9v14 prediction ledger missing required columns: {missing}")


__all__ = [
    "A9V14_LEDGER_REQUIRED_COLUMNS",
    "A9V14_MODEL_INPUT_FEATURE_NAMES",
    "CONSISTENCY_FEATURE_NAMES",
    "LOCAL_FEATURE_NAMES",
    "RANK_FEATURE_NAMES",
    "ChannelTopology",
    "ClinicalPositiveMixtureHead",
    "ConsistencyFeatureResult",
    "GatedShaftLocalResidualModule",
    "LocalFeatureResult",
    "MultiSeizureConsistencyResidualModule",
    "PatientContextResidualReranker",
    "build_consistency_features",
    "build_local_context_features",
    "build_positive_core_targets",
    "compute_patient_rank_features",
    "compute_rank_tensors",
    "parse_channel_topology",
    "rank_change_diagnostics",
    "validate_a9v14_ledger_columns",
]
