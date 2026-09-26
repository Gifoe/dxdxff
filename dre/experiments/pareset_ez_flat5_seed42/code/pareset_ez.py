"""PaReSet-EZ: a single-score, paired-reference EpiLENS research candidate.

Input: one patient's standardized four-view features [S,W,C,36], boolean
validity [S,W,C], and actual window-center times [S,W] in seconds from onset.
No labels, center IDs, patient IDs, coordinates, or true EZ counts enter forward.
This is an untrained research implementation, not an established SOTA model.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ModelConfig:
    input_dimension: int = 36
    dimension: int = 32
    heads: int = 2
    dropout: float = 0.4
    reference_modes: int = 4
    reference_key_dimension: int = 16
    temporal_hidden_dimension: int = 16
    seizure_hidden_dimension: int = 16
    early_end_seconds: float = 10.0
    late_end_seconds: float = 30.0
    temporal_residual_scale: float = 0.2
    uniform_seizure_fraction: float = 0.5
    use_reference_contrast: bool = True
    use_temporal_update: bool = True
    use_adaptive_seizure_pool: bool = True
    use_patient_relative: bool = True
    # Diagnostic/control only: default is a pre-onset-matched reference.
    reference_mode: str = "learned"  # learned | uniform


def masked_mean(x: Tensor, mask: Tensor, dim: int) -> Tensor:
    safe = torch.where(mask.unsqueeze(-1), x, torch.zeros_like(x))
    return safe.sum(dim) / mask.sum(dim).clamp_min(1).unsqueeze(-1)


def masked_mean_std(x: Tensor, mask: Tensor, dim: int) -> tuple[Tensor, Tensor]:
    """Population std with EXACT zero at zero variance and finite derivatives.

    clamp-before-sqrt plus the outer where avoids sqrt(0)'s singular gradient.
    The 1e-12 floor is only a numerical boundary, not an estimated uncertainty.
    """
    mean = masked_mean(x, mask, dim)
    diff = torch.where(mask.unsqueeze(-1), x - mean.unsqueeze(dim), torch.zeros_like(x))
    var = masked_mean(diff.square(), mask, dim).clamp_min(0)
    std = torch.where(var > 0, var.clamp_min(1e-12).sqrt(), torch.zeros_like(var))
    return mean, std


def patient_relative_z(x: Tensor, valid: Tensor) -> Tensor:
    """Last-but-one axis is channels; accepts [C,D] or [S,C,D]."""
    dim = x.ndim - 2
    mean = masked_mean(x, valid, dim)
    centered = torch.where(valid.unsqueeze(-1), x - mean.unsqueeze(dim), torch.zeros_like(x))
    var = masked_mean(centered.square(), valid, dim)
    return (centered / var.clamp_min(1e-6).sqrt().unsqueeze(dim)).masked_fill(~valid.unsqueeze(-1), 0)


def masked_softmax(scores: Tensor, valid: Tensor, dim: int) -> Tensor:
    """All-invalid slices return zeros rather than NaN or a uniform mass."""
    masked = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
    weights = torch.softmax(masked, dim=dim).masked_fill(~valid, 0)
    return weights / weights.sum(dim=dim, keepdim=True).clamp_min(1e-12)


class ChannelContextBlock(nn.Module):
    """Same topology as the supplied EpiLENS channel context block."""
    def __init__(self, dimension: int, heads: int, dropout: float):
        super().__init__()
        self.attention = nn.MultiheadAttention(dimension, heads, dropout=dropout, batch_first=True)
        self.normalization = nn.LayerNorm(dimension)

    def forward(self, values: Tensor, valid: Tensor) -> Tensor:
        safe = values.masked_fill(~valid.unsqueeze(-1), 0)
        if not bool(valid.any()):
            return safe * 0
        attended, _ = self.attention(safe.unsqueeze(0), safe.unsqueeze(0), safe.unsqueeze(0),
                                     key_padding_mask=(~valid).unsqueeze(0), need_weights=False)
        return self.normalization(safe + attended.squeeze(0)).masked_fill(~valid.unsqueeze(-1), 0)


class RetainedEncoder(nn.Module):
    """Retain both EpiLENS context blocks and its 64->32 patient projection."""
    def __init__(self, config: ModelConfig):
        super().__init__()
        d = config.dimension
        self.window_mlp = nn.Sequential(nn.LayerNorm(config.input_dimension),
            nn.Linear(config.input_dimension, d), nn.GELU(), nn.Dropout(config.dropout), nn.Linear(d, d))
        self.window_context = ChannelContextBlock(d, config.heads, config.dropout)
        self.patient_projection = nn.Sequential(nn.LayerNorm(2*d), nn.Linear(2*d, d),
                                               nn.GELU(), nn.Dropout(config.dropout))
        self.patient_context = ChannelContextBlock(d, config.heads, config.dropout)

    def encode_windows(self, features: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
        encoded = self.window_mlp(features).masked_fill(~valid.unsqueeze(-1), 0)
        rows = []
        for s in range(features.shape[0]):
            rows.append(torch.stack([self.window_context(encoded[s, w], valid[s, w])
                                     for w in range(features.shape[1])]))
        return encoded, torch.stack(rows)


class PairedReference(nn.Module):
    """Baseline-defined peer groups; compare each contact's CHANGE to peers.

    Peer membership is learned from pre-onset embeddings, BEFORE channel
    contextualization. It does not use post-onset activations or any labels.
    Within a seizure the same group weights follow contacts through time.
    Direct self-contribution is subtracted analytically from each pooled mode.
    These are comparison groups, NOT assumed healthy/NEZ prototypes.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.key = nn.Linear(config.dimension, config.reference_key_dimension, bias=False)
        self.mode_queries = nn.Parameter(torch.empty(config.reference_modes, config.reference_key_dimension))
        nn.init.orthogonal_(self.mode_queries)

    def forward(self, encoded: Tensor, valid: Tensor, times: Tensor) -> dict[str, Tensor]:
        pre = valid & (times[:, :, None] < 0)
        baseline_valid = pre.any(1)
        baseline = masked_mean(encoded, pre, 1)
        delta_valid = valid & baseline_valid[:, None, :]
        delta = (encoded - baseline[:, None, :, :]).masked_fill(~delta_valid.unsqueeze(-1), 0)
        keys = F.normalize(self.key(baseline), dim=-1, eps=1e-6)
        queries = F.normalize(self.mode_queries, dim=-1, eps=1e-6)
        group_scores = 2.0 * torch.einsum("scr,kr->sck", keys, queries)
        if self.config.reference_mode == "uniform":
            group_scores = group_scores * 0
        assignment = masked_softmax(group_scores, baseline_valid[:, :, None].expand_as(group_scores), dim=1)
        mixture = torch.softmax(group_scores, dim=-1)
        # [S,W,C,K]; no C x C attention matrix is formed in this module.
        weights = assignment[:, None, :, :] * delta_valid[:, :, :, None]
        total_weight = weights.sum(dim=2)
        total_delta = torch.einsum("swck,swcd->swkd", weights, delta)
        denominator = (total_weight[:, :, None, :] - weights).clamp_min(0)
        numerator = total_delta[:, :, None, :, :] - weights.unsqueeze(-1) * delta.unsqueeze(-2)
        mode_valid = denominator > 1e-7
        peer_modes = numerator / denominator.clamp_min(1e-7).unsqueeze(-1)
        peer_modes = peer_modes.masked_fill(~mode_valid.unsqueeze(-1), 0)
        mix = mixture[:, None, :, :] * mode_valid
        mix = mix / mix.sum(-1, keepdim=True).clamp_min(1e-7)
        peer_delta = (mix.unsqueeze(-1) * peer_modes).sum(-2)
        peer_valid = mode_valid.any(-1) & delta_valid
        peer_delta = peer_delta.masked_fill(~peer_valid.unsqueeze(-1), 0)
        contrast = (delta - peer_delta).masked_fill(~peer_valid.unsqueeze(-1), 0)
        # Reference aggregation ESS: a diagnostic, NOT a calibrated confidence.
        effective_peers = 1.0 / assignment.square().sum(1).clamp_min(1e-7)
        effective_peers = effective_peers.masked_fill(~baseline_valid.any(-1)[:, None], 0)
        return dict(delta=delta, delta_valid=delta_valid, contrast=contrast, peer_delta=peer_delta,
                    peer_valid=peer_valid, assignment=assignment, effective_peers=effective_peers,
                    baseline_valid=baseline_valid)


class PhaseEvidence(nn.Module):
    """Small time-aware REPRESENTATION update; no separate prediction head."""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        n = 5 * config.dimension + 6
        self.net = nn.Sequential(nn.LayerNorm(n), nn.Linear(n, config.temporal_hidden_dimension),
                                 nn.GELU(), nn.Linear(config.temporal_hidden_dimension, config.dimension))
        # Start at the retained base, not at an arbitrary reference correction.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, evidence: dict[str, Tensor], valid: Tensor, times: Tensor) -> dict[str, Tensor]:
        d, r = evidence['delta'], evidence['contrast']
        dv, rv = evidence['delta_valid'], evidence['peer_valid']
        if not self.config.use_reference_contrast:
            r, rv = torch.zeros_like(r), torch.zeros_like(rv)
        early = (times >= 0) & (times < self.config.early_end_seconds)
        late = (times >= self.config.early_end_seconds) & (times <= self.config.late_end_seconds)
        post = early | late
        masks = [dv & early[:, :, None], dv & late[:, :, None],
                 rv & early[:, :, None], rv & late[:, :, None]]
        summaries = [masked_mean(d, masks[0], 1), masked_mean(d, masks[1], 1),
                     masked_mean(r, masks[2], 1), masked_mean(r, masks[3], 1)]
        t = times[:, :, None].expand_as(rv)
        pm = rv & post[:, :, None]
        tmean = masked_mean(t.unsqueeze(-1), pm, 1).squeeze(-1)
        moment = masked_mean(r * ((t - tmean[:, None, :]) / self.config.late_end_seconds).unsqueeze(-1), pm, 1)
        denom = valid.sum(1).clamp_min(1)
        pre_fraction = (valid & (times[:, :, None] < 0)).sum(1) / denom
        post_fraction = (valid & post[:, :, None]).sum(1) / denom
        flags = torch.stack([m.any(1).to(d.dtype) for m in masks] + [pre_fraction, post_fraction], dim=-1)
        summary = torch.cat([*summaries, moment, flags], dim=-1)
        update = self.config.temporal_residual_scale * torch.tanh(self.net(summary))
        eligible = evidence['baseline_valid'] & (dv & post[:, :, None]).any(1)
        update = update.masked_fill(~eligible.unsqueeze(-1), 0)
        return dict(update=update, summary=summary, availability=torch.stack([pre_fraction, post_fraction], -1))


class SeizureEvidencePool(nn.Module):
    """Mean floor + learned cross-seizure weighting; keep unweighted std."""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        n = 2*config.dimension + 2
        self.score = nn.Sequential(nn.LayerNorm(n), nn.Linear(n, config.seizure_hidden_dimension),
                                   nn.GELU(), nn.Linear(config.seizure_hidden_dimension, 1))
        nn.init.zeros_(self.score[-1].weight)
        nn.init.zeros_(self.score[-1].bias)

    def forward(self, u: Tensor, valid: Tensor, availability: Tensor) -> dict[str, Tensor]:
        mean, std = masked_mean_std(u, valid, 0)
        rel = patient_relative_z(u, valid)
        relmean = masked_mean(rel, valid, 0)
        score_input = torch.cat([rel, (rel-relmean.unsqueeze(0)).abs(), availability], -1)
        logits = self.score(score_input).squeeze(-1)
        learned = masked_softmax(logits, valid, 0)
        uniform = valid / valid.sum(0).clamp_min(1)
        f = self.config.uniform_seizure_fraction
        weights = f*uniform + (1-f)*learned if self.config.use_adaptive_seizure_pool else uniform
        pooled = (u * weights.unsqueeze(-1)).sum(0)
        ess = 1 / weights.square().sum(0).clamp_min(1e-7)
        ess = ess.masked_fill(~valid.any(0), 0)
        return dict(mean=pooled, ordinary_mean=mean, std=std, weights=weights, effective_seizures=ess)


class PaReSetEZ(nn.Module):
    def __init__(self, config: ModelConfig | None = None):
        super().__init__()
        self.config = config or ModelConfig()
        c = self.config
        if c.dimension % c.heads or c.reference_modes < 1:
            raise ValueError('dimension must be divisible by heads; reference_modes must be positive')
        if c.reference_mode not in {'learned', 'uniform'}:
            raise ValueError('reference_mode must be learned or uniform')
        if not 0 <= c.dropout < 1 or not 0 <= c.uniform_seizure_fraction <= 1:
            raise ValueError('invalid dropout or uniform_seizure_fraction')
        if not 0 < c.early_end_seconds < c.late_end_seconds or c.temporal_residual_scale < 0:
            raise ValueError('invalid phase boundaries or residual scale')
        self.encoder = RetainedEncoder(c)
        self.reference = PairedReference(c)
        self.phase = PhaseEvidence(c)
        self.seizure_pool = SeizureEvidencePool(c)
        self.ez_head = nn.Linear(c.dimension, 1)

    def forward(self, features: Tensor, valid: Tensor, window_times: Tensor,
                return_diagnostics: bool = False) -> dict[str, Tensor]:
        if features.ndim != 4 or features.shape[-1] != self.config.input_dimension:
            raise ValueError('features must be [S,W,C,input_dimension]')
        if min(features.shape[:3]) < 1 or valid.shape != features.shape[:3] or valid.dtype != torch.bool:
            raise ValueError('nonempty dimensions and a boolean [S,W,C] mask are required')
        if window_times.shape != features.shape[:2]:
            raise ValueError('window_times must be [S,W], actual seconds from onset')
        if valid.device != features.device or window_times.device != features.device:
            raise ValueError('features, valid, and window_times must share a device')
        if not features.is_floating_point():
            raise ValueError('floating-point features required')
        if not bool(torch.isfinite(features[valid]).all()):
            raise ValueError('non-finite values in VALID features; repair the cache/mask first')
        if not bool(torch.isfinite(window_times[valid.any(-1)]).all()):
            raise ValueError('non-finite timestamps in VALID windows')
        # Mask before any arithmetic: NaN * 0 is still NaN.
        features = features.masked_fill(~valid.unsqueeze(-1), 0)
        times = window_times.masked_fill(~valid.any(-1), 0).to(features.dtype)
        encoded, contextual = self.encoder.encode_windows(features, valid)
        base_seizure = masked_mean(contextual, valid, 1)
        seizure_valid = valid.any(1)
        evidence = self.reference(encoded, valid, times)
        phase = self.phase(evidence, valid, times)
        u = base_seizure + phase['update'] if self.config.use_temporal_update else base_seizure
        u = u.masked_fill(~seizure_valid.unsqueeze(-1), 0)
        pooled = self.seizure_pool(u, seizure_valid, phase['availability'])
        channel_valid = seizure_valid.any(0)
        h = self.encoder.patient_projection(torch.cat([pooled['mean'], pooled['std']], -1))
        if self.config.use_patient_relative:
            h = patient_relative_z(h, channel_valid)
        h = self.encoder.patient_context(h, channel_valid)
        logit_ez = self.ez_head(h).squeeze(-1).masked_fill(~channel_valid, 0)
        # Invalid scores are placeholders; ALL exported metrics must use the mask.
        result = dict(logit_ez=logit_ez, logit_nez=-logit_ez,
                      probability_ez=torch.sigmoid(logit_ez), probability_nez=torch.sigmoid(-logit_ez),
                      channel_valid=channel_valid)
        if return_diagnostics:
            result.update(seizure_weights=pooled['weights'], effective_seizures=pooled['effective_seizures'],
                reference_assignment=evidence['assignment'], effective_reference_peers=evidence['effective_peers'],
                own_change=evidence['delta'], peer_contrast=evidence['contrast'], peer_valid=evidence['peer_valid'],
                phase_summary=phase['summary'], phase_update=phase['update'], channel_embedding=h)
        return result

    @torch.no_grad()
    def copy_retained_from_prq(self, prq: nn.Module) -> None:
        """Optional audit/transfer helper, NOT the default training protocol.

        Copy a dimension-matched supplied PRQ's shared topology and BASE head.
        The old Q10 head is not copied. Use only an authorized fit-fold checkpoint.
        At initialization, in eval mode, prediction equals PRQ's no-Q10 base
        apart from floating-point/numerical-std treatment at degenerate variance.
        """
        self.encoder.load_state_dict(prq.encoder.state_dict(), strict=True)
        self.ez_head.weight.copy_(-prq.base_head.weight)
        self.ez_head.bias.copy_(-prq.base_head.bias)


def patient_objective(logit_ez: Tensor, labels_nez: Tensor, valid: Tensor,
                      boundary_weight: float = 0.05, margin: float = 0.05) -> tuple[Tensor, dict[str, Tensor]]:
    """Original boundary mining; caller averages losses over patients equally."""
    if logit_ez.ndim != 1 or labels_nez.shape != logit_ez.shape or valid.shape != logit_ez.shape:
        raise ValueError('logits, labels, valid must be matching [C] vectors')
    if valid.dtype != torch.bool or not bool(valid.any()):
        raise ValueError('loss requires at least one valid channel and a boolean mask')
    if not bool(((labels_nez[valid] == 0) | (labels_nez[valid] == 1)).all()):
        raise ValueError('labels_nez must be 0=EZ,1=NEZ on valid channels')
    if boundary_weight < 0:
        raise ValueError('boundary_weight must be nonnegative')
    bce = F.binary_cross_entropy_with_logits(logit_ez[valid], (1-labels_nez[valid]).to(logit_ez.dtype))
    ez, nez = logit_ez[valid & (labels_nez == 0)], logit_ez[valid & (labels_nez == 1)]
    boundary = logit_ez[valid].sum()*0
    if ez.numel() and nez.numel():
        np = max(1, math.ceil(.3*ez.numel()))
        nn_ = min(nez.numel(), max(1, min(16, ez.numel())))
        boundary = F.softplus(margin + nez.topk(nn_).values.mean() - ez.topk(np, largest=False).values.mean())
    return bce+boundary_weight*boundary, dict(bce=bce, boundary=boundary)


def drop_seizures(features: Tensor, valid: Tensor, times: Tensor, probability: float = 0.15,
                  generator: torch.Generator | None = None) -> tuple[Tensor, Tensor, Tensor]:
    """Optional TRAIN-only availability augmentation; never drop the last seizure.

    Off by default in the first architecture comparison. This does not create
    additional independent patients and is never used during final inference.
    """
    if not 0 <= probability < 1:
        raise ValueError('probability must be in [0,1)')
    present = valid.any(-1).any(-1)
    keep = (torch.rand(len(present), device=features.device, generator=generator) >= probability) & present
    if bool(present.any()) and not bool(keep.any()):
        choices = present.nonzero(as_tuple=False).flatten()
        j = torch.randint(len(choices), (1,), device=features.device, generator=generator)
        keep[choices[j]] = True
    return features, valid & keep[:, None, None], times


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
