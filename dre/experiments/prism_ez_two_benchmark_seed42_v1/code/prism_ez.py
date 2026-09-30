"""PRiSM-EZ: a single <100K parameter patient-relative spectral mixer.

The module deliberately has no hidden raw encoder, attention, recurrence,
BatchNorm, graph operation, or label-conditioned coordinate.  It consumes the
precomputed 68-D window tokens stipulated by the protocol and emits one logit
per channel for a group of synchronized records.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor, nn


FEATURE_DIM = 68
PHYSIOLOGY_DIM = 36
SPECTRAL_DIM = 32
HIDDEN_DIM = 48
EMBEDDING_DIM = 64
SPECTRAL_EDGES_HZ = np.geomspace(1.0, 300.0, SPECTRAL_DIM + 1).astype(np.float64)


def spectral_availability(sampling_rate_hz: float) -> np.ndarray:
    """Return a label-free physical availability mask for the 32 sketch bins."""
    if sampling_rate_hz <= 0:
        raise ValueError("sampling rate must be positive")
    upper = 0.45 * float(sampling_rate_hz)
    # A partially observable bin remains valid and is integrated only over its
    # observable frequency samples.  A bin entirely above the trusted Nyquist
    # ceiling is explicitly masked, never fabricated by upsampling.
    return SPECTRAL_EDGES_HZ[:-1] < upper


def spectral_sketch(waveforms: np.ndarray, sampling_rate_hz: float,
                    window_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fixed 2-s/1-s-hop Hann-periodogram sketch, independent of labels.

    ``waveforms`` is ``[channels, samples]`` and ``window_mask`` is
    ``[channels, T]``.  The output has 32 log-spaced 1--300 Hz coordinates;
    unavailable bins are zero with a false availability mask.
    """
    x = np.asarray(waveforms, dtype=np.float64)
    valid = np.asarray(window_mask, dtype=bool)
    if x.ndim != 2 or valid.ndim != 2 or x.shape[0] != valid.shape[0]:
        raise ValueError("waveform/window-mask shape mismatch")
    samples = int(round(2.0 * float(sampling_rate_hz)))
    hop = int(round(float(sampling_rate_hz)))
    if samples < 4 or hop < 1:
        raise ValueError("sampling rate is too low for a two-second sketch")
    channels, steps = valid.shape
    result = np.zeros((channels, steps, SPECTRAL_DIM), dtype=np.float32)
    available = np.zeros_like(result, dtype=bool)
    trusted_max = 0.45 * float(sampling_rate_hz)
    physical = spectral_availability(sampling_rate_hz)
    taper = np.hanning(samples).astype(np.float64)
    taper_energy = max(float(np.square(taper).sum()), 1e-12)
    frequencies = np.fft.rfftfreq(samples, d=1.0 / float(sampling_rate_hz))
    # Periodogram integration is performed in physical Hz.  The common
    # frequency spacing is included so feature scale does not depend on FFT
    # length alone.
    df = float(frequencies[1] - frequencies[0]) if len(frequencies) > 1 else 0.0
    bin_indices = [np.flatnonzero((frequencies >= lo) & (frequencies < hi) &
                                  (frequencies <= trusted_max))
                   for lo, hi in zip(SPECTRAL_EDGES_HZ[:-1], SPECTRAL_EDGES_HZ[1:])]
    for step in range(steps):
        begin, end = step * hop, step * hop + samples
        if end > x.shape[1]:
            continue
        use = valid[:, step]
        if not bool(use.any()):
            continue
        segment = x[use, begin:end] * taper[None, :]
        power = np.square(np.abs(np.fft.rfft(segment, axis=-1))) / taper_energy
        for index, frequencies_in_bin in enumerate(bin_indices):
            if not physical[index] or not len(frequencies_in_bin):
                continue
            result[use, step, index] = np.log1p(
                power[:, frequencies_in_bin].sum(axis=-1) * df
            ).astype(np.float32)
            available[use, step, index] = True
    if not np.isfinite(result).all():
        raise RuntimeError("nonfinite spectral sketch")
    return result, available


def empirical_rank(values: Tensor, channel_mask: Tensor, window_mask: Tensor) -> Tensor:
    """Average-tie rank over label-free, signal-valid channels.

    Shapes are ``[C,T,D]``, ``[C]``, ``[C,T]``.  The explicit tie groups make
    the intended rank-average semantics auditable; no label or supervision
    array is accepted by this function.
    """
    if values.ndim != 3 or channel_mask.shape != values.shape[:1] or \
            window_mask.shape != values.shape[:2]:
        raise ValueError("rank input shape mismatch")
    channels, steps, features = values.shape
    out = torch.zeros_like(values)
    for step in range(steps):
        active = torch.nonzero(channel_mask & window_mask[:, step], as_tuple=False).flatten()
        count = int(active.numel())
        if count <= 1:
            continue
        table = values[active, step]
        ordered, order = torch.sort(table, dim=0, stable=True)
        ranks = torch.empty_like(table)
        # ``torch.unique_consecutive`` does not return group starts per column;
        # dimensions are small (68), so the transparent scalar grouping is
        # preferable to a label-dependent or unstable shortcut.
        for feature in range(features):
            column = ordered[:, feature]
            cursor = 0
            while cursor < count:
                end = cursor + 1
                while end < count and bool(column[end] == column[cursor]):
                    end += 1
                # ranks are one-based before the prescribed [-1, 1] mapping.
                average = (float(cursor + 1) + float(end)) / 2.0
                ranks[order[cursor:end, feature], feature] = average
                cursor = end
        out[active, step] = 2.0 * (ranks - 1.0) / float(count - 1) - 1.0
    return out


class DepthwiseTemporalMixer(nn.Module):
    def __init__(self):
        super().__init__()
        self.k3 = nn.Conv1d(HIDDEN_DIM, HIDDEN_DIM, 3, padding=1, groups=HIDDEN_DIM)
        self.k7 = nn.Conv1d(HIDDEN_DIM, HIDDEN_DIM, 7, padding=3, groups=HIDDEN_DIM)
        self.k15 = nn.Conv1d(HIDDEN_DIM, HIDDEN_DIM, 15, padding=7, groups=HIDDEN_DIM)
        self.pointwise = nn.Conv1d(HIDDEN_DIM * 3, HIDDEN_DIM, 1)
        self.dropout = nn.Dropout(0.1)
        self.norm = nn.LayerNorm(HIDDEN_DIM)

    def forward(self, value: Tensor) -> Tensor:
        # [C,T,48] -> convolutions are independent for every channel.
        x = value.transpose(1, 2)
        mixed = self.pointwise(torch.cat((self.k3(x), self.k7(x), self.k15(x)), dim=1))
        mixed = self.dropout(torch.nn.functional.gelu(mixed)).transpose(1, 2)
        return self.norm(value + mixed)


def _quantile_statistics(value: Tensor, valid: Tensor, *, dimension: int) -> Tensor:
    """Mean/Q25/Q50/Q75/max with explicit masking and finite singletons."""
    if dimension not in (0, 1):
        raise ValueError("only record/time pooling is supported")
    # In the model this helper is called on either [C,T,D] (time) or [R,D]
    # (records).  NaN-aware reductions exactly exclude padded entries, while
    # keeping the reductions batched.  The old channel-by-channel implementation
    # made three separate GPU quantile launches per channel and was needlessly
    # slow without changing the mathematical pooling rule.
    if dimension == 1:
        if not bool(valid.any(dim=1).all()):
            raise RuntimeError("channel without valid windows reached pooling")
        masked = value.masked_fill(~valid[..., None], float("nan"))
        maximum = masked.nan_to_num(nan=float("-inf")).max(dim=1).values
        return torch.cat((torch.nanmean(masked, dim=1),
                          torch.nanquantile(masked, 0.25, dim=1),
                          torch.nanquantile(masked, 0.50, dim=1),
                          torch.nanquantile(masked, 0.75, dim=1), maximum), dim=1)
    if not bool(valid.any()):
        raise RuntimeError("record pooling received no valid records")
    masked = value.masked_fill(~valid[:, None], float("nan"))
    return torch.cat((torch.nanmean(masked, dim=0),
                      torch.nanquantile(masked, 0.25, dim=0),
                      torch.nanquantile(masked, 0.50, dim=0),
                      torch.nanquantile(masked, 0.75, dim=0),
                      masked.nan_to_num(nan=float("-inf")).max(dim=0).values))


class PRiSMEZ(nn.Module):
    """The exact shared PRiSM-EZ topology for both benchmarks."""
    def __init__(self):
        super().__init__()
        self.absolute = nn.Linear(FEATURE_DIM, HIDDEN_DIM)
        self.rank_gate = nn.Linear(FEATURE_DIM, HIDDEN_DIM)
        self.value = nn.Linear(FEATURE_DIM, HIDDEN_DIM)
        self.input_norm = nn.LayerNorm(HIDDEN_DIM)
        self.mixers = nn.ModuleList((DepthwiseTemporalMixer(), DepthwiseTemporalMixer()))
        self.time_projection = nn.Sequential(nn.Linear(HIDDEN_DIM * 5, EMBEDDING_DIM), nn.GELU(),
                                             nn.LayerNorm(EMBEDDING_DIM), nn.Dropout(0.1))
        self.record_projection = nn.Sequential(nn.Linear(EMBEDDING_DIM * 5, EMBEDDING_DIM), nn.GELU(),
                                               nn.LayerNorm(EMBEDDING_DIM))
        self.classifier = nn.Sequential(nn.Linear(EMBEDDING_DIM * 3, EMBEDDING_DIM), nn.GELU(),
                                        nn.Dropout(0.1), nn.Linear(EMBEDDING_DIM, 1))

    def record_embedding(self, features: Tensor, channel_mask: Tensor, window_mask: Tensor,
                         *, return_gate: bool = False):
        if features.ndim != 3 or features.shape[-1] != FEATURE_DIM:
            raise ValueError("features must be [channels, windows, 68]")
        if channel_mask.shape != features.shape[:1] or window_mask.shape != features.shape[:2]:
            raise ValueError("record masks are not aligned")
        if not bool(channel_mask.any()):
            raise RuntimeError("record has no signal-valid channel")
        rank = empirical_rank(features, channel_mask.bool(), window_mask.bool())
        gate = torch.sigmoid(self.rank_gate(rank))
        hidden = self.input_norm(self.absolute(features) + gate * self.value(features))
        # Invalid time positions contribute neither signal nor a pooling value.
        hidden = hidden * window_mask[..., None].to(hidden.dtype)
        for mixer in self.mixers:
            hidden = mixer(hidden) * window_mask[..., None].to(hidden.dtype)
        active_channels = channel_mask.bool() & window_mask.bool().any(dim=1)
        # A source record may omit a canonical channel.  It remains outside
        # rank, pooling, context, loss, and output; padding it with a fake
        # quantile would violate the valid-channel contract.
        embedding = hidden.new_zeros((features.shape[0], EMBEDDING_DIM))
        embedding[active_channels] = self.time_projection(
            _quantile_statistics(hidden[active_channels], window_mask.bool()[active_channels], dimension=1)
        )
        if return_gate:
            return embedding, gate, rank
        return embedding

    def forward_group(self, records: list[dict[str, Tensor]], *, return_diagnostics: bool = False):
        """Pool repeated records for aligned channels, then contextualize once."""
        if not records:
            raise RuntimeError("empty patient/EDF record group")
        embeddings, masks, gates, ranks = [], [], [], []
        expected_channels = records[0]["features"].shape[0]
        for record in records:
            if record["features"].shape[0] != expected_channels:
                raise RuntimeError("group channel axis differs across records")
            result = self.record_embedding(record["features"], record["channel_mask"],
                                           record["window_mask"], return_gate=return_diagnostics)
            if return_diagnostics:
                embedding, gate, rank = result
                gates.append(gate[record["channel_mask"].bool() & record["window_mask"].any(1)].detach())
                ranks.append(rank[record["channel_mask"].bool() & record["window_mask"].any(1)].detach())
            else:
                embedding = result
            embeddings.append(embedding)
            masks.append(record["channel_mask"].bool())
        stacked, available = torch.stack(embeddings), torch.stack(masks)
        channel_available = available.any(0)
        record_statistics = stacked.new_zeros((expected_channels, EMBEDDING_DIM * 5))
        for channel in range(expected_channels):
            if bool(channel_available[channel]):
                record_statistics[channel] = _quantile_statistics(
                    stacked[:, channel], available[:, channel], dimension=0
                )
        representation = stacked.new_zeros((expected_channels, EMBEDDING_DIM))
        representation[channel_available] = self.record_projection(record_statistics[channel_available])
        context = representation[channel_available]
        mean, maximum = context.mean(0), context.max(0).values
        contextual = torch.cat((representation, representation - mean, representation - maximum), dim=-1)
        logits = self.classifier(contextual).squeeze(-1)
        output = logits.new_zeros((expected_channels,))
        output[channel_available] = logits[channel_available]
        if return_diagnostics:
            return output, {"gate": torch.cat(gates) if gates else output.new_empty((0, HIDDEN_DIM)),
                            "rank": torch.cat(ranks) if ranks else output.new_empty((0, FEATURE_DIM)),
                            "channel_mask": channel_available}
        return output


def parameter_audit() -> dict[str, int | bool]:
    model = PRiSMEZ()
    count = int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))
    return {"trainable_parameters": count, "parameter_budget": 100000,
            "budget_pass": bool(count < 100000), "recommended_lower_bound": 50000,
            "recommended_upper_bound": 90000,
            "in_recommended_range": bool(50000 <= count <= 90000),
            "no_batchnorm": not any(isinstance(module, nn.modules.batchnorm._BatchNorm)
                                     for module in model.modules())}

