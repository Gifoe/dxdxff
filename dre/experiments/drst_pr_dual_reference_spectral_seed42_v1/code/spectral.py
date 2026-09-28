"""Frozen, label-free DRST-PR raw preprocessing and spectral views.

The full 60 s channel scale is computed before extracting aligned 2 s windows.
FIT frequency moments must be supplied by a separate fold-specific source-only
scan; no target waveform contributes to their estimation.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch


def fixed_robust_channel_scales(raw: np.ndarray, valid_start: int = 0,
                                valid_samples: int | None = None) -> np.ndarray:
    if raw.ndim != 2 or raw.shape[1] != 15000 or not np.isfinite(raw).all():
        raise ValueError("Expected finite [channel,15000] 60 s raw waveform")
    valid_end = raw.shape[1] if valid_samples is None else valid_start + valid_samples
    if valid_start < 0 or valid_end > raw.shape[1] or valid_end - valid_start < 500:
        raise ValueError("Invalid full-waveform valid interval")
    valid = raw[:, valid_start:valid_end]
    med = np.median(valid, axis=1, keepdims=True)
    mad = np.median(np.abs(valid - med), axis=1)
    scale = (1.4826 * mad + 1e-6).astype(np.float32)
    if not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("Invalid full-waveform fixed scale")
    return scale


class FixedScaleRawAlignmentStore:
    """Composition around the audited RawAlignmentStore, with no label reads."""

    def __init__(self, base_store) -> None:
        from neuroez_c.dual_view_data import _channel_names, _record_key, _sample

        self.base = base_store
        self.audit = base_store.audit
        self.scales: dict[tuple[str, str, str, str], dict[str, np.float32]] = {}
        for key, rec in base_store.raw_records.items():
            raw = np.asarray(_sample(rec).get("raw_waveform"), dtype=np.float32)
            names = _channel_names(rec)
            if len(names) != raw.shape[0] or key != _record_key(rec):
                raise ValueError("Raw channel identity mismatch")
            sample = _sample(rec)
            values = fixed_robust_channel_scales(raw, int(sample.get("raw_valid_start_sample", 0)),
                                                 int(sample.get("raw_valid_samples", raw.shape[1])))
            self.scales[key] = dict(zip(names, values, strict=True))

    def __getattr__(self, name):
        return getattr(self.base, name)

    def aligned_windows(self, feature_sample: Mapping) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        from neuroez_c.dual_view_data import _channel_names, _record_key

        raw, mask, available = self.base.aligned_windows(feature_sample)
        key = _record_key(feature_sample)
        name_to_scale = self.scales.get(key)
        if name_to_scale is None:
            return raw, mask, available
        scales = np.asarray([name_to_scale.get(name, 1.0) for name in _channel_names(feature_sample)], dtype=np.float32)
        if (available & np.asarray([name not in name_to_scale for name in _channel_names(feature_sample)])).any():
            raise ValueError("Aligned channel without full-waveform scale")
        return raw / scales[None, :, None], mask, available


def log_power_stft(raw: torch.Tensor) -> torch.Tensor:
    """Return [*,51,12] from [*,500], preserving FFT bins 1..51."""
    if raw.shape[-1] != 500:
        raise ValueError("Spectral input must be exactly 500 samples")
    x = raw.float() - raw.float().mean(dim=-1, keepdim=True)
    flat = x.reshape(-1, 500)
    window = torch.hann_window(128, periodic=True, device=x.device, dtype=x.dtype)
    power = torch.stft(flat, n_fft=128, hop_length=32, win_length=128,
                       window=window, center=False, return_complex=True).abs().square()
    out = torch.log(power[:, 1:52] + 1e-6)
    if out.shape[-2:] != (51, 12):
        raise ValueError(f"Unexpected fixed STFT shape {tuple(out.shape)}")
    return out.reshape(*raw.shape[:-1], 51, 12)


def spectral_views(
    log_power: torch.Tensor,
    window_mask: torch.Tensor,
    centers: torch.Tensor,
    frequency_mean: torch.Tensor,
    frequency_std: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A/D/R from [B,S,C,W,F,T], without labels or target-fitted moments."""
    if log_power.ndim != 6 or log_power.shape[-2:] != (51, 12):
        raise ValueError("Expected [B,S,C,W,51,12]")
    if window_mask.shape != log_power.shape[:4] or centers.shape != log_power.shape[:2] + (log_power.shape[3],):
        raise ValueError("Mask or center shape mismatch")
    if frequency_mean.shape != (51,) or frequency_std.shape != (51,) or (frequency_std <= 0).any():
        raise ValueError("Invalid FIT-only frequency moments")
    frequency_mean = frequency_mean.to(device=log_power.device)
    frequency_std = frequency_std.to(device=log_power.device)
    a = (log_power - frequency_mean.view(1, 1, 1, 1, 51, 1)) / (frequency_std.view(1, 1, 1, 1, 51, 1) + 1e-6)
    baseline = window_mask & (centers[:, :, None, :] <= -5.0)
    baseline_count = baseline.sum(dim=3)
    if ((window_mask.any(dim=3)) & (baseline_count < 3)).any():
        raise ValueError("A valid seizure/channel has <3 pre-onset windows")
    b = a.masked_fill(~baseline[..., None, None], float("nan")).nanmedian(dim=3).values
    b = torch.nan_to_num(b, nan=0.0)
    d = a - b[:, :, :, None]
    patient_center = d.masked_fill(~window_mask[..., None, None], float("nan")).nanmedian(dim=2).values
    patient_center = torch.nan_to_num(patient_center, nan=0.0)
    r = d - patient_center[:, :, None]
    keep = window_mask[..., None, None]
    return a * keep, d * keep, r * keep


class FrequencyMoments:
    """Exact streaming mean/std per frequency over FIT windows and frames."""

    def __init__(self) -> None:
        self.count = 0
        self.sum = torch.zeros(51, dtype=torch.float64)
        self.sumsq = torch.zeros(51, dtype=torch.float64)

    def update(self, log_power: torch.Tensor, mask: torch.Tensor) -> None:
        if log_power.shape[:-2] != mask.shape or log_power.shape[-2:] != (51, 12):
            raise ValueError("Moment shape mismatch")
        values = log_power[mask].detach().to("cpu", dtype=torch.float64)
        self.count += int(values.shape[0] * 12)
        self.sum += values.sum(dim=(0, 2))
        self.sumsq += values.square().sum(dim=(0, 2))

    def finalize(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.count < 2:
            raise ValueError("Insufficient FIT spectral observations")
        mean = self.sum / self.count
        var = (self.sumsq / self.count - mean.square()).clamp_min(1e-12)
        return mean.float(), var.sqrt().float()
