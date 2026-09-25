from __future__ import annotations

import torch


def _robust_patient_z(values: torch.Tensor, valid: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    output = torch.zeros_like(values)
    for patient_idx in range(values.shape[0]):
        current = values[patient_idx, valid[patient_idx]]
        if current.numel() == 0:
            continue
        median = torch.median(current)
        mad = torch.median(torch.abs(current - median))
        scale = 1.4826 * mad
        if float(scale.detach()) < eps:
            scale = torch.std(current, unbiased=False)
        if float(scale.detach()) < eps:
            continue
        output[patient_idx, valid[patient_idx]] = ((current - median) / scale).clamp(-4.0, 4.0)
    return output


def compute_simple_q10(
    seizure_nez_probability: torch.Tensor,
    seizure_mask: torch.Tensor,
    seizure_channel_mask: torch.Tensor,
    channel_mask: torch.Tensor,
    *,
    eps: float = 1e-5,
) -> dict[str, torch.Tensor]:
    """Exact simple cross-seizure Q10; no robust-tail or reliability residual."""
    if seizure_nez_probability.ndim != 3:
        raise ValueError("seizure_nez_probability must be [B,S,C]")
    valid = (
        seizure_mask.bool().unsqueeze(-1)
        & seizure_channel_mask.bool()
        & channel_mask.bool().unsqueeze(1)
        & torch.isfinite(seizure_nez_probability)
    )
    bsz, _, channels = seizure_nez_probability.shape
    q10 = torch.full((bsz, channels), 0.5, dtype=seizure_nez_probability.dtype, device=seizure_nez_probability.device)
    for batch_idx in range(bsz):
        for channel_idx in range(channels):
            values = seizure_nez_probability[batch_idx, :, channel_idx][valid[batch_idx, :, channel_idx]]
            if values.numel() == 1:
                q10[batch_idx, channel_idx] = values[0]
            elif values.numel() > 1:
                q10[batch_idx, channel_idx] = torch.quantile(values, 0.10)
    count = valid.sum(dim=1)
    q10_valid = (count > 0) & channel_mask.bool()
    q10 = torch.where(channel_mask.bool(), q10, torch.full_like(q10, 0.5))
    q10_logit = torch.logit(q10.clamp(eps, 1.0 - eps))
    q10_z = _robust_patient_z(q10_logit, q10_valid, eps=eps).masked_fill(~channel_mask.bool(), 0.0)
    return {
        "simple_q10_nez_probability": q10,
        "simple_q10_nez_logit": q10_logit.masked_fill(~channel_mask.bool(), 0.0),
        "simple_q10_nez_z": q10_z,
        "q10_valid": q10_valid,
        "valid_seizure_count": count,
    }


__all__ = ["compute_simple_q10"]

