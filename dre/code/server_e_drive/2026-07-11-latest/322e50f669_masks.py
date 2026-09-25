from __future__ import annotations

import torch


def masked_mean(values: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    valid = mask.bool()
    while valid.ndim < values.ndim:
        valid = valid.unsqueeze(-1)
    safe = torch.where(valid, torch.nan_to_num(values), torch.zeros_like(values))
    denominator = valid.to(values.dtype).sum(dim=dim).clamp_min(1.0)
    return safe.sum(dim=dim) / denominator


def masked_std(values: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    valid = mask.bool()
    mean = masked_mean(values, valid, dim=dim)
    expanded = mean.unsqueeze(dim)
    while valid.ndim < values.ndim:
        valid = valid.unsqueeze(-1)
    centered = torch.where(valid, torch.nan_to_num(values) - expanded, torch.zeros_like(values))
    denominator = valid.to(values.dtype).sum(dim=dim).clamp_min(1.0)
    variance = centered.square().sum(dim=dim) / denominator
    # sqrt(0) has an infinite derivative; torch.where still evaluates that
    # branch for padded/singleton rows and can produce 0*inf -> NaN gradients.
    stable_std = variance.clamp_min(torch.finfo(values.dtype).eps).sqrt()
    return torch.where(denominator > 1.0, stable_std, torch.zeros_like(variance))


def apply_channel_dropout(mask: torch.Tensor, probability: float, *, generator: torch.Generator | None = None, minimum: int = 4) -> torch.Tensor:
    if probability <= 0.0:
        return mask.bool()
    output = mask.bool().clone()
    flat = output.reshape(-1, output.shape[-1])
    for row in flat:
        indices = torch.where(row)[0]
        if indices.numel() <= minimum:
            continue
        keep = torch.rand(indices.numel(), device=row.device, generator=generator) >= float(probability)
        if int(keep.sum()) < minimum:
            order = torch.randperm(indices.numel(), device=row.device, generator=generator)
            keep[order[:minimum]] = True
        row[indices] = keep
    return output


def apply_seizure_dropout(mask: torch.Tensor, probability: float, *, generator: torch.Generator | None = None) -> torch.Tensor:
    if probability <= 0.0:
        return mask.bool()
    output = mask.bool().clone()
    for row in output:
        indices = torch.where(row)[0]
        if indices.numel() >= 2 and float(torch.rand((), device=row.device, generator=generator)) < probability:
            selected = indices[torch.randint(indices.numel(), (), device=row.device, generator=generator)]
            row[selected] = False
    return output


__all__ = ["apply_channel_dropout", "apply_seizure_dropout", "masked_mean", "masked_std"]
