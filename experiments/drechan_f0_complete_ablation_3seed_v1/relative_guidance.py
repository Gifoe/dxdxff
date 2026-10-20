"""Label-free same-patient/seizure/window physiology context, no new weights."""
import torch

EPSILON = 1e-8


def conditioning_delta(delta, mask, scale, keep, collect=False):
    """Preserve the original globally scaled evidence and add relative context.

    The caller supplies one COMPLETE patient, never a channel minibatch. Statistics
    run only over the channel axis, separately for every seizure/window/coordinate.
    Unsupported coordinates keep their original global input. Invalid windows are
    never included in statistics or converted into additional observations.

    Return to original feature units so the unchanged F0 feature function performs
    its original divide-by-FIT-scale and [-8,8] clamp. This FP32 roundtrip is part
    of the fixed executable definition, not an approximate replacement of F0.
    """
    assert delta.ndim == 4 and delta.shape[-1] == 7 and mask.shape == delta.shape[:3]
    assert delta.dtype == torch.float32 and mask.dtype == torch.bool
    assert scale.shape == keep.shape == (7,) and not delta.requires_grad
    assert torch.isfinite(scale).all() and (scale > 0).all()
    masked = torch.where(keep.bool(), delta, torch.zeros_like(delta))
    assert torch.isfinite(masked[mask]).all()
    valid = mask[..., None]
    global_input = torch.where(valid, masked.double() / scale.double(), 0.)
    count = mask.sum(1, keepdim=True)[..., None]
    mean = global_input.sum(1, keepdim=True) / count.clamp_min(1)
    centered = torch.where(valid, global_input - mean, 0.)
    variance = centered.square().sum(1, keepdim=True) / count.clamp_min(1)
    sd = variance.sqrt()
    eligible = (count >= 2) & (sd > EPSILON) & keep.bool()
    relative = centered / torch.where(eligible, sd, torch.ones_like(sd))
    mixed = torch.where(eligible, .5 * (global_input + relative), global_input)
    adjusted = torch.where(valid, mixed * scale.double(), 0.).to(delta.dtype)
    assert torch.isfinite(adjusted).all()
    audit = {}
    if collect:
        active = valid & keep.bool()
        total = int(active.sum())
        used = int((active & eligible).sum())
        audit = dict(relative_active_coordinates=total, relative_used_coordinates=used,
                     global_fallback_coordinates=total-used,
                     singleton_valid_windows=int((mask & (count[..., 0] < 2)).sum()),
                     valid_windows=int(mask.sum()))
    return adjusted, audit
