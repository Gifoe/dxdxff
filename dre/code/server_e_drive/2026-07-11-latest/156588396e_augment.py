from __future__ import annotations
import torch

def augment(windows: torch.Tensor, side: torch.Tensor, mask: torch.Tensor, generator=None):
    """Shape/identity preserving native SSL augmentation."""
    generator = generator or torch.Generator(device=windows.device)
    scale = .90 + .20 * torch.rand((), generator=generator, device=windows.device)
    robust = (windows - windows.median(-1, keepdim=True).values).abs().median(-1).values * 1.4826
    noise = torch.randn_like(windows) * (.02 * torch.rand((), generator=generator, device=windows.device) * robust[..., None])
    masked = mask.bool().clone(); valid = torch.where(masked.any(0))[0]
    if len(valid):
        n = max(1, round(.25 * len(valid))); chosen = valid[torch.randperm(len(valid), generator=generator, device=windows.device)[:n]]
        masked[:, chosen] = False
        length = max(1, int(.15 * len(valid))); start = int(torch.randint(len(valid), (), generator=generator, device=windows.device))
        masked[:, valid[start:start + length]] = False
    out_side = side.float().clone(); out_side[..., torch.rand(out_side.shape[-1], generator=generator, device=side.device) < .10] = 0
    return windows.float() * scale + noise, out_side, masked
