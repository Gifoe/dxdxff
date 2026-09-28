"""Source-only synthetic invariant checks; no labels or outcomes are opened."""

import torch

from model import DRSTPatientModel, parameter_count
from spectral import FrequencyMoments, log_power_stft, spectral_views


def run() -> None:
    torch.manual_seed(42)
    x = torch.randn(2, 500) * 3
    lp = log_power_stft(x)
    assert lp.shape == (2, 51, 12)
    assert torch.isfinite(lp).all()
    moments = FrequencyMoments()
    moments.update(lp, torch.ones(2, dtype=torch.bool))
    mean, std = moments.finalize()
    assert mean.shape == std.shape == (51,) and (std > 0).all()
    # Two channels share an identical pre-onset segment. Both relative views
    # must be exactly zero at that reference segment.
    base = lp[0].view(1, 1, 1, 1, 51, 12).expand(1, 1, 2, 3, 51, 12).clone()
    centers = torch.tensor([[[-9.0, -7.0, -5.0]]])
    mask = torch.ones(1, 1, 2, 3, dtype=torch.bool)
    a, d, r = spectral_views(base, mask, centers, mean, std)
    assert a.shape == (1, 1, 2, 3, 51, 12)
    assert d.abs().max() < 1e-7 and r.abs().max() < 1e-7
    model = DRSTPatientModel("S4_DRST_PR", mean, std, chunk_size=4)
    raw = torch.randn(1, 1, 2, 4, 500)
    batch = {"raw_windows": raw, "raw_window_mask": torch.ones(1, 1, 2, 4, dtype=torch.bool),
             "window_centers": torch.tensor([[[-9.0, -7.0, -5.0, 1.0]]]),
             "seizure_mask": torch.ones(1, 1, dtype=torch.bool),
             "raw_seizure_channel_mask": torch.ones(1, 1, 2, dtype=torch.bool),
             "channel_mask": torch.ones(1, 2, dtype=torch.bool)}
    model.eval()
    with torch.no_grad():
        out = model(batch)
        permutation = torch.tensor([1, 0])
        pb = dict(batch)
        pb["raw_windows"] = raw[:, :, permutation]
        pb["raw_window_mask"] = batch["raw_window_mask"][:, :, permutation]
        pb["raw_seizure_channel_mask"] = batch["raw_seizure_channel_mask"][:, :, permutation]
        pb["channel_mask"] = batch["channel_mask"][:, permutation]
        p = model(pb)
        assert torch.allclose(out["logits"][:, permutation], p["logits"], atol=1e-5)
    model.train()
    loss = model(batch)["logits"].square().sum()
    loss.backward()
    assert model.window.tokenizer[0].weight.grad is not None
    print({"pass": True, "S4_parameters": parameter_count(model), "logit_shape": tuple(out["logits"].shape)}, flush=True)


if __name__ == "__main__":
    run()
