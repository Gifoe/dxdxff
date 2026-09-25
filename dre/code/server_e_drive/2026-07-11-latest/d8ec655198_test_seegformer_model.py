from __future__ import annotations

import torch

from task1_baselines.seegformer.model import SEEGformerTask1


def test_seegformer_fft_and_masked_forward_are_finite() -> None:
    model = SEEGformerTask1(embed_dim=16, num_heads=4, num_blocks=1)
    waveform = torch.zeros((2, 17, 800))
    mask = torch.tensor([[True] * 17, [True] * 3 + [False] * 14])
    real, imag, amplitude = model.frontend(waveform, mask)
    assert real.shape[:2] == imag.shape[:2] == amplitude.shape[:2] == (2, 17)
    assert torch.isfinite(real).all() and torch.isfinite(imag).all() and torch.isfinite(amplitude).all()
    output = model(waveform, mask)["channel_logits"]
    assert output.shape == (2, 17)
    assert torch.equal(output[1, 3:], torch.zeros(14))


def test_seegformer_is_channel_permutation_equivariant_and_padding_invariant() -> None:
    torch.manual_seed(7)
    model = SEEGformerTask1(embed_dim=16, num_heads=4, num_blocks=1).eval()
    waveform = torch.randn((1, 5, 800)); mask = torch.ones((1, 5), dtype=torch.bool)
    original = model(waveform, mask)["channel_logits"]
    order = torch.tensor([3, 0, 4, 1, 2])
    permuted = model(waveform[:, order], mask[:, order])["channel_logits"]
    assert torch.allclose(permuted, original[:, order], atol=1e-5)
    padded = torch.cat([waveform, torch.zeros((1, 4, 800))], dim=1)
    padded_mask = torch.tensor([[True] * 5 + [False] * 4])
    padded_output = model(padded, padded_mask)["channel_logits"]
    assert torch.allclose(padded_output[:, :5], original, atol=1e-5)
