from __future__ import annotations

import torch

from task1_baselines.seegformer.fft_frontend import SEEGformerFFTFrontend


def test_fft_frontend_respects_channel_mask_and_frequency_band() -> None:
    frontend = SEEGformerFFTFrontend(sfreq=200.0, n_times=800, n_fft=1600, freq_min=0.5, freq_max=80.0)
    waveform = torch.randn(2, 3, 800)
    mask = torch.tensor([[True, True, False], [True, False, False]])
    real, imag, amplitude = frontend(waveform, mask)
    assert real.shape == imag.shape == amplitude.shape
    assert real.shape[-1] == 637
    assert torch.equal(real[~mask], torch.zeros_like(real[~mask]))
    assert torch.isfinite(amplitude).all()
