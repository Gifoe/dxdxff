import torch

from neuroez_c.p23_noisy_ez_loss import noise_ramp, observed_ez_reliability


def test_reliability_is_detached_and_bounded():
    score = torch.tensor([[0.1, 0.9]], requires_grad=True)
    observed = torch.tensor([[True, True]])
    value = observed_ez_reliability(score, score, score, score, observed)
    assert not value.requires_grad
    assert torch.all((value >= 0.10) & (value <= 0.95))
    assert noise_ramp(5, 6, 10) == 0.0
    assert noise_ramp(10, 6, 10) == 1.0
