import torch

from neuroez_c.task2.masks import apply_channel_dropout, apply_seizure_dropout, masked_mean, masked_std


def test_masked_statistics_ignore_padding():
    values = torch.tensor([[[1.0], [3.0], [999.0]]])
    mask = torch.tensor([[True, True, False]])
    assert torch.allclose(masked_mean(values, mask, dim=1), torch.tensor([[2.0]]))
    assert torch.allclose(masked_std(values, mask, dim=1), torch.tensor([[1.0]]))


def test_dropout_retains_minimum_valid_items():
    generator = torch.Generator().manual_seed(5)
    seizures = apply_seizure_dropout(torch.ones(3, 4, dtype=torch.bool), 1.0, generator=generator)
    channels = apply_channel_dropout(torch.ones(2, 3, 8, dtype=torch.bool), 1.0, generator=generator, minimum=4)
    assert torch.all(seizures.sum(-1) >= 1)
    assert torch.all(channels.sum(-1) >= 4)


def test_masked_std_has_finite_gradient_for_empty_and_singleton_rows():
    values = torch.randn(2, 3, 4, requires_grad=True)
    mask = torch.tensor([[True, False, False], [False, False, False]])
    masked_std(values, mask, dim=1).sum().backward()
    assert torch.isfinite(values.grad).all()
