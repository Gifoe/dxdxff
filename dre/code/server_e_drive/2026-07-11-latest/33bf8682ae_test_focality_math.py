from __future__ import annotations

import torch

from outcome_hifos.models.focality import WindowFocalityDescriptor


def _descriptor(anchors: torch.Tensor, responsibilities: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    batch, seizures, windows, channels, targets = responsibilities.shape
    cores = targets - 1
    masses = responsibilities[..., :cores].sum(dim=-2)
    background = responsibilities[..., -1].sum(dim=-1)
    return WindowFocalityDescriptor()(masses, background, responsibilities, anchors, valid)


def test_inter_core_separation_excludes_diagonal_and_handles_single_core() -> None:
    identical = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
    responsibility = torch.tensor([[[[[0.5, 0.5, 0.0]]]]])
    valid = torch.ones((1, 1, 1, 1), dtype=torch.bool)
    assert _descriptor(identical, responsibility, valid)[0, 0, 0, 7].item() == 0.0

    orthogonal = torch.eye(2).unsqueeze(0)
    assert _descriptor(orthogonal, responsibility, valid)[0, 0, 0, 7].item() > 0.9

    single = torch.tensor([[[1.0, 0.0]]])
    single_responsibility = torch.tensor([[[[[1.0, 0.0]]]]])
    assert _descriptor(single, single_responsibility, valid)[0, 0, 0, 7].item() == 0.0


def test_channel_entropy_is_normalized_by_each_windows_valid_channel_count() -> None:
    anchors = torch.eye(2).unsqueeze(0).expand(2, -1, -1)
    responsibilities = torch.zeros((2, 1, 1, 4, 3))
    valid = torch.zeros((2, 1, 1, 4), dtype=torch.bool)
    valid[0, ..., :2] = True
    valid[1, ..., :4] = True
    responsibilities[0, ..., :2, :2] = 0.5
    responsibilities[1, ..., :4, :2] = 0.5
    descriptor = _descriptor(anchors, responsibilities, valid)
    torch.testing.assert_close(descriptor[0, 0, 0, 6], descriptor[1, 0, 0, 6], atol=1e-6, rtol=1e-6)
    assert torch.isfinite(descriptor).all()


def test_channel_entropy_single_valid_channel_is_zero() -> None:
    anchors = torch.eye(2).unsqueeze(0)
    responsibilities = torch.tensor([[[[[0.5, 0.5, 0.0], [99.0, 99.0, 99.0]]]]])
    valid = torch.tensor([[[[True, False]]]])
    descriptor = _descriptor(anchors, responsibilities, valid)
    assert descriptor[0, 0, 0, 6].item() == 0.0
    assert torch.isfinite(descriptor).all()

