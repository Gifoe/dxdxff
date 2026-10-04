import torch

from modeling import A1ContextResidual, patient_relative_z


def test_zero_projection_recovers_zero_residual() -> None:
    torch.manual_seed(42)
    plugin = A1ContextResidual()
    anchors = torch.randn(2, 7, 32)
    mask = torch.tensor([[True, True, True, False, False, False, False],
                         [True, True, True, True, True, True, True]])
    residual = plugin(anchors, mask)
    assert torch.max(torch.abs(residual)).item() == 0.0


def test_relative_z_excludes_masked_channels() -> None:
    values = torch.tensor([[[1.0], [3.0], [999.0]]])
    mask = torch.tensor([[True, True, False]])
    result = patient_relative_z(values, mask)
    assert torch.allclose(result[0, :2, 0], torch.tensor([-1.0, 1.0]), atol=1e-5)
    assert result[0, 2, 0].item() == 0.0
