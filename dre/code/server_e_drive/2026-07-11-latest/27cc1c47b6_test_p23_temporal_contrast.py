import torch

from neuroez_c.p23_temporal_contrast import P23TemporalContrastEncoder


def test_real_centers_define_bins_and_zero_init_delta():
    module = P23TemporalContrastEncoder(4)
    windows = torch.randn(1, 1, 5, 2, 4)
    centers = torch.tensor([[[-2.0, 0.0, 10.0, 20.0, 40.0]]])
    output = module(windows, centers, torch.ones(1, 1, 5, dtype=torch.bool), torch.tensor([[[True, False]]]))
    assert output["valid_pre"][0, 0, 0]
    assert output["valid_onset"][0, 0, 0]
    assert output["valid_spread"][0, 0, 0]
    assert output["valid_late"][0, 0, 0]
    assert torch.allclose(output["temporal_delta"], torch.zeros_like(output["temporal_delta"]))
    assert not output["valid_pre"][0, 0, 1]


def test_single_window_slope_is_invalid():
    module = P23TemporalContrastEncoder(3)
    result = module(torch.ones(1, 1, 1, 1, 3), torch.zeros(1, 1, 1), torch.ones(1, 1, 1, dtype=torch.bool), torch.ones(1, 1, 1, dtype=torch.bool))
    assert not result["slope_valid"].item()
    assert result["slope_norm"].item() == 0.0


def test_padding_and_missing_bins_do_not_create_temporal_signal():
    module = P23TemporalContrastEncoder(2)
    windows = torch.ones(1, 1, 3, 1, 2)
    centers = torch.tensor([[[-5.0, 8.0, 99.0]]])
    mask = torch.tensor([[[True, True, False]]])
    result = module(windows, centers, mask, torch.ones(1, 1, 1, dtype=torch.bool))
    assert result["valid_pre"].item()
    assert result["valid_onset"].item()
    assert not result["valid_spread"].item()
    assert not result["valid_late"].item()
    assert torch.allclose(result["temporal_delta"], torch.zeros_like(result["temporal_delta"]))


def test_nan_padding_is_excluded_before_temporal_reductions():
    module = P23TemporalContrastEncoder(2)
    windows = torch.tensor([[[[[1.0, 2.0]], [[float("nan"), float("nan")]]]]])
    centers = torch.tensor([[[-2.0, 40.0]]])
    mask = torch.tensor([[[True, False]]])
    result = module(windows, centers, mask, torch.ones(1, 1, 1, dtype=torch.bool))
    assert all(torch.isfinite(value).all() for value in result.values() if torch.is_tensor(value))


def test_nonfinite_nominal_window_is_excluded_like_legacy_temporal_encoder():
    module = P23TemporalContrastEncoder(2)
    windows = torch.tensor([[[[[1.0, 2.0]], [[float("nan"), float("nan")]]]]])
    centers = torch.tensor([[[-2.0, 40.0]]])
    mask = torch.tensor([[[True, True]]])
    result = module(windows, centers, mask, torch.ones(1, 1, 1, dtype=torch.bool))
    assert result["valid_pre"].item()
    assert not result["valid_late"].item()
    assert torch.isfinite(result["temporal_delta"]).all()
