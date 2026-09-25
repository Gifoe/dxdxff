import torch

from neuroez_c.task2.simple_q10 import compute_simple_q10


def test_q10_zero_one_many_and_masks():
    values = torch.tensor([[[0.1, 0.8, 0.9, 0.2], [0.5, 0.4, 0.1, 0.9], [0.9, 0.2, 0.5, 0.7]]])
    seizure_mask = torch.tensor([[True, True, False]])
    seizure_channel_mask = torch.tensor([[[True, True, False, True], [True, False, False, True], [True, True, True, True]]])
    channel_mask = torch.tensor([[True, True, True, False]])
    out = compute_simple_q10(values, seizure_mask, seizure_channel_mask, channel_mask)
    assert torch.allclose(out["simple_q10_nez_probability"][0], torch.tensor([0.14, 0.8, 0.5, 0.5]), atol=1e-6)
    assert out["valid_seizure_count"].tolist() == [[2, 1, 0, 0]]
    assert out["q10_valid"].tolist() == [[True, True, False, False]]
    assert torch.isfinite(out["simple_q10_nez_z"]).all()
    assert out["simple_q10_nez_z"].abs().max() <= 4.0

