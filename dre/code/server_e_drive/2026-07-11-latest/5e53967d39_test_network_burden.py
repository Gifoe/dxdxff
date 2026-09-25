import torch

from neuroez_c.task2.network_burden import compute_network_burden


def test_network_burdens_are_masked_and_bounded():
    adjacency = torch.zeros(1, 1, 3, 4, 4)
    adjacency[..., 0, 1] = adjacency[..., 1, 0] = 1.0
    adjacency[..., 2, 3] = adjacency[..., 3, 2] = 0.5
    risk = torch.tensor([[[1.0, 1.0, 0.0, 0.0]]])
    channel_mask = torch.ones(1, 1, 4, dtype=torch.bool)
    phase_mask = torch.ones(1, 1, 3, 4, dtype=torch.bool)
    out = compute_network_burden(adjacency, risk, channel_mask, phase_mask)
    assert out["network_burden"].shape == (1, 1, 3, 5)
    assert torch.isfinite(out["network_burden"]).all()
    assert torch.all((out["network_burden"] >= 0) & (out["network_burden"] <= 1))
    assert torch.all(out["B_edge"] > out["B_cross"])
