import torch

from neuroez_c.task2.bounded_graph_residual import BoundedPhaseGraphResidual


def test_single_bounded_residual_and_invalid_identity():
    module = BoundedPhaseGraphResidual(6).eval()
    h = torch.randn(2, 3, 3, 5, 6)
    adjacency = torch.rand(2, 3, 3, 5, 5)
    adjacency = (adjacency + adjacency.transpose(-1, -2)) / 2
    adjacency.diagonal(dim1=-2, dim2=-1).zero_()
    node_mask = torch.ones(2, 3, 3, 5, dtype=torch.bool)
    phase_valid = torch.ones(2, 3, 3, dtype=torch.bool)
    phase_valid[:, :, 2] = False
    out = module(h, adjacency, node_mask, phase_valid)
    assert 0.0 < float(out["graph_gate"].detach()) < 0.2
    assert out["graph_embedding"].shape == (2, 3, 64)
    assert torch.allclose(out["phase_graph_embedding"][:, :, 2], h[:, :, 2])
    assert torch.all(out["graph_update"][:, :, 2] == 0)
