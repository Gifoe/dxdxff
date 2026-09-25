import torch

from neuroez_c.task2.graph_stability import compute_graph_stability


def test_stability_identical_seizures_and_single_seizure():
    risk = torch.tensor([[[0.1, 0.2, 0.4, 0.8], [0.1, 0.2, 0.4, 0.8]], [[0.2, 0.3, 0.4, 0.5], [0.0, 0.0, 0.0, 0.0]]])
    adjacency = torch.ones(2, 2, 3, 4, 4) - torch.eye(4).view(1, 1, 1, 4, 4)
    seizure_mask = torch.tensor([[True, True], [True, False]])
    channel_mask = seizure_mask.unsqueeze(-1).expand(2, 2, 4)
    valid = seizure_mask.unsqueeze(-1).expand(2, 2, 3)
    out = compute_graph_stability(risk, adjacency, seizure_mask, channel_mask, valid)
    assert torch.isclose(out["risk_stability_mean"][0], torch.tensor(1.0))
    assert torch.isclose(out["edge_stability_mean"][0], torch.tensor(1.0))
    assert out["has_graph_stability"].tolist() == [1.0, 0.0]
