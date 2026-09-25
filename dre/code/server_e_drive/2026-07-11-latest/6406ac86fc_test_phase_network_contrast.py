import torch

from neuroez_c.task2.phase_network import assemble_phase_network_features


def test_phase_delta_order_and_validity():
    signatures = torch.arange(15, dtype=torch.float32).reshape(1, 1, 3, 5)
    valid = torch.tensor([[[True, True, False]]])
    out = assemble_phase_network_features(signatures, valid, include_deltas=True)
    assert out["network_feature_vector"].shape[-1] == 30
    assert torch.allclose(out["phase_delta"][0, 0, 0], torch.full((5,), 5.0))
    assert out["delta_valid"].tolist() == [[[True, False]]]
    assert torch.all(out["phase_delta"][0, 0, 1:] == 0)
    assert assemble_phase_network_features(signatures, valid, include_deltas=False)["network_feature_vector"].shape[-1] == 18
