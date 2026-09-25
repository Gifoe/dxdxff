import torch

from neuroez_c.task2.channel_pam import ChannelOutcomePAM


def test_pam_q10_direction_bounds_and_masks():
    pam = ChannelOutcomePAM(4)
    with torch.no_grad():
        pam.base_failure_evidence_head[1].weight.zero_()
        pam.base_failure_evidence_head[1].bias.zero_()
    embeddings = torch.ones(1, 1, 3, 4)
    valid = torch.tensor([[[True, True, False]]])
    out = pam(embeddings, torch.tensor([[2.0, -2.0, 0.0]]), valid)
    assert out["failure_risk_logit"][0, 0, 0] < out["failure_risk_logit"][0, 0, 1]
    assert torch.allclose(out["q10_adjustment"], -out["beta_q"] * torch.tensor([[[2.0, -2.0, 0.0]]]) * valid)
    assert 0.0 < float(out["beta_q"].detach()) < 0.2
    assert torch.all((out["risk_membership"] >= 0) & (out["risk_membership"] <= 1))
    assert out["risk_membership"][0, 0, 2] == 0
    assert out["node_failure_burden"].shape == (1, 1)


def test_pam_variable_channel_count():
    for channels in (4, 11):
        out = ChannelOutcomePAM(5)(torch.randn(2, 3, channels, 5), torch.randn(2, channels), torch.ones(2, 3, channels, dtype=torch.bool))
        assert out["risk_embedding"].shape == (2, 3, 5)
