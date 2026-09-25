import torch

from neuroez_c.p23_selective_fusion import P23SelectiveFusion


def test_fusion_weights_and_delta_are_bounded():
    module = P23SelectiveFusion(4)
    mask = torch.tensor([[True, True, False]])
    result = module(torch.randn(1, 3, 4), torch.tensor([[0.2, -0.1, 0.0]]), torch.ones(1, 3), -torch.ones(1, 3), torch.ones(1, 3), torch.zeros(1, 3), mask)
    weights = result["w_noop"] + result["w_anchor"] + result["w_seizure"]
    assert torch.allclose(weights[mask], torch.ones_like(weights[mask]))
    assert result["delta"].abs().max().item() <= 0.20 + 1e-6
    assert torch.allclose(result["final_nez_logit"][mask], (torch.tensor([[0.2, -0.1, 0.0]]) + result["delta"])[mask])


def test_causal_gate_has_four_weights_and_masks_invalid_channels():
    module = P23SelectiveFusion(3, use_causal=True)
    mask = torch.tensor([[True, False]])
    result = module(
        torch.randn(1, 2, 3), torch.zeros(1, 2), torch.zeros(1, 2),
        torch.zeros(1, 2), torch.ones(1, 2), torch.zeros(1, 2), mask,
        u_causal=torch.ones(1, 2), q_cp=torch.ones(1, 2),
    )
    weights = result["w_noop"] + result["w_anchor"] + result["w_seizure"] + result["w_causal"]
    assert torch.allclose(weights[mask], torch.ones_like(weights[mask]))
    assert result["w_causal"][~mask].item() == 0.0
