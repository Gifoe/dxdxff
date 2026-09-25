from __future__ import annotations

import pytest
import torch

from neuroez_c.cane_path_cp_heads import CausalPropagationResidual, MultiSeizureNEZEvidenceResidual
from neuroez_c.p21_selective_fusion import SelectiveResidualFusion, compute_causal_quality_gate


def test_quality_gate_and_causal_invalid_are_bounded():
    gate = compute_causal_quality_gate(
        torch.tensor([[1.0, 0.5]]), torch.tensor([[1.0, 3.0]]),
        torch.tensor([[0.0, 0.5]]), torch.tensor([[1, 0]], dtype=torch.bool),
    )
    assert gate["q_seizure"][0, 0] <= 0.334
    assert gate["q_cp"][0, 0] == 0
    assert gate["q_cp"][0, 1] == 0
    head = CausalPropagationResidual(enable_p21_head=True)
    output = head.forward_p21(torch.randn(1, 2, 6), torch.tensor([[1, 0]]), torch.ones(1, 2, dtype=torch.bool), torch.ones(1, 2), torch.ones(1, 2), torch.ones(1, 2))
    assert output["u_causal"][0, 1] == 0
    assert output["raw_u_causal"].abs().max() <= 1


def test_q10_single_and_missing_seizure():
    head = MultiSeizureNEZEvidenceResidual(4, enable_p21_head=True)
    embedding = torch.randn(2, 3, 2, 4)
    seizure_mask = torch.tensor([[1, 0, 0], [0, 0, 0]], dtype=torch.bool)
    channel_mask = torch.tensor([[[1, 1], [0, 0], [0, 0]], [[0, 0], [0, 0], [0, 0]]], dtype=torch.bool)
    out = head.forward_p21(embedding, seizure_mask, channel_mask, use_q10=True)
    assert torch.allclose(out["seizure_nez_probability_q10"][0], out["seizure_nez_probability_mean"][0])
    assert out["u_seizure"][1].abs().sum() == 0
    assert out["normalized_valid_seizure_count"][0, 0] == pytest.approx(1 / 3)


def test_simplex_initialization_sum_mask_and_bound():
    fusion = SelectiveResidualFusion(6, hidden_dim=8, max_total_correction=0.2).eval()
    mask = torch.tensor([[1, 1, 0]], dtype=torch.bool)
    base = torch.randn(1, 3)
    out = fusion(torch.randn(1, 3, 6), base, torch.randn(1, 3), torch.ones(1, 3), -torch.ones(1, 3), torch.ones(1, 3), torch.ones(1, 3), torch.ones(1, 3), mask)
    weights = torch.stack([out[name] for name in ("w_noop", "w_anchor", "w_seizure", "w_causal")], dim=-1)
    assert torch.allclose(weights[0, :2].sum(-1), torch.ones(2))
    assert weights[0, 0].tolist() == pytest.approx([0.90, 0.04, 0.04, 0.02], abs=1e-6)
    assert out["delta"].abs().max() <= 0.2 + 1e-7
    assert out["delta"][0, 2] == 0
    assert out["final_nez_logit"][0, 0].item() == pytest.approx((base[0, 0] + out["delta"][0, 0]).item())
