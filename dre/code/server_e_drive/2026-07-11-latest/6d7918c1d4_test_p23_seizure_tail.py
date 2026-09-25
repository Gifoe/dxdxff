import pytest
import torch
from pathlib import Path
from torch import nn

from neuroez_c.p23_seizure_tail import P23CrossSeizureTailEvidence, validate_seizure_tail_quantile


class _FirstFeature(nn.Module):
    def forward(self, value):
        return value[..., :1]


def test_q10_single_and_missing_seizure_are_safe():
    module = P23CrossSeizureTailEvidence(4)
    output = module(torch.randn(1, 2, 2, 4), torch.tensor([[True, False]]), torch.tensor([[[True, False], [False, False]]]))
    assert torch.allclose(output["seizure_nez_probability_q10"][0, 0], output["seizure_nez_probability_mean"][0, 0])
    assert output["valid_seizure_count"][0, 1].item() == 0
    assert output["u_seizure"][0, 1].item() == 0.0
    assert torch.allclose(output["u_seizure"], torch.zeros_like(output["u_seizure"]))


def test_single_seizure_backward_has_finite_gradients():
    module = P23CrossSeizureTailEvidence(4)
    embedding = torch.randn(2, 2, 3, 4, requires_grad=True)
    seizure_mask = torch.tensor([[True, False], [True, False]])
    channel_mask = torch.ones(2, 2, 3, dtype=torch.bool)
    output = module(embedding, seizure_mask, channel_mask)
    (output["seizure_nez_probability_std"].sum() + output["u_seizure"].sum()).backward()
    assert torch.isfinite(embedding.grad).all()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in module.parameters()
    )


def test_q10_and_agreement_are_bounded_for_multiple_seizures():
    module = P23CrossSeizureTailEvidence(3)
    output = module(torch.randn(1, 3, 1, 3), torch.ones(1, 3, dtype=torch.bool), torch.ones(1, 3, 1, dtype=torch.bool))
    assert output["seizure_nez_probability_q10"].shape == (1, 1)
    assert torch.all((output["seizure_agreement"] >= 0.0) & (output["seizure_agreement"] <= 1.0))
    assert output["valid_seizure_count"].item() == 3


@pytest.mark.parametrize("quantile", [0.05, 0.10, 0.20])
def test_allowed_tail_quantiles_match_torch_quantile(quantile):
    module = P23CrossSeizureTailEvidence(1, quantile=quantile)
    module.scorer = _FirstFeature()
    logits = torch.tensor([[-2.0, -0.5, 0.5, 2.0]])
    embedding = logits.view(1, 4, 1, 1)
    output = module(embedding, torch.ones(1, 4, dtype=torch.bool), torch.ones(1, 4, 1, dtype=torch.bool))
    expected = torch.quantile(torch.sigmoid(logits[0]), quantile)
    assert torch.allclose(output["seizure_nez_probability_tail_quantile"][0, 0], expected)
    assert output["seizure_tail_quantile"][0, 0].item() == pytest.approx(quantile)


def test_single_seizure_tail_equals_that_seizure_and_invalid_channels_do_not_contribute():
    module = P23CrossSeizureTailEvidence(1, quantile=0.20)
    module.scorer = _FirstFeature()
    embedding = torch.tensor([[[[-1.0], [10.0]], [[3.0], [-10.0]]]])
    output = module(embedding, torch.tensor([[True, False]]), torch.tensor([[[True, False], [False, False]]]))
    assert torch.allclose(output["seizure_nez_probability_tail_quantile"][0, 0], torch.sigmoid(torch.tensor(-1.0)))
    assert output["seizure_nez_probability_tail_quantile"][0, 1].item() == 0.0
    assert "seizure_nez_probability_q10" not in output


def test_q10_default_preserves_legacy_output_keys_and_invalid_quantile_is_rejected():
    output = P23CrossSeizureTailEvidence(1)(torch.zeros(1, 1, 1, 1), torch.ones(1, 1, dtype=torch.bool), torch.ones(1, 1, 1, dtype=torch.bool))
    assert "seizure_nez_probability_q10" in output
    assert torch.equal(output["seizure_nez_probability_q10"], output["seizure_nez_probability_tail_quantile"])
    with pytest.raises(ValueError, match="p23_seizure_quantile"):
        validate_seizure_tail_quantile(0.15)


def test_generic_tail_fields_are_collected_for_q05_q20_ledgers():
    source = Path("neuroez_c/cane_path_cp_trainer.py").read_text(encoding="utf-8")
    assert '"seizure_nez_probability_tail_quantile"' in source
    assert '"seizure_tail_quantile"' in source
