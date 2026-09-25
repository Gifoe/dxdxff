from __future__ import annotations

import torch

from outcome_hifos.baselines.hierarchical import SharedHierPoolV1, TokenHierarchicalOutcomeModel
from outcome_hifos.models.model_registry import build_model


class TinyEncoder(torch.nn.Module):
    def __init__(self, output_dim: int = 4) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.projection = torch.nn.Linear(8, output_dim)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.projection(values.squeeze(1))


def _batch() -> dict[str, torch.Tensor]:
    torch.manual_seed(5)
    values = torch.randn(2, 2, 3, 4, 8)
    mask = torch.ones(2, 2, 3, 4, dtype=torch.bool)
    mask[1, 1] = False
    return {
        "feature_x": values,
        "window_channel_mask": mask,
        "window_mask": mask.any(dim=-1),
        "seizure_mask": mask.any(dim=(-1, -2)),
    }


def test_shared_hier_pool_emits_explicit_hierarchy_and_is_channel_permutation_invariant() -> None:
    torch.manual_seed(7)
    pool = SharedHierPoolV1(token_dim=4, hidden_dim=6, dropout=0.0).eval()
    tokens = torch.randn(2, 2, 3, 4, 4)
    mask = _batch()["window_channel_mask"]
    output = pool(tokens, mask)
    permutation = torch.tensor([2, 0, 3, 1])
    permuted = pool(tokens.index_select(3, permutation), mask.index_select(3, permutation))
    assert output["window_embeddings"].shape == (2, 2, 3, 12)
    assert output["seizure_embeddings"].shape == (2, 2, 36)
    assert output["patient_embedding"].shape == (2, 108)
    torch.testing.assert_close(output["logits"], permuted["logits"], atol=1e-6, rtol=1e-6)


def test_token_hierarchical_model_encodes_only_valid_tokens_and_freezes_fm() -> None:
    encoder = TinyEncoder()
    model = TokenHierarchicalOutcomeModel(encoder, SharedHierPoolV1(4, hidden_dim=6, dropout=0.0), frozen_backbone=True).eval()
    output = model(_batch())
    assert output["logits"].shape == (2,)
    assert output["valid_token_count"].item() == 36
    assert all(not parameter.requires_grad for parameter in model.encoder.parameters())
    assert any(parameter.requires_grad for parameter in model.aggregator.parameters())


def test_fm_registry_builds_frozen_identity_backbone_with_trainable_shared_pool() -> None:
    model = build_model("T2_CBramod_Frozen_HierPool", {"model_dim": 6, "dropout": 0.0}, input_dim=4)
    assert isinstance(model, TokenHierarchicalOutcomeModel)
    assert model.frozen_backbone is True
    assert all(not parameter.requires_grad for parameter in model.encoder.parameters())
    assert any(parameter.requires_grad for parameter in model.aggregator.parameters())
