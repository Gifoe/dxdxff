from __future__ import annotations

import torch

from outcome_hifos.models.model_registry import build_model


def _batch() -> dict[str, torch.Tensor]:
    torch.manual_seed(23)
    feature_x = torch.randn(2, 2, 3, 4, 5)
    seizure_mask = torch.tensor([[True, True], [True, False]])
    window_mask = torch.tensor([[[True, True, True], [True, True, False]], [[True, False, False], [False, False, False]]])
    seizure_channel_mask = torch.tensor([[[True, True, True, True], [True, False, True, True]], [[True, True, False, False], [False, False, False, False]]])
    window_channel_mask = seizure_mask[:, :, None, None] & window_mask[:, :, :, None] & seizure_channel_mask[:, :, None, :]
    return {
        "feature_x": feature_x,
        "seizure_mask": seizure_mask,
        "window_mask": window_mask,
        "channel_mask": seizure_channel_mask.any(dim=1),
        "seizure_channel_mask": seizure_channel_mask,
        "window_channel_mask": window_channel_mask,
        "window_centers": torch.tensor([[[-1.0, 0.0, 2.0], [-2.0, 1.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]),
        "canonical_index": torch.arange(4)[None, :].expand(2, 4),
    }


def _config() -> dict[str, float | int]:
    return {"model_dim": 12, "num_cores": 3, "dropout": 0.0}


def test_h2_h3_emit_explicit_window_seizure_patient_hierarchy() -> None:
    batch = _batch()
    for variant in ("H2_HIER_POOL", "H3_ATTENTION_MIL"):
        output = build_model(variant, _config(), input_dim=5).eval()(batch)
        assert output["window_embeddings"].shape == (2, 2, 3, 12)
        assert output["seizure_embeddings"].shape == (2, 2, 12)
        assert output["patient_embedding"].shape == (2, 48)
        assert output["logits"].shape == (2,)


def test_h2_h3_are_channel_permutation_invariant_and_padding_safe() -> None:
    batch = _batch()
    permutation = torch.tensor([2, 0, 3, 1])
    permuted = {key: value.clone() if torch.is_tensor(value) else value for key, value in batch.items()}
    for key in ("feature_x", "window_channel_mask"):
        permuted[key] = permuted[key].index_select(3, permutation)
    for key in ("channel_mask", "seizure_channel_mask"):
        permuted[key] = permuted[key].index_select(-1, permutation)
    poisoned = {key: value.clone() if torch.is_tensor(value) else value for key, value in batch.items()}
    poisoned["feature_x"][~batch["window_channel_mask"]] = 99999.0

    for variant in ("H2_HIER_POOL", "H3_ATTENTION_MIL"):
        torch.manual_seed(9)
        model = build_model(variant, _config(), input_dim=5).eval()
        reference = model(batch)["logits"]
        torch.testing.assert_close(reference, model(permuted)["logits"], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(reference, model(poisoned)["logits"], atol=1e-5, rtol=1e-5)


def test_h3_single_seizure_and_single_window_attention_is_finite() -> None:
    batch = _batch()
    batch["seizure_mask"][:] = False
    batch["seizure_mask"][:, 0] = True
    batch["window_mask"][:] = False
    batch["window_mask"][:, 0, 0] = True
    batch["window_channel_mask"] = (
        batch["seizure_mask"][:, :, None, None]
        & batch["window_mask"][:, :, :, None]
        & batch["seizure_channel_mask"][:, :, None, :]
    )
    output = build_model("H3_ATTENTION_MIL", _config(), input_dim=5).eval()(batch)
    assert torch.isfinite(output["logits"]).all()
    torch.testing.assert_close(output["seizure_attention"][:, 0], torch.ones(2))
    assert torch.count_nonzero(output["seizure_attention"][:, 1:]).item() == 0
