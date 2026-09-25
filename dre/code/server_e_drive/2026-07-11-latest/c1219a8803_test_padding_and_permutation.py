from __future__ import annotations

import copy

import torch

from outcome_hifos.models.model_registry import build_model


def _batch() -> dict[str, torch.Tensor]:
    torch.manual_seed(11)
    feature_x = torch.randn(1, 2, 3, 5, 4)
    seizure_mask = torch.tensor([[True, True]])
    window_mask = torch.tensor([[[True, True, True], [True, False, False]]])
    channel_mask = torch.tensor([[True, True, True, True, False]])
    seizure_channel_mask = torch.tensor([[[True, True, True, True, False], [True, False, True, True, False]]])
    window_channel_mask = seizure_mask[:, :, None, None] & window_mask[:, :, :, None] & seizure_channel_mask[:, :, None, :] & channel_mask[:, None, None, :]
    return {
        "feature_x": feature_x,
        "seizure_mask": seizure_mask,
        "window_mask": window_mask,
        "channel_mask": channel_mask,
        "seizure_channel_mask": seizure_channel_mask,
        "window_channel_mask": window_channel_mask,
        "window_centers": torch.tensor([[[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]]]),
        "canonical_index": torch.arange(5)[None, :],
    }


def _config() -> dict:
    return {"model_dim": 12, "num_cores": 3, "dropout": 0.0, "num_heads": 2}


def test_padding_values_do_not_change_h2_or_h5_outputs() -> None:
    base_batch = _batch()
    changed = copy.deepcopy(base_batch)
    changed["feature_x"] = changed["feature_x"].clone()
    changed["feature_x"][~changed["window_channel_mask"]] = 1e8
    for variant in ("H2_HIER_POOL", "H5_ANCHORED_CORE"):
        torch.manual_seed(3)
        model = build_model(variant, _config(), input_dim=4).eval()
        base = model(base_batch)
        other = model(changed)
        torch.testing.assert_close(base["logits"], other["logits"], atol=1e-6, rtol=1e-6)


def test_channel_permutation_preserves_logits_and_permutates_responsibilities() -> None:
    batch = _batch()
    order = torch.tensor([2, 0, 3, 1, 4])
    permuted = {key: value.clone() if torch.is_tensor(value) else value for key, value in batch.items()}
    permuted["feature_x"] = batch["feature_x"][:, :, :, order, :]
    permuted["channel_mask"] = batch["channel_mask"][:, order]
    permuted["seizure_channel_mask"] = batch["seizure_channel_mask"][:, :, order]
    permuted["window_channel_mask"] = batch["window_channel_mask"][:, :, :, order]
    permuted["canonical_index"] = batch["canonical_index"][:, order]
    torch.manual_seed(5)
    model = build_model("H5_ANCHORED_CORE", _config(), input_dim=4).eval()
    base = model(batch)
    changed = model(permuted)
    torch.testing.assert_close(base["logits"], changed["logits"], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(base["core_masses"], changed["core_masses"], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(base["responsibilities"][:, :, :, order, :], changed["responsibilities"], atol=1e-6, rtol=1e-6)
