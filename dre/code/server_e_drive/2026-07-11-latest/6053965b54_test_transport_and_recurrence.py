from __future__ import annotations

import torch

from outcome_hifos.models.model_registry import build_model


def _batch(single_seizure: bool = False) -> dict[str, torch.Tensor]:
    torch.manual_seed(19)
    seizures = 1 if single_seizure else 2
    feature_x = torch.randn(2, seizures, 4, 3, 5)
    seizure_mask = torch.ones((2, seizures), dtype=torch.bool)
    window_mask = torch.ones((2, seizures, 4), dtype=torch.bool)
    channel_mask = torch.ones((2, 3), dtype=torch.bool)
    seizure_channel_mask = torch.ones((2, seizures, 3), dtype=torch.bool)
    window_channel_mask = seizure_mask[:, :, None, None] & window_mask[:, :, :, None] & seizure_channel_mask[:, :, None, :]
    return {
        "feature_x": feature_x,
        "seizure_mask": seizure_mask,
        "window_mask": window_mask,
        "channel_mask": channel_mask,
        "seizure_channel_mask": seizure_channel_mask,
        "window_channel_mask": window_channel_mask,
        "window_centers": torch.arange(4, dtype=torch.float32)[None, None, :].expand(2, seizures, 4),
        "canonical_index": torch.arange(3)[None, :].expand(2, 3),
    }


def _config() -> dict:
    return {
        "model_dim": 12,
        "num_cores": 3,
        "dropout": 0.0,
        "uot_epsilon": 0.1,
        "uot_tau": 0.8,
        "sinkhorn_iterations": 30,
    }


def test_h6_h7_h8_emit_transport_structures() -> None:
    batch = _batch()
    for variant in ("H6_ANCHORED_UOT_DESC", "H7_TRANSPORT_GRAPH", "H8_RECURRENCE"):
        torch.manual_seed(2)
        output = build_model(variant, _config(), input_dim=5).eval()(batch)
        assert output["logits"].shape == (2,)
        assert output["transport_plans"].shape == (2, 2, 3, 3, 3)
        assert output["transport_descriptors"].shape[:3] == (2, 2, 3)
        assert output["seizure_embeddings"].shape[:2] == (2, 2)
        assert torch.isfinite(output["transport_plans"]).all()
        assert torch.isfinite(output["logits"]).all()


def test_h8_single_seizure_fallback_is_finite_and_zero_distance() -> None:
    output = build_model("H8_RECURRENCE", _config(), input_dim=5).eval()(_batch(single_seizure=True))
    assert output["recurrence_distances"].shape == (2, 1, 1)
    assert torch.count_nonzero(output["recurrence_distances"]).item() == 0
    assert torch.isfinite(output["patient_embedding"]).all()
