from __future__ import annotations

import numpy as np
import torch

from outcome_hifos.models.baselines import SummaryMLModel, patient_summary_features
from outcome_hifos.models.model_registry import build_model


def _batch() -> dict[str, torch.Tensor]:
    torch.manual_seed(7)
    feature_x = torch.randn(2, 2, 3, 4, 5)
    seizure_mask = torch.tensor([[True, True], [True, False]])
    window_mask = torch.tensor([[[True, True, True], [True, True, False]], [[True, True, False], [False, False, False]]])
    channel_mask = torch.tensor([[True, True, True, True], [True, True, True, False]])
    seizure_channel_mask = torch.tensor([[[True, True, True, True], [True, False, True, True]], [[True, True, True, False], [False, False, False, False]]])
    window_channel_mask = seizure_mask[:, :, None, None] & window_mask[:, :, :, None] & seizure_channel_mask[:, :, None, :] & channel_mask[:, None, None, :]
    return {
        "feature_x": feature_x,
        "seizure_mask": seizure_mask,
        "window_mask": window_mask,
        "channel_mask": channel_mask,
        "seizure_channel_mask": seizure_channel_mask,
        "window_channel_mask": window_channel_mask,
        "window_centers": torch.arange(3, dtype=torch.float32)[None, None, :].expand(2, 2, 3),
        "canonical_index": torch.arange(4)[None, :].expand(2, 4),
    }


def _config() -> dict:
    return {"model_dim": 16, "num_cores": 4, "dropout": 0.0, "num_heads": 2}


def test_h1_summary_ml_fits_patient_level_probabilities() -> None:
    batch = _batch()
    features = patient_summary_features(batch)
    model = SummaryMLModel(random_seed=4).fit(features.detach().numpy(), np.asarray([0, 1]))
    probabilities = model.predict_proba(features.detach().numpy())
    assert probabilities.shape == (2,)
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))


def test_h2_to_h5_registry_models_return_patient_logits() -> None:
    batch = _batch()
    for variant in ("H2_HIER_POOL", "H3_ATTENTION_MIL", "H4_MULTI_CORE", "H5_ANCHORED_CORE"):
        model = build_model(variant, _config(), input_dim=5).eval()
        output = model(batch)
        assert output["logits"].shape == (2,)
        assert output["patient_embedding"].shape[0] == 2
        assert torch.isfinite(output["logits"]).all()


def test_competitive_assignment_includes_null_and_masks_invalid_channels() -> None:
    batch = _batch()
    model = build_model("H5_ANCHORED_CORE", _config(), input_dim=5).eval()
    output = model(batch)
    responsibility = output["responsibilities"]
    assert responsibility.shape == (2, 2, 3, 4, 5)
    valid = batch["window_channel_mask"]
    torch.testing.assert_close(responsibility.sum(dim=-1)[valid], torch.ones_like(responsibility.sum(dim=-1)[valid]))
    assert torch.count_nonzero(responsibility[~valid]).item() == 0
    assert output["core_masses"].shape == (2, 2, 3, 4)


def test_patient_conditioned_anchors_change_but_global_anchors_do_not() -> None:
    batch = _batch()
    batch["feature_x"][1] += 10.0
    h4 = build_model("H4_MULTI_CORE", _config(), input_dim=5).eval()(batch)
    h5 = build_model("H5_ANCHORED_CORE", _config(), input_dim=5).eval()(batch)
    torch.testing.assert_close(h4["core_anchors"][0], h4["core_anchors"][1])
    assert not torch.allclose(h5["core_anchors"][0], h5["core_anchors"][1])
