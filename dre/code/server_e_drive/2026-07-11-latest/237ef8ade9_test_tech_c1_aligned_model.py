from types import SimpleNamespace

import numpy as np
import torch

from neuroez_c.task2.tech_outcome_c1_aligned.model import TeChOutcomeC1Aligned
from neuroez_c.task2.tech_outcome_c1_aligned.trainer import better_checkpoint
from test_tech_c1_schema import patient


def test_model_averages_logits_only_after_independent_patient_views():
    torch.manual_seed(7)
    model = TeChOutcomeC1Aligned().eval()
    view = patient()
    with torch.inference_mode():
        first = model.forward_patient_views([[view] * 8])[0]
        second = model.forward_patient_views([[view] * 8])[0]
    assert len(first["view_outputs"]) == 8
    assert torch.equal(first["view_logits"], second["view_logits"])
    assert abs(float(first["logit"] - first["view_logits"].mean())) <= 1e-7


def test_one_bce_on_mean_logit_differs_from_mean_view_bce():
    logits = torch.tensor([-3.0, 1.0])
    target = torch.tensor(1.0)
    required = torch.nn.functional.binary_cross_entropy_with_logits(logits.mean(), target)
    forbidden = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, target.expand_as(logits)
    ).mean()
    assert not torch.isclose(required, forbidden)


def test_regularization_and_augmentation_defaults_are_aligned():
    model = TeChOutcomeC1Aligned()
    dropouts = [m.p for m in model.modules() if isinstance(m, torch.nn.Dropout)]
    assert model.augmentations_enabled is False
    assert max(dropouts) <= 0.10
    assert np.isclose(model.layers[0].drop.p, 0.05)


def test_checkpoint_uses_auroc_then_bce_tie_break():
    assert better_checkpoint(0.61, 0.8, 0.60, 0.5)
    assert not better_checkpoint(0.59, 0.1, 0.60, 0.5)
    assert better_checkpoint(0.60, 0.49, 0.60, 0.5)
    assert not better_checkpoint(0.60, 0.49995, 0.60, 0.5)
    assert not better_checkpoint(float("nan"), 0.1, 0.60, 0.5)
