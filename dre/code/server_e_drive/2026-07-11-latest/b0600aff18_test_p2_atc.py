from types import SimpleNamespace

import torch

from neuroez_c.cane_path_cp_loss import atc_tail_losses
from neuroez_c.p23_seizure_tail import P23CrossSeizureTailEvidence


def _args(profile: str) -> SimpleNamespace:
    return SimpleNamespace(
        use_p2_atc=True, p2_atc_profile=profile, p2_atc_loss_start_epoch=8,
        p2_atc_loss_ramp_epochs=5, p2_atc_clean_tail_margin=.10,
        p2_atc_clean_tail_weight=.020, p2_atc_pair_margin=.15,
        p2_atc_pair_weight=.020, p2_atc_trusted_ez_fraction=.20,
        p2_atc_max_pairs_per_patient=256,
    )


def _outputs() -> dict[str, torch.Tensor]:
    return {
        "final_nez_logit": torch.zeros(2, 4, requires_grad=True),
        "seizure_nez_robust_tail_logit": torch.tensor([[.8, -.4, .6, -.2], [.7, -.3, .5, -.1]], requires_grad=True),
        "tail_valid": torch.ones(2, 4, dtype=torch.bool),
        "valid_seizure_count": torch.tensor([[2, 2, 1, 2], [3, 2, 2, 1]]),
        "direct_score_nez": torch.tensor([[.9, .1, .8, .2], [.8, .2, .7, .3]], requires_grad=True),
        "anchor_nez_evidence": torch.tensor([[.8, .1, .7, .2], [.7, .2, .6, .3]], requires_grad=True),
    }


def test_a0_uses_legacy_five_feature_head_and_a1_uses_atc_seven_feature_head():
    assert P23CrossSeizureTailEvidence(4, robust_tail=False).evidence[0].in_features == 5
    assert P23CrossSeizureTailEvidence(4, robust_tail=True, feature_mode="atc7").evidence[0].in_features == 7


def test_clean_nez_tail_ignores_single_seizure_and_is_patient_balanced():
    outputs = _outputs(); labels = torch.tensor([[1., 0., 1., 0.], [1., 0., 1., 0.]])
    floor, _, parts = atc_tail_losses(outputs, labels, torch.ones(2, 4, dtype=torch.bool), 12, _args("A2"))
    # The single-seizure clean channel at [0, 2] is excluded.
    assert parts["clean_nez_tail_valid_channel_count"] == 3.0
    assert floor.requires_grad and torch.isfinite(floor)


def test_trusted_ez_pair_uses_valid_cross_seizure_channels_only():
    outputs = _outputs(); labels = torch.tensor([[1., 0., 1., 0.], [1., 0., 1., 0.]])
    _, pair, parts = atc_tail_losses(outputs, labels, torch.ones(2, 4, dtype=torch.bool), 12, _args("A3"))
    assert parts["trusted_ez_valid_patient_count"] == 2.0
    assert parts["tail_pair_count"] > 0
    assert pair.requires_grad and torch.isfinite(pair)


def test_atc_ramp_is_zero_before_epoch_eight():
    outputs = _outputs(); labels = torch.tensor([[1., 0., 1., 0.], [1., 0., 1., 0.]])
    floor, pair, parts = atc_tail_losses(outputs, labels, torch.ones(2, 4, dtype=torch.bool), 7, _args("A3"))
    assert parts["clean_nez_tail_active_weight"] == 0.0
    assert parts["trusted_ez_tail_pair_active_weight"] == 0.0
    assert floor.item() == 0.0 and pair.item() == 0.0
