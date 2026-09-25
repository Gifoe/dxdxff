from __future__ import annotations

import torch

from outcome_hifos.models.patient_context import _masked_token_statistics


def test_patient_top_fraction_is_independent_of_other_patient_padding() -> None:
    small = torch.tensor([1.0, 3.0]).reshape(1, 1, 1, 2, 1)
    small_mask = torch.ones((1, 1, 1, 2), dtype=torch.bool)
    alone = _masked_token_statistics(small, small_mask, top_fraction=0.2)

    padded = torch.zeros((2, 1, 1, 10, 1))
    mask = torch.zeros((2, 1, 1, 10), dtype=torch.bool)
    padded[0, 0, 0, :2, 0] = torch.tensor([1.0, 3.0])
    mask[0, 0, 0, :2] = True
    padded[1, 0, 0, :, 0] = torch.arange(10, dtype=torch.float32)
    mask[1] = True
    batched = _masked_token_statistics(padded, mask, top_fraction=0.2)

    torch.testing.assert_close(alone[0], batched[0])


def test_patient_top_fraction_ignores_large_invalid_padding_values() -> None:
    values = torch.tensor([1.0, 3.0, 99999.0, 99999.0]).reshape(1, 1, 1, 4, 1)
    mask = torch.tensor([True, True, False, False]).reshape(1, 1, 1, 4)
    statistics = _masked_token_statistics(values, mask, top_fraction=0.5)
    top_value = statistics[0, -1]
    assert top_value.item() == 3.0
