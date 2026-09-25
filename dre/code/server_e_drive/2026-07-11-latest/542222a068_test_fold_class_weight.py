from __future__ import annotations

import json

import pytest
import torch

from outcome_hifos.training.losses import compute_fold_class_weight, compute_outcome_loss


def test_fold_weight_is_fixed_across_single_patient_batches(tmp_path) -> None:
    audit = compute_fold_class_weight([1.0, 0.0, 0.0], ["success", "failure-a", "failure-b"])
    assert audit.positive_count == 1
    assert audit.negative_count == 2
    assert audit.pos_weight == pytest.approx(2.0)

    positive_loss, _ = compute_outcome_loss(
        {"logits": torch.tensor([0.0])},
        torch.tensor([1.0]),
        {},
        pos_weight=audit.pos_weight,
    )
    negative_loss, _ = compute_outcome_loss(
        {"logits": torch.tensor([0.0])},
        torch.tensor([0.0]),
        {},
        pos_weight=audit.pos_weight,
    )
    assert positive_loss == pytest.approx(2.0 * negative_loss.item())

    output = tmp_path / "class_weight_audit.json"
    audit.save(output)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["fit_subject_ids"] == ["success", "failure-a", "failure-b"]
    assert "validation" not in payload["fit_subject_ids"]


@pytest.mark.parametrize("targets", ([0.0, 0.0], [1.0, 1.0]))
def test_single_class_fold_weight_fails_fast(targets) -> None:
    with pytest.raises(ValueError, match="both outcome classes"):
        compute_fold_class_weight(targets, ["a", "b"])

