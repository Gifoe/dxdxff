import torch

from neuroez_c.task2.p2_q10_npam_loss import compute_npam_loss


def test_loss_uses_success_logit_and_only_valid_q10_adjustment():
    logit = torch.tensor([0.0, 0.0], requires_grad=True)
    adjustment = torch.tensor([[[1.0, 1000.0]], [[2.0, 1000.0]]], requires_grad=True)
    valid = torch.tensor([[[True, False]], [[True, False]]])
    output = {
        "outcome_logit_success": logit,
        "q10_adjustment": adjustment,
        "q10_residual": adjustment,
        "risk_membership": torch.zeros_like(adjustment),
        "seizure_channel_valid": valid,
    }
    parts = compute_npam_loss(output, torch.tensor([1.0, 0.0]), pos_weight=1.0, lambda_loc=0.0, lambda_residual=1.0)
    assert torch.isclose(parts["residual_loss"], torch.tensor(2.5))
    parts["loss"].backward()
    assert torch.isfinite(logit.grad).all() and torch.isfinite(adjustment.grad).all()
    assert torch.all(adjustment.grad[~valid] == 0)
