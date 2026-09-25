from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_npam_loss(
    output: dict[str, torch.Tensor],
    outcome_target: torch.Tensor,
    *,
    pos_weight: torch.Tensor | float,
    localization_target_nez: torch.Tensor | None = None,
    localization_mask: torch.Tensor | None = None,
    lambda_loc: float = 0.20,
    lambda_residual: float = 1e-4,
) -> dict[str, torch.Tensor]:
    logit = output["outcome_logit_success"]
    weight = torch.as_tensor(pos_weight, dtype=logit.dtype, device=logit.device)
    outcome_loss = F.binary_cross_entropy_with_logits(logit, outcome_target.to(logit.dtype), pos_weight=weight)
    zero = outcome_loss * 0.0
    localization_loss = zero
    if localization_target_nez is not None and localization_mask is not None and "final_nez_logit" in output:
        valid = localization_mask.bool()
        if torch.any(valid):
            localization_loss = F.binary_cross_entropy_with_logits(output["final_nez_logit"][valid], localization_target_nez.to(output["final_nez_logit"].dtype)[valid])
    residual_terms = []
    if "graph_update" in output and output["graph_update"].numel():
        valid = output.get("graph_phase_valid")
        residual_terms.append(output["graph_update"][valid].square().mean() if valid is not None and torch.any(valid) else zero)
    if "q10_adjustment" in output and output["q10_adjustment"].numel():
        valid = output.get("seizure_channel_valid")
        if valid is None:
            valid = output["risk_membership"] >= 0.0
        residual_terms.append(output["q10_adjustment"][valid].square().mean() if torch.any(valid) else zero)
    residual_loss = torch.stack(residual_terms).sum() if residual_terms else zero
    total = outcome_loss + float(lambda_loc) * localization_loss + float(lambda_residual) * residual_loss
    return {"loss": total, "outcome_loss": outcome_loss, "localization_loss": localization_loss, "residual_loss": residual_loss}


__all__ = ["compute_npam_loss"]
