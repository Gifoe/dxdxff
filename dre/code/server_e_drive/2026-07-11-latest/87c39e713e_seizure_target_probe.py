from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


CROSS_SEIZURE_FEATURES = (
    "stable_target_coverage", "stable_target_residual", "target_stability_inside", "target_stability_outside", "stability_inside_outside_gap",
    "coverage_across_seizures_mean", "coverage_across_seizures_std", "coverage_across_seizures_min",
    "residual_across_seizures_mean", "residual_across_seizures_std", "residual_across_seizures_max",
)


class SeizureTargetProbe(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__(); self.probe = nn.Sequential(nn.LayerNorm(embedding_dim), nn.Linear(embedding_dim, 1))

    def forward(self, embedding: torch.Tensor, valid: torch.Tensor) -> dict[str, torch.Tensor]:
        logits = self.probe(embedding).squeeze(-1).masked_fill(~valid.bool(), 0.0)
        return {"seizure_target_logit": logits, "seizure_target_probability": torch.sigmoid(logits).masked_fill(~valid.bool(), 0.0)}


def patient_balanced_probe_loss(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    losses=[]
    expanded=target.unsqueeze(1).expand_as(logits)
    for index in range(logits.shape[0]):
        mask=valid[index].bool()
        if mask.any(): losses.append(F.binary_cross_entropy_with_logits(logits[index][mask],expanded[index][mask]))
    return torch.stack(losses).mean() if losses else logits.sum()*0.0


def compute_cross_seizure_features(probability: torch.Tensor, target: torch.Tensor, valid: torch.Tensor, eps: float=1e-8) -> dict[str, torch.Tensor]:
    output={name:[] for name in CROSS_SEIZURE_FEATURES}; qstats={name:[] for name in ("target_probability_mean","target_probability_std","target_probability_q10","target_probability_q25","target_probability_q50","target_probability_q75","target_probability_q90","valid_seizure_count","new_q10_std","new_q10_min","new_q10_max")}
    for p in range(probability.shape[0]):
        channel_valid=valid[p].any(dim=0); t=target[p][channel_valid].to(probability.dtype); values=probability[p][:,channel_valid]; masks=valid[p][:,channel_valid]
        stable=[]
        for c in range(values.shape[1]):
            v=values[:,c][masks[:,c]]; stable.append(torch.quantile(v,.10) if v.numel() else values.new_tensor(0.0))
        stable=torch.stack(stable); total=stable.sum().clamp_min(eps)
        qstats["new_q10_std"].append(stable.std(unbiased=False)); qstats["new_q10_min"].append(stable.min()); qstats["new_q10_max"].append(stable.max())
        inside=(stable*t).sum()/t.sum().clamp_min(eps); outside=(stable*(1-t)).sum()/(1-t).sum().clamp_min(eps)
        output["stable_target_coverage"].append((stable*t).sum()/total); output["stable_target_residual"].append((stable*(1-t)).sum()/total)
        output["target_stability_inside"].append(inside); output["target_stability_outside"].append(outside); output["stability_inside_outside_gap"].append(inside-outside)
        coverage=[]; residual=[]
        for s in range(values.shape[0]):
            m=masks[s]
            if not m.any(): continue
            score=values[s][m]; tt=t[m]; denom=score.sum().clamp_min(eps)
            coverage.append((score*tt).sum()/denom); residual.append((score*(1-tt)).sum()/denom)
        cov=torch.stack(coverage) if coverage else values.new_zeros(1); res=torch.stack(residual) if residual else values.new_zeros(1)
        output["coverage_across_seizures_mean"].append(cov.mean()); output["coverage_across_seizures_std"].append(cov.std(unbiased=False)); output["coverage_across_seizures_min"].append(cov.min())
        output["residual_across_seizures_mean"].append(res.mean()); output["residual_across_seizures_std"].append(res.std(unbiased=False)); output["residual_across_seizures_max"].append(res.max())
        flat=values[masks]; qstats["target_probability_mean"].append(flat.mean()); qstats["target_probability_std"].append(flat.std(unbiased=False))
        for q,name in ((.1,"q10"),(.25,"q25"),(.5,"q50"),(.75,"q75"),(.9,"q90")): qstats[f"target_probability_{name}"].append(torch.quantile(flat,q))
        qstats["valid_seizure_count"].append(valid[p].any(dim=-1).sum().to(probability.dtype))
    return {name:torch.stack(values) for name,values in {**output,**qstats}.items()}


__all__=["CROSS_SEIZURE_FEATURES","SeizureTargetProbe","compute_cross_seizure_features","patient_balanced_probe_loss"]
