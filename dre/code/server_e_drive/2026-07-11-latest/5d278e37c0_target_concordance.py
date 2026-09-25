from __future__ import annotations

import math

import torch
from sklearn.metrics import average_precision_score, roc_auc_score


ABNORMALITY_FEATURES = (
    "abnormality_mean", "abnormality_std", "abnormality_min", "abnormality_max", "abnormality_q05", "abnormality_q10",
    "abnormality_q25", "abnormality_q50", "abnormality_q75", "abnormality_q90", "abnormality_q95",
    "abnormality_top05_mean", "abnormality_top10_mean", "abnormality_top20_mean", "abnormality_entropy", "abnormality_iqr",
)
TARGET_FEATURES = ("target_count", "target_fraction", "non_target_count", "non_target_fraction")
CONCORDANCE_FEATURES = (
    "coverage_mass", "residual_mass", "target_precision_mass", "outside_abnormality_mean", "inside_abnormality_mean",
    "inside_outside_gap", "inside_outside_ratio", "top05_target_coverage", "top10_target_coverage", "top20_target_coverage",
    "top05_target_miss", "top10_target_miss", "top20_target_miss", "target_count_precision", "target_count_recall",
    "target_count_jaccard", "target_count_ndcg", "within_patient_target_auroc", "within_patient_target_auprc",
    "target_score_margin", "ranking_metric_valid",
)


def _entropy(values: torch.Tensor, eps: float) -> torch.Tensor:
    mass = values.clamp_min(0)
    probability = mass / mass.sum().clamp_min(eps)
    return -(probability * probability.clamp_min(eps).log()).sum()


def compute_target_concordance(abnormality: torch.Tensor, target: torch.Tensor, channel_mask: torch.Tensor, eps: float = 1e-8) -> dict[str, torch.Tensor]:
    rows: dict[str, list[torch.Tensor]] = {name: [] for name in ABNORMALITY_FEATURES + TARGET_FEATURES + CONCORDANCE_FEATURES}
    quantiles = ((.05, "q05"), (.10, "q10"), (.25, "q25"), (.50, "q50"), (.75, "q75"), (.90, "q90"), (.95, "q95"))
    for patient_idx in range(abnormality.shape[0]):
        valid = channel_mask[patient_idx].bool()
        a, t = abnormality[patient_idx][valid], target[patient_idx][valid].to(abnormality.dtype)
        if a.numel() == 0 or not torch.isfinite(t).all() or not torch.all((t == 0) | (t == 1)):
            raise ValueError("clinical_target_mask must be complete binary values on every valid channel")
        rows["abnormality_mean"].append(a.mean()); rows["abnormality_std"].append(a.std(unbiased=False))
        rows["abnormality_min"].append(a.min()); rows["abnormality_max"].append(a.max())
        for q, name in quantiles: rows[f"abnormality_{name}"].append(torch.quantile(a, q))
        for fraction, name in ((.05, "05"), (.10, "10"), (.20, "20")):
            k = min(a.numel(), max(1, math.ceil(a.numel() * fraction)))
            rows[f"abnormality_top{name}_mean"].append(torch.topk(a, k).values.mean())
        rows["abnormality_entropy"].append(_entropy(a, eps)); rows["abnormality_iqr"].append(torch.quantile(a, .75) - torch.quantile(a, .25))
        n, k_target = a.numel(), int(t.sum().item())
        rows["target_count"].append(a.new_tensor(float(k_target))); rows["target_fraction"].append(t.mean())
        rows["non_target_count"].append(a.new_tensor(float(n - k_target))); rows["non_target_fraction"].append(1 - t.mean())
        covered, residual, total = (a * t).sum(), (a * (1-t)).sum(), a.sum().clamp_min(eps)
        rows["coverage_mass"].append(covered/total); rows["residual_mass"].append(residual/total)
        inside = covered/t.sum().clamp_min(eps); outside = residual/(1-t).sum().clamp_min(eps)
        rows["target_precision_mass"].append(inside); rows["outside_abnormality_mean"].append(outside); rows["inside_abnormality_mean"].append(inside)
        rows["inside_outside_gap"].append(inside-outside); rows["inside_outside_ratio"].append(inside/outside.clamp_min(eps))
        for fraction, name in ((.05, "05"), (.10, "10"), (.20, "20")):
            k = min(n, max(1, math.ceil(n * fraction))); chosen = torch.topk(a, k).indices
            coverage = t[chosen].mean(); rows[f"top{name}_target_coverage"].append(coverage); rows[f"top{name}_target_miss"].append(1-coverage)
        chosen = torch.topk(a, max(1, min(n, k_target))).indices
        intersection = t[chosen].sum(); predicted = float(chosen.numel())
        rows["target_count_precision"].append(intersection/predicted); rows["target_count_recall"].append(intersection/max(k_target, 1))
        rows["target_count_jaccard"].append(intersection/(predicted+k_target-intersection).clamp_min(eps))
        order = torch.argsort(a, descending=True)[:max(k_target, 1)]; gains = t[order]
        discounts = 1.0/torch.log2(torch.arange(gains.numel(), device=a.device, dtype=a.dtype)+2.0)
        dcg = (gains*discounts).sum(); ideal = discounts[:max(k_target, 1)].sum().clamp_min(eps)
        rows["target_count_ndcg"].append(dcg/ideal)
        both = bool((t == 0).any() and (t == 1).any())
        if both:
            y, score = t.detach().cpu().numpy(), a.detach().cpu().numpy()
            auroc, auprc = roc_auc_score(y, score), average_precision_score(y, score)
        else: auroc = auprc = 0.0
        rows["within_patient_target_auroc"].append(a.new_tensor(auroc)); rows["within_patient_target_auprc"].append(a.new_tensor(auprc))
        rows["target_score_margin"].append(inside-outside); rows["ranking_metric_valid"].append(a.new_tensor(float(both)))
    return {name: torch.stack(values) for name, values in rows.items()}


def stack_features(features: dict[str, torch.Tensor], names: tuple[str, ...]) -> torch.Tensor:
    return torch.stack([features[name] for name in names], dim=-1)


__all__ = ["ABNORMALITY_FEATURES", "TARGET_FEATURES", "CONCORDANCE_FEATURES", "compute_target_concordance", "stack_features"]
