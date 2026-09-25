"""Frozen probability-space CDEL fusion for PRQ-Net and BCR-Net."""
from __future__ import annotations

import numpy as np

# CDEL has one formal fusion configuration.  Keep the historical P2/V3 aliases
# because several evaluators still use those neutral column names.
LOCKED_BCR_WEIGHT = 0.20
LOCKED_PRQ_WEIGHT = 1.0 - LOCKED_BCR_WEIGHT
LOCKED_P2_WEIGHT = LOCKED_PRQ_WEIGHT
LOCKED_V3_WEIGHT = LOCKED_BCR_WEIGHT
FORMAL_DECISION_RULE = "fixed_80_20_probability_fusion_with_fold_validation_global_threshold"
TRUEK_DECISION_RULE = "true_ez_count_topk_on_fixed_80_20_fused_score"


def bcr_ez_logit_to_nez_probability(ez_logit: np.ndarray) -> np.ndarray:
    """Convert BCR's EZ-oriented logit to the NEZ probability used by CDEL."""
    values = np.asarray(ez_logit, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("BCR EZ logits must be finite")
    return 1.0 / (1.0 + np.exp(values))


def conservative_probability_fusion(
    p2_score_nez: np.ndarray,
    v3_score_nez: np.ndarray,
    *,
    p2_weight: float = LOCKED_P2_WEIGHT,
    v3_weight: float = LOCKED_V3_WEIGHT,
) -> tuple[np.ndarray, np.ndarray]:
    """Fuse frozen NEZ probabilities; logit fusion is intentionally unsupported."""
    p2 = np.asarray(p2_score_nez, dtype=np.float64)
    v3 = np.asarray(v3_score_nez, dtype=np.float64)
    if p2.shape != v3.shape:
        raise ValueError("P2 and V3 probability shapes must match")
    if not np.isfinite(p2).all() or not np.isfinite(v3).all():
        raise ValueError("Fusion probabilities must be finite")
    if ((p2 < 0.0) | (p2 > 1.0)).any() or ((v3 < 0.0) | (v3 > 1.0)).any():
        raise ValueError("Fusion probabilities must lie in [0, 1]")
    if p2_weight < 0.0 or v3_weight < 0.0 or not np.isclose(p2_weight + v3_weight, 1.0, atol=1e-12):
        raise ValueError("Fusion weights must be non-negative and sum to exactly one")
    fused_nez = p2_weight * p2 + v3_weight * v3
    if ((fused_nez < 0.0) | (fused_nez > 1.0)).any():
        raise RuntimeError("Probability fusion escaped [0, 1]")
    return fused_nez, 1.0 - fused_nez


def require_locked_weights(
    p2_weight: float,
    v3_weight: float,
    *,
    diagnostic_allow_nonlocked_weights: bool = False,
) -> str:
    if np.isclose(p2_weight, LOCKED_P2_WEIGHT, atol=1e-12) and np.isclose(v3_weight, LOCKED_V3_WEIGHT, atol=1e-12):
        return "EXPLORATORY_LOCKED_FUSION"
    if not diagnostic_allow_nonlocked_weights:
        raise ValueError("Formal CDEL requires locked weights PRQ=0.80 and BCR=0.20")
    return "DIAGNOSTIC_WEIGHT_ABLATION_NOT_PRIMARY"


__all__ = [
    "LOCKED_BCR_WEIGHT", "LOCKED_PRQ_WEIGHT", "LOCKED_P2_WEIGHT", "LOCKED_V3_WEIGHT",
    "FORMAL_DECISION_RULE", "TRUEK_DECISION_RULE", "bcr_ez_logit_to_nez_probability",
    "conservative_probability_fusion", "require_locked_weights",
]
