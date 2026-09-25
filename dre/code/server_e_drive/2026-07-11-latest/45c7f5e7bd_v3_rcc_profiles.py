"""Frozen V3-RCC profile contract."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class V3RCCProfile:
    name: str
    use_coverage: bool
    use_hybrid_bce: bool


V3_RCC_PROFILES = {
    "R0_BASE": V3RCCProfile("R0_BASE", use_coverage=False, use_hybrid_bce=False),
    "R1_RANK_COVERAGE": V3RCCProfile("R1_RANK_COVERAGE", use_coverage=True, use_hybrid_bce=False),
    "R2_HYBRID_CALIBRATION": V3RCCProfile("R2_HYBRID_CALIBRATION", use_coverage=True, use_hybrid_bce=True),
}


def get_v3_rcc_profile(name: str) -> V3RCCProfile:
    key = str(name).strip().upper()
    if key not in V3_RCC_PROFILES:
        raise ValueError(f"Unknown V3-RCC profile {name!r}; expected {sorted(V3_RCC_PROFILES)}")
    return V3_RCC_PROFILES[key]


def validate_v3_rcc_args(args: object) -> V3RCCProfile:
    profile = get_v3_rcc_profile(getattr(args, "v3_rcc_profile", "R0_BASE"))
    if str(getattr(args, "positive_label", "ez")).lower() != "ez":
        raise ValueError("V3-RCC requires the V3 EZ-positive supervised adapter.")
    forbidden = (
        "use_v3_qbc", "use_n6_dual_view_ema", "use_two_expert_router",
        "use_feature_separated_two_expert", "use_a9v8_lcbo", "use_broad_ez_mil_loss",
        "use_diffusion_residual", "use_view_gated_fusion", "use_hard_topk_loss",
    )
    enabled = [name for name in forbidden if bool(getattr(args, name, False))]
    if enabled:
        raise ValueError(f"V3-RCC is incompatible with enabled modules: {enabled}")
    if not bool(getattr(args, "use_ez_ranking_loss", False)) or float(getattr(args, "ez_ranking_loss_weight", 0.0)) != 0.05:
        raise ValueError("All V3-RCC profiles require original EZ pairwise ranking at weight 0.05.")
    return profile


__all__ = ["V3RCCProfile", "V3_RCC_PROFILES", "get_v3_rcc_profile", "validate_v3_rcc_args"]
