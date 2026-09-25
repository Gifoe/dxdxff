"""Final BCR-Net objective profiles.

BCR-Net is intentionally independent from PRQ-Net's lower-tail Q10 path.
It contains channel BCE supervision plus optional boundary and coverage
objectives only.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BCRProfile:
    name: str
    use_boundary: bool
    use_coverage: bool


BCR_PROFILES = {
    "BCR_BC_ONLY": BCRProfile("BCR_BC_ONLY", False, False),
    "BCR_BOUNDARY_ONLY": BCRProfile("BCR_BOUNDARY_ONLY", True, False),
    "BCR_COVERAGE_ONLY": BCRProfile("BCR_COVERAGE_ONLY", False, True),
    "BCR_BOUNDARY_COVERAGE": BCRProfile("BCR_BOUNDARY_COVERAGE", True, True),
}
# Backward-compatible import name only.  Old A*-QBC profile names are rejected.
V3_QBC_PROFILES = BCR_PROFILES


def get_v3_qbc_profile(name: str) -> BCRProfile:
    key = str(name).strip().upper()
    if key not in V3_QBC_PROFILES:
        raise ValueError(
            f"Unknown BCR profile {name!r}; expected one of {sorted(BCR_PROFILES)}"
        )
    return V3_QBC_PROFILES[key]


def validate_v3_qbc_args(args: object) -> BCRProfile:
    profile = get_v3_qbc_profile(getattr(args, "v3_qbc_profile", "BCR_BC_ONLY"))
    forbidden = {
        "use_n6_dual_view_ema": bool(getattr(args, "use_n6_dual_view_ema", False)),
        "use_two_expert_router": bool(getattr(args, "use_two_expert_router", False)),
        "use_feature_separated_two_expert": bool(
            getattr(args, "use_feature_separated_two_expert", False)
        ),
        "use_a9v8_lcbo": bool(getattr(args, "use_a9v8_lcbo", False)),
        "use_broad_ez_mil_loss": bool(getattr(args, "use_broad_ez_mil_loss", False)),
        "use_diffusion_residual": bool(getattr(args, "use_diffusion_residual", False)),
        "use_view_gated_fusion": bool(getattr(args, "use_view_gated_fusion", False)),
    }
    enabled = sorted(key for key, value in forbidden.items() if value)
    if enabled:
        raise ValueError(f"BCR-Net is incompatible with enabled modules: {enabled}")
    if str(getattr(args, "positive_label", "ez")).lower() != "ez":
        raise ValueError(
            "BCR-Net uses EZ-oriented supervision and requires --positive_label ez"
        )
    return profile


V3QBCProfile = BCRProfile

__all__ = [
    "BCRProfile", "BCR_PROFILES", "V3QBCProfile", "V3_QBC_PROFILES",
    "get_v3_qbc_profile", "validate_v3_qbc_args",
]
