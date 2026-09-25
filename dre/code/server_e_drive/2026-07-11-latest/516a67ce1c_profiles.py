from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class COPProfile:
    name: str
    feature: bool = True
    raw: bool = False
    p2: bool = False
    group_gated: bool = False


_PROFILES = {
    "O0_FEATURE_PHENOTYPE": COPProfile("O0_FEATURE_PHENOTYPE"),
    "O1_FEATURE_RAW_PROPAGATION": COPProfile("O1_FEATURE_RAW_PROPAGATION", raw=True),
    "O2_COP_FULL": COPProfile("O2_COP_FULL", raw=True, p2=True),
    "O3_COP_GROUP_GATED": COPProfile("O3_COP_GROUP_GATED", raw=True, p2=True, group_gated=True),
}


def get_cop_profile(name: str) -> COPProfile:
    key = str(name).strip().upper()
    if key not in _PROFILES: raise ValueError(f"Unknown COP profile {name!r}; expected one of {tuple(_PROFILES)}")
    return _PROFILES[key]


def cop_profile_names() -> tuple[str, ...]: return tuple(_PROFILES)


__all__ = ["COPProfile", "cop_profile_names", "get_cop_profile"]
