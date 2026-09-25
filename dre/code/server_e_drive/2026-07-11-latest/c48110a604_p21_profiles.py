"""Single-source profile switches for P2.1 V3-ASRR."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class P21Profile:
    name: str
    legacy_p2: bool
    bounded_simplex: bool
    weighted_rank: bool
    preserve: bool
    v3_anchor: bool
    q10_seizure: bool
    center_alignment: bool


P21_PROFILES = {
    "R0_CURRENT_P2": P21Profile("R0_CURRENT_P2", True, False, False, False, False, False, False),
    "R1_BOUNDED_SIMPLEX": P21Profile("R1_BOUNDED_SIMPLEX", False, True, False, False, False, False, False),
    "R2_WEIGHTED_RANK_PRESERVE": P21Profile("R2_WEIGHTED_RANK_PRESERVE", False, True, True, True, False, False, False),
    "R3_V3_ANCHORED": P21Profile("R3_V3_ANCHORED", False, True, True, True, True, False, False),
    "R4_Q10_MULTI_SEIZURE": P21Profile("R4_Q10_MULTI_SEIZURE", False, True, True, True, True, True, False),
    "R5_FULL": P21Profile("R5_FULL", False, True, True, True, True, True, True),
}


def get_p21_profile(name: str) -> P21Profile:
    key = str(name).strip().upper()
    if key not in P21_PROFILES:
        raise ValueError(f"Unknown P2.1 profile {name!r}; choices={list(P21_PROFILES)}")
    return P21_PROFILES[key]


__all__ = ["P21Profile", "P21_PROFILES", "get_p21_profile"]
