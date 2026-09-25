"""Predeclared, no-calibration profiles for P2_ATC."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ATCProfile:
    name: str
    robust_tail: bool
    clean_nez_tail_loss: bool
    trusted_ez_tail_loss: bool


_PROFILES = {
    "A0": ATCProfile("A0_P2_TEMPORAL_Q10", False, False, False),
    "A1": ATCProfile("A1_P2_ROBUST_TAIL", True, False, False),
    "A2": ATCProfile("A2_P2_NEZ_TAIL", True, True, False),
    "A3": ATCProfile("A3_P2_ATC", True, True, True),
}


def get_atc_profile(value: str) -> ATCProfile:
    key = str(value).strip().upper()
    if key in _PROFILES:
        return _PROFILES[key]
    for profile in _PROFILES.values():
        if key == profile.name:
            return profile
    raise ValueError(f"Unknown P2_ATC profile: {value!r}")


def atc_profile_names() -> tuple[str, ...]:
    return tuple(_PROFILES)
