"""Predeclared one-module-at-a-time profiles for P2_RTC_SHIFT."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RTCProfile:
    name: str
    robust_tail: bool
    tail_reliability: bool
    tail_rank: bool
    ez_aware_selection: bool
    shift_calibration: bool


_PROFILES = {
    "A0": RTCProfile("A0_P2_TEMPORAL_Q10", False, False, False, False, False),
    "A1": RTCProfile("A1_ROBUST_TAIL", True, False, False, False, False),
    "A2": RTCProfile("A2_TAIL_RELIABILITY", True, True, False, False, False),
    "A3": RTCProfile("A3_TAIL_RANK", True, True, True, False, False),
    "A4": RTCProfile("A4_EZ_AWARE_SELECTION", True, True, True, True, False),
    "A5": RTCProfile("A5_P2_RTC_SHIFT", True, True, True, True, True),
}


def get_rtc_profile(value: str) -> RTCProfile:
    key = str(value).strip().upper()
    if key in _PROFILES:
        return _PROFILES[key]
    for profile in _PROFILES.values():
        if key == profile.name:
            return profile
    raise ValueError(f"Unknown P2_RTC_SHIFT profile: {value!r}")


def rtc_profile_names() -> tuple[str, ...]:
    return tuple(_PROFILES)

