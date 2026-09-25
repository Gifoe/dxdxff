"""Small, explicit P23-TRN profile surface."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class P23Profile:
    name: str
    temporal: bool
    seizure_tail: bool
    bounded_fusion: bool
    noise_aware: bool
    causal: bool
    min_seizures_for_tail: int = 1


_PROFILES = {
    "P0_CURRENT_P2": P23Profile("P0_CURRENT_P2", False, False, False, False, False),
    "P1_TEMPORAL": P23Profile("P1_TEMPORAL", True, False, False, False, False),
    "P2_TEMPORAL_Q10": P23Profile("P2_TEMPORAL_Q10", True, True, False, False, False),
    "P3_BOUNDED_FUSION": P23Profile("P3_BOUNDED_FUSION", True, True, True, False, False),
    "P4_NOISE_AWARE": P23Profile("P4_NOISE_AWARE", True, True, True, True, False),
    "P5_FULL": P23Profile("P5_FULL", True, True, True, True, False),
    "P6_FULL_WITH_CAUSAL": P23Profile("P6_FULL_WITH_CAUSAL", True, True, True, True, True),
    # Regression fallback: no EMA/noisy-EZ objective, no core-existence loss.
    # Its bounded gate is a safety constraint, not a new evidence source.
    "P23_LITE": P23Profile("P23_LITE", True, True, True, False, False, 2),
}


def get_p23_profile(name: str) -> P23Profile:
    try:
        return _PROFILES[str(name).upper()]
    except KeyError as error:
        raise ValueError(f"Unknown P23 profile: {name!r}") from error


def profile_names() -> tuple[str, ...]:
    return tuple(_PROFILES)
