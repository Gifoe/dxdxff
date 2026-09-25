from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NPAMProfile:
    name: str
    network_stats: bool
    phase_contrast: bool
    graph_residual: bool
    graph_stability: bool
    limited_finetune: bool
    clinical_target: bool = False
    p2_signal: bool = True
    cross_seizure: bool = False
    target_network: bool = False
    outcome_epochs: int = 30


@dataclass(frozen=True)
class NVRProfile:
    """Configuration for the separate frozen-P2 NVR-Outcome mainline."""

    name: str
    consensus: bool
    seizure_stability: bool
    virtual_network: bool
    monotone_head: bool
    set_residual: bool
    robust_consistency: bool
    outcome_epochs: int = 25


_PROFILES = {
    "M0_PAM": NPAMProfile("M0_PAM", False, False, False, False, False),
    "M1_NETWORK_STATS": NPAMProfile("M1_NETWORK_STATS", True, False, False, False, False),
    "M2_PHASE_NETWORK": NPAMProfile("M2_PHASE_NETWORK", True, True, False, False, False),
    "M3_GRAPH_RESIDUAL": NPAMProfile("M3_GRAPH_RESIDUAL", True, True, True, False, False),
    "M4_GRAPH_STABILITY": NPAMProfile("M4_GRAPH_STABILITY", True, True, True, True, False),
    "M5_LIMITED_FINETUNE": NPAMProfile("M5_LIMITED_FINETUNE", True, True, True, True, True),
    "C0_TARGET_ONLY": NPAMProfile("C0_TARGET_ONLY", False, False, False, False, False, True, False, False, False, 40),
    "C1_P2_ONLY": NPAMProfile("C1_P2_ONLY", False, False, False, False, False, False, True, False, False, 40),
    "C2_P2_TARGET_CONCORDANCE": NPAMProfile("C2_P2_TARGET_CONCORDANCE", False, False, False, False, False, True, True, False, False, 40),
    "C3_CROSS_SEIZURE_CONCORDANCE": NPAMProfile("C3_CROSS_SEIZURE_CONCORDANCE", False, False, False, False, False, True, True, True, False, 40),
    "C4_TARGET_NETWORK": NPAMProfile("C4_TARGET_NETWORK", True, False, False, False, False, True, True, True, True, 50),
    "C5_FULL": NPAMProfile("C5_FULL", True, False, False, False, True, True, True, True, True, 40),
}

_NVR_PROFILES = {
    "R0_NEZ_OUTSIDE": NVRProfile("R0_NEZ_OUTSIDE", False, False, False, False, False, False),
    "R1_NEZ_CONSENSUS": NVRProfile("R1_NEZ_CONSENSUS", True, False, False, False, False, False),
    "R2_PERSISTENT_RESIDUAL": NVRProfile("R2_PERSISTENT_RESIDUAL", True, True, False, False, False, False),
    "R3_VIRTUAL_RESECTION": NVRProfile("R3_VIRTUAL_RESECTION", True, True, True, False, False, False),
    "R4_MONOTONE_STRUCTURED": NVRProfile("R4_MONOTONE_STRUCTURED", True, True, True, True, False, False),
    "R5_BOUNDED_SET_RESIDUAL": NVRProfile("R5_BOUNDED_SET_RESIDUAL", True, True, True, True, True, False),
    "R6_ROBUST_FULL": NVRProfile("R6_ROBUST_FULL", True, True, True, True, True, True),
}


def get_profile(name: str) -> NPAMProfile:
    key = str(name).strip().upper()
    if key not in _PROFILES:
        raise ValueError(f"Unknown P2-Q10-NPAM profile {name!r}; expected one of {tuple(_PROFILES)}")
    return _PROFILES[key]


def profile_names() -> tuple[str, ...]:
    # Deliberately excludes R profiles: the historical runner must never route
    # an NVR experiment through its M fallback model.
    return tuple(_PROFILES)


def get_nvr_profile(name: str) -> NVRProfile:
    key = str(name).strip().upper()
    if key not in _NVR_PROFILES:
        raise ValueError(f"Unknown NVR-Outcome profile {name!r}; expected one of {tuple(_NVR_PROFILES)}")
    return _NVR_PROFILES[key]


def nvr_profile_names() -> tuple[str, ...]:
    return tuple(_NVR_PROFILES)


__all__ = ["NPAMProfile", "NVRProfile", "get_profile", "get_nvr_profile", "profile_names", "nvr_profile_names"]
