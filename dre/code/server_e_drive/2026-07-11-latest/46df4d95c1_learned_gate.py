"""Model-facing name for the low-capacity cross-fitted patient gate."""

from __future__ import annotations

from ..patient_gate import LearnedPatientGate


class LearnedGateModel(LearnedPatientGate):
    """Alias with an explicit model module for A12 variant manifests."""

    pass
