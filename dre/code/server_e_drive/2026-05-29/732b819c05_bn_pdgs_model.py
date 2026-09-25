from __future__ import annotations

from typing import Any

try:
    from .neuroez_hybrid_model import NeuroEZConfig, NeuroEZHybridModel
except ImportError:
    from neuroez_hybrid_model import NeuroEZConfig, NeuroEZHybridModel


class BNPDGSModel(NeuroEZHybridModel):
    """Backward-compatible alias for the NeuroEZ hybrid patient-level model.

    The previous BN-PDGS skeleton expected graph-spectral features, adjacency,
    temporal pooling, cross-seizure MIL, and a patient channel ranker.  The new
    implementation keeps that path and adds an optional raw waveform CNN branch.
    V3-small output scores are p_NEZ by default: label 1 is NEZ and label 0 is
    EZ, so EZ ranking sorts channels by ascending p_NEZ.
    """

    def __init__(self, args: Any | None = None) -> None:
        super().__init__(args=args)


__all__ = ["BNPDGSModel", "NeuroEZConfig", "NeuroEZHybridModel"]
