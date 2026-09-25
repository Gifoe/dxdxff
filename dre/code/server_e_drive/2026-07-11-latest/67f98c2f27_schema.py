from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from ..functional_graph import normalize_channel_name


@dataclass
class MosaicRunRecord:
    patient_key: str
    center: str
    seizure_id: str
    signal: np.ndarray
    sampling_rate: float
    onset_sample: int
    channel_names: list[str]
    valid_channel_mask: np.ndarray
    true_ez_mask: np.ndarray

    def __post_init__(self) -> None:
        self.signal = np.asarray(self.signal, dtype=float)
        self.channel_names = [normalize_channel_name(x) for x in self.channel_names]
        self.valid_channel_mask = np.asarray(self.valid_channel_mask, dtype=bool)
        self.true_ez_mask = np.asarray(self.true_ez_mask, dtype=bool)
        n = len(self.channel_names)
        if self.signal.ndim != 2 or self.signal.shape[0] != n:
            raise ValueError("signal must be [C,T] and match channel_names")
        if self.valid_channel_mask.shape != (n,) or self.true_ez_mask.shape != (n,):
            raise ValueError("channel masks must have shape [C]")
        target = self.true_ez_mask[self.valid_channel_mask]
        if not target.any() or target.all():
            raise ValueError("true_ez_mask must contain both target and outside valid channels")


@dataclass
class TemporalTrajectory:
    patient_key: str
    seizure_id: str
    mechanism: str
    phase: str
    times_sec: np.ndarray
    values: np.ndarray
    valid_mask: np.ndarray


@dataclass
class VirtualInterventionFeatures:
    capture: np.ndarray
    outside_residual: np.ndarray
    inside_outside_gap: np.ndarray
    outside_extent: np.ndarray
    nez_concordant_residual: np.ndarray


@dataclass
class PatientExpertFeatures:
    patient_key: str
    center: str
    feature_names: list[str]
    feature_values: np.ndarray


@dataclass
class ExpertOOFRecord:
    patient_key: str
    outer_fold: int
    expert: str
    probability_success: float
    true_label: int


def from_ngbr(record) -> MosaicRunRecord:
    return MosaicRunRecord(record.patient_key, record.center, record.seizure_id,
                           record.signal, record.sampling_rate, record.onset_sample,
                           list(record.channel_names), record.valid_channel_mask,
                           record.clinical_target_mask)


__all__ = ["MosaicRunRecord", "TemporalTrajectory", "VirtualInterventionFeatures",
           "PatientExpertFeatures", "ExpertOOFRecord", "from_ngbr"]
