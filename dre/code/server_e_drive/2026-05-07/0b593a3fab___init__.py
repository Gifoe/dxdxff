"""BN-PDGS ranker package."""

from .config import BNPDGSConfig
from .data_interface import PatientRecord, SeizureRecord, load_patient_records


def __getattr__(name: str):
    if name == "BNPDGSModel":
        from .model import BNPDGSModel

        return BNPDGSModel
    if name in {"DynamicPatientSample", "DynamicSeizureSample"}:
        from .dynamic_dataset import DynamicPatientSample, DynamicSeizureSample

        return {"DynamicPatientSample": DynamicPatientSample, "DynamicSeizureSample": DynamicSeizureSample}[name]
    raise AttributeError(name)

__all__ = [
    "BNPDGSConfig",
    "BNPDGSModel",
    "DynamicPatientSample",
    "DynamicSeizureSample",
    "PatientRecord",
    "SeizureRecord",
    "load_patient_records",
]
