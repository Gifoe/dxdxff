"""Generic, center-anonymized input format used by all experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import Any, Iterable

import numpy as np
import pandas as pd


@dataclass
class PatientRecord:
    """One patient's multi-seizure descriptors.

    ``descriptors`` has shape [seizure, window, channel, 9].
    ``window_times`` has shape [seizure, window], in seconds from onset.
    ``valid`` has shape [seizure, window, channel].
    The label convention is NEZ=1 and EZ=0.
    """

    patient_id: str
    center: str
    descriptors: np.ndarray
    window_times: np.ndarray
    valid: np.ndarray
    channel_names: tuple[str, ...]
    label_nez: np.ndarray

    def validate(self) -> None:
        x = np.asarray(self.descriptors)
        times = np.asarray(self.window_times)
        valid = np.asarray(self.valid)
        labels = np.asarray(self.label_nez)
        if x.ndim != 4 or x.shape[-1] != 9:
            raise ValueError(f"{self.patient_id}: descriptors must be [S,W,C,9]")
        if times.shape != x.shape[:2]:
            raise ValueError(f"{self.patient_id}: window_times shape mismatch")
        if valid.shape != x.shape[:3]:
            raise ValueError(f"{self.patient_id}: valid-mask shape mismatch")
        if labels.shape != (x.shape[2],):
            raise ValueError(f"{self.patient_id}: label shape mismatch")
        if len(self.channel_names) != x.shape[2]:
            raise ValueError(f"{self.patient_id}: channel-name count mismatch")
        if not set(np.unique(labels)).issubset({0, 1}):
            raise ValueError(f"{self.patient_id}: labels must use NEZ=1, EZ=0")
        if not np.isfinite(times[np.any(valid, axis=2)]).all():
            raise ValueError(f"{self.patient_id}: valid windows require finite times")


def _coerce_record(value: PatientRecord | dict[str, Any]) -> PatientRecord:
    if isinstance(value, PatientRecord):
        record = value
    elif isinstance(value, dict):
        record = PatientRecord(
            patient_id=str(value["patient_id"]),
            center=str(value["center"]),
            descriptors=np.asarray(value["descriptors"], dtype=np.float32),
            window_times=np.asarray(value["window_times"], dtype=np.float32),
            valid=np.asarray(value["valid"], dtype=bool),
            channel_names=tuple(map(str, value["channel_names"])),
            label_nez=np.asarray(value["label_nez"], dtype=np.int64),
        )
    else:
        raise TypeError(f"Unsupported record type: {type(value)!r}")
    record.validate()
    return record


def load_records(path: str | Path) -> list[PatientRecord]:
    """Load a pickle containing a list of ``PatientRecord``-compatible dicts."""
    with Path(path).open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, (list, tuple)):
        raise TypeError("Input pickle must contain a list of patient records")
    records = [_coerce_record(item) for item in payload]
    identifiers = [record.patient_id for record in records]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Duplicate patient_id values are not allowed")
    return records


def load_partition_manifest(path: str | Path) -> pd.DataFrame:
    """Load explicit fit/validation/test assignments for every outer fold."""
    frame = pd.read_csv(path)
    aliases = {
        "subject_id": "patient_id",
        "fold": "outer_fold",
        "fold_idx": "outer_fold",
        "role": "partition",
        "split_role": "partition",
    }
    frame = frame.rename(columns={k: v for k, v in aliases.items() if k in frame})
    required = {"patient_id", "outer_fold", "partition"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Partition manifest requires {sorted(required)}")
    frame = frame[list(required)].copy()
    frame["patient_id"] = frame["patient_id"].astype(str)
    frame["outer_fold"] = pd.to_numeric(frame["outer_fold"], errors="raise").astype(int)
    frame["partition"] = frame["partition"].astype(str).str.lower()
    if not set(frame["partition"]).issubset({"fit", "validation", "test"}):
        raise ValueError("partition must be fit, validation, or test")
    if frame.duplicated(["patient_id", "outer_fold"]).any():
        raise ValueError("A patient may occur once per outer fold")
    return frame


def select_records(
    records: Iterable[PatientRecord],
    manifest: pd.DataFrame,
    outer_fold: int,
    partition: str,
) -> list[PatientRecord]:
    ids = set(
        manifest.loc[
            (manifest["outer_fold"] == outer_fold)
            & (manifest["partition"] == partition),
            "patient_id",
        ]
    )
    selected = [record for record in records if record.patient_id in ids]
    if {record.patient_id for record in selected} != ids:
        raise ValueError(f"Missing records for fold={outer_fold}, partition={partition}")
    return selected


def validate_protocol(
    records: Iterable[PatientRecord], manifest: pd.DataFrame, expected_patients: int = 80
) -> None:
    """Fail closed on patient overlap or incomplete outer-test coverage."""
    record_ids = {record.patient_id for record in records}
    if len(record_ids) != expected_patients:
        raise ValueError(f"Expected {expected_patients} patients, found {len(record_ids)}")
    if not set(manifest["patient_id"]).issubset(record_ids):
        raise ValueError("Manifest contains patients absent from the input records")
    test = manifest.loc[manifest["partition"] == "test"]
    counts = test.groupby("patient_id").size()
    if set(counts.index) != record_ids or not counts.eq(1).all():
        raise ValueError("Every patient must be outer-test exactly once")
    for fold, group in manifest.groupby("outer_fold"):
        parts = {
            name: set(group.loc[group["partition"] == name, "patient_id"])
            for name in ("fit", "validation", "test")
        }
        if any(parts[a] & parts[b] for a, b in (("fit", "validation"), ("fit", "test"), ("validation", "test"))):
            raise ValueError(f"Patient overlap detected in outer fold {fold}")
