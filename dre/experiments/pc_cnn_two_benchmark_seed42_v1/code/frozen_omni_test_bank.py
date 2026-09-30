"""Official Omni TEST loader exists only behind a verified pre-test freeze."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from train_rawcnn import digest
from prepare_omni_train import flag


HISTORICAL_COHORT_SHA256 = "e10241ce0e823ced7dd262ed6eda4eeb0ffdbe52082aa4ec771590e253ffaf00"


def numeric(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


class FrozenOmniTestBank:
    def __init__(self, cache: Path, official_split: Path, cohort_audit: Path,
                 freeze_path: Path, protocol_path: Path):
        self.cache = Path(cache)
        freeze = json.loads(Path(freeze_path).read_text(encoding="utf-8"))
        protocol = json.loads(Path(protocol_path).read_text(encoding="utf-8"))
        if freeze.get("status") != "FROZEN_BEFORE_OFFICIAL_TEST" or \
                freeze.get("protocol_sha256") != digest(protocol_path) or \
                not freeze.get("model_frozen_before_final_test") or \
                digest(official_split) != protocol["omni_official_split_sha256"]:
            raise RuntimeError("Omni official TEST cannot be opened before valid freeze")
        if digest(cohort_audit) != HISTORICAL_COHORT_SHA256:
            raise RuntimeError("Historical supervised test cohort audit changed")
        with Path(official_split).open(newline="", encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))
        rows = [row for row in rows if row["split"] == "test" and
                row["dataset"] != "Multicenter" and numeric(row["frequency"]) > 900 and
                flag(row["interictal"]) and numeric(row["length"]) >= 62]
        if len(rows) != 237 or len({row["patient_name"] for row in rows}) != 102:
            raise RuntimeError("Frozen official Omni metadata cohort differs")
        with Path(cohort_audit).open(newline="", encoding="utf-8-sig") as stream:
            audit_rows = list(csv.DictReader(stream))
        audited = {row["edf"]: row["patient"] for row in audit_rows
                   if row["official_split"] == "test" and
                   numeric(row["official_labeled_channels"]) > 0}
        if len(audited) != 174 or len(set(audited.values())) != 96:
            raise RuntimeError("Historical 174-EDF/96-patient cohort differs")
        self.official = {row["edf_name"]: row for row in rows
                         if row["edf_name"] in audited and
                         row["patient_name"] == audited[row["edf_name"]]}
        if len(self.official) != 174:
            raise RuntimeError("Audited cohort is not an official-split subset")
        self.center_by_patient = {}
        self.patient_files = defaultdict(list)
        provenance = digest(protocol_path) + "|" + digest(freeze_path) + "|" + digest(cohort_audit)
        for path in sorted(self.cache.glob("edf_*.npz")):
            marker = path.with_suffix(".json")
            if not marker.is_file():
                raise RuntimeError("Partial official test cache")
            status = json.loads(marker.read_text(encoding="utf-8"))
            if status.get("protocol_sha256") != provenance or status.get("sha256") != digest(path):
                raise RuntimeError("Omni test cache freeze/content provenance mismatch")
            with np.load(path, allow_pickle=False) as sample:
                patient, edf = str(sample["patient"]), str(sample["edf"])
                if edf not in self.official or self.official[edf]["patient_name"] != patient:
                    raise RuntimeError("Private test file outside official cohort")
                center = self.official[edf]["dataset"]
                if patient in self.center_by_patient and self.center_by_patient[patient] != center:
                    raise RuntimeError("Patient spans multiple official centers")
                self.center_by_patient[patient] = center
                self.patient_files[patient].append(path)
        if len(list(self.cache.glob("edf_*.npz"))) != 174 or len(self.patient_files) != 96:
            raise RuntimeError("Official Omni test extraction not complete")

    def patients(self):
        return sorted(self.patient_files)

    def records(self, patient: str, epoch: int, *, all_clips: bool):
        del epoch
        if not all_clips:
            raise RuntimeError("Frozen official test always evaluates all extracted clips")
        for path in self.patient_files[patient]:
            with np.load(path, allow_pickle=False) as sample:
                names = [str(value) for value in sample["channel_names"]]
                labels = sample["labels"].astype(np.int8)
                waves, descriptors = sample["waveforms"], sample["descriptors"]
                for index in range(len(sample["starts"])):
                    descriptor = descriptors[index].astype(np.float32)
                    yield {"waveforms": waves[index].astype(np.float32),
                           "descriptors": descriptor,
                           "descriptor_mask": np.ones_like(descriptor, dtype=bool),
                           "channel_mask": np.ones(len(names), dtype=bool),
                           "channel_names": names, "labels": labels,
                           "sampling_rate_hz": 1000.0,
                           "edf": str(sample["edf"]), "patient": patient}
