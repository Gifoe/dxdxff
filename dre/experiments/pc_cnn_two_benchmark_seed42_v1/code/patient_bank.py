"""Private patient/record loader for PC-CNN; no outcome or test file is opened.

The training bank is deliberately split-locked. The Omni constructor rejects
any EDF outside the frozen official TRAIN membership and the 141-patient
inner split. Test loading must live in a separate post-freeze entry point.
"""

from __future__ import annotations

import csv
import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np

from descriptor_norm import DescriptorMoments
from official_spectrum import descriptor_frequency_mask


def patient_hash(patient: str) -> str:
    return hashlib.sha256(patient.encode()).hexdigest()[:20]


class IctalBank:
    def __init__(self, cache: Path, manifest: Path):
        self.cache = Path(cache)
        self.folds = {}
        with Path(manifest).open(newline="", encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))
        for fold in range(1, 6):
            group = defaultdict(list)
            for row in rows:
                if int(row["outer_fold"]) == fold:
                    group[row["split_role"]].append(str(row["subject_id"]))
            if set(group) != {"fit", "validation", "test"} or \
                    sum(map(len, group.values())) != 80 or \
                    len(set(sum(group.values(), []))) != 80:
                raise RuntimeError(f"Frozen ictal fold {fold} is invalid")
            self.folds[fold] = {key: sorted(value) for key, value in group.items()}
        if len(list(self.cache.glob("patient_*.npz"))) != 80:
            raise RuntimeError("Ictal private TRAIN cache must have exactly 80 patients")

    def path(self, patient: str) -> Path:
        result = self.cache / f"patient_{patient_hash(patient)}.npz"
        if not result.is_file() or not result.with_suffix(".json").is_file():
            raise RuntimeError("Frozen ictal patient cache/marker absent")
        return result

    def records(self, patient: str):
        with np.load(self.path(patient), allow_pickle=False) as sample:
            if str(sample["patient"]) != patient or float(sample["sampling_rate_hz"]) != 250:
                raise RuntimeError("Ictal patient/sampling-rate mismatch")
            labels = sample["labels"].astype(np.int8)
            names = [str(value) for value in sample["channel_names"]]
            starts, lengths = sample["valid_start"], sample["valid_samples"]
            presence, window_masks = sample["channel_present"], sample["window_mask"]
            descriptors, waveforms = sample["descriptors"], sample["waveforms"]
            for ri in range(len(lengths)):
                begin = int(starts[ri])
                n = int(lengths[ri])
                present = presence[ri].astype(bool)
                windows = window_masks[ri].astype(bool)
                descriptor = descriptors[ri].astype(np.float32)
                mask = np.repeat(windows[:, :, None], 36, axis=2)
                physical = descriptor_frequency_mask(250).numpy().astype(bool)
                mask &= physical[None, None, :]
                yield {"waveforms": waveforms[ri, :, begin:begin + n].astype(np.float32),
                       "descriptors": descriptor, "descriptor_mask": mask,
                       "channel_mask": present, "labels": labels,
                       "channel_names": names,
                       "sampling_rate_hz": 250.0, "record_index": ri,
                       "patient": patient}

    def fit_moments(self, fold: int) -> DescriptorMoments:
        moments = DescriptorMoments()
        for patient in self.folds[fold]["fit"]:
            with np.load(self.path(patient), allow_pickle=False) as sample:
                descriptors = sample["descriptors"]
                windows = sample["window_mask"]
                physical = descriptor_frequency_mask(250).numpy().astype(bool)
                mask = windows[..., None] & physical[None, None, None, :]
                moments.update(descriptors, np.broadcast_to(mask, descriptors.shape))
        return moments

    def labeled_observations(self, patient: str) -> int:
        with np.load(self.path(patient), allow_pickle=False) as sample:
            labeled = sample["labels"] >= 0
            return int(np.sum(sample["channel_present"] & labeled[None, :]))


class OmniTrainBank:
    def __init__(self, cache: Path, frozen_train_val_split: Path,
                 official_split: Path):
        self.cache = Path(cache)
        with Path(frozen_train_val_split).open(newline="", encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))
        self.roles = {row["patient"]: row["role"] for row in rows}
        if len(self.roles) != 141 or set(self.roles.values()) != {"inner_train", "inner_val"}:
            raise RuntimeError("Frozen Omni TRAIN/inner split differs")
        with Path(official_split).open(newline="", encoding="utf-8-sig") as stream:
            official = list(csv.DictReader(stream))
        allowed = {row["edf_name"] for row in official if row["split"] == "train"
                   and row["patient_name"] in self.roles}
        self.patient_files = defaultdict(list)
        for path in sorted(self.cache.glob("edf_*.npz")):
            if not path.with_suffix(".json").is_file():
                raise RuntimeError("Partial Omni train cache")
            with np.load(path, allow_pickle=False) as sample:
                patient, edf = str(sample["patient"]), str(sample["edf"])
                if patient not in self.roles or edf not in allowed or \
                        float(sample["sampling_rate_hz"]) != 1000:
                    raise RuntimeError("Omni private file outside frozen TRAIN scope")
                if (path.name != "edf_" + hashlib.sha256(edf.encode()).hexdigest()[:24] + ".npz"):
                    raise RuntimeError("Omni private EDF hash mismatch")
            self.patient_files[patient].append(path)
        if set(self.patient_files) != set(self.roles):
            raise RuntimeError("Omni TRAIN cache has not covered all 141 patients")

    def patients(self, role: str) -> list[str]:
        return sorted(patient for patient, value in self.roles.items() if value == role)

    def records(self, patient: str, epoch: int, *, all_clips: bool):
        if patient not in self.patient_files:
            raise RuntimeError("Unknown Omni TRAIN patient")
        for path in self.patient_files[patient]:
            with np.load(path, allow_pickle=False) as sample:
                if str(sample["patient"]) != patient:
                    raise RuntimeError("Omni private patient identity changed")
                n_clips = len(sample["starts"])
                if n_clips < 1:
                    raise RuntimeError("Empty Omni train EDF")
                if all_clips:
                    chosen = range(n_clips)
                else:
                    key = f"42|{patient}|{path.name}|{epoch}"
                    chosen = [int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % n_clips]
                labels = sample["labels"].astype(np.int8)
                names = [str(value) for value in sample["channel_names"]]
                all_descriptors = sample["descriptors"]
                all_waveforms = sample["waveforms"]
                for index in chosen:
                    descriptor = all_descriptors[index].astype(np.float32)
                    mask = np.ones_like(descriptor, dtype=bool)
                    yield {"waveforms": all_waveforms[index].astype(np.float32),
                           "descriptors": descriptor, "descriptor_mask": mask,
                           "channel_mask": np.ones(len(labels), dtype=bool),
                           "labels": labels, "sampling_rate_hz": 1000.0,
                           "channel_names": names,
                           "record_index": index, "edf": str(sample["edf"]),
                           "patient": patient}

    def fit_moments(self) -> DescriptorMoments:
        moments = DescriptorMoments()
        for patient in self.patients("inner_train"):
            for path in self.patient_files[patient]:
                with np.load(path, allow_pickle=False) as sample:
                    descriptor = sample["descriptors"].astype(np.float32)
                    moments.update(descriptor, np.ones_like(descriptor, dtype=bool))
        return moments

    def labeled_observations(self, patient: str) -> int:
        count = 0
        for path in self.patient_files[patient]:
            with np.load(path, allow_pickle=False) as sample:
                count += int(np.sum(sample["labels"] >= 0))
        return count
