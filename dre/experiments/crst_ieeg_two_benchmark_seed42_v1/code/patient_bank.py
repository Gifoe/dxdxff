"""Private patient-aware spectral inputs; never serializes patient rows to Git."""

from __future__ import annotations

import csv
import hashlib
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


def patient_hash(patient: str) -> str:
    return hashlib.sha256(patient.encode()).hexdigest()[:20]


class IctalPatientBank:
    def __init__(self, cache: Path, manifest: Path, feature_cache: Path):
        self.cache = Path(cache)
        with Path(manifest).open(newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        self.folds = {}
        for fold in range(1, 6):
            group = {row["split_role"]: [] for row in rows if int(row["outer_fold"]) == fold}
            for row in rows:
                if int(row["outer_fold"]) == fold:
                    group[row["split_role"]].append(row["subject_id"])
            if not (len(group["fit"]) + len(group["validation"]) + len(group["test"]) == 80):
                raise RuntimeError("Frozen ictal fold membership broken")
            self.folds[fold] = group
        with Path(feature_cache).open("rb") as f:
            payload = pickle.load(f)
        self.center = {patient: str(payload["patient_index"][patient]["source_center"])
                       for patient in self.folds[1]["fit"] + self.folds[1]["validation"] +
                       self.folds[1]["test"]}
        if len(self.center) != 80 or len(list(self.cache.glob("patient_*.npz"))) != 80:
            raise RuntimeError("Private ictal cache incomplete")

    def load(self, patient: str) -> dict:
        path = self.cache / f"patient_{patient_hash(patient)}.npz"
        marker = path.with_suffix(".json")
        if not path.is_file() or not marker.is_file():
            raise RuntimeError("Audited ictal patient cache absent")
        with np.load(path) as z:
            if str(z["subject_id"]) != patient:
                raise RuntimeError("Patient hash/file identity mismatch")
            sample = {key: np.asarray(z[key]) for key in
                      ("patches", "window_mask", "edges", "frequency_mask", "labels")}
        sample["center"] = self.center[patient]
        sample["patient"] = patient
        return sample


class OmniPatientBank:
    def __init__(self, cache: Path, train_val_split: Path):
        self.cache = Path(cache)
        with Path(train_val_split).open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.roles = {row["patient"]: row["role"] for row in rows}
        if len(self.roles) != 141:
            raise RuntimeError("Frozen Omni inner split not 141")
        self.patient_files = defaultdict(list)
        for path in self.cache.glob("edf_*.npz"):
            marker = path.with_suffix(".json")
            if not marker.is_file():
                raise RuntimeError("Partial Omni spectral cache")
            with np.load(path) as z:
                patient = str(z["patient"])
            if patient not in self.roles:
                raise RuntimeError("Omni cache patient outside official train roles")
            self.patient_files[patient].append(path)
        if set(self.patient_files) != set(self.roles):
            raise RuntimeError("Omni TRAIN spectral cache incomplete")

    def load(self, patient: str, epoch: int, *, all_clips: bool = False) -> dict:
        paths = sorted(self.patient_files[patient])
        records = []
        labels = {}
        center = None
        for path in paths:
            with np.load(path) as z:
                patches = np.asarray(z["patches"])
                edges = np.asarray(z["edges"])
                names = [str(n) for n in z["channel_names"]]
                current_labels = np.asarray(z["labels"], dtype=np.int8)
                if center is None:
                    center = str(z["dataset"])
                elif center != str(z["dataset"]):
                    raise RuntimeError("One Omni patient spans multiple source centers")
                for name, label in zip(names, current_labels):
                    if label >= 0:
                        labels.setdefault(name, []).append(int(label))
                if all_clips:
                    selected = range(len(patches))
                else:
                    # Deterministic one clip per EDF/epoch; never label-driven.
                    selected = [int(hashlib.sha256(f"42|{patient}|{path.name}|{epoch}".encode()).hexdigest()[:8], 16)
                                % len(patches)]
                for clip in selected:
                    records.append((names, patches[clip], edges[clip], current_labels))
        canonical = sorted({name for names, _, _, _ in records for name in names})
        idx = {name: i for i, name in enumerate(canonical)}
        r, c = len(records), len(canonical)
        patches = np.zeros((r, c, 59, 64, 8), dtype=np.float16)
        edges = np.zeros((r, c, c, 15), dtype=np.float16)
        mask = np.zeros((r, c, 59), dtype=bool)
        record_labels = np.full((r, c), -1, dtype=np.int8)
        for ri, (names, p, e, labels_in_edf) in enumerate(records):
            local = [idx[name] for name in names]
            patches[ri, local] = p
            edges[ri][np.ix_(local, local)] = e
            mask[ri, local] = True
            record_labels[ri, local] = labels_in_edf
        # A common patient-channel label is definable only when all official
        # EDF-channel observations agree; disagreements are retained separately
        # in private diagnostics, never silently majority-voted.
        target = np.full(c, -1, dtype=np.int8)
        conflicts = 0
        for name, values in labels.items():
            if len(set(values)) == 1:
                target[idx[name]] = values[0]
            else:
                conflicts += 1
        return {"patches": patches, "edges": edges, "window_mask": mask,
                "frequency_mask": np.ones(64, dtype=np.float32),
                "labels": target, "record_labels": record_labels,
                "center": center, "patient": patient,
                "label_conflicts": conflicts, "records": r}


def to_device(sample: dict, device: torch.device) -> dict:
    result = {"patches": torch.from_numpy(sample["patches"].astype(np.float32))[None].to(device),
            "edges": torch.from_numpy(sample["edges"].astype(np.float32))[None].to(device),
            "window_mask": torch.from_numpy(sample["window_mask"])[None].to(device),
            "frequency_mask": torch.from_numpy(sample["frequency_mask"].astype(np.float32))[None].to(device),
            "labels": torch.from_numpy(sample["labels"].astype(np.float32))[None].to(device),
            "center": sample["center"], "patient": sample["patient"],
            "label_conflicts": sample.get("label_conflicts", 0)}
    if "record_labels" in sample:
        result["record_labels"] = torch.from_numpy(sample["record_labels"].astype(np.float32))[None].to(device)
    return result
