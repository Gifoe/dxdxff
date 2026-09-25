from __future__ import annotations

import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .exclusions import apply_exclusions
from .functional_graph import NETWORK_PHASES, normalize_channel_name


def load_cache(path: str | Path) -> dict[str, Any]:
    with Path(path).expanduser().open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict) or not isinstance(payload.get("run_records"), list) or not isinstance(payload.get("patient_index"), dict):
        raise ValueError(f"Unsupported cache contract: {path}")
    return payload


def load_graph_cache(path: str | Path | None) -> dict[tuple[str, str, str], dict[str, Any]]:
    if not path:
        return {}
    with Path(path).expanduser().open("rb") as handle:
        payload = pickle.load(handle)
    records = payload.get("graphs", payload) if isinstance(payload, Mapping) else payload
    output: dict[tuple[str, str, str], dict[str, Any]] = {}
    iterable = records.values() if isinstance(records, Mapping) else records
    for record in iterable:
        key = (str(record["patient_key"]), str(record["seizure_id"]), str(record["phase"]))
        if key in output:
            raise ValueError(f"Duplicate graph cache key: {key}")
        output[key] = dict(record)
    return output


class NPAMCollator:
    def __init__(self, runtime: Any, outcome_by_subject: Mapping[str, int], graph_lookup: Mapping[tuple[str, str, str], Mapping[str, Any]] | None = None, clinical_target_lookup: Mapping[str, Mapping[str, Any]] | None = None) -> None:
        self.runtime = runtime
        self.outcome_by_subject = {str(key): int(value) for key, value in outcome_by_subject.items()}
        self.graph_lookup = dict(graph_lookup or {})
        self.clinical_target_lookup = dict(clinical_target_lookup or {})

    def __call__(self, examples: Sequence[dict[str, Any]]) -> dict[str, Any]:
        batch = self.runtime.collate(examples)
        subjects = [str(value) for value in batch["subject_id"]]
        batch["outcome_target"] = torch.tensor([self.outcome_by_subject[value] for value in subjects], dtype=torch.float32)
        success = batch["outcome_target"] == 1
        batch["localization_mask"] = success.unsqueeze(-1) & batch["channel_mask"].bool() & torch.isfinite(batch["labels_nez"]) & (batch["labels_nez"] >= 0.0)
        if self.clinical_target_lookup:
            from .clinical_target import collate_clinical_targets
            batch["clinical_target_mask"] = collate_clinical_targets(subjects, batch["canonical_channels"], self.clinical_target_lookup, batch["channel_mask"].shape[1])
            valid = batch["channel_mask"].bool()
            if not torch.isfinite(batch["clinical_target_mask"][valid]).all():
                raise ValueError("Missing clinical target value on a valid channel")
        if self.graph_lookup:
            batch.update(self._collate_graphs(batch))
        return batch

    def _collate_graphs(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        bsz, max_seizures, max_channels = batch["seizure_channel_mask"].shape
        n_phases = len(NETWORK_PHASES)
        adjacency = torch.zeros((bsz, max_seizures, n_phases, max_channels, max_channels), dtype=torch.float32)
        phase_channel_mask = torch.zeros((bsz, max_seizures, n_phases, max_channels), dtype=torch.bool)
        graph_valid = torch.zeros((bsz, max_seizures, n_phases), dtype=torch.bool)
        for patient_idx, subject in enumerate(batch["subject_id"]):
            canonical = [normalize_channel_name(name) for name in batch["canonical_channels"][patient_idx]]
            canonical_index = {name: idx for idx, name in enumerate(canonical)}
            for seizure_idx, run_id in enumerate(batch["run_ids"][patient_idx]):
                for phase_idx, phase in enumerate(NETWORK_PHASES):
                    record = self.graph_lookup.get((str(subject), str(run_id), phase))
                    if not record:
                        continue
                    local_names = [normalize_channel_name(name) for name in record.get("channel_names", [])]
                    local_indices, patient_indices = [], []
                    local_valid = np.asarray(record.get("valid_channel_mask", np.ones(len(local_names), dtype=bool)), dtype=bool)
                    for local_idx, name in enumerate(local_names):
                        patient_channel_idx = canonical_index.get(name)
                        if patient_channel_idx is not None and local_idx < local_valid.size and local_valid[local_idx]:
                            local_indices.append(local_idx)
                            patient_indices.append(patient_channel_idx)
                    if not patient_indices:
                        continue
                    matrix = np.asarray(record.get("adjacency"), dtype=np.float32)
                    local = matrix[np.ix_(local_indices, local_indices)]
                    index_tensor = torch.tensor(patient_indices, dtype=torch.long)
                    adjacency[patient_idx, seizure_idx, phase_idx][index_tensor[:, None], index_tensor[None, :]] = torch.from_numpy(local)
                    phase_channel_mask[patient_idx, seizure_idx, phase_idx, index_tensor] = True
                    graph_valid[patient_idx, seizure_idx, phase_idx] = bool(record.get("graph_valid", False)) and len(patient_indices) >= 4 and bool(np.any(local > 0.0))
        return {"graph_adjacency": adjacency, "graph_phase_channel_mask": phase_channel_mask, "graph_valid": graph_valid}


def group_records_by_subject(run_records: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in run_records:
        grouped[str(record.get("subject_id", ""))].append(record)
    return dict(grouped)


def filtered_cache(cache: Mapping[str, Any], exclusion_manifest: Any) -> dict[str, Any]:
    output = dict(cache)
    output["run_records"], _ = apply_exclusions(cache.get("run_records", []), exclusion_manifest)
    return output


__all__ = ["NPAMCollator", "filtered_cache", "group_records_by_subject", "load_cache", "load_graph_cache"]
