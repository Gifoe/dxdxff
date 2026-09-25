from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


def build_patient_index(run_records: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in run_records:
        grouped.setdefault(str(record["subject_id"]), []).append(record)
    patient_index: dict[str, dict[str, Any]] = {}
    for subject_id, records in grouped.items():
        channel_meta_map: dict[str, dict[str, Any]] = {}
        label_map: dict[str, float] = {}
        for record in records:
            for idx, channel_name in enumerate(record["channel_names_norm"]):
                if channel_name not in channel_meta_map:
                    channel_meta_map[channel_name] = {
                        "channel_name_norm": channel_name,
                        "contact_group": record["contact_groups"][idx],
                        "contact_number": record["contact_numbers"][idx],
                    }
                label_map[channel_name] = max(float(label_map.get(channel_name, 0.0)), float(record["labels"][idx]))
        canonical_meta = sorted(
            channel_meta_map.values(),
            key=lambda item: (
                str(item.get("contact_group", "")),
                int(item.get("contact_number") if item.get("contact_number") is not None else 10**9),
                str(item.get("channel_name_norm", "")),
            ),
        )
        channels = [item["channel_name_norm"] for item in canonical_meta]
        patient_index[subject_id] = {
            "subject_id": subject_id,
            "canonical_channels": channels,
            "channel_meta": canonical_meta,
            "labels": np.asarray([label_map.get(name, 0.0) for name in channels], dtype=np.float32),
            "label_mask": np.ones((len(channels),), dtype=bool),
        }
    return patient_index


def write_cache(path: Path, *, source_center: str, run_records: Sequence[dict[str, Any]], extra: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "cache_version": "hup_strict_interictal_v1",
        "source_center": source_center,
        "run_records": list(run_records),
        "patient_index": build_patient_index(run_records),
        **extra,
    }
    with path.open("wb") as fout:
        pickle.dump(payload, fout, protocol=pickle.HIGHEST_PROTOCOL)
