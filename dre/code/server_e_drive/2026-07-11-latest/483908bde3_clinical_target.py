from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from .functional_graph import normalize_channel_name


def task1_nez_to_clinical_target(labels_nez: Any) -> np.ndarray:
    values = np.asarray(labels_nez, dtype=np.float32)
    valid = np.isfinite(values) & np.isin(values, (0.0, 1.0))
    if not bool(valid.all()):
        raise ValueError("Task-1 labels must be complete binary NEZ=1/EZ=0 labels; missing values cannot become non-target")
    return (1.0 - values).astype(np.float32, copy=False)


def center_clinical_target(channel_meta: Sequence[Mapping[str, Any]], center: str, fallback_labels_nez: Any) -> tuple[np.ndarray, list[str]]:
    """Map all centers to one clinician-defined target without outcome access."""
    fallback=task1_nez_to_clinical_target(fallback_labels_nez); values=[]; sources=[]; center=str(center).lower()
    for index,item in enumerate(channel_meta):
        if center=="hup" and (item.get("soz") is not None or item.get("resection") is not None):
            values.append(float(item.get("soz")==1 or item.get("resection")==1)); sources.append("HUP:SOZ_OR_RESECTION")
        elif item.get("is_ez_or_soz") is not None:
            values.append(float(bool(item.get("is_ez_or_soz")))); sources.append(f"{center}:TASK1_EZ_COMPOSITE")
        elif item.get("final_label") is not None and float(item.get("final_label")) in (0.0,1.0):
            values.append(1.0-float(item["final_label"])); sources.append(f"{center}:TASK1_FINAL_LABEL")
        else:
            values.append(float(fallback[index])); sources.append(f"{center}:PATIENT_INDEX_TASK1_LABEL")
    return np.asarray(values,dtype=np.float32),sources


def _unique_index(names: Sequence[Any], *, patient: str, role: str) -> dict[str, int]:
    normalized = [normalize_channel_name(value) for value in names]
    duplicates = sorted({value for value in normalized if normalized.count(value) > 1})
    if "" in normalized or duplicates:
        raise ValueError(f"{patient}: ambiguous {role} channel names after normalization: {duplicates or ['<empty>']}")
    return {value: index for index, value in enumerate(normalized)}


def build_clinical_target_lookup(
    cache: Mapping[str, Any], patients: Sequence[str] | None = None, *, strict: bool = True, minimum_match_fraction: float = 0.95,
) -> tuple[dict[str, dict[str, Any]], pd.DataFrame, pd.DataFrame]:
    selected = set(map(str, patients)) if patients is not None else None
    records_by_patient: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in cache.get("run_records", []):
        records_by_patient[str(record.get("subject_id", ""))].append(record)
    lookup: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for patient, meta in cache.get("patient_index", {}).items():
        patient = str(patient)
        if selected is not None and patient not in selected:
            continue
        center = str(meta.get("source_center", patient.split(":", 1)[0])).lower()
        signal_names = list(meta.get("canonical_channels", []))
        signal_index = _unique_index(signal_names, patient=patient, role="signal")
        label_names: list[Any] = []
        labels: np.ndarray | None = None
        sources: list[str] = []
        channel_meta = list(meta.get("channel_meta", []))
        if channel_meta and len(channel_meta) == len(meta.get("labels", [])):
            label_names = [item.get("channel_name_norm", item.get("channel_name_orig")) for item in channel_meta]
            target_values,sources = center_clinical_target(channel_meta,center,meta.get("labels"))
            labels = 1.0-target_values
        elif signal_names and len(signal_names) == len(meta.get("labels", [])):
            label_names, labels = signal_names, np.asarray(meta.get("labels"), dtype=np.float32)
            sources = ["patient_index.labels"] * len(label_names)
        if labels is None:
            raise ValueError(f"{patient}: clinical target labels are missing")
        label_index = _unique_index(label_names, patient=patient, role="label")
        matched = [name for name in signal_index if name in label_index]
        match_fraction = len(matched) / max(len(signal_index), 1)
        aligned = np.full(len(signal_names), np.nan, dtype=np.float32)
        aligned_sources: list[str] = ["missing"] * len(signal_names)
        for name in matched:
            si, li = signal_index[name], label_index[name]
            aligned[si] = labels[li]
            aligned_sources[si] = sources[li] if li < len(sources) else "unknown"
        status = "ok"
        if match_fraction < minimum_match_fraction:
            status = "low_match_fraction"
        elif not np.isfinite(aligned).all():
            status = "missing_channel_label"
        elif not np.isin(aligned, (0.0, 1.0)).all():
            status = "non_binary_label"
        target = task1_nez_to_clinical_target(aligned) if status == "ok" else np.zeros_like(aligned)
        has_target, has_non_target = bool((target == 1).any()), bool((target == 0).any())
        if status == "ok" and not has_target:
            status = "all_zero_target"
        if status == "ok" and not has_non_target:
            status = "all_one_target"
        outcome = meta.get("surgery_success")
        row = {
            "patient_key": patient, "center": center,
            "outcome_label": int(bool(outcome)) if outcome is not None else np.nan,
            "n_signal_channels": len(signal_names), "n_label_channels": len(label_names),
            "n_matched_channels": len(matched), "match_fraction": match_fraction,
            "n_target_channels": int(target.sum()), "target_fraction": float(target.mean()) if target.size else np.nan,
            "has_target": has_target, "has_non_target": has_non_target,
            "label_source": "|".join(sorted(set(aligned_sources) - {"missing"})), "alignment_status": status,
        }
        rows.append(row)
        if strict and status != "ok":
            raise ValueError(f"{patient}: invalid clinical_target_mask ({status}, match_fraction={match_fraction:.3f})")
        if status == "ok":
            lookup[patient] = {"channel_names": signal_names, "mask": target, "label_source": row["label_source"], "center": center}
    audit = pd.DataFrame(rows)
    distribution_rows: list[dict[str, Any]] = []
    if not audit.empty:
        for (center, outcome), group in audit.groupby(["center", "outcome_label"], dropna=False):
            values = pd.to_numeric(group["target_fraction"], errors="coerce")
            distribution_rows.append({
                "center": center, "outcome": outcome, "n_patients": len(group),
                "mean_target_fraction": values.mean(), "std_target_fraction": values.std(ddof=1),
                "q10_target_fraction": values.quantile(.10), "q50_target_fraction": values.quantile(.50), "q90_target_fraction": values.quantile(.90),
                "missing_count": int(values.isna().sum()), "all_zero_count": int((group["n_target_channels"] == 0).sum()),
                "all_one_count": int((group["has_non_target"] == False).sum()),
            })
    return lookup, audit, pd.DataFrame(distribution_rows)


def collate_clinical_targets(subjects: Sequence[str], canonical_channels: Sequence[Sequence[str]], lookup: Mapping[str, Mapping[str, Any]], max_channels: int) -> torch.Tensor:
    output = torch.full((len(subjects), max_channels), float("nan"), dtype=torch.float32)
    for patient_idx, (subject, channels) in enumerate(zip(subjects, canonical_channels)):
        record = lookup.get(str(subject))
        if record is None:
            raise KeyError(f"Missing clinical_target_mask for {subject}")
        source_index = _unique_index(record["channel_names"], patient=str(subject), role="clinical-target")
        for channel_idx, channel in enumerate(channels):
            key = normalize_channel_name(channel)
            if key not in source_index:
                raise ValueError(f"{subject}: P2 channel {channel!r} has no clinical target label")
            output[patient_idx, channel_idx] = float(record["mask"][source_index[key]])
    return output


__all__ = ["build_clinical_target_lookup", "center_clinical_target", "collate_clinical_targets", "task1_nez_to_clinical_target"]
