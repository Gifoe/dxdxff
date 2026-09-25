from __future__ import annotations

import math
from typing import Any

import numpy as np

from .schema import BiomarkerRunMaps, P2NEZRecord, RawRunRecord


def top_fraction_set(values: np.ndarray, channels: list[str], valid: np.ndarray, fraction: float = .10) -> set[str]:
    indices = np.where(np.asarray(valid, dtype=bool) & np.isfinite(values))[0]
    if not indices.size: return set()
    k = min(indices.size, max(1, math.ceil(indices.size * fraction))); top = indices[np.argsort(values[indices], kind="stable")[-k:]]
    return {channels[index] for index in top}


def top10_mean(values: np.ndarray, valid: np.ndarray | None = None) -> float:
    data = np.asarray(values, dtype=float); mask = np.isfinite(data) if valid is None else np.asarray(valid, dtype=bool) & np.isfinite(data); local = data[mask]
    if not local.size: return np.nan
    k = min(local.size, max(1, math.ceil(local.size * .10))); return float(np.sort(local)[-k:].mean())


def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else np.nan


def aggregate_seizure(maps: BiomarkerRunMaps, raw: RawRunRecord, p2: P2NEZRecord, propagation_audit: dict, hfo_hub: np.ndarray) -> dict[str, Any]:
    valid = raw.valid_channel_mask; target = raw.clinical_target_mask & valid; outside = ~raw.clinical_target_mask & valid; consensus = maps.biomarker_consensus; non_nez = 1.0 - p2.final_nez_probability
    bio_set = top_fraction_set(consensus, maps.channel_names, valid); nez_set = top_fraction_set(non_nez, maps.channel_names, valid); joint_set = top_fraction_set(maps.joint_residual, maps.channel_names, outside)
    denominator = float(np.nansum(consensus[valid])); alignment = float(np.nansum(consensus[target]) / denominator) if denominator > 0 else np.nan
    hfo_denominator = float(np.nansum(maps.hfo[valid])); hfo_alignment = float(np.nansum(maps.hfo[target]) / hfo_denominator) if hfo_denominator > 0 else np.nan
    return {
        "patient_key": maps.patient_key, "center": maps.center, "seizure_id": maps.seizure_id,
        "bio_top10": top10_mean(maps.biomarker_only_residual, outside), "joint_top10": top10_mean(maps.joint_residual, outside),
        "joint_high_risk_fraction": float((maps.joint_residual[outside] > .5).mean()) if outside.any() else np.nan,
        "biomarker_target_alignment": alignment, "biomarker_nez_top10_overlap": jaccard(bio_set, nez_set),
        "top_joint_channels": joint_set, "top_biomarker_channels": bio_set,
        "fragility_outside_top10": top10_mean(maps.fragility, outside), "ei_outside_top10": top10_mean(maps.ei, outside),
        "low_entropy_outside_top10": top10_mean(maps.low_entropy, outside),
        "target_to_outside_delay": propagation_audit.get("target_to_outside_delay", np.nan),
        "outside_recruited_3s_fraction": propagation_audit.get("outside_recruited_3s_fraction", np.nan),
        "hfo_outside_top10": top10_mean(maps.hfo, outside), "hfo_target_alignment": hfo_alignment,
        "hfo_hub_outside_burden": top10_mean(hfo_hub, outside),
        "fragility_valid": bool(np.isfinite(maps.fragility).sum() >= 4), "ei_valid": bool(np.isfinite(maps.ei).sum() >= 4),
        "entropy_valid": bool(np.isfinite(maps.low_entropy).sum() >= 4), "hfo_valid": bool(np.isfinite(maps.hfo).sum() >= 4),
    }


__all__ = ["aggregate_seizure", "jaccard", "top10_mean", "top_fraction_set"]
