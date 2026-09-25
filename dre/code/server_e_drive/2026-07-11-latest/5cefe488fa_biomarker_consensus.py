from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import BiomarkerRunMaps, P2NEZRecord, RawRunRecord


def consensus_map(*biomarkers: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.stack([np.asarray(x, dtype=float) for x in biomarkers], axis=0)
    count = np.sum(np.isfinite(values), axis=0); output = np.full(values.shape[1], np.nan); agreement = np.full(values.shape[1], np.nan)
    for channel in range(values.shape[1]):
        valid = values[:, channel][np.isfinite(values[:, channel])]
        if valid.size < 2: continue
        highest = np.sort(valid)[-2:]; output[channel] = float(np.sqrt(highest[0] * highest[1])); agreement[channel] = float((valid > .8).mean())
    return output, count.astype(int), agreement


def build_joint_maps(
    raw: RawRunRecord, p2: P2NEZRecord, fragility: np.ndarray, ei: np.ndarray,
    propagation: np.ndarray, low_entropy: np.ndarray, hfo: np.ndarray,
) -> tuple[BiomarkerRunMaps, pd.DataFrame]:
    if list(raw.channel_names) != list(p2.channel_names): raise ValueError(f"P2/raw channels must be aligned before consensus: {raw.patient_key}/{raw.seizure_id}")
    consensus, count, agreement = consensus_map(fragility, ei, propagation, low_entropy)
    target = np.asarray(raw.clinical_target_mask, dtype=bool); outside = ~target; q_nez = np.asarray(p2.final_nez_probability, dtype=float); non_nez = 1.0 - q_nez
    bio = np.where(outside, consensus, 0.0); nez = np.where(outside, non_nez, 0.0); joint = np.where(outside, consensus * non_nez, 0.0)
    bio[~np.isfinite(consensus)] = np.nan; joint[~np.isfinite(consensus)] = np.nan
    maps = BiomarkerRunMaps(raw.patient_key, raw.center, raw.seizure_id, list(raw.channel_names), np.asarray(fragility), np.asarray(ei), np.asarray(propagation), np.asarray(low_entropy), np.asarray(hfo), consensus, bio, nez, joint, count)
    table = pd.DataFrame({"patient_key": raw.patient_key, "center": raw.center, "seizure_id": raw.seizure_id, "channel": raw.channel_names, "clinical_target": target.astype(int), "fragility": fragility, "ei": ei, "propagation": propagation, "low_entropy": low_entropy, "hfo": hfo, "q_nez": q_nez, "non_nez": non_nez, "valid_biomarker_count": count, "biomarker_consensus": consensus, "agreement_fraction": agreement, "biomarker_only_residual": bio, "nez_only_residual": nez, "joint_residual": joint})
    return maps, table


__all__ = ["build_joint_maps", "consensus_map"]
