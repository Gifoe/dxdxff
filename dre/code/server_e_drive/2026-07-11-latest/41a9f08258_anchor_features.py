from __future__ import annotations

import hashlib
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .schemas import normalize_channel_name


ANCHOR_COLUMNS = ("distance_to_patient_nez_anchor_l1", "distance_to_patient_nez_anchor_l2",
                  "distance_to_patient_nez_anchor_cosine", "robust_anchor_z_mean")


def _hash_subject(value: object) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:16]


def build_patient_nez_anchor_features(
    ledger: pd.DataFrame, channel_features: Mapping[str, Mapping[str, Sequence[float]]], *,
    quantile: float = .50, min_channels: int = 8,
    minimum_channel_coverage: float = 0., minimum_feature_finite_ratio: float = 0.,
    strict: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build label-free anchors and an explicit per-patient availability table."""
    output = ledger.copy()
    for column in ANCHOR_COLUMNS:
        output[column] = np.nan
    coverage_rows: list[dict[str, object]] = []
    for subject_id, indices in output.groupby("subject_id", sort=True).groups.items():
        group = output.loc[indices]
        raw = channel_features.get(str(subject_id), {})
        available = {normalize_channel_name(name): np.asarray(value, dtype=float).reshape(-1)
                     for name, value in raw.items()}
        ledger_names = set(group["channel_name_norm"].astype(str))
        matched_names = sorted(ledger_names & set(available))
        finite_total = sum(vector.size for name, vector in available.items() if name in matched_names)
        finite_count = sum(int(np.isfinite(available[name]).sum()) for name in matched_names)
        finite_ratio = float(finite_count / finite_total) if finite_total else 0.
        coverage_ratio = float(len(matched_names) / len(ledger_names)) if ledger_names else 0.
        candidates = group[group["old_v3_selected"].astype(int) == 0].sort_values("old_v3_score_ez", kind="mergesort")
        requested = min(len(candidates), max(int(min_channels), int(np.ceil(len(candidates) * float(quantile)))))
        anchor_names = [str(name) for name in candidates.head(requested)["channel_name_norm"] if str(name) in available]
        status = "available"
        if coverage_ratio < float(minimum_channel_coverage):
            status = "unavailable_low_channel_coverage"
        elif finite_ratio < float(minimum_feature_finite_ratio):
            status = "unavailable_low_feature_finite_ratio"
        elif not anchor_names:
            status = "unavailable_no_anchor_channels"
        coverage_rows.append({
            "subject_id_hash": _hash_subject(subject_id), "n_ledger_channels": len(ledger_names),
            "n_cache_channels": len(available), "n_matched_channels": len(matched_names),
            "n_anchor_candidates": len(candidates), "n_anchor_used": len(anchor_names),
            "coverage_ratio": coverage_ratio, "anchor_feature_finite_ratio": finite_ratio,
            "missing_channel_count": len(ledger_names - set(available)), "status": status,
        })
        if status != "available":
            if strict:
                raise ValueError(f"anchor unavailable for subject hash {_hash_subject(subject_id)}: {status}")
            continue
        anchors = np.vstack([available[name] for name in anchor_names])
        with np.errstate(invalid="ignore"):
            median = np.nanmedian(anchors, axis=0)
            mad = np.nanmedian(np.abs(anchors - median), axis=0)
        mad = np.where(np.isfinite(mad) & (mad > 1e-6), mad, 1e-6)
        for index, row in group.iterrows():
            vector = available.get(str(row["channel_name_norm"]))
            if vector is None:
                continue
            valid = np.isfinite(vector) & np.isfinite(median)
            if not valid.any():
                continue
            x, center = vector[valid], median[valid]
            l1 = float(np.abs(x - center).sum())
            l2 = float(np.linalg.norm(x - center))
            denom = max(float(np.linalg.norm(x) * np.linalg.norm(center)), 1e-12)
            cosine = float(1. - np.dot(x, center) / denom)
            robust = float(np.mean(np.abs((x - center) / mad[valid])))
            output.loc[index, list(ANCHOR_COLUMNS)] = [l1, l2, cosine, robust]
    return output, pd.DataFrame(coverage_rows)
