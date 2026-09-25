from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np
import pandas as pd

from ..functional_graph import normalize_channel_name
from .schema import FeatureRunRecord, P2NEZRecord, RawRunRecord


def _unique(names: Sequence[str], patient: str, seizure: str, role: str) -> dict[str, int]:
    normalized = [normalize_channel_name(x) for x in names]
    duplicate = sorted({x for x in normalized if normalized.count(x) > 1})
    if "" in normalized or duplicate:
        raise ValueError(f"AMBIGUOUS_CHANNEL_NAME: patient={patient}, seizure={seizure}, role={role}, values={duplicate}")
    return {name: index for index, name in enumerate(normalized)}


def audit_raw_feature_alignment(
    features: Sequence[FeatureRunRecord], raws: Sequence[RawRunRecord], *, strict: bool,
) -> pd.DataFrame:
    raw_map = {(x.patient_key, x.seizure_id): x for x in raws}
    rows: list[dict] = []
    patient_rates: dict[str, list[float]] = defaultdict(list)
    for feature in features:
        raw = raw_map.get((feature.patient_key, feature.seizure_id))
        f = _unique(feature.channel_names, feature.patient_key, feature.seizure_id, "feature")
        r = _unique(raw.channel_names, feature.patient_key, feature.seizure_id, "raw") if raw else {}
        common = sorted(set(f) & set(r)); rate = len(common) / max(len(f), 1)
        row = {
            "patient_key": feature.patient_key, "seizure_id": feature.seizure_id,
            "n_feature_channels": len(f), "n_raw_channels": len(r),
            "n_matched_channels": len(common), "match_rate": rate,
            "missing_feature_channels": ";".join(sorted(set(r) - set(f))),
            "missing_raw_channels": ";".join(sorted(set(f) - set(r))),
        }
        rows.append(row); patient_rates[feature.patient_key].append(rate)
        if strict and rate < .90:
            raise ValueError(f"RAW_FEATURE_ALIGNMENT_BELOW_0P90: patient={feature.patient_key}, seizure={feature.seizure_id}, rate={rate:.3f}")
    if strict:
        bad = {key: float(np.mean(value)) for key, value in patient_rates.items() if np.mean(value) < .95}
        if bad:
            patient, rate = next(iter(bad.items()))
            raise ValueError(f"RAW_FEATURE_PATIENT_ALIGNMENT_BELOW_0P95: patient={patient}, mean_rate={rate:.3f}")
    return pd.DataFrame(rows)


def align_p2_to_raw(p2: P2NEZRecord, raw: RawRunRecord, *, strict: bool = True) -> P2NEZRecord:
    p = _unique(p2.channel_names, p2.patient_key, raw.seizure_id, "p2")
    r = _unique(raw.channel_names, raw.patient_key, raw.seizure_id, "raw")
    rate = len(set(p) & set(r)) / max(len(r), 1)
    if strict and rate < .95:
        raise ValueError(f"P2_CHANNEL_ALIGNMENT_BELOW_0P95: patient={raw.patient_key}, seizure={raw.seizure_id}, rate={rate:.3f}")
    final = np.full(len(raw.channel_names), np.nan); direct = np.full(len(raw.channel_names), np.nan) if p2.direct_nez_probability is not None else None
    valid = np.zeros(len(raw.channel_names), dtype=bool)
    for index, name in enumerate(raw.channel_names):
        source = p.get(normalize_channel_name(name))
        if source is None:
            continue
        final[index] = p2.final_nez_probability[source]
        if direct is not None:
            direct[index] = p2.direct_nez_probability[source]
        valid[index] = bool(p2.valid_channel_mask[source])
    return P2NEZRecord(raw.patient_key, raw.center, list(raw.channel_names), final, direct, valid, raw.clinical_target_mask.copy())


__all__ = ["align_p2_to_raw", "audit_raw_feature_alignment"]
