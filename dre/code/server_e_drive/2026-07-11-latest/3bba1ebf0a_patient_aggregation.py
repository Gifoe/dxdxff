from __future__ import annotations

from collections import defaultdict
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .schema import P2NEZRecord
from .seizure_aggregation import jaccard, top10_mean


CORE_FEATURES = (
    "fragility_outside_persistent_top10", "ei_outside_persistent_top10",
    "low_entropy_outside_persistent_top10", "target_to_outside_delay_mean",
    "outside_recruited_3s_mean", "biomarker_outside_mean_top10",
    "biomarker_outside_persistent_top10", "biomarker_target_alignment_mean",
    "biomarker_channel_recurrence", "outside_nez_q10", "outside_non_nez_top10",
    "joint_outside_mean_top10", "joint_outside_persistent_top10",
    "joint_high_risk_fraction_mean", "biomarker_nez_top10_overlap",
    "joint_channel_recurrence",
)
HFO_FEATURES = ("hfo_outside_persistent_top10", "hfo_target_alignment", "hfo_hub_outside_burden")


def _stat(values: Sequence[Any], kind: str) -> float:
    array = np.asarray(values, dtype=float); array = array[np.isfinite(array)]
    if not array.size: return np.nan
    if kind == "q25": return float(np.quantile(array, .25))
    return float(getattr(np, kind)(array))


def mean_pairwise_jaccard(sets: Sequence[set[str]]) -> float:
    valid = [value for value in sets if value]
    if len(valid) < 2: return np.nan
    return float(np.mean([jaccard(valid[i], valid[j]) for i in range(len(valid)) for j in range(i + 1, len(valid))]))


def aggregate_patients(seizures: Sequence[dict[str, Any]], p2_by_patient: dict[str, P2NEZRecord]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in seizures: grouped[str(row["patient_key"])].append(row)
    patients, recurrence_rows = [], []
    for patient, values in sorted(grouped.items()):
        p2 = p2_by_patient[patient]; outside = p2.valid_channel_mask & ~p2.clinical_target_mask; q = p2.final_nez_probability
        get = lambda name: [row.get(name, np.nan) for row in values]
        bio_sets = [row["top_biomarker_channels"] for row in values]; joint_sets = [row["top_joint_channels"] for row in values]
        bio_recurrence = mean_pairwise_jaccard(bio_sets); joint_recurrence = mean_pairwise_jaccard(joint_sets)
        row = {
            "patient_key": patient, "center": values[0]["center"],
            "fragility_outside_persistent_top10": _stat(get("fragility_outside_top10"), "q25"),
            "ei_outside_persistent_top10": _stat(get("ei_outside_top10"), "q25"),
            "low_entropy_outside_persistent_top10": _stat(get("low_entropy_outside_top10"), "q25"),
            "target_to_outside_delay_mean": _stat(get("target_to_outside_delay"), "mean"),
            "outside_recruited_3s_mean": _stat(get("outside_recruited_3s_fraction"), "mean"),
            "biomarker_outside_mean_top10": _stat(get("bio_top10"), "mean"),
            "biomarker_outside_persistent_top10": _stat(get("bio_top10"), "q25"),
            "biomarker_target_alignment_mean": _stat(get("biomarker_target_alignment"), "mean"),
            "biomarker_channel_recurrence": bio_recurrence,
            "outside_nez_q10": float(np.nanquantile(q[outside], .10)) if np.isfinite(q[outside]).any() else np.nan,
            "outside_non_nez_top10": top10_mean(1.0 - q, outside),
            "joint_outside_mean_top10": _stat(get("joint_top10"), "mean"),
            "joint_outside_persistent_top10": _stat(get("joint_top10"), "q25"),
            "joint_high_risk_fraction_mean": _stat(get("joint_high_risk_fraction"), "mean"),
            "biomarker_nez_top10_overlap": _stat(get("biomarker_nez_top10_overlap"), "mean"),
            "joint_channel_recurrence": joint_recurrence,
            "hfo_outside_persistent_top10": _stat(get("hfo_outside_top10"), "q25"),
            "hfo_target_alignment": _stat(get("hfo_target_alignment"), "mean"),
            "hfo_hub_outside_burden": _stat(get("hfo_hub_outside_burden"), "mean"),
            "n_valid_seizures": len(values),
            "n_fragility_valid_seizures": int(sum(bool(x.get("fragility_valid")) for x in values)),
            "n_ei_valid_seizures": int(sum(bool(x.get("ei_valid")) for x in values)),
            "n_entropy_valid_seizures": int(sum(bool(x.get("entropy_valid")) for x in values)),
            "n_hfo_valid_seizures": int(sum(bool(x.get("hfo_valid")) for x in values)),
        }
        patients.append(row); recurrence_rows.append({"patient_key": patient, "n_seizures": len(values), "biomarker_channel_recurrence": bio_recurrence, "joint_channel_recurrence": joint_recurrence, "recurrence_valid": len(values) >= 2})
    definition = []
    for order, name in enumerate(CORE_FEATURES + HFO_FEATURES, 1):
        definition.append({"order": order, "feature": name, "group": "hfo_conditional" if name in HFO_FEATURES else "core", "model_input": True, "aggregation": "fixed_protocol"})
    return pd.DataFrame(patients), pd.DataFrame(definition), pd.DataFrame(recurrence_rows)


__all__ = ["CORE_FEATURES", "HFO_FEATURES", "aggregate_patients", "mean_pairwise_jaccard"]
