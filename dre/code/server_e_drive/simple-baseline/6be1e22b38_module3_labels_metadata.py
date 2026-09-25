from __future__ import annotations

from functools import lru_cache
import re
from typing import Dict, Iterable, Tuple

import pandas as pd

from module2_preprocessing import normalize_channel_name


def _normalize_patient_id(value) -> str:
    return str(value).strip()


def _extract_contact_ids(value) -> set[int]:
    return {int(match) for match in re.findall(r"\d+", str(value))}


def _map_outcome_binary(engel_value: str) -> float:
    normalized = str(engel_value).strip().upper()
    if re.match(r"^ENGEL\s*I[A-D]?$", normalized):
        return 1.0
    if re.match(r"^ENGEL\s*(II|III|IV)[A-D]?$", normalized):
        return 0.0
    return float("nan")


def parse_contact_topology(channel_name_norm: str):
    match = re.match(r"^([A-Z]+)(\d+)$", channel_name_norm)
    if match:
        return match.group(1), int(match.group(2))
    return channel_name_norm, None


def _make_unique_channel_name(
    channel_name_base_norm: str,
    contact_id: int,
    seen_norms: set[str],
) -> str:
    if channel_name_base_norm not in seen_norms:
        seen_norms.add(channel_name_base_norm)
        return channel_name_base_norm

    unique_name = f"{channel_name_base_norm}__C{contact_id}"
    seen_norms.add(unique_name)
    return unique_name


@lru_cache(maxsize=None)
def _load_patient_dictionaries_cached(
    ez_annotation_path: str,
    clinical_info_path: str,
) -> Tuple[Dict[str, Dict[str, object]], Dict[str, Dict[str, object]]]:
    ez_df = pd.read_excel(ez_annotation_path)
    patient_ez_dict: Dict[str, Dict[str, object]] = {}
    for _, row in ez_df.iterrows():
        patient_id = _normalize_patient_id(row.get("Patient ID"))
        patient_ez_dict[patient_id] = {
            "ez_contacts": _extract_contact_ids(row.get("EZ Channel ID", "")),
            "bad_contacts": _extract_contact_ids(row.get("Deleted Channels \n(Bad contacts)", "")),
        }

    clinical_df = pd.read_excel(clinical_info_path)
    patient_clinical_dict: Dict[str, Dict[str, object]] = {}
    for _, row in clinical_df.iterrows():
        patient_id = _normalize_patient_id(row.get("Patient ID"))
        engel = str(row.get("Outcomes", "UNKNOWN")).strip()
        outcome_binary = _map_outcome_binary(engel)
        if pd.isna(outcome_binary):
            outcome = "unknown"
        else:
            outcome = "good" if float(outcome_binary) == 1.0 else "poor"

        patient_clinical_dict[patient_id] = {
            "outcome": outcome,
            "outcome_binary": outcome_binary,
            "engel": engel,
            "therapy": str(row.get("Therapy", "N/A")),
            "implant": "N/A",
            "target": str(row.get("EZ", "N/A")),
            "lesion_status": str(row.get("MRI", "N/A")),
            "age": row.get("Age of Surgery", "N/A"),
            "sex": str(row.get("Sex", "N/A")),
            "hand": "N/A",
            "age_onset": row.get("Age of Onset", "N/A"),
        }

    return patient_ez_dict, patient_clinical_dict


def build_patient_dictionaries(ez_annotation_path, clinical_info_path):
    return _load_patient_dictionaries_cached(
        str(ez_annotation_path),
        str(clinical_info_path),
    )


def parse_mat_channel_labels(
    patient_id,
    ch_names_raw: Iterable[str],
    patient_ez_dict,
):
    patient_id = _normalize_patient_id(patient_id)
    patient_info = patient_ez_dict.get(
        patient_id,
        {"ez_contacts": set(), "bad_contacts": set()},
    )
    ez_contacts = set(patient_info.get("ez_contacts", set()))
    bad_contacts = set(patient_info.get("bad_contacts", set()))

    meta_records = []
    seen_norms = set()
    for row_idx, channel_name_orig in enumerate(ch_names_raw):
        contact_id = row_idx + 1
        channel_name_base_norm = normalize_channel_name(channel_name_orig)
        channel_name_norm = _make_unique_channel_name(
            channel_name_base_norm,
            contact_id,
            seen_norms,
        )
        is_bad = 1 if contact_id in bad_contacts else 0
        is_ez = 1 if contact_id in ez_contacts else 0
        contact_group, contact_number = parse_contact_topology(channel_name_base_norm)

        meta_records.append(
            {
                "channel_name_orig": str(channel_name_orig),
                "channel_name_norm": channel_name_norm,
                "channel_name_base_norm": channel_name_base_norm,
                "status_description": "ez" if is_ez else "",
                "is_soz": is_ez,
                "is_resected": 0,
                "has_non_na_label": is_ez,
                "is_ez": is_ez,
                "is_valid": 0 if is_bad else 1,
                "type": "SEEG",
                "contact_id": contact_id,
                "contact_group": contact_group,
                "contact_number": contact_number,
                "channel_order": row_idx,
            }
        )

    return pd.DataFrame(meta_records)


__all__ = [
    "build_patient_dictionaries",
    "parse_contact_topology",
    "parse_mat_channel_labels",
]
