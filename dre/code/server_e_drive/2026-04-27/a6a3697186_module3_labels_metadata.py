from __future__ import annotations

import re

import pandas as pd

from module2_preprocessing import normalize_channel_name


def _has_non_na_status_label(status_description: str) -> bool:
    normalized = str(status_description).strip().lower()
    return normalized not in {"", "n/a", "na", "nan", "none", "unknown"}


def _as_binary_label(value, *, positive_value: int = 1):
    try:
        if pd.isna(value):
            return None
        value_i = int(float(value))
    except Exception:
        text = str(value).strip().lower()
        if text in {"true", "yes", "y"}:
            value_i = 1
        elif text in {"false", "no", "n"}:
            value_i = 0
        else:
            return None
    if value_i < 0:
        return None
    return 1 if value_i == int(positive_value) else 0


def parse_contact_topology(channel_name_norm: str):
    match = re.match(r"^([A-Z]+)(\d+)$", channel_name_norm)
    if match:
        return match.group(1), int(match.group(2))
    return channel_name_norm, None


def parse_channel_labels(channels_path: str, ez_definition: str = "soz_or_resected") -> pd.DataFrame:
    channels_df = pd.read_csv(channels_path, sep="\t")
    meta_records = []
    seen_norms = set()

    for _, row in channels_df.iterrows():
        channel_name_orig = str(row["name"])
        channel_name_norm = normalize_channel_name(channel_name_orig)

        if channel_name_norm in seen_norms:
            print(f"Warning: Duplicate normalized channel {channel_name_norm} in {channels_path}.")
            continue
        seen_norms.add(channel_name_norm)

        channel_type = str(row.get("type", "SEEG")).upper()
        status = str(row.get("status", "good")).strip().lower()
        status_desc = str(row.get("status_description", "")).strip().lower()

        soz_value = _as_binary_label(row.get("soz", None))
        resection_value = _as_binary_label(row.get("resection", None))
        is_soz = int(soz_value) if soz_value is not None else 1 if "soz" in status_desc else 0
        is_resected = (
            int(resection_value) if resection_value is not None else 1 if "resect" in status_desc else 0
        )
        has_non_na_label = 1 if (
            soz_value is not None
            or resection_value is not None
            or _has_non_na_status_label(status_desc)
        ) else 0

        if ez_definition in {"any_non_na_status", "non_na"}:
            is_ez = has_non_na_label
        elif ez_definition == "soz_or_resected":
            is_ez = 1 if (is_soz or is_resected) else 0
        elif ez_definition == "soz_only":
            is_ez = 1 if is_soz else 0
        else:
            is_ez = 0

        good_value = _as_binary_label(row.get("good", None))
        good_channel = bool(good_value == 1) if good_value is not None else status != "bad"
        is_valid = 1 if good_channel and channel_type in ["SEEG", "ECOG", "STEREOEEG"] else 0
        contact_group, contact_number = parse_contact_topology(channel_name_norm)

        meta_records.append(
            {
                "channel_name_orig": channel_name_orig,
                "channel_name_norm": channel_name_norm,
                "status_description": status_desc,
                "is_soz": is_soz,
                "is_resected": is_resected,
                "has_non_na_label": has_non_na_label,
                "is_ez": is_ez,
                "is_valid": is_valid,
                "type": channel_type,
                "contact_group": contact_group,
                "contact_number": contact_number,
                "channel_order": len(meta_records),
            }
        )

    return pd.DataFrame(meta_records)


__all__ = ["parse_channel_labels", "parse_contact_topology"]
