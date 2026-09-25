from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


OUTCOME_FIELDS = ("outcome_group", "surgery_success", "outcome_raw", "outcome", "lzu_outcome_raw", "engel_score", "engel", "surgery_result", "seizure_free")


def normalize_outcome(value: Any, *, field: str = "outcome") -> str:
    if value is None:
        return "unknown"
    field_lower = str(field).lower()
    if isinstance(value, (bool, np.bool_)):
        return "success" if bool(value) else "failure"
    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            return "unknown"
        if "engel" in field_lower:
            return "success" if number == 1 else "failure" if number in {2, 3, 4} else "unknown"
        if "failure" in field_lower:
            return "failure" if number == 1 else "success" if number == 0 else "unknown"
        return "success" if number == 1 else "failure" if number == 0 else "unknown"
    text = str(value).strip().lower().replace("_", " ").replace("-", " ")
    compact = "".join(char for char in text if char.isalnum())
    if not text or text in {"unknown", "nan", "none", "null", "na", "n/a", "nr"}:
        return "unknown"
    if compact.startswith("engel"):
        compact = compact[5:]
    if compact in {"i", "1", "success", "successful", "yes", "true", "seizurefree", "s"} or any(token in text for token in ("成功", "无发作", "無發作")):
        return "success"
    if compact in {"ii", "iii", "iv", "2", "3", "4", "failure", "failed", "fail", "no", "false", "recurrence", "f"} or any(token in text for token in ("失败", "失敗", "复发", "復發", "仍发作", "仍發作")):
        return "failure"
    return "unknown"


def _mapping_candidates(mapping: Any, prefix: str) -> list[tuple[str, str, Any]]:
    if not isinstance(mapping, Mapping):
        return []
    output = []
    for field in OUTCOME_FIELDS:
        if field in mapping:
            output.append((f"{prefix}.{field}", normalize_outcome(mapping.get(field), field=field), mapping.get(field)))
    return output


def resolve_cache_outcomes(cache: Mapping[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    patient_index = cache.get("patient_index", {})
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in cache.get("run_records", []):
        grouped[str(record.get("subject_id", ""))].append(record)
    rows, candidate_rows = [], []
    subjects = sorted(set(map(str, patient_index)) | set(grouped))
    for subject in subjects:
        meta = patient_index.get(subject, {})
        candidates = _mapping_candidates(meta, "patient")
        for index, record in enumerate(grouped.get(subject, [])):
            candidates.extend(_mapping_candidates(record, f"run[{index}]"))
            candidates.extend(_mapping_candidates(record.get("metadata"), f"run[{index}].metadata"))
            candidates.extend(_mapping_candidates(record.get("sample"), f"run[{index}].sample"))
        informative = {group for _, group, _ in candidates if group in {"success", "failure"}}
        group = "conflict" if informative == {"success", "failure"} else next(iter(informative)) if informative else "unknown"
        center = str(meta.get("source_center", meta.get("center", subject.split(":", 1)[0] if ":" in subject else "unknown"))).lower()
        rows.append({"patient_id": subject, "patient_key": subject, "center": center, "outcome_group": group, "outcome_label": 1 if group == "success" else 0 if group == "failure" else np.nan})
        candidate_rows.extend({"patient_key": subject, "source": source, "parsed_group": parsed, "raw_value": raw} for source, parsed, raw in candidates)
    return pd.DataFrame(rows), pd.DataFrame(candidate_rows)


def load_outcome_table(path: str | Path | None, *, cache: Mapping[str, Any] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    if path is None or str(path).strip() in {"", "cache", "cache://patient_index"}:
        if cache is None:
            raise ValueError("cache outcome source requires a loaded feature cache")
        return resolve_cache_outcomes(cache)
    source = Path(path).expanduser()
    if source.suffix.lower() == ".csv":
        frame = pd.read_csv(source)
    elif source.suffix.lower() in {".parquet", ".pq"}:
        frame = pd.read_parquet(source)
    elif source.suffix.lower() in {".json", ".jsonl"}:
        frame = pd.read_json(source, lines=source.suffix.lower() == ".jsonl")
    else:
        raise ValueError(f"Unsupported outcome table format: {source}")
    id_column = next((column for column in ("patient_key", "patient_id", "subject_id") if column in frame.columns), None)
    if id_column is None:
        raise ValueError("Outcome table needs patient_key, patient_id, or subject_id")
    center_column = "center" if "center" in frame.columns else None
    group_column = next((column for column in ("outcome_group", "outcome_label_failure", "engel", "engel_score", "outcome", "surgery_success") if column in frame.columns), None)
    if group_column is None:
        raise ValueError("Outcome table needs outcome_group/Engel/outcome/surgery_success")
    output = pd.DataFrame({
        "patient_id": frame[id_column].astype(str),
        "patient_key": frame[id_column].astype(str),
        "center": frame[center_column].astype(str).str.lower() if center_column else frame[id_column].astype(str).str.split(":").str[0].str.lower(),
        "outcome_group": [normalize_outcome(value, field=group_column) for value in frame[group_column]],
    })
    grouped = output.groupby("patient_key")["outcome_group"].agg(lambda values: set(value for value in values if value in {"success", "failure"}))
    conflicts = set(grouped[grouped.map(len) > 1].index)
    output.loc[output["patient_key"].isin(conflicts), "outcome_group"] = "conflict"
    output = output.drop_duplicates("patient_key", keep="first").reset_index(drop=True)
    output["outcome_label"] = output["outcome_group"].map({"success": 1, "failure": 0})
    return output, pd.DataFrame(columns=("patient_key", "source", "parsed_group", "raw_value"))


__all__ = ["load_outcome_table", "normalize_outcome", "resolve_cache_outcomes"]
