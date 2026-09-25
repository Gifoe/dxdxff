from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def load_success_subject_manifest(path: str | Path) -> set[str]:
    """Load an audited success-patient set from CSV or sensitivity audit JSON."""
    source = Path(path)
    if source.suffix.lower() == ".json":
        payload = json.loads(source.read_text(encoding="utf-8"))
        values = payload.get("retained_subjects")
        if not isinstance(values, list) or not values:
            raise ValueError(f"Success manifest JSON needs a non-empty retained_subjects list: {source}")
        return {str(value) for value in values}
    table = pd.read_csv(source)
    if "subject_id" not in table:
        raise ValueError(f"Success manifest must contain subject_id: {source}")
    if "status" in table:
        table = table[table["status"].astype(str).str.lower().eq("retained")]
    subjects = {str(value) for value in table["subject_id"].dropna()}
    if not subjects:
        raise ValueError(f"Success manifest has no retained subjects: {source}")
    return subjects


__all__ = ["load_success_subject_manifest"]
