from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from module3_labels_metadata import build_patient_dictionaries


def _default_clinical_record() -> Dict[str, object]:
    return {
        "outcome": "unknown",
        "outcome_binary": float("nan"),
        "engel": "UNKNOWN",
        "therapy": "N/A",
        "implant": "N/A",
        "target": "N/A",
        "lesion_status": "N/A",
        "age": "N/A",
        "sex": "N/A",
        "hand": "N/A",
        "age_onset": "N/A",
    }


def _resolve_subject_dirs(dataset_path: Path, subject_filter: Optional[str]) -> List[Path]:
    if subject_filter:
        return [dataset_path / str(subject_filter)]
    return sorted(
        [path for path in dataset_path.iterdir() if path.is_dir() and path.name.startswith("P")],
        key=lambda path: path.name,
    )


def _phase_group_from_state(state_name: str) -> str:
    lowered = str(state_name).strip().lower()
    if "interictal" in lowered:
        return "interictal"
    if "ictal" in lowered:
        return "ictal"
    return lowered or "unknown"


def discover_all_mat_files(
    dataset_dir,
    ez_annotation_path,
    clinical_info_path,
    subject_filter=None,
    success_only=True,
):
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        return pd.DataFrame()

    _, patient_clinical_dict = build_patient_dictionaries(
        ez_annotation_path,
        clinical_info_path,
    )

    records = []
    for subject_dir in _resolve_subject_dirs(dataset_path, subject_filter):
        if not subject_dir.exists() or not subject_dir.is_dir():
            continue

        subject_id = subject_dir.name
        clinical_record = _default_clinical_record()
        clinical_record.update(patient_clinical_dict.get(subject_id, {}))

        state_dirs = sorted(
            [
                path
                for path in subject_dir.iterdir()
                if path.is_dir() and ("ictal" in path.name.lower() or "interictal" in path.name.lower())
            ],
            key=lambda path: path.name,
        )
        for state_dir in state_dirs:
            task = state_dir.name
            phase_group = _phase_group_from_state(task)
            task_slug = task.strip().lower().replace(" ", "-")

            for mat_path in sorted(state_dir.glob("*.mat"), key=lambda path: path.name):
                record = {
                    "subject_id": subject_id,
                    "run_id": f"{subject_id}__{task_slug}__{mat_path.stem}",
                    "task": task,
                    "phase_group": phase_group,
                    "mat_path": str(mat_path),
                    "seizure_onset": None,
                    "seizure_offset": None,
                }
                record.update(clinical_record)
                records.append(record)

    runs_df = pd.DataFrame(records)
    if runs_df.empty:
        return runs_df

    if bool(success_only):
        runs_df = runs_df[runs_df["outcome_binary"].eq(1.0)].copy()

    runs_df = runs_df.sort_values(["subject_id", "task", "run_id"]).reset_index(drop=True)
    return runs_df


__all__ = ["discover_all_mat_files"]
