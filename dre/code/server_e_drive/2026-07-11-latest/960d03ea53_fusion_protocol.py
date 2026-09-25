from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pandas as pd

from outcome_hifos.dataset import OutcomePatientExample


@dataclass(frozen=True)
class FusionCohort:
    feature_examples: tuple[OutcomePatientExample, ...]
    fm_examples: tuple[OutcomePatientExample, ...]
    manifest: pd.DataFrame
    alignment_audit: pd.DataFrame


def build_fusion_cohort(
    feature_examples: Sequence[OutcomePatientExample],
    fm_examples: Sequence[OutcomePatientExample],
    *,
    minimum_run_alignment_ratio: float = 1.0,
) -> FusionCohort:
    if not 0.0 < float(minimum_run_alignment_ratio) <= 1.0:
        raise ValueError("minimum_run_alignment_ratio must be in (0, 1].")
    feature_by_subject = {example.subject_id: example for example in feature_examples}
    fm_by_subject = {example.subject_id: example for example in fm_examples}
    if len(feature_by_subject) != len(feature_examples) or len(fm_by_subject) != len(fm_examples):
        raise ValueError("Fusion inputs contain duplicate subject IDs.")
    subjects = sorted(set(feature_by_subject) & set(fm_by_subject))
    if not subjects:
        raise ValueError("Feature and FM cohorts have an empty patient intersection.")
    rows = []
    selected_feature = []
    selected_fm = []
    audit_rows = []
    for subject in subjects:
        feature = feature_by_subject[subject]
        fm = fm_by_subject[subject]
        if int(feature.target) != int(fm.target):
            raise ValueError(f"Fusion outcome mismatch for subject {subject}.")
        if str(feature.center) != str(fm.center):
            raise ValueError(f"Fusion center mismatch for subject {subject}.")
        feature_runs = set(map(str, feature.side_metadata.get("run_ids", ())))
        fm_runs = set(map(str, fm.side_metadata.get("run_ids", ())))
        overlap = feature_runs & fm_runs
        if feature_runs and fm_runs and not overlap:
            raise ValueError(f"Fusion run alignment is empty for subject {subject}.")
        alignment_ratio = len(overlap) / max(len(feature_runs | fm_runs), 1)
        action = "include" if alignment_ratio >= float(minimum_run_alignment_ratio) else "exclude"
        reason = "meets_minimum_run_alignment_ratio" if action == "include" else "below_minimum_run_alignment_ratio"
        audit_rows.append(
            {
                "subject_id": subject,
                "feature_run_count": len(feature_runs),
                "fm_run_count": len(fm_runs),
                "aligned_run_count": len(overlap),
                "alignment_ratio": alignment_ratio,
                "action": action,
                "reason": reason,
            }
        )
        if action == "exclude":
            continue
        rows.append(
            {
                "subject_id": subject,
                "center": feature.center,
                "outcome_label": int(feature.target),
                "feature_run_count": len(feature_runs),
                "fm_run_count": len(fm_runs),
                "aligned_run_count": len(overlap),
                "run_alignment_ratio": alignment_ratio,
            }
        )
        selected_feature.append(feature)
        selected_fm.append(fm)
    if not selected_feature:
        raise ValueError("No fusion patients satisfy the configured run alignment policy.")
    return FusionCohort(tuple(selected_feature), tuple(selected_fm), pd.DataFrame(rows), pd.DataFrame(audit_rows))


__all__ = ["FusionCohort", "build_fusion_cohort"]
