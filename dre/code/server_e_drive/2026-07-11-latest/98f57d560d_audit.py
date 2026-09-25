from __future__ import annotations

from typing import Any

import pandas as pd

from .candidate_pool import CandidateConfig, build_candidate_pools
from .pair_dataset import build_pair_dataset
from .protocol import anchor_parity, assert_frozen_anchor


def audit_anchor(ledger: pd.DataFrame, *, expected_macro_f1: float | None = None, expected_patients: int | None = 90, expected_folds: int = 5, tolerance: float = 1e-6) -> dict[str, Any]:
    contract = assert_frozen_anchor(ledger, expected_patients=expected_patients, expected_folds=expected_folds)
    if expected_macro_f1 is None:
        return {"contract": contract, "passed": True, "metric_parity": {"passed": False, "status": "not_claimed_without_independent_expected_metric", "expected_patient_macro_f1": None}}
    metric = anchor_parity(ledger, expected_macro_f1=float(expected_macro_f1), tolerance=tolerance)
    return {"contract": contract, "passed": bool(metric["passed"]), "metric_parity": metric, **metric}


def candidate_ceiling_audit(ledger: pd.DataFrame, config: CandidateConfig = CandidateConfig()) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    candidates = build_candidate_pools(ledger, config)
    pairs = build_pair_dataset(ledger, candidates, include_labels=True)
    by_patient = []
    for subject_id, group in pairs.groupby("subject_id", sort=True):
        improving = group[group["beneficial_label"].astype(int) == 1]
        by_patient.append({"subject_id": subject_id, "candidate_pairs": len(group), "beneficial_pairs": len(improving), "best_delta": float(group["delta_patient_macro_f1"].max()) if not group.empty else 0.0, "complete_corrective_pair_coverage": bool(not improving.empty)})
    patient = pd.DataFrame(by_patient)
    summary = {"diagnostic_only": True, "uses_ground_truth": True, "not_a_model_result": True, "n_candidate_pairs": int(len(pairs)), "candidate_patient_coverage": float(patient["complete_corrective_pair_coverage"].mean()) if not patient.empty else 0.0}
    return summary, patient, pairs
