from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .matching import select_max_weight_swaps
from .evaluation import evaluate_patient_predictions
from .schemas import normalize_channel_name


def recompute_utility(frame: pd.DataFrame, *, risk_lambda: float, uncertainty_lambda: float = 0.0) -> pd.DataFrame:
    """Derive utility only from calibrated probabilities and ensemble uncertainty."""
    out = frame.copy()
    uncertainty = pd.to_numeric(out.get("pred_delta_std", 0.0), errors="coerce").fillna(0.0) if isinstance(out.get("pred_delta_std", 0.0), pd.Series) else 0.0
    out["utility"] = (pd.to_numeric(out["p_benefit"], errors="coerce").fillna(0.0) * np.maximum(pd.to_numeric(out["pred_delta"], errors="coerce").fillna(0.0), 0.0) - float(risk_lambda) * pd.to_numeric(out["p_harm"], errors="coerce").fillna(1.0) - float(uncertainty_lambda) * uncertainty)
    out["utility_semantics"] = "calibrated_benefit_times_positive_delta_minus_risk_minus_uncertainty"
    return out


@dataclass(frozen=True)
class PatientPolicyResult:
    subject_id: str
    action_count: int
    patient_macro_f1: float
    anchor_patient_macro_f1: float
    delta_patient_macro_f1: float
    actions: pd.DataFrame


def simulate_patient_policy(ledger: pd.DataFrame, pairs: pd.DataFrame, *, tau_benefit: float, tau_harm: float, tau_utility: float, tau_delta: float, tau_positive_seed_fraction: float, max_swaps: int) -> PatientPolicyResult:
    if ledger["subject_id"].nunique() != 1:
        raise ValueError("patient policy simulation requires exactly one patient ledger")
    subject = str(ledger["subject_id"].iloc[0])
    positive_fraction = pairs["seed_positive_fraction"] if "seed_positive_fraction" in pairs else pd.Series(1.0, index=pairs.index)
    eligible = pairs[(pairs["p_benefit"] >= tau_benefit) & (pairs["p_harm"] <= tau_harm) & (pairs["utility"] >= tau_utility) & (pairs["pred_delta"] >= tau_delta) & (positive_fraction >= tau_positive_seed_fraction)]
    actions = select_max_weight_swaps(eligible, max_swaps=max_swaps, utility_threshold=-np.inf)
    final = ledger.copy()
    final["pred_ez"] = final["old_v3_selected"].astype(int)
    normalized = final.get("channel_name_norm", final["channel_name_original"].map(normalize_channel_name)).astype(str)
    for _, action in actions.iterrows():
        final.loc[normalized == normalize_channel_name(action.get("eject_channel_norm", action["eject_channel"])), "pred_ez"] = 0
        final.loc[normalized == normalize_channel_name(action.get("add_channel_norm", action["add_channel"])), "pred_ez"] = 1
    anchor = evaluate_patient_predictions(ledger, "old_v3_selected")["patient_macro_f1"]
    current = evaluate_patient_predictions(final, "pred_ez")["patient_macro_f1"]
    return PatientPolicyResult(subject, len(actions), float(current), float(anchor), float(current - anchor), actions)
