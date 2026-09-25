from __future__ import annotations

import pandas as pd

from .evaluation import evaluate_patient_predictions
from .schemas import normalize_channel_name


def compute_pair_label(patient_ledger: pd.DataFrame, eject_channel: str, add_channel: str, *, epsilon: float = 1e-12) -> dict[str, float | int]:
    """Label one swap by calling the shared patient evaluator before and after it."""
    before = evaluate_patient_predictions(patient_ledger, "old_v3_selected")["patient_macro_f1"]
    after_ledger = patient_ledger.copy()
    keys = after_ledger.get("channel_name_norm", after_ledger["channel_name_original"].map(normalize_channel_name)).astype(str)
    eject = keys.eq(normalize_channel_name(eject_channel))
    add = keys.eq(normalize_channel_name(add_channel))
    if eject.sum() != 1 or add.sum() != 1 or bool((eject & add).any()):
        raise ValueError("swap channels must be distinct, unique patient channels")
    after_ledger.loc[eject, "old_v3_selected"] = 0
    after_ledger.loc[add, "old_v3_selected"] = 1
    after = evaluate_patient_predictions(after_ledger, "old_v3_selected")["patient_macro_f1"]
    delta = float(after - before)
    return {"delta_patient_macro_f1": delta, "beneficial_label": int(delta > epsilon), "harmful_label": int(delta < -epsilon), "neutral_label": int(abs(delta) <= epsilon)}
