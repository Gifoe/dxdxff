"""Leak-free adapter split and strict non-LZU invariance audit."""
from __future__ import annotations
import hashlib
import numpy as np

def deterministic_lzu_split(subjects: list[str], seed: int) -> tuple[list[str], list[str]]:
    values=sorted(set(map(str, subjects)))
    if len(values)<3: raise ValueError("Each outer-train must contain at least three LZU patients for adapter fit/validation")
    order=sorted(values,key=lambda x: hashlib.sha256(f"{seed}|{x}".encode()).hexdigest())
    n_val=max(1,round(.20*len(order))); return order[n_val:],order[:n_val]

def non_lzu_invariance_audit(base_logit: np.ndarray, adapted_logit: np.ndarray, base_score: np.ndarray, adapted_score: np.ndarray, base_pred: np.ndarray, adapted_pred: np.ndarray) -> dict:
    logit_equal=np.array_equal(base_logit,adapted_logit); score_equal=np.array_equal(base_score,adapted_score); pred_equal=np.array_equal(base_pred,adapted_pred)
    return {"n_non_lzu_channels":int(base_logit.size),"max_abs_logit_delta":float(np.max(np.abs(base_logit-adapted_logit))) if base_logit.size else 0.0,"max_abs_probability_delta":float(np.max(np.abs(base_score-adapted_score))) if base_logit.size else 0.0,"n_prediction_changes":int(np.count_nonzero(base_pred!=adapted_pred)),"bitwise_logit_equal":logit_equal,"bitwise_probability_equal":score_equal,"bitwise_prediction_equal":pred_equal,"passed":bool(logit_equal and score_equal and pred_equal)}

__all__=["deterministic_lzu_split","non_lzu_invariance_audit"]
