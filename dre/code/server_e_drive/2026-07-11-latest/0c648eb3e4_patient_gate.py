from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class RuleGate:
    tau_benefit: float = .70
    tau_harm: float = .20
    tau_utility: float = 0.
    tau_delta: float = 0.
    tau_positive_seed_fraction: float = 1.

    def accept(self, action: Mapping[str, float]) -> bool:
        return bool(float(action.get("p_benefit", 0.)) >= self.tau_benefit
                    and float(action.get("p_harm", 1.)) <= self.tau_harm
                    and float(action.get("utility", float("-inf"))) >= self.tau_utility
                    and float(action.get("pred_delta", float("-inf"))) >= self.tau_delta
                    and float(action.get("seed_positive_fraction", 0.)) >= self.tau_positive_seed_fraction)


class LearnedPatientGate:
    def __init__(self, threshold: float = .5) -> None:
        self.threshold = float(threshold)
        self.model: LogisticRegression | None = None
        self.scaler = StandardScaler()
        self.constant: float | None = None
        self.feature_columns: list[str] = []
        self.fit_subjects: set[str] = set()
        self.scaler_fit_subjects: set[str] = set()

    def fit(self, frame: pd.DataFrame, feature_columns: list[str], target_column: str = "net_beneficial") -> "LearnedPatientGate":
        self.feature_columns = list(feature_columns)
        self.fit_subjects = set(frame["subject_id"].astype(str))
        matrix = np.nan_to_num(frame[self.feature_columns].to_numpy(dtype=float), nan=0., posinf=0., neginf=0.)
        matrix = self.scaler.fit_transform(matrix)
        self.scaler_fit_subjects = set(self.fit_subjects)
        target = frame[target_column].astype(int).to_numpy()
        if len(np.unique(target)) < 2:
            self.constant = float(target.mean()) if len(target) else 0.
        else:
            self.model = LogisticRegression(C=.25, max_iter=1000, random_state=0).fit(matrix, target)
        return self

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        matrix = np.nan_to_num(frame[self.feature_columns].to_numpy(dtype=float), nan=0., posinf=0., neginf=0.)
        matrix = self.scaler.transform(matrix)
        if self.model is not None:
            return self.model.predict_proba(matrix)[:, 1]
        return np.full(len(frame), .5 if self.constant is None else self.constant)

    def accept(self, frame: pd.DataFrame) -> np.ndarray:
        return self.predict_proba(frame) >= self.threshold


def build_patient_action_summary(patient_ledger: pd.DataFrame, all_scored_pairs: pd.DataFrame,
                                 eligible_pairs: pd.DataFrame | None = None,
                                 matched_actions: pd.DataFrame | None = None,
                                 selected_policy: Mapping[str, object] | object | None = None) -> pd.DataFrame:
    """Return exactly one gate row describing all, eligible, and matched pairs."""
    # Compatibility with the former three-argument API.
    if matched_actions is None:
        matched_actions = eligible_pairs if eligible_pairs is not None else all_scored_pairs.iloc[0:0]
        eligible_pairs = matched_actions
    eligible_pairs = eligible_pairs if eligible_pairs is not None else all_scored_pairs.iloc[0:0]
    matched_actions = matched_actions if matched_actions is not None else all_scored_pairs.iloc[0:0]
    selected = matched_actions
    utilities = pd.to_numeric(selected.get("utility", pd.Series(dtype=float)), errors="coerce")
    ordered = utilities.sort_values(ascending=False).to_numpy()
    patient_k = float(patient_ledger["old_v3_selected"].sum()); n_channels = float(len(patient_ledger))
    selected_scores = pd.to_numeric(patient_ledger["old_v3_score_ez"], errors="coerce")
    selected_mask = patient_ledger["old_v3_selected"].astype(int) == 1
    probabilities = np.clip(selected_scores.to_numpy(dtype=float), 1e-12, 1. - 1e-12)
    entropy = float(np.mean(-(probabilities * np.log(probabilities) + (1. - probabilities) * np.log(1. - probabilities)))) if len(probabilities) else 0.
    row = {"subject_id": str(patient_ledger["subject_id"].iloc[0]), "top1_utility": float(ordered[0]) if len(ordered) else 0., "top2_utility": float(ordered[1]) if len(ordered) > 1 else 0., "top1_top2_gap": float(ordered[0] - ordered[1]) if len(ordered) > 1 else float(ordered[0]) if len(ordered) else 0., "sum_selected_utility": float(utilities.sum()), "mean_selected_utility": float(utilities.mean()) if len(utilities) else 0., "selected_utility_std": float(utilities.std(ddof=0)) if len(utilities) else 0., "mean_selected_p_benefit": float(selected.get("p_benefit", pd.Series(dtype=float)).mean()) if len(selected) else 0., "max_selected_p_harm": float(selected.get("p_harm", pd.Series(dtype=float)).max()) if len(selected) else 0., "mean_selected_pred_delta": float(selected.get("pred_delta", pd.Series(dtype=float)).mean()) if len(selected) else 0., "pred_delta_std": float(selected.get("pred_delta", pd.Series(dtype=float)).std(ddof=0)) if len(selected) else 0., "seed_positive_fraction": float(selected.get("seed_positive_fraction", pd.Series(dtype=float)).mean()) if len(selected) else 0., "seed_consensus": float(selected.get("seed_consensus", pd.Series(dtype=float)).mean()) if len(selected) else 0., "seed_disagreement": 1. - float(selected.get("seed_consensus", pd.Series(dtype=float)).mean()) if len(selected) else 1., "candidate_count": int(len(all_scored_pairs)), "eligible_candidate_count": int(len(eligible_pairs)), "matched_action_count": int(len(matched_actions)), "unique_eject_count": int(matched_actions.get("eject_channel", pd.Series(dtype=str)).nunique()), "unique_add_count": int(matched_actions.get("add_channel", pd.Series(dtype=str)).nunique()), "patient_k": patient_k, "patient_n_channels": n_channels, "patient_k_fraction": patient_k / max(n_channels, 1.), "boundary_gap": float(selected_scores[~selected_mask].max() - selected_scores[selected_mask].min()) if (~selected_mask).any() and selected_mask.any() else 0., "selected_score_entropy": entropy}
    for column in sorted(name for name in all_scored_pairs if name.startswith(("eject_source_", "add_source_"))):
        row[f"candidate_source_count_{column}"] = int(pd.to_numeric(all_scored_pairs[column], errors="coerce").fillna(0).astype(bool).sum())
    if selected_policy is not None:
        policy = selected_policy if isinstance(selected_policy, Mapping) else vars(selected_policy)
        row["selected_policy"] = str(dict(policy))
    return pd.DataFrame([row])


def _select_threshold(frame: pd.DataFrame, *, harm_limit: float) -> float:
    candidates = []
    for threshold in (.30, .40, .50, .60, .70):
        passed = frame["cross_fitted_gate_probability"] >= threshold
        delta = pd.to_numeric(frame.get("net_delta_patient_macro_f1", 0.), errors="coerce").fillna(0.)
        anchor = pd.to_numeric(frame.get("anchor_patient_macro_f1", 0.), errors="coerce").fillna(0.)
        objective = float((anchor + delta.where(passed, 0.)).mean())
        harm = float((passed & delta.lt(0.)).mean())
        actions = float(pd.to_numeric(frame.get("matched_action_count", 0.), errors="coerce").fillna(0.).where(passed, 0.).mean())
        if harm <= harm_limit:
            candidates.append((objective, -harm, -actions, threshold))
    return max(candidates)[3] if candidates else .70


def cross_fit_patient_gate(frame: pd.DataFrame, feature_columns: list[str], *, n_folds: int = 5,
                           seed: int = 0, harm_limit: float = .15) -> pd.DataFrame:
    subjects = np.asarray(sorted(frame["subject_id"].astype(str).unique()))
    if len(subjects) < 2:
        result = frame.copy(); probability = float(frame["net_beneficial"].mean()) if len(frame) else 0.
        result["cross_fitted_gate_probability"] = probability; result["gate_fold"] = 0
        result["gate_fit_subjects"] = [[] for _ in range(len(result))]
    else:
        folds = min(max(2, int(n_folds)), len(subjects))
        result_parts = []
        for fold, (fit_index, holdout_index) in enumerate(KFold(folds, shuffle=True, random_state=seed).split(subjects), start=1):
            fit_subjects, holdout_subjects = set(subjects[fit_index]), set(subjects[holdout_index])
            fit = frame[frame["subject_id"].astype(str).isin(fit_subjects)]
            holdout = frame[frame["subject_id"].astype(str).isin(holdout_subjects)].copy()
            gate = LearnedPatientGate().fit(fit, feature_columns)
            holdout["cross_fitted_gate_probability"] = gate.predict_proba(holdout)
            holdout["gate_fold"] = fold
            holdout["gate_fit_subjects"] = [sorted(gate.fit_subjects) for _ in range(len(holdout))]
            result_parts.append(holdout)
        result = pd.concat(result_parts, ignore_index=True)
    threshold = _select_threshold(result, harm_limit=harm_limit)
    result["cross_fitted_gate_prediction"] = (result["cross_fitted_gate_probability"] >= threshold).astype(int)
    result["selected_gate_threshold"] = threshold
    return result.sort_values("subject_id", kind="mergesort").reset_index(drop=True)


def apply_patient_gate(actions: pd.DataFrame, probabilities: pd.DataFrame, *, threshold: float) -> pd.DataFrame:
    decisions = probabilities[["subject_id", "gate_probability"]].copy()
    decisions["gate_pass"] = decisions["gate_probability"] >= float(threshold)
    output = actions.merge(decisions, on="subject_id", how="left", validate="many_to_one")
    if output["gate_pass"].isna().any():
        raise ValueError("missing patient gate decision for action subjects")
    return output[output["gate_pass"]].copy().reset_index(drop=True)
