from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .action_simulation import simulate_patient_policy


@dataclass(frozen=True)
class SelectedThresholds:
    tau_benefit: float = .70
    tau_harm: float = .20
    tau_utility: float = 0.
    max_swaps: int = 0
    tau_delta: float = 0.
    tau_positive_seed_fraction: float = .67
    objective: float = 0.
    patient_harm_rate: float = 0.
    action_harm_rate: float = 0.
    patient_coverage: float = 0.
    action_coverage: float = 0.

    def to_dict(self) -> dict[str, object]: return asdict(self)


def select_thresholds(inner_oof: pd.DataFrame, *, ledger: pd.DataFrame | None = None, minimum_action_coverage: float = 0.0, harm_limit: float = .15, max_swaps_limit: int = 2, return_grid: bool = False, smoke_mode: bool = False):
    """Select only with inner-OOF patient-level matched action simulations."""
    if inner_oof.empty or ledger is None:
        result = SelectedThresholds()
        return (result, pd.DataFrame()) if return_grid else result
    utility_values = inner_oof.loc[inner_oof["utility"] > 0, "utility"]
    delta_values = inner_oof.loc[inner_oof["pred_delta"] > 0, "pred_delta"]
    utilities = sorted({0.0, *np.quantile(utility_values, [.25, .5, .75]).tolist()}) if len(utility_values) else [0.0]
    deltas = sorted({0.0, *np.quantile(delta_values, [.25, .5, .75]).tolist()}) if len(delta_values) else [0.0]
    rows: list[dict[str, object]] = []
    best: SelectedThresholds | None = None
    benefits = (.5, .7) if smoke_mode else (.5, .6, .7, .8, .9)
    harms = (.2,) if smoke_mode else (.1, .2, .3, .4)
    if smoke_mode:
        utilities, deltas, agreements = [0.0], [0.0], (.67,)
    else:
        agreements = (.67, 1.0)
    for benefit in benefits:
        for harm in harms:
            for utility in utilities:
                for delta in deltas:
                    for agreement in agreements:
                        for swaps in range(0, int(max_swaps_limit) + 1):
                            patient_results = [simulate_patient_policy(patient, inner_oof[inner_oof["subject_id"].astype(str) == str(subject)], tau_benefit=benefit, tau_harm=harm, tau_utility=float(utility), tau_delta=float(delta), tau_positive_seed_fraction=agreement, max_swaps=swaps) for subject, patient in ledger.groupby("subject_id", sort=True)]
                            deltas_by_patient = np.asarray([item.delta_patient_macro_f1 for item in patient_results])
                            counts = np.asarray([item.action_count for item in patient_results])
                            patient_coverage = float(np.mean(counts > 0)) if len(counts) else 0.
                            action_count = int(counts.sum())
                            harmful_actions = sum(int(pd.to_numeric(item.actions.get("harmful_label", pd.Series(dtype=float)), errors="coerce").fillna(0).astype(bool).sum()) for item in patient_results)
                            action_harm_rate = float(harmful_actions / action_count) if action_count else 0.
                            action_coverage = float(action_count / max(len(inner_oof), 1))
                            harm_rate = float(np.mean(deltas_by_patient < 0)) if len(deltas_by_patient) else 0.
                            objective = float(np.mean([item.patient_macro_f1 for item in patient_results])) if patient_results else 0.
                            row = {"tau_benefit": benefit, "tau_harm": harm, "tau_utility": float(utility), "tau_delta": float(delta), "tau_positive_seed_fraction": agreement, "max_swaps": swaps, "patient_macro_f1": objective, "objective": objective, "patient_harm_rate": harm_rate, "action_harm_rate": action_harm_rate, "patient_coverage": patient_coverage, "action_coverage": action_coverage, "mean_swaps": float(counts.mean()) if len(counts) else 0., "n_improved": int((deltas_by_patient > 0).sum()), "n_harmed": int((deltas_by_patient < 0).sum()), "n_unchanged": int((deltas_by_patient == 0).sum())}
                            rows.append(row)
                            candidate = SelectedThresholds(benefit, harm, float(utility), swaps, float(delta), agreement, objective, harm_rate, action_harm_rate, patient_coverage, action_coverage)
                            candidate_key = (candidate.objective, -candidate.patient_harm_rate, -row["mean_swaps"], candidate.tau_benefit)
                            best_key = None if best is None else (best.objective, -best.patient_harm_rate, -next((item["mean_swaps"] for item in rows if item["tau_benefit"] == best.tau_benefit and item["tau_harm"] == best.tau_harm and item["tau_utility"] == best.tau_utility and item["tau_delta"] == best.tau_delta and item["tau_positive_seed_fraction"] == best.tau_positive_seed_fraction and item["max_swaps"] == best.max_swaps), 0.), best.tau_benefit)
                            if patient_coverage >= minimum_action_coverage and harm_rate <= harm_limit and (best is None or candidate_key > best_key): best = candidate
    result = best if best is not None else SelectedThresholds(max_swaps=0)
    grid = pd.DataFrame(rows).sort_values(["objective", "patient_harm_rate"], ascending=[False, True], kind="mergesort") if rows else pd.DataFrame()
    return (result, grid) if return_grid else result
