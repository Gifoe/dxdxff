"""Synthetic consistency test for score grouping and patient-cluster bootstrap."""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from evaluate_official_test import bootstrap, patient_metrics


def main():
    frame = pd.DataFrame({
        "patient": ["a", "a", "a", "b", "b", "b", "c", "c", "c"],
        "dataset": ["one"] * 9,
        "soz": [1, 0, 0, 0, 1, 0, 0, 1, 0],
        "score_ez": [0.8, 0.8, 0.1, 0.9, 0.7, 0.3, 0.9, 0.7, 0.2],
    })
    patient = pd.DataFrame([patient_metrics(g) for _, g in frame.groupby("patient", sort=True)])
    ci = bootstrap(frame, patient, draws=1)
    rng = np.random.default_rng(42)
    selected = rng.integers(0, len(patient), size=len(patient))
    replicated = pd.concat([frame.loc[frame["patient"] == patient.iloc[i]["patient"]]
                            for i in selected], ignore_index=True)
    ap = average_precision_score(replicated["soz"], replicated["score_ez"])
    auc = roc_auc_score(replicated["soz"], replicated["score_ez"])
    actual = ci.set_index("metric")
    assert np.isclose(actual.loc["pooled_ez_ap", "ci_lower"], ap, atol=1e-12)
    assert np.isclose(actual.loc["pooled_auroc", "ci_lower"], auc, atol=1e-12)
    print({"pass": True, "tied_score_ap": ap, "tied_score_auc": auc})


if __name__ == "__main__":
    main()
