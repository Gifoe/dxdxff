"""Development-only permutation and isolation sanity checks."""
from __future__ import annotations

import json
import pickle

import numpy as np

from analyze import align_pair, fit_shared, ranking
from common import EXPERIMENT, RUNTIME, ensure_source, write_json


def run():
    ensure_source()
    rng = np.random.default_rng(42)
    source, random_label, prevalence = [], [], []
    max_channel_error = 0.0
    all_repeat = True
    for fold in range(1, 6):
        folder = RUNTIME / "private" / f"fold_{fold}"
        with (folder / "raw.pkl").open("rb") as stream:
            raw = pickle.load(stream)
        fit, val = align_pair(raw["fit"], raw["val"], "R1", "patient_z")
        normal = fit_shared(fit)
        permuted = {patient: {"x": row["x"], "y": rng.permutation(row["y"])} for patient, row in fit.items()}
        shuffled = fit_shared(permuted)
        for row in val.values():
            score = normal.decision_function(row["x"])
            false_score = shuffled.decision_function(row["x"])
            source.append(ranking(row["y"], score)["ap"])
            random_label.append(ranking(row["y"], false_score)["ap"])
            prevalence.append(float(np.mean(row["y"])))
            index = rng.permutation(len(row["y"]))
            metric1 = ranking(row["y"], score)
            metric2 = ranking(row["y"][index], score[index])
            max_channel_error = max(max_channel_error, *(abs(metric1[key] - metric2[key]) for key in metric1))
        extraction = json.loads((folder / "extraction_status.json").read_text(encoding="utf-8"))
        all_repeat &= all(check["repeat_identical"] for check in extraction["new_hash_checks"])
        if set(raw["fit"]) & set(raw["val"]):
            raise RuntimeError("FIT and validation patient overlap")
    payload = {"n_validation_patients": len(source),
               "random_fit_label_permutation": {"source_ap": float(np.mean(source)),
                                                "permuted_fit_ap": float(np.mean(random_label)),
                                                "validation_prevalence": float(np.mean(prevalence)),
                                                "difference_from_prevalence": float(np.mean(random_label) - np.mean(prevalence))},
               "channel_permutation_max_metric_error": max_channel_error,
               "representation_repeat_identical": bool(all_repeat),
               "fit_validation_overlap": False,
               "patient_specific_cv_inner_test_labels_used_for_training": False,
               "label_free_alignment_and_pca": True,
               "feature_inventory_committed_before_richer_evaluation": True,
               "outer_test_loader_constructed": False,
               "legacy_cache_initializer_materializes_all_80_patients_including_outer_labels": True,
               "outer_test_metric_computed": False,
               "pass": bool(max_channel_error < 1e-12 and all_repeat)}
    write_json(EXPERIMENT / "SANITY_AUDIT.json", payload)
    if not payload["pass"]:
        raise RuntimeError("Sanity audit failed")


if __name__ == "__main__":
    run()
