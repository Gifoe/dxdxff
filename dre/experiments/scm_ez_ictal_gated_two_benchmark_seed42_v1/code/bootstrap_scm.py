"""Patient-cluster bootstrap for SCM; paired A1 cells are not available."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


REFERENCE = {"auroc": 0.746382, "ap": 0.576743, "macro_f1": 0.620810,
             "mrr": 0.740038, "top1": 0.654771}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--private-cells", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    with args.private_cells.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    clustered = defaultdict(list)
    for row in rows:
        clustered[row["patient_private"]].append(row)
    patients = sorted(clustered)
    if len(patients) != 47 or len(rows) != 1300:
        raise RuntimeError("historical 47-person/65-target cluster scope changed")
    names = ("auroc", "ap", "macro_f1", "mrr", "top1")
    patient_values = {}
    for patient in patients:
        patient_values[patient] = {name: np.asarray([float(row[name]) for row in clustered[patient]], dtype=np.float64)
                                   for name in names}
    rng = np.random.default_rng(42)
    draws = {name: np.empty(10000, dtype=np.float64) for name in names}
    for draw in range(10000):
        selected = rng.integers(0, len(patients), size=len(patients))
        for name in names:
            draws[name][draw] = np.nanmean(np.concatenate(
                [patient_values[patients[index]][name] for index in selected]))
    output = []
    for name in names:
        values = draws[name]
        estimate = float(np.nanmean(np.concatenate([patient_values[p][name] for p in patients])))
        lo, hi = np.quantile(values, (0.025, 0.975))
        reference = REFERENCE[name]
        output.append({"metric": name, "SCM_patient_cluster_estimate": estimate,
                       "SCM_ci_low": float(lo), "SCM_ci_high": float(hi),
                       "historical_A1_public_point": reference,
                       "SCM_minus_A1_point": estimate - reference,
                       "difference_ci_low_using_fixed_A1_point": float(lo - reference),
                       "difference_ci_high_using_fixed_A1_point": float(hi - reference),
                       "draws": 10000, "seed": 42,
                       "comparison_type": "NOT_PAIRED_A1_cells_unavailable"})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output[0])); writer.writeheader(); writer.writerows(output)
    args.audit.write_text(json.dumps({"status": "SCM_CLUSTER_BOOTSTRAP_COMPLETE_A1_PAIRED_NOT_ESTIMABLE",
        "draws": 10000, "seed": 42, "cluster_unit": "patient ID across repeated fold targets",
        "unique_patient_clusters": 47, "SCM_query_cells": 1300,
        "reason_unpaired": "Only compact A1 fold/overall aggregates are retained; matched A1 patient-query cells are absent. A paired CI cannot be reconstructed without fabrication.",
        "A1_values_in_csv_are_fixed_public_points": True, "outer_test_accessed": False}, indent=2) + "\n",
        encoding="utf-8")


if __name__ == "__main__":
    main()
