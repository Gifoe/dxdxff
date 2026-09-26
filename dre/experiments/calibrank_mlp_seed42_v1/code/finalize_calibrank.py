"""Engineering-only finalizer for completed, protocol-locked CalibRank cells.

The training runner reached all 20 COMPLETE markers but its final positive-fold
count used pandas.query with a missing comprehension-scope variable. This script
does not train, select, evaluate a checkpoint anew, or alter any scientific rule.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_calibrank import VARIANTS, bootstrap, summarize


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--diagnostics", required=True, type=Path)
    args = parser.parse_args()
    lock = json.loads((args.train / "PROTOCOL_LOCK.json").read_text(encoding="utf-8"))
    if lock["variants"] != list(VARIANTS) or lock["seed"] != 42 or lock["folds"] != [1, 2, 3, 4, 5]:
        raise RuntimeError("Frozen protocol lock is not the expected matrix")
    fold_rows, development_rows, patients = [], [], []
    for variant in VARIANTS:
        for fold in range(1, 6):
            cell = args.train / variant / f"fold{fold}"
            completed = json.loads((cell / "COMPLETE.json").read_text(encoding="utf-8"))
            test = pd.read_csv(cell / "TEST_PATIENT_PRIVATE.csv")
            recorded = completed["fold_result"]
            if recorded["variant"] != variant or recorded["fold"] != fold or len(test) != recorded["test_patients"]:
                raise RuntimeError(f"Cell identity/count mismatch {variant}/{fold}")
            measured = summarize(test)
            for metric, actual in measured.items():
                if not np.isclose(actual, recorded[metric], rtol=0, atol=1e-10, equal_nan=True):
                    raise RuntimeError(f"Cell metric mismatch {variant}/{fold}/{metric}")
            fold_rows.append(recorded)
            development_rows.append(completed["development_result"])
            patients.append(test)
    fold_table = pd.DataFrame(fold_rows).sort_values(["variant", "fold"])
    development_table = pd.DataFrame(development_rows).sort_values(["variant", "fold"])
    all_patients = pd.concat(patients, ignore_index=True)
    identities = None
    for variant in VARIANTS:
        part = all_patients[all_patients.variant == variant]
        current = set(part.subject_id)
        if len(part) != 80 or len(current) != 80 or (identities is not None and current != identities):
            raise RuntimeError(f"Incomplete or unmatched 80-patient cohort for {variant}")
        identities = current
    original = json.loads(args.diagnostics.read_text(encoding="utf-8"))["b0"]
    reference = all_patients[all_patients.variant == "B0"]
    for metric in ("macro_f1", "ez_f1", "balanced_accuracy", "ez_auprc", "ez_auroc"):
        if not np.isclose(float(reference[metric].mean()), float(original[metric]), rtol=0, atol=1e-7):
            raise RuntimeError(f"B0 aggregate differs from original diagnostic {metric}")
    summary_rows, paired_rows = [], []
    indexed = fold_table.set_index(["variant", "fold"])
    for variant in VARIANTS:
        part = all_patients[all_patients.variant == variant]
        positive = 0 if variant == "B0" else sum(
            bool(indexed.loc[(variant, fold), "macro_f1"] > indexed.loc[("B0", fold), "macro_f1"])
            for fold in range(1, 6))
        summary_rows.append({"variant": variant, "seed": 42, "patients": 80,
                             "positive_folds_vs_b0": positive, **summarize(part)})
        if variant != "B0":
            for metric, result in bootstrap(part, reference).items():
                paired_rows.append({"variant": variant, "metric": metric, "seed": 42, "patients": 80, **result})
    development_table.to_csv(args.train / "DEVELOPMENT_RESULTS.csv", index=False)
    fold_table.to_csv(args.train / "OUTER_FOLD_RESULTS.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(args.train / "OUTER_SUMMARY.csv", index=False)
    pd.DataFrame(paired_rows).to_csv(args.train / "PAIRED_BOOTSTRAP.csv", index=False)
    (args.train / "FINALIZATION_STATUS.json").write_text(json.dumps({
        "status": "PASS", "cells": 20, "variants": list(VARIANTS), "patients_per_variant": 80,
        "engineering_repair": "Post-training pandas.query scope error in positive-fold count; no model work rerun.",
        "training_lock_preserved": True}, indent=2), encoding="utf-8")
    print("FINALIZATION_PASS " + pd.DataFrame(summary_rows).to_json(orient="records"), flush=True)


if __name__ == "__main__":
    main()
