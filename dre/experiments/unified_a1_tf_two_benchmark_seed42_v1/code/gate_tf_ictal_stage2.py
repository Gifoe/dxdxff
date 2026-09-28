"""Pre-specified leave-one-validation-patient-out I1 Stage-2 AP gate.

This reads validation labels only after all Stage-1 score grids exist. For
each target patient, the gate uses the other 12 patients; it never uses that
target's own labels to decide whether Stage 2 is an eligible candidate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

from train_tf_ictal_stage1 import BASE_LOCK_SHA, AMENDMENT_SHA


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    a = parser.parse_args()
    if sha256(a.lock) != BASE_LOCK_SHA or sha256(a.amendment) != AMENDMENT_SHA:
        raise RuntimeError("Ictal protocol/amendment changed")
    rows = []
    for fold in range(1, 6):
        folder = a.runtime / f"fold_{fold}" / "stage1"
        summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
        if summary["fold"] != fold or not summary["stage1_complete"]:
            raise RuntimeError(f"Incomplete Stage 1 fold {fold}")
        historical = list(csv.DictReader((a.history / "private" / f"fold_{fold}" /
                                          "A1_VLOO_PRIVATE.csv").open(encoding="utf-8")))
        if len(historical) != 13:
            raise RuntimeError("Historical A1 validation membership mismatch")
        names = sorted(row["subject_id"] for row in historical)
        historical_ap = {row["subject_id"]: float(row["patient_ez_auprc"])
                         for row in historical}
        labels = {}
        for row in historical:
            path = a.history / "private" / f"fold_{fold}" / \
                f"epoch_{int(row['selected_epoch']):02d}_representations.pkl"
            with path.open("rb") as handle:
                payload = pickle.load(handle)
            labels[row["subject_id"]] = np.asarray(payload["val"][row["subject_id"]]["y"],
                                                    dtype=np.int8)
        score_files = [folder / f"epoch_{epoch:02d}_SCORES_PRIVATE.pkl"
                       for epoch in range(1, 16)]
        if not all(path.is_file() for path in score_files):
            raise RuntimeError(f"Incomplete Stage 1 score grid fold {fold}")
        scores = []
        for path in score_files:
            with path.open("rb") as handle:
                snap = pickle.load(handle)
            if sorted(snap) != names:
                raise RuntimeError("Validation subject membership changed")
            scores.append(snap)
        for target in names:
            other = [name for name in names if name != target]
            baseline = float(np.mean([historical_ap[name] for name in other]))
            candidates = []
            for epoch, snap in enumerate(scores, 1):
                ap = []
                for name in other:
                    y = labels[name]
                    s = np.asarray(snap[name]["score_ez"], dtype=np.float64)
                    if len(y) != len(s) or not np.isfinite(s).all():
                        raise RuntimeError("Stage 1 score/label alignment mismatch")
                    ap.append(float(average_precision_score(y, s)))
                candidates.append((float(np.mean(ap)), epoch))
            best_ap, best_epoch = max(candidates, key=lambda item: (item[0], -item[1]))
            rows.append({"fold": fold, "target_private": target,
                         "other_12_original_a1_ap": baseline,
                         "other_12_stage1_best_ap": best_ap,
                         "other_12_stage1_best_epoch": best_epoch,
                         "stage2_eligible": best_ap >= baseline - 0.01,
                         "target_labels_used_for_gate": False})
        print(f"I1 Stage2 gate fold={fold} eligible={sum(r['stage2_eligible'] for r in rows if r['fold']==fold)}/13", flush=True)
    if len(rows) != 65:
        raise RuntimeError("Expected exact 65 historical validation targets")
    a.runtime.joinpath("private").mkdir(parents=True, exist_ok=True)
    private = a.runtime / "private" / "ICTAL_STAGE2_GATE_PRIVATE.csv"
    with private.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    public = {"protocol_sha256": BASE_LOCK_SHA, "amendment_sha256": AMENDMENT_SHA,
              "stage1_score_files": 75, "target_cells": 65,
              "eligible_cells": sum(r["stage2_eligible"] for r in rows),
              "eligible_folds": [fold for fold in range(1, 6)
                                 if any(r["stage2_eligible"] for r in rows if r["fold"] == fold)],
              "per_fold_eligible": {str(fold): sum(r["stage2_eligible"] for r in rows
                                              if r["fold"] == fold) for fold in range(1, 6)},
              "private_gate_sha256": sha256(private),
              "target_labels_used_for_own_gate": False,
              "historical_loader_materialized_validation_labels": True}
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / "ICTAL_STAGE2_GATE_AUDIT.json").write_text(
        json.dumps(public, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(public, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
