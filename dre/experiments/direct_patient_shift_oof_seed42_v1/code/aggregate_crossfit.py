"""Verify the complete 5x5 private cross-fit run and emit public aggregate audits."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--crossfit-dir", required=True, type=Path)
    p.add_argument("--protocol-lock", required=True, type=Path)
    p.add_argument("--source-code", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    a = p.parse_args()
    lock_hash = digest(a.protocol_lock)
    folds = []
    heldout_total = 0
    channel_total = 0
    selected_epochs = []
    for fold in range(1, 6):
        path = a.crossfit_dir / f"fold{fold}" / "FOLD_AUDIT.json"
        audit = json.loads(path.read_text(encoding="utf-8"))
        if audit["outer_fold"] != fold or audit["protocol_lock_sha256"] != lock_hash:
            raise RuntimeError(f"Fold {fold} scientific lock mismatch")
        if len(audit["crossfit_cells"]) != 5 or not audit["heldout_from_every_predictor"]:
            raise RuntimeError(f"Fold {fold} missing cells or holdout invariant")
        if sum(cell["heldout_patients"] for cell in audit["crossfit_cells"]) != audit["fit_patients"]:
            raise RuntimeError(f"Fold {fold} patient coverage mismatch")
        for cell in audit["crossfit_cells"]:
            if not cell["model_excluded_heldout_patients"] or cell["protocol_lock_sha256"] != lock_hash:
                raise RuntimeError("Cell leakage or lock mismatch")
            selected_epochs.append(cell["nested_selected_epoch"])
        with np.load(a.crossfit_dir / f"fold{fold}" / "FOLD_OOF_PRIVATE.npz", allow_pickle=False) as archive:
            if len(archive["subject_id"]) != audit["fit_channels"] or len(np.unique(archive["subject_id"])) != audit["fit_patients"]:
                raise RuntimeError("Combined OOF patient/channel coverage mismatch")
        if digest(a.crossfit_dir / f"fold{fold}" / "FOLD_OOF_PRIVATE.npz") != audit["combined_oof_sha256"]:
            raise RuntimeError("Combined OOF hash mismatch")
        heldout_total += audit["fit_patients"]
        channel_total += audit["fit_channels"]
        folds.append({key: value for key, value in audit.items() if key != "crossfit_cells"})
        folds[-1]["crossfit_cells"] = [{key: cell[key] for key in (
            "inner_fold", "nested_train_patients", "nested_validation_patients", "refit_train_patients",
            "heldout_patients", "heldout_channels", "heldout_membership_sha256", "refit_membership_sha256",
            "nested_selected_epoch", "model_excluded_heldout_patients", "oof_sha256")}
            for cell in audit["crossfit_cells"]]
    public = {"status": "PASS", "outer_folds": 5, "crossfit_cells": 25,
              "outer_fit_patient_fold_episodes": heldout_total, "outer_fit_channel_predictions": channel_total,
              "all_meta_predictions_out_of_sample_by_patient": True,
              "nested_selected_epoch_min": min(selected_epochs), "nested_selected_epoch_median": float(np.median(selected_epochs)),
              "nested_selected_epoch_max": max(selected_epochs),
              "protocol_lock_sha256": lock_hash, "crossfit_code_sha256": digest(a.source_code),
              "folds": folds,
              "privacy": "Only counts, hashes and score-distribution aggregates; private membership and channel predictions remain server-side."}
    a.output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("CROSS_FIT_AUDIT.json", "OOF_B0_META_TRAIN_AUDIT.json"):
        (a.output_dir / name).write_text(json.dumps(public, indent=2, sort_keys=True), encoding="utf-8")
    print(f"CROSSFIT_AUDIT_PASS cells=25 patient_fold_episodes={heldout_total} channels={channel_total}", flush=True)


if __name__ == "__main__":
    main()
