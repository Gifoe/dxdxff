"""One-time R0-only outer evaluation under the explicit post-gate amendment.

This does not run R1 or convert the failed R1 validation gate into a pass.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from run_matched import EXPERIMENT, RUNTIME, SOURCE_ROOT, assert_sources, install_interleaved_hlv_view, sha256
from evaluate_outer import csv_write, evaluate_variant


AMENDMENT_SHA256 = "5b45a59c77a83d3416693588fc39cab0dac7e55e2fe39c02772b105c5e2bbad4"
METRICS = (
    "patient_macro_f1",
    "patient_macro_ez_f1",
    "patient_macro_balanced_accuracy",
    "patient_macro_auprc_ez",
)
PUBLIC = EXPERIMENT / "outer"
PRIVATE_MARKER = RUNTIME / "R0" / "R0_OUTER_STARTED.json"


def preflight() -> dict:
    amendment_path = EXPERIMENT / "R0_ONLY_OUTER_PROTOCOL_AMENDMENT.json"
    if sha256(amendment_path) != AMENDMENT_SHA256:
        raise RuntimeError("R0-only amendment hash changed")
    amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
    if sha256(EXPERIMENT / "PROTOCOL_LOCK.json") != amendment["original_protocol_lock_sha256"]:
        raise RuntimeError("Original protocol lock changed")
    gate = json.loads((EXPERIMENT / "validation" / "VALIDATION_GATE.json").read_text(encoding="utf-8"))
    if gate.get("pass") is not False or gate.get("outer_test_evaluated") is not False:
        raise RuntimeError("Unexpected original validation-gate state")
    if amendment["scope"]["variant"] != "R0" or amendment["scope"]["R1_outer_evaluation_authorized"]:
        raise RuntimeError("Amendment scope changed")
    if [entry["fold"] for entry in amendment["frozen_r0_folds"]] != [1, 2, 3, 4, 5]:
        raise RuntimeError("Frozen R0 folds changed")
    if PRIVATE_MARKER.exists() or (PUBLIC / "R0_OUTER_RESULTS.csv").exists():
        raise RuntimeError("R0 outer evaluation already started; refusing automatic rerun")
    assert_sources()
    with (EXPERIMENT / "validation" / "R0_VALIDATION_RESULTS.csv").open(newline="", encoding="utf-8") as stream:
        validation = {int(row["fold"]): row for row in csv.DictReader(stream)}
    if set(validation) != {1, 2, 3, 4, 5}:
        raise RuntimeError("R0 validation rows incomplete")
    for entry in amendment["frozen_r0_folds"]:
        fold = entry["fold"]
        checkpoint_path = RUNTIME / "R0" / f"fold_{fold}" / "selected.pt"
        if sha256(checkpoint_path) != entry["checkpoint_sha256"]:
            raise RuntimeError(f"R0 fold {fold} checkpoint changed")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint["variant"] != "R0" or int(checkpoint["fold"]) != fold:
            raise RuntimeError(f"R0 fold {fold} checkpoint identity changed")
        if int(checkpoint["selected_epoch"]) != entry["selected_epoch"]:
            raise RuntimeError(f"R0 fold {fold} selected epoch changed")
        if not math.isclose(float(checkpoint["threshold"]), entry["threshold"], abs_tol=1e-12):
            raise RuntimeError(f"R0 fold {fold} threshold changed")
        if int(validation[fold]["selected_epoch"]) != entry["selected_epoch"] or not math.isclose(
            float(validation[fold]["validation_selected_threshold"]), entry["threshold"], abs_tol=1e-12
        ):
            raise RuntimeError(f"R0 fold {fold} validation record disagrees with amendment")
    return amendment


def summarize(fold_rows: list[dict], patients: list[dict], amendment: dict) -> None:
    if len(fold_rows) != 5 or {int(row["fold"]) for row in fold_rows} != {1, 2, 3, 4, 5}:
        raise RuntimeError("Expected exactly five distinct outer folds")
    if len(patients) != 80 or len({row["subject_id"] for row in patients}) != 80:
        raise RuntimeError("Expected 80 distinct outer patients")
    counts = {fold: sum(int(patient["fold"]) == fold for patient in patients) for fold in range(1, 6)}
    if any(count == 0 for count in counts.values()) or sum(counts.values()) != 80:
        raise RuntimeError("Outer fold coverage is incomplete")
    if any(int(row["n_patients"]) != counts[int(row["fold"])] for row in fold_rows):
        raise RuntimeError("Fold aggregate patient counts disagree with private patient metrics")
    rng = np.random.default_rng(42042)
    draws = rng.integers(0, 80, size=(2000, 80))
    summaries = []
    for metric in METRICS:
        values = np.asarray([row[metric] for row in patients], dtype=float)
        if not np.isfinite(values).all():
            raise RuntimeError(f"Non-finite outer metric: {metric}")
        replicates = values[draws].mean(axis=1)
        summaries.append({
            "variant": "R0", "metric": metric, "n_patients": 80,
            "mean": float(values.mean()),
            "descriptive_ci95_low": float(np.quantile(replicates, 0.025)),
            "descriptive_ci95_high": float(np.quantile(replicates, 0.975)),
            "bootstrap_replicates": 2000,
        })
    PUBLIC.mkdir(parents=True, exist_ok=True)
    csv_write(PUBLIC / "R0_ONLY_SUMMARY.csv", summaries)
    payload = {
        "status": "R0_OUTER_EVALUATED_ONCE",
        "original_R1_validation_gate_pass": False,
        "R1_outer_evaluated": False,
        "historical_outer_already_viewed": True,
        "interpretation": "exploratory post-hoc R0-only test; not a fresh sealed confirmation",
        "amendment_sha256": AMENDMENT_SHA256,
        "n_folds": 5,
        "n_unique_patients": 80,
        "summary": summaries,
    }
    (PUBLIC / "R0_ONLY_SUMMARY.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    result = {row["metric"]: row for row in summaries}
    report = [
        "# R0-only outer test (post-gate amendment)", "",
        "The five validation-selected R0 checkpoints and thresholds were hash-frozen before this one-time test.",
        "The original R1 validation gate failed and remains failed. R1 outer test was not run.",
        "This is exploratory: prior outer outcomes on the same cohort were already viewed, and the R0-only test was requested after the gate failure.",
        "No test outcome was used for training, selection, or tuning.", "",
        "| Metric | 80-patient mean | Descriptive patient-bootstrap 95% CI |",
        "| --- | ---: | ---: |",
    ]
    for metric in METRICS:
        row = result[metric]
        report.append(f"| {metric} | {row['mean']:.4f} | [{row['descriptive_ci95_low']:.4f}, {row['descriptive_ci95_high']:.4f}] |")
    report.extend(["", "Five-fold aggregate results: `R0_OUTER_RESULTS.csv`.",
                   f"Amendment SHA-256: `{AMENDMENT_SHA256}`.", ""])
    (PUBLIC / "R0_ONLY_REPORT.md").write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    amendment = preflight()
    PRIVATE_MARKER.parent.mkdir(parents=True, exist_ok=True)
    marker = {"status": "R0_OUTER_STARTED", "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "amendment_sha256": AMENDMENT_SHA256}
    with os.fdopen(os.open(PRIVATE_MARKER, os.O_CREAT | os.O_EXCL | os.O_WRONLY), "w", encoding="utf-8") as stream:
        json.dump(marker, stream)
    sys.path.insert(0, str(SOURCE_ROOT))
    install_interleaved_hlv_view()
    fold_rows, patients = evaluate_variant("R0")
    summarize(fold_rows, patients, amendment)
    marker["status"] = "R0_OUTER_COMPLETED"
    marker["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    PRIVATE_MARKER.write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": marker["status"], "n_patients": len(patients)}, indent=2), flush=True)


def summarize_existing() -> None:
    """Recover summary from the completed one-shot evaluation; never re-enter test inference."""
    amendment_path = EXPERIMENT / "R0_ONLY_OUTER_PROTOCOL_AMENDMENT.json"
    if sha256(amendment_path) != AMENDMENT_SHA256:
        raise RuntimeError("R0-only amendment hash changed")
    amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
    if sha256(EXPERIMENT / "PROTOCOL_LOCK.json") != amendment["original_protocol_lock_sha256"]:
        raise RuntimeError("Original protocol lock changed")
    marker = json.loads(PRIVATE_MARKER.read_text(encoding="utf-8"))
    if marker.get("status") != "R0_OUTER_STARTED" or marker.get("amendment_sha256") != AMENDMENT_SHA256:
        raise RuntimeError("Expected a single started R0-only test")
    with (PUBLIC / "R0_OUTER_RESULTS.csv").open(newline="", encoding="utf-8") as stream:
        fold_rows = list(csv.DictReader(stream))
    patients = json.loads((RUNTIME / "R0" / "outer_patient_metrics.json").read_text(encoding="utf-8"))
    summarize(fold_rows, patients, amendment)
    marker["status"] = "R0_OUTER_COMPLETED"
    marker["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    marker["recovered_summary_without_repeat_inference"] = True
    PRIVATE_MARKER.write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": marker["status"], "n_patients": len(patients),
                      "repeat_outer_inference": False}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--summarize-existing", action="store_true")
    args = parser.parse_args()
    if args.summarize_existing:
        summarize_existing()
    else:
        main()
