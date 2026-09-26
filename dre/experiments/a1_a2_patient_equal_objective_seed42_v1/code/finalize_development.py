"""Aggregate 5x3 completed private development summaries and apply locked gates."""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path

import numpy as np


EXPERIMENT = Path(__file__).resolve().parents[1]
RUNTIME = Path(os.environ.get("A1_A2_RUNTIME", ""))
DEVELOPMENT = EXPERIMENT / "development"
VARIANTS = ("A0", "A1", "A2")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"No rows for {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def avg(rows: list[dict], metric: str) -> float:
    return float(np.mean([float(row[metric]) for row in rows]))


def load() -> tuple[dict, list[dict], list[dict], list[dict], list[dict]]:
    if not os.environ.get("A1_A2_RUNTIME") or not RUNTIME.is_absolute():
        raise RuntimeError("Set absolute A1_A2_RUNTIME")
    summaries = {}
    vloo, fullval, loss_diag, epoch_train = [], [], [], []
    private = RUNTIME / "development_private"
    for fold in range(1, 6):
        for variant in VARIANTS:
            directory = RUNTIME / variant / f"fold_{fold}"
            record = json.loads((directory / "development_summary.json").read_text(encoding="utf-8"))
            if not record["all_30_epochs_completed"]:
                raise RuntimeError("A variant did not train for all 30 epochs")
            summaries[(variant, fold)] = record
            vloo.append(record["vloo"])
            fullval.append(record["fullval"])
            loss_diag.append(record["loss_diagnostics"])
            for epoch in range(1, 31):
                if not (directory / f"epoch_{epoch:02d}.pt").exists():
                    raise RuntimeError("Per-epoch checkpoint missing")
                epoch_train.append(json.loads((directory / f"epoch_{epoch:02d}_train_metrics.json").read_text(encoding="utf-8")))
    if len({record["parameter_count"] for record in summaries.values()}) != 1:
        raise RuntimeError("A0/A1/A2 parameter counts differ")
    for variant in VARIANTS:
        rows = [row for row in vloo if row["variant"] == variant]
        write_csv(DEVELOPMENT / f"{variant}_VLOO_BY_FOLD.csv", rows)
        patient_rows = []
        for fold in range(1, 6):
            with (private / f"{variant}_fold_{fold}_VLOO_PATIENT_PRIVATE.csv").open(newline="", encoding="utf-8") as stream:
                patient_rows.extend(csv.DictReader(stream))
        if len(patient_rows) != 65:
            raise RuntimeError("Expected 65 VLOO patient-fold rows per variant")
        write_csv(private / f"{variant}_VLOO_PATIENT_PRIVATE.csv", patient_rows)
    write_csv(DEVELOPMENT / "FULLVAL_SELECTION.csv", fullval)
    write_csv(DEVELOPMENT / "LOSS_DIAGNOSTICS.csv", loss_diag)
    write_csv(DEVELOPMENT / "TRAINING_LOSS_BY_EPOCH.csv", epoch_train)
    return summaries, vloo, fullval, loss_diag, epoch_train


def decide(summaries: dict, vloo: list[dict], fullval: list[dict], epoch_train: list[dict]) -> tuple[dict, dict, list[dict]]:
    by = {variant: [summaries[(variant, fold)]["vloo"] for fold in range(1, 6)] for variant in VARIANTS}
    apparent = {variant: [summaries[(variant, fold)]["fullval"] for fold in range(1, 6)] for variant in VARIANTS}
    comparison = []
    for fold in range(1, 6):
        a0, a1, a2 = (summaries[(variant, fold)]["vloo"] for variant in VARIANTS)
        comparison.append({
            "fold": fold,
            "A0_vloo_macro_f1": a0["patient_macro_f1"],
            "A1_vloo_macro_f1": a1["patient_macro_f1"],
            "A2_vloo_macro_f1": a2["patient_macro_f1"],
            "delta_A1_minus_A0": a1["patient_macro_f1"] - a0["patient_macro_f1"],
            "delta_A2_minus_A1": a2["patient_macro_f1"] - a1["patient_macro_f1"],
            "delta_A2_minus_A0": a2["patient_macro_f1"] - a0["patient_macro_f1"],
            "A0_vloo_ez_f1": a0["patient_ez_f1"],
            "A1_vloo_ez_f1": a1["patient_ez_f1"],
            "A2_vloo_ez_f1": a2["patient_ez_f1"],
            "A0_vloo_ez_auprc": a0["patient_ez_auprc"],
            "A1_vloo_ez_auprc": a1["patient_ez_auprc"],
            "A2_vloo_ez_auprc": a2["patient_ez_auprc"],
        })
    write_csv(DEVELOPMENT / "OBJECTIVE_COMPARISON.csv", comparison)
    a2_delta = float(np.mean([row["delta_A2_minus_A1"] for row in comparison]))
    a2_positive = sum(row["delta_A2_minus_A1"] > 0 for row in comparison)
    a2_ez_nondecreasing = avg(by["A2"], "patient_ez_f1") >= avg(by["A1"], "patient_ez_f1") - 1e-12
    a2_checks = {"delta_macro_f1_ge_0_005": a2_delta >= 0.005 - 1e-12,
                 "positive_folds_ge_3": a2_positive >= 3,
                 "mean_ez_f1_nondecreasing": a2_ez_nondecreasing}
    candidate = "A2" if all(a2_checks.values()) else "A1"
    candidate_payload = {"candidate": candidate, "A1_is_primary": True,
                         "A2_minus_A1_mean_macro_f1": a2_delta, "A2_positive_folds": a2_positive,
                         "A2_checks": a2_checks}
    write_json(DEVELOPMENT / "CANDIDATE_SELECTION.json", candidate_payload)
    delta = avg(by[candidate], "patient_macro_f1") - avg(by["A0"], "patient_macro_f1")
    positive = sum(summaries[(candidate, fold)]["vloo"]["patient_macro_f1"] >
                   summaries[("A0", fold)]["vloo"]["patient_macro_f1"] for fold in range(1, 6))
    ez_delta = avg(by[candidate], "patient_ez_f1") - avg(by["A0"], "patient_ez_f1")
    auprc_delta = avg(by[candidate], "patient_ez_auprc") - avg(by["A0"], "patient_ez_auprc")
    apparent_mean = avg(apparent[candidate], "apparent_patient_macro_f1")
    apparent_worst = min(float(row["apparent_patient_macro_f1"]) for row in apparent[candidate])
    finite = all(math.isfinite(float(value)) for row in vloo for value in row.values()
                 if isinstance(value, (int, float))) and all(math.isfinite(float(value)) for row in epoch_train
                                                        for value in row.values() if isinstance(value, (int, float)))
    unit = json.loads((EXPERIMENT / "A1_LOSS_UNIT_TEST.json").read_text(encoding="utf-8"))
    support = json.loads((EXPERIMENT / "CLASS_SUPPORT_AUDIT.json").read_text(encoding="utf-8"))
    no_pathology = finite and unit.get("pass") is True and support.get("all_active_fit_validation_patients_have_both_classes") is True
    checks = {
        "mean_vloo_macro_f1_gain_ge_0_010": delta >= 0.010 - 1e-12,
        "positive_folds_ge_4": positive >= 4,
        "mean_ez_f1_nondecreasing": ez_delta >= -1e-12,
        "ez_auprc_delta_ge_minus_0_005": auprc_delta >= -0.005 - 1e-12,
        "apparent_fullval_mean_macro_f1_ge_0_665": apparent_mean >= 0.665 - 1e-12,
        "apparent_fullval_worst_fold_macro_f1_ge_0_620": apparent_worst >= 0.620 - 1e-12,
        "no_pathology": no_pathology,
    }
    gate = {"pass": all(checks.values()), "candidate": candidate, "checks": checks,
            "A0_vloo_macro_f1": avg(by["A0"], "patient_macro_f1"),
            "candidate_vloo_macro_f1": avg(by[candidate], "patient_macro_f1"),
            "candidate_minus_A0_vloo_macro_f1": delta, "positive_folds": positive,
            "candidate_minus_A0_ez_f1": ez_delta, "candidate_minus_A0_ez_auprc": auprc_delta,
            "candidate_apparent_fullval_macro_f1_mean": apparent_mean,
            "candidate_apparent_fullval_macro_f1_worst_fold": apparent_worst,
            "outer_test_evaluated": False,
            "terminal": ("A1_SELECTED_FOR_OUTER" if candidate == "A1" else "A2_SELECTED_FOR_OUTER")
                        if all(checks.values()) else "A1_A2_DEVELOPMENT_GATE_FAILED"}
    write_json(DEVELOPMENT / "TEST_READINESS_GATE.json", gate)
    return candidate_payload, gate, comparison


def write_report(summaries: dict, candidate: dict, gate: dict, comparison: list[dict]) -> None:
    by = {variant: [summaries[(variant, fold)]["vloo"] for fold in range(1, 6)] for variant in VARIANTS}
    lines = ["# A1/A2 objective-only seed-42 development", "",
             "Exploratory fixed-80 development. No current outer-test result was read or evaluated.",
             "A0 was retrained for 30 epochs under this VLOO selection protocol; historical R0 validation/test numbers are not the matched control.", "",
             "| Variant | VLOO Macro-F1 | EZ-F1 | NEZ-F1 | BA | EZ-AUPRC | EZ-AUROC | EZ-MRR | Top-1 EZ | Pred. EZ fraction | Positive folds vs A0 |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for variant in VARIANTS:
        positive = "n/a" if variant == "A0" else str(sum(
            summaries[(variant, fold)]["vloo"]["patient_macro_f1"] >
            summaries[("A0", fold)]["vloo"]["patient_macro_f1"] for fold in range(1, 6))) + "/5"
        lines.append(f"| {variant} | {avg(by[variant], 'patient_macro_f1'):.6f} | "
                     f"{avg(by[variant], 'patient_ez_f1'):.6f} | {avg(by[variant], 'patient_nez_f1'):.6f} | "
                     f"{avg(by[variant], 'patient_balanced_accuracy'):.6f} | "
                     f"{avg(by[variant], 'patient_ez_auprc'):.6f} | {avg(by[variant], 'patient_ez_auroc'):.6f} | "
                     f"{avg(by[variant], 'patient_ez_mrr'):.6f} | {avg(by[variant], 'top1_is_ez'):.6f} | "
                     f"{avg(by[variant], 'predicted_ez_fraction'):.6f} | {positive} |")
    lines.extend(["", "| Fold | A0 Macro-F1 | A1 Macro-F1 | A2 Macro-F1 | A1−A0 | A2−A1 |",
                  "| --- | ---: | ---: | ---: | ---: | ---: |"])
    for row in comparison:
        lines.append(f"| {row['fold']} | {row['A0_vloo_macro_f1']:.6f} | {row['A1_vloo_macro_f1']:.6f} | "
                     f"{row['A2_vloo_macro_f1']:.6f} | {row['delta_A1_minus_A0']:+.6f} | "
                     f"{row['delta_A2_minus_A1']:+.6f} |")
    lines.extend(["", f"A1−A0 mean VLOO Macro-F1: {np.mean([row['delta_A1_minus_A0'] for row in comparison]):+.6f} "
                  f"({sum(row['delta_A1_minus_A0'] > 0 for row in comparison)}/5 positive folds).",
                  f"A1−A0 EZ-F1: {avg(by['A1'], 'patient_ez_f1') - avg(by['A0'], 'patient_ez_f1'):+.6f}; "
                  f"EZ-AUPRC: {avg(by['A1'], 'patient_ez_auprc') - avg(by['A0'], 'patient_ez_auprc'):+.6f}.",
                  f"A2−A1 mean VLOO Macro-F1: {candidate['A2_minus_A1_mean_macro_f1']:+.6f}; "
                  f"positive in {candidate['A2_positive_folds']}/5 folds.",
                  f"A2 replacement minimum +0.005 met: {candidate['A2_checks']['delta_macro_f1_ge_0_005']}.",
                  f"Locked candidate: {candidate['candidate']}.",
                  f"Candidate−A0 mean VLOO Macro-F1: {gate['candidate_minus_A0_vloo_macro_f1']:+.6f}; "
                  f"positive in {gate['positive_folds']}/5 folds.",
                  f"Candidate APPARENT_FULLVAL mean/worst-fold Macro-F1: "
                  f"{gate['candidate_apparent_fullval_macro_f1_mean']:.6f}/"
                  f"{gate['candidate_apparent_fullval_macro_f1_worst_fold']:.6f}.",
                  "", "Locked readiness checks:"])
    lines.extend(f"- {key}: {'PASS' if passed else 'FAIL'}" for key, passed in gate["checks"].items())
    lines.extend(["", f"**Terminal: `{gate['terminal']}`.**", ""])
    if not gate["pass"]:
        lines.extend(["Stop at development. No seed52/62 or current outer test is authorized. The >0.650 test target is not evaluated in this experiment.", ""])
    else:
        lines.extend(["Seed52/62 sensitivity checks are required before any outer test; no outer result exists yet.", ""])
    lines.extend(["Per-fold selected epochs, VLOO threshold distributions, full-validation epoch/threshold choices, "
                  "and loss diagnostics are in the aggregate CSV files under `development/`.",
                  "Patient identifiers, channel records, checkpoints and private cache paths are not published."])
    (EXPERIMENT / "FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    summaries, vloo, fullval, diagnostics, epoch_train = load()
    candidate, gate, comparison = decide(summaries, vloo, fullval, epoch_train)
    write_report(summaries, candidate, gate, comparison)
    print(json.dumps({"terminal": gate["terminal"], "candidate": gate["candidate"],
                      "development_gate_pass": gate["pass"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
