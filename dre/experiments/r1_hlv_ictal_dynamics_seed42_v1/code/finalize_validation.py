"""Aggregate private fold summaries and apply the immutable R1 outer gate."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import mean


ROOT = Path(r"D:\nips-temp\r1_hlv_ictal_dynamics_seed42_v1")
PUBLIC = Path(__file__).resolve().parents[1] / "validation"
FIELDS = (
    "patient_macro_f1", "patient_macro_ez_f1", "patient_macro_nez_f1",
    "patient_macro_balanced_accuracy", "patient_macro_auprc_ez", "patient_macro_auroc_ez",
    "patient_macro_ez_mrr", "top1_is_ez_rate", "predicted_ez_fraction",
)
DIAGNOSTICS = (
    "gate_mean", "gate_median", "gate_q10", "gate_q90",
    "mean_abs_gated_residual", "gated_residual_to_base_norm_ratio",
)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"No rows for {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_rows(variant: str) -> list[dict]:
    rows = []
    for fold in range(1, 6):
        path = ROOT / variant / f"fold_{fold}" / "validation_summary.json"
        if not path.is_file():
            raise RuntimeError(f"Incomplete validation: {path}")
        row = json.loads(path.read_text(encoding="utf-8"))
        if row["fold"] != fold or row["variant"] != variant:
            raise RuntimeError(f"Incorrect fold/variant metadata: {path}")
        rows.append(row)
    return rows


def main() -> None:
    r0, r1 = load_rows("R0"), load_rows("R1")
    write_csv(PUBLIC / "R0_VALIDATION_RESULTS.csv", r0)
    write_csv(PUBLIC / "R1_VALIDATION_RESULTS.csv", r1)
    comparisons = []
    gates = []
    for a, b in zip(r0, r1, strict=True):
        if a["fold"] != b["fold"] or a["n_patients"] != b["n_patients"]:
            raise RuntimeError("R0/R1 validation membership differs")
        row = {"fold": a["fold"], "n_patients": a["n_patients"]}
        for key in FIELDS:
            row[f"R0_{key}"] = a[key]
            row[f"R1_{key}"] = b[key]
            row[f"delta_{key}"] = float(b[key]) - float(a[key])
        comparisons.append(row)
        gates.append({"fold": b["fold"], **{key: b[key] for key in DIAGNOSTICS}})
    write_csv(PUBLIC / "VALIDATION_COMPARISON.csv", comparisons)
    write_csv(PUBLIC / "GATE_DIAGNOSTICS.csv", gates)
    aggregate = {
        "R0_mean_macro_f1": mean(row["patient_macro_f1"] for row in r0),
        "R1_mean_macro_f1": mean(row["patient_macro_f1"] for row in r1),
        "delta_macro_f1": mean(row["delta_patient_macro_f1"] for row in comparisons),
        "positive_folds": sum(row["delta_patient_macro_f1"] > 0 for row in comparisons),
        "delta_ez_f1": mean(row["delta_patient_macro_ez_f1"] for row in comparisons),
        "delta_ez_auprc": mean(row["delta_patient_macro_auprc_ez"] for row in comparisons),
        "delta_ez_mrr": mean(row["delta_patient_macro_ez_mrr"] for row in comparisons),
        "mean_gate": mean(row["gate_mean"] for row in gates),
        "mean_residual_to_base_norm_ratio": mean(row["gated_residual_to_base_norm_ratio"] for row in gates),
    }
    finite = all(math.isfinite(float(value)) for row in r0 + r1 for value in row.values() if isinstance(value, (float, int)))
    checks = {
        "mean_macro_f1_ge_0_645": aggregate["R1_mean_macro_f1"] >= 0.645,
        "delta_macro_f1_ge_0_010": aggregate["delta_macro_f1"] >= 0.010,
        "positive_folds_ge_3": aggregate["positive_folds"] >= 3,
        "ez_f1_non_decreasing": aggregate["delta_ez_f1"] >= 0,
        "auprc_or_mrr_improves": aggregate["delta_ez_auprc"] > 0 or aggregate["delta_ez_mrr"] > 0,
        "no_pathology": finite and all(
            row["gate_mean"] > 1e-6 and row["gate_q90"] < 0.95
            and row["gated_residual_to_base_norm_ratio"] < 1.0 for row in gates
        ),
    }
    result = {"pass": all(checks.values()), "checks": checks, "aggregate": aggregate,
              "outer_test_evaluated": False, "historical_outer_already_viewed": True}
    (PUBLIC / "VALIDATION_GATE.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# R1 HLV validation result", "",
        f"R0 mean patient Macro-F1: `{aggregate['R0_mean_macro_f1']:.6f}`.  "
        f"R1: `{aggregate['R1_mean_macro_f1']:.6f}`.  "
        f"Delta: `{aggregate['delta_macro_f1']:+.6f}`; positive folds: `{aggregate['positive_folds']}/5`.",
        f"EZ-F1 delta: `{aggregate['delta_ez_f1']:+.6f}`; EZ-AUPRC delta: `{aggregate['delta_ez_auprc']:+.6f}`; "
        f"EZ-MRR delta: `{aggregate['delta_ez_mrr']:+.6f}`.",
        f"Mean HLV gate: `{aggregate['mean_gate']:.6f}`; mean gated-residual/base norm ratio: "
        f"`{aggregate['mean_residual_to_base_norm_ratio']:.6f}`.",
        "", "## Frozen outer-test gate", "",
    ]
    lines.extend(f"- {key}: {'PASS' if value else 'FAIL'}" for key, value in checks.items())
    lines += ["", f"Overall: **{'PASS' if result['pass'] else 'FAIL'}**.", ""]
    if not result["pass"]:
        lines += ["Outer-test evaluation was not run; R1 is unsupported under the locked gate.", ""]
    (PUBLIC.parent / "FINAL_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
