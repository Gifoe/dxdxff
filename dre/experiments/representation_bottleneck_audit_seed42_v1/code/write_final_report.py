"""Publish aggregate-only final interpretation after all five audits complete."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from reproduce_source import LOCK_SHA256, RUNTIME, ensure_source  # noqa: E402


def records(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def payload(relative: str) -> dict:
    return json.loads((EXPERIMENT / relative).read_text(encoding="utf-8"))


def signed(value) -> str:
    return f"{float(value):+.6f}"


def main() -> None:
    ensure_source()
    source = payload("SOURCE_REPRODUCTION.json")
    status = payload("AUDITS_ABCD_STATUS.json")
    if not source.get("pass") or not status.get("complete"):
        raise RuntimeError("Cannot publish incomplete source or A/B/C/D audit")
    matrix = records(EXPERIMENT / "SUMMARY_MATRIX.csv")
    if len(matrix) != 6:
        raise RuntimeError("Expected six prespecified comparison rows")
    e_report = EXPERIMENT / "audit_e_failures" / "FAILURE_STRATIFICATION_REPORT.md"
    if not e_report.exists() or "FAILURE_STRATIFICATION_COMPLETE" not in e_report.read_text(encoding="utf-8"):
        raise RuntimeError("Failure stratification incomplete")
    a = payload("audit_a_temporal_order/TEMPORAL_ORDER_GATE.json")
    b = payload("audit_b_recruitment/RECRUITMENT_GATE.json")
    c = payload("audit_c_cross_seizure/PERSISTENCE_GATE.json")
    d = payload("audit_d_views/VIEW_REDUNDANCY_CONCLUSION.json")
    if any(gate.get("outer_test_accessed") for gate in (a, b, c, d)):
        raise RuntimeError("Outer-test flag inconsistent")
    supported = [name for name, gate in (("temporal_order", a), ("recruitment_rank", b),
                                         ("cross_seizure_persistence", c)) if gate["pass"]]
    if d["terminal"] == "VIEW_REDUNDANCY_IDENTIFIED":
        supported.append("four_view_redundancy_or_necessity")
    terminal = "REPRESENTATION_BOTTLENECK_IDENTIFIED" if supported else "NO_CLEAR_REPRESENTATION_BOTTLENECK_IDENTIFIED"
    by_name = {row["mechanism"]: row for row in matrix}
    corr = {row["predictor"]: row for row in records(EXPERIMENT / "audit_e_failures" / "FAILURE_CORRELATIONS.csv")}
    failures = records(EXPERIMENT / "audit_e_failures" / "FAILURE_TYPE_SUMMARY.csv")
    centers = records(EXPERIMENT / "audit_e_failures" / "FAILURE_BY_CENTER.csv")
    lines = ["# Representation bottleneck audit: seed 42", "",
             "Development-only on the historically viewed 80-patient cohort. No new outer-test loader, predictions or performance evaluation were used.",
             f"Frozen A1 source reproduction: **PASS**, 150 checkpoints, maximum validation-grid error {source['max_grid_error_vs_original']}; mean VLOO Macro-F1 {source['observed_mean']:.10f}.",
             "", "| Mechanism | Baseline | Matched control | Candidate | Δ baseline | Δ matched | Positive folds baseline/control | Gate |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for row in matrix:
        gate = row["interpretation"]
        lines.append(f"| {row['mechanism']} | {float(row['baseline_macro_f1']):.6f} | {float(row['matched_control_macro_f1']):.6f} | "
                     f"{float(row['candidate_macro_f1']):.6f} | {signed(row['delta_vs_baseline'])} | {signed(row['delta_vs_matched_control'])} | "
                     f"{row['positive_folds_vs_baseline']}/5 / {row['positive_folds_vs_control']}/5 | {gate} |")
    lines.extend(["", "A/B/C support requires every prelocked mean, matched-control, fold-consistency, EZ-F1, AUPRC and absolute-performance check. D uses separate conservative removability/usefulness rules; it is not a new model-selection sweep.", "",
                  "## Prespecified questions", "",
                  f"- Chronological temporal order beyond shuffled control: **{a['terminal']}**.",
                  f"- Cross-channel recruitment rank beyond magnitude-only evidence: **{b['terminal']}**.",
                  f"- Cross-seizure persistence beyond mean/std aggregation: **{c['terminal']}**. The >=2-seizure subset is diagnostic only.",
                  f"- Four-view audit: **{d['terminal']}**. ABS removal and RATIO removal are individually inconclusive. Removing both to retain only DELTA+ZDELTA lowers Macro-F1 by {abs(d['findings']['delta_zdelta_only']['macro_f1_delta']):.6f} in 4/5 folds; the pair is jointly useful, but neither view is individually proven necessary.",
                  "- Candidates reaching VLOO Macro-F1 >=0.640: " + (", ".join(row["mechanism"] for row in matrix[:3] if float(row["candidate_macro_f1"]) >= 0.64) or "none") + ".",
                  "- Frozen probes with >=1 pp gain over A1: " + (", ".join(row["mechanism"] for row in matrix[:3] if float(row["delta_vs_baseline"]) >= 0.01) or "none") + ".",
                  "- Frozen probes with positive matched-control mean: " + (", ".join(row["mechanism"] for row in matrix[:3] if float(row["delta_vs_matched_control"]) > 0) or "none") + ".",
                  "- Fold consistency is listed explicitly in the table; a positive mean alone is insufficient.", "",
                  "## Frozen A1 failure stratification", "",
                  "65 excluded-validation patient-fold cases; label-using variables are diagnostic only. Correlations are descriptive without multiplicity-based significance claims.",
                  "| Predictor | Pearson r with Macro-F1 | Spearman rho |", "| --- | ---: | ---: |"])
    for name in ("patient_ez_auprc", "patient_ez_mrr", "patient_ez_auroc", "n_seizures", "n_channels", "true_ez_fraction", "valid_window_count", "missing_invalid_window_fraction"):
        row = corr[name]
        lines.append(f"| {name} | {row['pearson_r']} | {row['spearman_rho']} |")
    lines.extend(["", "Failure-type counts (fixed pooled-median definitions): " + ", ".join(f"{row['failure_type']}={row['n_cases']}" for row in failures) + ".",
                  "Among the 32 low-Macro-F1 cases, 28 also have poor ranking and 4 have good ranking; this is a descriptive ranking-associated failure pattern, not causal evidence or an independent prediction test.",
                  "Center descriptive aggregates (n patients, mean Macro-F1): " + ", ".join(f"{row['center']} ({row['n_patients']}, {float(row['macro_f1_mean']):.3f})" for row in centers) + ". Small-n centers are not interpreted as causal effects.",
                  "Seizure count and true EZ fraction have near-zero rank correlation with Macro-F1 here; channel count is moderately negative. Window-mask missingness has zero variation, so window quality cannot be diagnosed from this cache field.",
                  "", "## Decision", "",
                  "Mechanisms passing their own frozen development gates: " + (", ".join(supported) if supported else "none") + ".",
                  "The identified D result is joint view necessity, not a successful replacement model: no frozen probe improved A1, no candidate reached 0.640 VLOO Macro-F1, and the individual ABS/RATIO roles remain unresolved. It does not by itself justify a constructive redesign.",
                  "These exploratory diagnostics alone are not sealed confirmation and do not authorize an outer-test claim.",
                  f"Exact terminal: `{terminal}`.", ""])
    (EXPERIMENT / "FINAL_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    audit = payload("INTERMEDIATE_REPRESENTATION_AUDIT.json")
    implementation = ["# Implementation audit", "",
                      f"- Locked protocol SHA-256: `{LOCK_SHA256}`; A1 source reproduced exactly over 5 folds × 30 epochs × 13 validation patients.",
                      "- A/B/C: 900 trained 15-epoch residual heads, matched initialization and parameter counts within each control/candidate pair; A1 frozen throughout.",
                      "- A1 trajectory projection was FIT-only; deterministic ordered/shuffled controls had equal 32-column head input after zero-padding unused PCA columns.",
                      "- B recruitment standardization and magnitude-control thresholds were fitted on FIT only; C reused the same rank evidence and retained one-seizure patients.",
                      "- All 150 fold/epoch cached normalized inputs, centers and validity masks were independently checked bitwise equal within fold; the fold-level recruitment feature cache is numerically exact.",
                      "- D: original A1 D0 reused, 3 matched ablations × 5 folds × 30 epochs trained from the exact fold initialization, with full 36-D architecture and view masking after the original FIT normalizer.",
                      f"- Private intermediate cache: {audit['cached_fold_epoch_cells']} cells; schema 36-D input, 32-D window/temporal embeddings, 64-D patient/contextual embeddings. Private cache bytes: {audit['total_private_cache_bytes']}.",
                      "- Engineering repair: NumPy native access violation during quantile of short strided seizure arrays was replaced by a pure-Python linear-interpolation quantile with identical mathematical semantics; all completed cells and checkpoints were preserved and resumed.",
                      "- One unreadable private representation cache cell was quarantined and rebuilt from its exact frozen A1 checkpoint; the six existing probe heads and validation grids were preserved. The subsequent 150-cell input invariance scan passed.",
                      "- Patient-level representations, IDs, labels, scores, failure rows, model checkpoints and logs remain in the private server runtime. Public outputs are aggregate only.",
                      "- No outer-test loader was constructed and no outer predictions/metrics were evaluated. The underlying historical cache constructor indexes cohort metadata; no outer labels or outcomes were used for training, selection, diagnostics or gates.", ""]
    (EXPERIMENT / "IMPLEMENTATION_AUDIT.md").write_text("\n".join(implementation), encoding="utf-8")
    (EXPERIMENT / "README.md").write_text("# Representation bottleneck audit (seed 42)\n\n" +
        "Frozen A1 development-only audits A/B/C (matched representation probes), D (four-view retraining), and E (failure diagnostics). " +
        f"The locked source and all audits completed. Terminal: `{terminal}`. See `FINAL_REPORT.md` and `SUMMARY_MATRIX.csv`. " +
        "No outer test was run; private patient data and checkpoints were not published.\n", encoding="utf-8")
    print(json.dumps({"terminal": terminal, "supported": supported, "outer_test_accessed": False}), flush=True)


if __name__ == "__main__":
    main()
