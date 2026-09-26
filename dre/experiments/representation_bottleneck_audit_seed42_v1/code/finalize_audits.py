"""Summarize fully completed A/B/C/D validation audits; never touch outer test."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve()
EXPERIMENT = HERE.parents[1]
sys.path.insert(0, str(HERE.parents[2] / "a1_a2_patient_equal_objective_seed42_v1" / "code"))
from development_metrics import METRICS, finalize_fold  # noqa: E402
from reproduce_source import A1_RUNTIME, RUNTIME, ensure_source  # noqa: E402

BASE = {"A0": "A1", "B0": "A1", "C0": "A1", "D0": "A1"}
PROBES = {"A1": "A", "A2": "A", "B1": "B", "B2": "B", "C1": "C", "C2": "C"}
DIRS = {"A": "audit_a_temporal_order", "B": "audit_b_recruitment", "C": "audit_c_cross_seizure", "D": "audit_d_views"}
NAMES = {"A0": "A0_VLOO_BY_FOLD.csv", "A1": "A1_ORDERED_VLOO_BY_FOLD.csv", "A2": "A2_SHUFFLED_VLOO_BY_FOLD.csv",
         "B0": "B0_VLOO_BY_FOLD.csv", "B1": "B1_MAGNITUDE_VLOO_BY_FOLD.csv", "B2": "B2_RANK_VLOO_BY_FOLD.csv",
         "C0": "C0_VLOO_BY_FOLD.csv", "C1": "C1_SIMPLE_VLOO_BY_FOLD.csv", "C2": "C2_PERSISTENCE_VLOO_BY_FOLD.csv",
         "D0": "D0_ALL_VLOO_BY_FOLD.csv", "D1": "D1_NO_ABS_VLOO_BY_FOLD.csv", "D2": "D2_NO_RATIO_VLOO_BY_FOLD.csv", "D3": "D3_DELTA_ZDELTA_VLOO_BY_FOLD.csv"}


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"No aggregate rows for {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def source_grid(fold: int, epoch: int) -> dict:
    path = A1_RUNTIME / "A1" / f"fold_{fold}" / f"epoch_{epoch:02d}_validation_grid.json"
    return json.loads(path.read_text(encoding="utf-8"))


def probe_grid(variant: str, fold: int, epoch: int) -> dict:
    path = RUNTIME / "private" / f"fold_{fold}" / f"epoch_{epoch:02d}" / f"{variant}_validation_grid.json"
    return json.loads(path.read_text(encoding="utf-8"))


def build_variant(variant: str, folder: Path) -> tuple[list[dict], list[dict]]:
    public, apparent = [], []
    for fold in range(1, 6):
        if variant in BASE:
            grids = [source_grid(fold, epoch) for epoch in range(1, 31)]
        elif variant in PROBES:
            grids = [probe_grid(variant, fold, epoch) for epoch in range(1, 31)]
        else:
            directory = RUNTIME / "private" / "views" / variant / f"fold_{fold}"
            summary = json.loads((directory / "development_summary.json").read_text(encoding="utf-8"))
            if summary["epochs"] != 30 or summary["outer_test_accessed"]:
                raise RuntimeError("D ablation incomplete or leakage flag changed")
            public.append(summary["vloo"])
            apparent.append(summary["fullval"])
            continue
        private = RUNTIME / "private" / "finalization" / f"{variant}_fold_{fold}_VLOO_PATIENT.csv"
        row, fullval = finalize_fold(grids, variant, fold, private)
        public.append(row)
        apparent.append(fullval)
    write_csv(folder / NAMES[variant], public)
    return public, apparent


def avg(rows: list[dict], metric: str) -> float:
    return float(np.mean([float(row[metric]) for row in rows]))


def comparison(base: list[dict], control: list[dict], candidate: list[dict], label: str) -> dict:
    b, c, x = (avg(rows, "patient_macro_f1") for rows in (base, control, candidate))
    return {"mechanism": label, "baseline_macro_f1": b, "candidate_macro_f1": x,
            "delta_vs_baseline": x - b, "matched_control_macro_f1": c, "delta_vs_matched_control": x - c,
            "positive_folds_vs_baseline": sum(float(xr["patient_macro_f1"]) > float(br["patient_macro_f1"]) for xr, br in zip(candidate, base, strict=True)),
            "positive_folds_vs_control": sum(float(xr["patient_macro_f1"]) > float(cr["patient_macro_f1"]) for xr, cr in zip(candidate, control, strict=True)),
            "EZ_F1_delta": avg(candidate, "patient_ez_f1") - avg(base, "patient_ez_f1"),
            "AUPRC_delta": avg(candidate, "patient_ez_auprc") - avg(base, "patient_ez_auprc")}


def simple_gate(result: dict, prefix: str, multi: bool | None = None) -> dict:
    checks = {"gain_ge_0_010": result["delta_vs_baseline"] >= 0.010,
              "matched_gain_ge_0_005": result["delta_vs_matched_control"] >= 0.005,
              "positive_vs_baseline_ge_4": result["positive_folds_vs_baseline"] >= 4,
              "positive_vs_control_ge_3": result["positive_folds_vs_control"] >= 3,
              "EZ_F1_nondecreasing": result["EZ_F1_delta"] >= 0,
              "EZ_AUPRC_preserved": result["AUPRC_delta"] >= -0.005,
              "VLOO_macro_f1_ge_0_640": result["candidate_macro_f1"] >= 0.640}
    if multi is not None:
        checks["multiseizure_subset_C2_minus_C1_positive"] = multi
    passed = all(checks.values())
    return {"pass": passed, "terminal": f"{prefix}_{'SUPPORTED' if passed else 'NOT_SUPPORTED'}",
            "checks": checks, "summary": result, "outer_test_accessed": False}


def c_subset() -> tuple[list[dict], bool]:
    output = []
    total_control, total_candidate = [], []
    for fold in range(1, 6):
        path = RUNTIME / "private" / f"fold_{fold}" / "epoch_01" / "representation.pt"
        payload = torch.load(path, map_location="cpu", weights_only=False)
        counts = {row["subject_id"]: len(row["seizures"]) for row in payload["validation"]}
        c1 = {row["subject_id"]: row for row in read_csv(RUNTIME / "private" / "finalization" / f"C1_fold_{fold}_VLOO_PATIENT.csv")}
        c2 = {row["subject_id"]: row for row in read_csv(RUNTIME / "private" / "finalization" / f"C2_fold_{fold}_VLOO_PATIENT.csv")}
        ids = [subject_id for subject_id, count in counts.items() if count >= 2]
        if set(c1) != set(c2) or set(c1) != set(counts):
            raise RuntimeError("C matched multi-seizure patient membership changed")
        v1 = [float(c1[i]["patient_macro_f1"]) for i in ids]
        v2 = [float(c2[i]["patient_macro_f1"]) for i in ids]
        output.append({"fold": fold, "n_validation_patients_ge_2_seizures": len(ids),
                       "C1_macro_f1": float(np.mean(v1)) if ids else "",
                       "C2_macro_f1": float(np.mean(v2)) if ids else "",
                       "C2_minus_C1": float(np.mean(np.asarray(v2) - v1)) if ids else ""})
        total_control.extend(v1); total_candidate.extend(v2)
    if not total_control:
        raise RuntimeError("No multi-seizure validation patients")
    delta = float(np.mean(np.asarray(total_candidate) - total_control))
    output.append({"fold": "ALL", "n_validation_patients_ge_2_seizures": len(total_control),
                   "C1_macro_f1": float(np.mean(total_control)), "C2_macro_f1": float(np.mean(total_candidate)),
                   "C2_minus_C1": delta})
    return output, delta > 0


def view_decisions(rows: dict[str, list[dict]]) -> dict:
    base = rows["D0"]
    findings = {}
    for variant, question in (("D1", "remove_ABS"), ("D2", "remove_RATIO"), ("D3", "delta_zdelta_only")):
        ablated = rows[variant]
        delta = avg(ablated, "patient_macro_f1") - avg(base, "patient_macro_f1")
        ez = avg(ablated, "patient_ez_f1") - avg(base, "patient_ez_f1")
        auprc = avg(ablated, "patient_ez_auprc") - avg(base, "patient_ez_auprc")
        fold_deltas = [float(x["patient_macro_f1"]) - float(b["patient_macro_f1"]) for x, b in zip(ablated, base, strict=True)]
        removable = delta >= -0.003 and ez >= -0.005 and auprc >= -0.005 and sum(v < -0.01 for v in fold_deltas) <= 1
        useful = delta <= -0.005 and sum(v < 0 for v in fold_deltas) >= 3
        findings[question] = {"ablated_variant": variant, "macro_f1_delta": delta, "EZ_F1_delta": ez,
                              "EZ_AUPRC_delta": auprc, "fold_deltas": fold_deltas,
                              "view_removable": removable, "removed_view_positively_useful": useful,
                              "interpretation": "removable" if removable else "useful" if useful else "inconclusive"}
    identified = any(v["view_removable"] or v["removed_view_positively_useful"] for v in findings.values())
    return {"terminal": "VIEW_REDUNDANCY_IDENTIFIED" if identified else "VIEW_REDUNDANCY_INCONCLUSIVE",
            "findings": findings, "outer_test_accessed": False}


def fold_comparison(base: list[dict], control: list[dict], candidate: list[dict]) -> list[dict]:
    output = []
    for b, c, x in zip(base, control, candidate, strict=True):
        output.append({"fold": b["fold"], "baseline_macro_f1": b["patient_macro_f1"],
                       "control_macro_f1": c["patient_macro_f1"], "candidate_macro_f1": x["patient_macro_f1"],
                       "candidate_minus_baseline": float(x["patient_macro_f1"]) - float(b["patient_macro_f1"]),
                       "candidate_minus_control": float(x["patient_macro_f1"]) - float(c["patient_macro_f1"]),
                       "candidate_EZ_F1": x["patient_ez_f1"], "candidate_EZ_AUPRC": x["patient_ez_auprc"]})
    return output


def representation_audit() -> dict:
    paths = [RUNTIME / "private" / f"fold_{fold}" / f"epoch_{epoch:02d}" / "representation.pt"
             for fold in range(1, 6) for epoch in range(1, 31)]
    if any(not path.is_file() for path in paths):
        raise RuntimeError("Missing intermediate representation cache cell")
    sample = torch.load(paths[0], map_location="cpu", weights_only=False)
    first = sample["fit"][0]
    seizure = first["seizures"][0]
    if seizure["x"].shape[-1] != 36 or seizure["w"].shape[-1] != 32 or seizure["q"].shape[-1] != 32 or first["h"].shape[-1] != 64 or first["r"].shape[-1] != 64:
        raise RuntimeError("Intermediate representation schema changed")
    return {"cached_fold_epoch_cells": 150, "total_private_cache_bytes": sum(path.stat().st_size for path in paths),
            "fit_patients_first_cell": len(sample["fit"]), "validation_patients_first_cell": len(sample["validation"]),
            "window_input_dim": 36, "four_views": ["ABS", "DELTA", "ZDELTA", "RATIO"],
            "view_dim_each": 9, "window_encoder_output_dim": 32, "temporal_pooled_dim": 32,
            "cross_seizure_patient_channel_dim": 64, "contextual_preclassifier_dim": 64,
            "frozen_logit_replay_checked_during_extraction": True, "outer_test_accessed": False}


def feature_diagnostics() -> None:
    for audit, folder, filename in (("A", DIRS["A"], "TEMPORAL_SUMMARY_DIAGNOSTICS.csv"),
                                    ("B", DIRS["B"], "RECRUITMENT_DIAGNOSTICS.csv")):
        rows = []
        for fold in range(1, 6):
            items = [json.loads((RUNTIME / "private" / f"fold_{fold}" / f"epoch_{epoch:02d}" /
                                 f"{audit}_feature_diagnostics.json").read_text(encoding="utf-8")) for epoch in range(1, 31)]
            if audit == "A":
                rows.append({"fold": fold, "epochs": 30,
                             "ordered_PCA_components_mean": float(np.mean([r["A1_PCA"]["selected_components"] for r in items])),
                             "ordered_PCA_components_min": min(r["A1_PCA"]["selected_components"] for r in items),
                             "ordered_PCA_components_max": max(r["A1_PCA"]["selected_components"] for r in items),
                             "shuffled_PCA_components_mean": float(np.mean([r["A2_PCA"]["selected_components"] for r in items])),
                             "ordered_explained_variance_mean": float(np.mean([r["A1_PCA"]["explained_variance_selected"] for r in items]))})
            else:
                rows.append({"fold": fold, "epochs": 30,
                             "fit_magnitude_q80_mean": float(np.mean([r["fit_magnitude_q80"] for r in items])),
                             "fit_magnitude_q90_mean": float(np.mean([r["fit_magnitude_q90"] for r in items])),
                             "fit_change_std_min_across_epochs": float(min(r["fit_change_std_min"] for r in items)),
                             "fit_change_std_max_across_epochs": float(max(r["fit_change_std_max"] for r in items))})
        write_csv(EXPERIMENT / folder / filename, rows)


def main() -> None:
    ensure_source()
    source = json.loads((EXPERIMENT / "SOURCE_REPRODUCTION.json").read_text(encoding="utf-8"))
    if not source.get("pass") or source["max_grid_error_vs_original"] != 0:
        raise RuntimeError("SOURCE_A1_REPRODUCTION_FAILED")
    all_rows: dict[str, list[dict]] = {}
    fullvals: dict[str, list[dict]] = {}
    for variant in NAMES:
        folder = EXPERIMENT / DIRS[variant[0]]
        all_rows[variant], fullvals[variant] = build_variant(variant, folder)
        print(f"[FINALIZE] {variant} VLOO={avg(all_rows[variant], 'patient_macro_f1'):.6f}", flush=True)
    if any(abs(float(row["patient_macro_f1"]) - source["folds"][i]["observed"]) > 1e-8 for i, row in enumerate(all_rows["A0"])):
        raise RuntimeError("A0 source reproduction mismatch")
    subset_rows, subset_positive = c_subset()
    write_csv(EXPERIMENT / DIRS["C"] / "MULTISEIZURE_SUBSET.csv", subset_rows)
    comps = {}
    gates = {}
    for audit, baseline, control, candidate, label, prefix, gatefile in (
        ("A", "A0", "A2", "A1", "temporal_order", "TEMPORAL_ORDER", "TEMPORAL_ORDER_GATE.json"),
        ("B", "B0", "B1", "B2", "recruitment_rank", "RECRUITMENT_RANK", "RECRUITMENT_GATE.json"),
        ("C", "C0", "C1", "C2", "cross_seizure_persistence", "CROSS_SEIZURE_PERSISTENCE", "PERSISTENCE_GATE.json"),
    ):
        result = comparison(all_rows[baseline], all_rows[control], all_rows[candidate], label)
        gate = simple_gate(result, prefix, subset_positive if audit == "C" else None)
        comps[audit], gates[audit] = result, gate
        write_csv(EXPERIMENT / DIRS[audit] / "COMPARISON.csv", fold_comparison(all_rows[baseline], all_rows[control], all_rows[candidate]))
        write_json(EXPERIMENT / DIRS[audit] / gatefile, gate)
    view_rows = []
    for candidate, label in (("D1", "remove_ABS"), ("D2", "remove_RATIO"), ("D3", "delta_zdelta_only")):
        result = comparison(all_rows["D0"], all_rows["D0"], all_rows[candidate], label)
        view_rows.append(result)
    decisions = view_decisions(all_rows)
    for row in view_rows:
        finding = decisions["findings"][row["mechanism"]]
        row["gate_pass"] = finding["view_removable"] or finding["removed_view_positively_useful"]
        row["interpretation"] = finding["interpretation"]
    write_csv(EXPERIMENT / DIRS["D"] / "VIEW_COMPARISON.csv", view_rows)
    write_json(EXPERIMENT / DIRS["D"] / "VIEW_REDUNDANCY_CONCLUSION.json", decisions)
    feature_diagnostics()
    write_json(EXPERIMENT / "INTERMEDIATE_REPRESENTATION_AUDIT.json", representation_audit())
    matrix = [{**comps[audit], "gate_pass": gates[audit]["pass"],
               "interpretation": gates[audit]["terminal"]} for audit in ("A", "B", "C")]
    matrix.extend(view_rows)
    write_csv(EXPERIMENT / "SUMMARY_MATRIX.csv", matrix)
    write_json(EXPERIMENT / "AUDITS_ABCD_STATUS.json", {"complete": True,
               "source_reproduced": True, "probe_cells": 900, "view_retraining_cells": 450,
               "terminals": {"A": gates["A"]["terminal"], "B": gates["B"]["terminal"],
                             "C": gates["C"]["terminal"], "D": decisions["terminal"]},
               "outer_test_accessed": False})


if __name__ == "__main__":
    main()
