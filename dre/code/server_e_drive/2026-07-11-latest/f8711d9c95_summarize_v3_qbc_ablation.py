#!/usr/bin/env python3
"""Summarize the frozen V3-QBC profiles with patient-paired bootstrap."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROFILES = ["A0_BASE", "A1_Q10", "A2_BOUNDARY_COVERAGE", "A3_QBC_FULL"]
V3_REFERENCE = {
    "truek_patient_macro_f1": 0.669,
    "truek_patient_ez_auprc": 0.5184277652,
    "truek_patient_ez_mrr": 0.7099510182,
}


def paired_bootstrap(base: np.ndarray, candidate: np.ndarray, *, repeats: int, seed: int) -> dict[str, float]:
    if base.shape != candidate.shape:
        raise ValueError("Paired bootstrap arrays must have identical shapes")
    rng = np.random.default_rng(seed)
    deltas = candidate - base
    samples = np.empty(repeats, dtype=np.float64)
    for idx in range(repeats):
        sampled = rng.integers(0, deltas.size, size=deltas.size)
        samples[idx] = deltas[sampled].mean()
    return {
        "mean_delta": float(deltas.mean()),
        "ci_2_5": float(np.quantile(samples, 0.025)),
        "ci_97_5": float(np.quantile(samples, 0.975)),
        "probability_delta_gt_zero": float(np.mean(samples > 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", "--root_dir", dest="root", required=True)
    parser.add_argument("--bootstrap-repeats", "--bootstrap_repeats", dest="bootstrap_repeats", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    root = Path(args.root)
    overall = []
    by_fold = []
    by_center = []
    patients: dict[str, pd.DataFrame] = {}
    for profile in PROFILES:
        profile_dir = root / profile
        required = [
            profile_dir / "formal_summary.csv",
            profile_dir / "truek_summary.csv",
            profile_dir / "formal_by_patient.csv",
        ]
        if not all(path.exists() for path in required):
            continue
        formal = pd.read_csv(required[0]).iloc[0].to_dict()
        truek = pd.read_csv(required[1]).iloc[0].to_dict()
        overall.append({"profile": profile, **{f"formal_{k}": v for k, v in formal.items()}, **{
            f"truek_{k}": v for k, v in truek.items()
        }})
        for path, target in ((profile_dir / "formal_by_fold.csv", by_fold), (profile_dir / "formal_by_center.csv", by_center)):
            frame = pd.read_csv(path)
            frame.insert(0, "profile", profile)
            target.extend(frame.to_dict("records"))
        patients[profile] = pd.read_csv(required[2]).sort_values("subject_id").reset_index(drop=True)
    pd.DataFrame(overall).to_csv(root / "ablation_summary.csv", index=False)
    pd.DataFrame(by_fold).to_csv(root / "ablation_by_fold.csv", index=False)
    pd.DataFrame(by_center).to_csv(root / "ablation_by_center.csv", index=False)
    bootstrap = []
    if "A0_BASE" in patients:
        base = patients["A0_BASE"]
        for profile, frame in patients.items():
            if profile == "A0_BASE":
                continue
            merged = base[["subject_id", "patient_macro_f1"]].merge(
                frame[["subject_id", "patient_macro_f1"]], on="subject_id", suffixes=("_base", "_candidate"), validate="one_to_one"
            )
            bootstrap.append({"profile": profile, **paired_bootstrap(
                merged["patient_macro_f1_base"].to_numpy(),
                merged["patient_macro_f1_candidate"].to_numpy(),
                repeats=args.bootstrap_repeats,
                seed=args.seed,
            )})
    pd.DataFrame(bootstrap).to_csv(root / "ablation_bootstrap.csv", index=False)
    lines = ["# V3-QBC Ablation Report", "", "Formal and true-K diagnostic results are reported separately.", ""]
    a0 = next((row for row in overall if row["profile"] == "A0_BASE"), None)
    if a0 is not None:
        deviations = {
            key: abs(float(a0.get(key, float("nan"))) - reference)
            for key, reference in V3_REFERENCE.items()
        }
        reproduction_passed = all(np.isfinite(value) and value <= 0.003 for value in deviations.values())
        lines.extend([
            f"A0 reproduction gate: {'PASSED' if reproduction_passed else 'FAILED'}.",
            f"Absolute deviations: {deviations}.",
            (
                "A1-A3 are eligible for interpretation."
                if reproduction_passed
                else "A1-A3 must not be interpreted until the A0 protocol drift is resolved."
            ),
            "",
        ])
    for row in overall:
        lines.append(
            f"- {row['profile']}: formal Macro-F1={float(row.get('formal_patient_macro_f1', 0.0)):.6f}; "
            f"true-K diagnostic Macro-F1={float(row.get('truek_patient_macro_f1', 0.0)):.6f}"
        )
    (root / "V3_QBC_ABLATION_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
