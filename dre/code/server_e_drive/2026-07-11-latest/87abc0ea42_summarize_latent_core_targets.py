from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def normalize_center(value: object) -> str:
    return str(value).strip().lower().replace(" ", "_")


def _entropy(q_values: pd.Series) -> float:
    vals = q_values.astype(float).to_numpy()
    vals = vals[vals > 1e-12]
    if len(vals) == 0:
        return 0.0
    probs = vals / max(float(vals.sum()), 1e-12)
    return float(-np.sum(probs * np.log(probs)))


def validate_targets(targets: pd.DataFrame, *, rho: float, broad_centers: set[str]) -> list[str]:
    warnings: list[str] = []
    df = targets.copy()
    df["center"] = df["center"].map(normalize_center)
    broad_centers = {normalize_center(center) for center in broad_centers}
    strong = df[~df["center"].isin(broad_centers)]
    if not strong.empty:
        if not np.allclose(strong["pseudo_core_q"].astype(float), strong["label_ez"].astype(float)):
            raise ValueError("Strong-center targets must satisfy pseudo_core_q == label_ez.")
        if set(strong["core_target_type"].astype(str)) != {"strong_label"}:
            raise ValueError("Strong-center targets must have core_target_type == strong_label.")

    broad = df[df["center"].isin(broad_centers)]
    for pid, group in broad.groupby("patient_id", sort=False):
        q = group["pseudo_core_q"].astype(float)
        labels = group["label_ez"].astype(float)
        target_types = set(group["core_target_type"].astype(str))
        if not target_types.issubset({"latent_core", "teacher_only_latent_core"}):
            raise ValueError(f"Broad-center patient {pid} has invalid core_target_type={sorted(target_types)}.")
        if (q < -1e-8).any() or (q > 1.0 + 1e-8).any():
            raise ValueError(f"Broad-center patient {pid} has pseudo_core_q outside [0,1].")
        if (q[labels <= 0.5] != 0.0).any():
            raise ValueError(f"Broad-center patient {pid} has nonzero NEZ pseudo_core_q.")
        n_ez = int(round(float(labels.sum())))
        if n_ez > 0:
            expected = max(1.0, float(rho) * float(n_ez))
            if float(q.sum()) + 1e-5 < expected * 0.70:
                raise ValueError(f"Broad-center patient {pid} has too little pseudo-core mass.")
    return warnings


def summarize_targets(targets: pd.DataFrame, *, rho: float = 0.30, broad_centers: set[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    broad_centers = broad_centers or {"lzu", "pediatric"}
    validate_targets(targets, rho=rho, broad_centers=broad_centers)
    targets = targets.copy()
    targets["center"] = targets["center"].map(normalize_center)
    patient = (
        targets.groupby(["patient_id", "center", "fold_id"], as_index=False)
        .agg(
            n_channels=("channel_name", "count"),
            n_ez=("label_ez", "sum"),
            sum_q=("pseudo_core_q", "sum"),
            q_max=("pseudo_core_q", "max"),
            mean_teacher_score=("a9v3_oof_score", "mean"),
        )
    )
    entropies = []
    q_gt_05 = []
    q_gt_02 = []
    for _, group in targets.groupby("patient_id", sort=False):
        labels = group["label_ez"].astype(float)
        q_ez = group.loc[labels > 0.5, "pseudo_core_q"].astype(float)
        entropies.append({"patient_id": group["patient_id"].iloc[0], "q_entropy": _entropy(q_ez)})
        q_gt_05.append({"patient_id": group["patient_id"].iloc[0], "n_core_q_gt_05": int((q_ez > 0.5).sum())})
        q_gt_02.append({"patient_id": group["patient_id"].iloc[0], "n_core_q_gt_02": int((q_ez > 0.2).sum())})
    patient = patient.merge(pd.DataFrame(entropies), on="patient_id", how="left")
    patient = patient.merge(pd.DataFrame(q_gt_05), on="patient_id", how="left")
    patient = patient.merge(pd.DataFrame(q_gt_02), on="patient_id", how="left")
    patient["mean_effective_core_fraction"] = patient["sum_q"].astype(float) / patient["n_ez"].astype(float).clip(lower=1.0)
    center = (
        patient.groupby("center", as_index=False)
        .agg(
            n_patients=("patient_id", "nunique"),
            n_channels=("n_channels", "sum"),
            mean_n_ez=("n_ez", "mean"),
            mean_sum_q=("sum_q", "mean"),
            mean_effective_core_fraction=("mean_effective_core_fraction", "mean"),
            mean_q_entropy=("q_entropy", "mean"),
            mean_q_max=("q_max", "mean"),
            mean_n_core_q_gt_05=("n_core_q_gt_05", "mean"),
            mean_n_core_q_gt_02=("n_core_q_gt_02", "mean"),
            mean_teacher_score=("mean_teacher_score", "mean"),
        )
    )
    center["q_uniformity_warning"] = center.apply(
        lambda row: "possible uniform pseudo-core" if float(row["mean_q_max"]) < 0.35 and normalize_center(row["center"]) in broad_centers else "",
        axis=1,
    )
    center["q_collapse_warning"] = center.apply(
        lambda row: "possible collapse" if float(row["mean_q_entropy"]) < 0.25 and float(row["mean_n_core_q_gt_05"]) <= 1.0 and normalize_center(row["center"]) in broad_centers else "",
        axis=1,
    )
    report = {
        "n_rows": int(len(targets)),
        "n_patients": int(targets["patient_id"].nunique()),
        "broad_centers": sorted(broad_centers),
        "warnings": [
            str(value)
            for value in pd.concat([center["q_uniformity_warning"], center["q_collapse_warning"]]).tolist()
            if str(value)
        ],
    }
    return patient, center, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize A9v8 latent core targets.")
    parser.add_argument("--target_csv", "--latent_core_csv", dest="target_csv", type=str, required=True)
    parser.add_argument("--audit_json", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--broad_centers", type=str, default="lzu,pediatric")
    parser.add_argument("--rho", type=float, default=0.30)
    args = parser.parse_args()

    target_csv = Path(args.target_csv)
    output_dir = Path(args.output_dir) if args.output_dir else target_csv.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = pd.read_csv(target_csv)
    broad_centers = {item.strip() for item in args.broad_centers.split(",") if item.strip()}
    if args.audit_json and Path(args.audit_json).exists():
        with open(args.audit_json, "r", encoding="utf-8") as fin:
            audit = json.load(fin)
        broad_centers = set(audit.get("broad_centers", sorted(broad_centers)))
        args.rho = float(audit.get("rho", args.rho))
    patient, center, report = summarize_targets(targets, rho=args.rho, broad_centers=broad_centers)
    patient.to_csv(output_dir / "latent_core_patient_summary.csv", index=False)
    center.to_csv(output_dir / "latent_core_center_summary.csv", index=False)
    with open(output_dir / "latent_core_q_quality_report.json", "w", encoding="utf-8") as fout:
        json.dump(report, fout, indent=2, ensure_ascii=False, sort_keys=True)
    print(f"Wrote {output_dir / 'latent_core_patient_summary.csv'}")
    print(f"Wrote {output_dir / 'latent_core_center_summary.csv'}")
    print(f"Wrote {output_dir / 'latent_core_q_quality_report.json'}")


if __name__ == "__main__":
    main()
