"""Patient-level multi-seed statistics. Patient x seed is never treated as iid."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


CORE_METRICS = ["patient_macro_f1", "patient_ez_f1", "patient_nez_f1", "patient_ez_auprc", "patient_ez_auroc", "patient_ez_mrr"]
PRQ_MODEL = "PRQ-Net"
BCR_MODEL = "BCR-Net"
CDEL_MODEL = "CDEL"


def _normalize_model_column(frame: pd.DataFrame, *, source: Path) -> pd.DataFrame:
    """Canonicalize evaluator `experiment` rows to the statistics `model` key."""
    data = frame.copy()
    if "model" not in data.columns:
        if "experiment" not in data.columns:
            raise ValueError(f"Patient metrics lack model/experiment column: {source}")
        data = data.rename(columns={"experiment": "model"})
    elif "experiment" in data.columns:
        mismatch = data["model"].astype(str) != data["experiment"].astype(str)
        if mismatch.any():
            raise ValueError(f"Conflicting model and experiment columns: {source}")
        data = data.drop(columns=["experiment"])
    return data


def _patient_average(root: Path, seeds: Iterable[int]) -> pd.DataFrame:
    frames = []
    for seed in seeds:
        path = root / "pooled_cv" / f"seed_{seed}" / "metrics" / "patient_level.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing completed seed metrics: {path}")
        frame = _normalize_model_column(pd.read_csv(path), source=path)
        frame["seed"] = int(seed); frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    keys = ["model", "subject_id", "center", "outer_fold"]
    metrics = [column for column in CORE_METRICS if column in combined.columns]
    return combined.groupby(keys, as_index=False)[metrics].mean(), combined


def _bootstrap_delta(frame: pd.DataFrame, *, model_a: str, model_b: str, metric: str, repeats: int, seed: int) -> dict:
    pivot = frame.pivot(index="subject_id", columns="model", values=metric).dropna()
    delta = pivot[model_a].to_numpy(float) - pivot[model_b].to_numpy(float)
    rng = np.random.default_rng(seed); values = np.empty(repeats, dtype=float)
    for index in range(repeats): values[index] = delta[rng.integers(0, len(delta), len(delta))].mean()
    return {"comparison": f"{model_a}_vs_{model_b}", "metric": metric, "estimate_a": float(pivot[model_a].mean()), "estimate_b": float(pivot[model_b].mean()), "delta": float(delta.mean()), "ci_low": float(np.quantile(values, .025)), "ci_high": float(np.quantile(values, .975)), "probability_delta_gt_zero": float((values > 0).mean()), "bootstrap_repeats": int(repeats), "n_unique_patients": int(len(delta))}


def _sign_flip(frame: pd.DataFrame, *, repeats: int, seed: int) -> dict:
    pivot = frame.pivot(index="subject_id", columns="model", values="patient_macro_f1").dropna()
    delta = (pivot[CDEL_MODEL] - pivot[PRQ_MODEL]).to_numpy(float); observed = abs(delta.mean())
    rng = np.random.default_rng(seed); signs = rng.choice((-1.0, 1.0), size=(repeats, len(delta)))
    permuted = np.abs((signs * delta).mean(axis=1))
    return {"protocol": "patient_wise_5fold_seed_averaged", "comparison": "Fusion_vs_P2", "metric": "patient_macro_f1", "observed_delta": float(delta.mean()), "p_value_two_sided": float((1 + (permuted >= observed).sum()) / (1 + repeats)), "permutation_repeats": int(repeats), "n_unique_patients": int(len(delta))}


def summarize_statistics(*, output_root: str | Path, seeds: list[int], bootstrap_repeats: int, permutation_repeats: int) -> dict:
    root = Path(output_root); statistics = root / "statistics"; statistics.mkdir(parents=True, exist_ok=True)
    averaged, raw = _patient_average(root, seeds)
    summary_rows = []
    for model, group in raw.groupby("model"):
        for metric in CORE_METRICS:
            values = group.groupby("seed")[metric].mean()
            summary_rows.append({"model": model, "metric": metric, **{f"seed_{seed}": float(values.get(seed, np.nan)) for seed in seeds}, "mean": float(values.mean()), "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0, "min": float(values.min()), "max": float(values.max())})
    pd.DataFrame(summary_rows).to_csv(statistics / "multiseed_overall.csv", index=False)
    deltas = raw.pivot_table(index="seed", columns="model", values="patient_macro_f1", aggfunc="mean").reset_index()
    deltas["fusion_minus_prq"] = deltas[CDEL_MODEL] - deltas[PRQ_MODEL]; deltas["fusion_minus_bcr"] = deltas[CDEL_MODEL] - deltas[BCR_MODEL]
    deltas.to_csv(statistics / "multiseed_fusion_delta.csv", index=False)
    boot = [_bootstrap_delta(averaged, model_a=CDEL_MODEL, model_b=PRQ_MODEL, metric=metric, repeats=bootstrap_repeats, seed=42 + index) for index, metric in enumerate(CORE_METRICS)]
    pd.DataFrame(boot).to_csv(statistics / "paired_patient_bootstrap.csv", index=False)
    pd.DataFrame([_sign_flip(averaged, repeats=permutation_repeats, seed=42)]).to_csv(statistics / "paired_permutation_test.csv", index=False)
    merged = averaged.pivot(index=["subject_id", "center", "outer_fold"], columns="model", values="patient_macro_f1").reset_index()
    merged["delta_patient"] = merged[CDEL_MODEL] - merged[PRQ_MODEL]
    merged["status"] = np.select([merged.delta_patient > 1e-6, merged.delta_patient < -1e-6], ["improved", "worsened"], default="unchanged")
    merged.to_csv(statistics / "patient_delta_detail.csv", index=False)
    merged.groupby("status").size().rename("n_patients").reset_index().to_csv(statistics / "patient_delta_summary.csv", index=False)
    thresholds = []
    for seed in seeds:
        path = root / "pooled_cv" / f"seed_{seed}" / "thresholds" / "selected_thresholds.csv"
        if path.is_file():
            frame = pd.read_csv(path); frame["seed"] = seed; thresholds.append(frame)
    if thresholds:
        selected = pd.concat(thresholds, ignore_index=True); selected.to_csv(statistics / "threshold_grid_sensitivity.csv", index=False)
        selected.groupby("model").threshold.agg(["median", "std", "min", "max", "nunique"]).reset_index().to_csv(statistics / "threshold_bootstrap_summary.csv", index=False)
    loco_rows = []
    loco_patients = []
    loco_root = root / "loco"
    if loco_root.is_dir():
        for center_dir in sorted(loco_root.glob("heldout_*")):
            center = center_dir.name.removeprefix("heldout_")
            for seed in seeds:
                overall_path = center_dir / f"seed_{seed}" / "metrics" / "overall.csv"
                patient_path = center_dir / f"seed_{seed}" / "metrics" / "patient_level.csv"
                if overall_path.is_file():
                    frame = pd.read_csv(overall_path); frame["held_out_center"] = center; frame["seed"] = seed; loco_rows.append(frame)
                if patient_path.is_file():
                    frame = pd.read_csv(patient_path); frame["held_out_center"] = center; frame["seed"] = seed; loco_patients.append(frame)
    if loco_rows:
        loco = pd.concat(loco_rows, ignore_index=True)
        pivot = loco.pivot_table(index=["held_out_center", "seed"], columns="experiment", values="patient_macro_f1").reset_index()
        if {PRQ_MODEL, CDEL_MODEL}.issubset(pivot.columns):
            pivot["fusion_minus_prq"] = pivot[CDEL_MODEL] - pivot[PRQ_MODEL]
        pivot.to_csv(statistics / "loco_by_center_seed.csv", index=False)
        pivot.groupby("held_out_center", as_index=False).mean(numeric_only=True).to_csv(statistics / "loco_summary.csv", index=False)
    if loco_patients:
        pd.concat(loco_patients, ignore_index=True).to_csv(statistics / "loco_patient_level_long.csv", index=False)
    return {"status": "passed", "n_seeds": len(seeds), "n_patients": int(averaged.subject_id.nunique()), "statistics_dir": str(statistics), "loco_completed_rows": len(loco_rows)}
