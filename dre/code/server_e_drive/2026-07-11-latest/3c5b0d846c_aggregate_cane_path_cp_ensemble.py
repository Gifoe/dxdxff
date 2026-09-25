from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exp_ez_hybrid import _summarize_prediction_records


SEEDS = (42, 43, 44)
KEY = ["outer_fold", "subject_id", "channel_name", "channel_id"]
CP = [
    "cp_preictal_suppression_rank", "cp_suppression_release_shift",
    "cp_causal_early_activation_rank", "cp_early_source_rank",
    "cp_propagation_persistence", "cp_cross_seizure_source_consistency",
]


def _read_seed(root: Path, seed: int) -> pd.DataFrame:
    paths = sorted((root / f"seed_{seed}").glob("test_channel_predictions_neuroez_v2_fold_*.csv"))
    if not paths:
        raise FileNotFoundError(f"No held-out channel predictions for seed {seed}")
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    required = set(KEY + ["label_nez", "label_ez", "center", "valid", "standardized_nez_logit", "predicted_patient_threshold", *CP])
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Seed {seed} ledger missing columns: {sorted(missing)}")
    if frame.duplicated(KEY).any():
        raise ValueError(f"Seed {seed} ledger has duplicate channel keys")
    return frame.sort_values(KEY).reset_index(drop=True)


def _assert_alignment(frames: list[pd.DataFrame]) -> None:
    reference = frames[0]
    for seed, frame in zip(SEEDS[1:], frames[1:]):
        if not reference[KEY].equals(frame[KEY]):
            raise ValueError(f"Seed {seed} channel keys do not match seed 42")
        for column in ("label_nez", "label_ez", "center", "valid", *CP):
            left, right = reference[column].to_numpy(), frame[column].to_numpy()
            equal = np.allclose(left, right, equal_nan=True) if np.issubdtype(left.dtype, np.number) else np.array_equal(left, right)
            if not equal:
                raise ValueError(f"Seed alignment mismatch in {column} for seed {seed}")
    counts = [frame.groupby(["outer_fold", "subject_id"]).size().sort_index() for frame in frames]
    if any(not counts[0].equals(value) for value in counts[1:]):
        raise ValueError("Per-patient valid channel counts differ across seeds")


def _records(frame: pd.DataFrame) -> list[dict[str, object]]:
    records = []
    for (fold, subject), group in frame.groupby(["outer_fold", "subject_id"], sort=True):
        group = group.sort_values("channel_id")
        predicted_nez = group["predicted_nez"].to_numpy(dtype=bool)
        records.append({
            "outer_fold": int(fold), "subject_id": str(subject), "center": str(group["center"].iloc[0]),
            "canonical_channels": group["channel_name"].astype(str).tolist(),
            "channel_mask": np.ones(len(group), dtype=bool),
            "labels_nez": group["label_nez"].to_numpy(dtype=np.float32),
            "labels_ez": group["label_ez"].to_numpy(dtype=np.float32),
            "score_nez": 1.0 / (1.0 + np.exp(-group["ensemble_standardized_nez_logit"].to_numpy())),
            "score_ez": 1.0 - 1.0 / (1.0 + np.exp(-group["ensemble_standardized_nez_logit"].to_numpy())),
            "predicted_nez_mask": predicted_nez, "predicted_ez_mask": ~predicted_nez,
            "decision_rule": "patient_adaptive_standardized_nez_threshold",
            "threshold_source": "cross_fitted_patient_adaptive_head", "positive_label": "nez",
            "true_count_used_for_prediction": False, "oracle_threshold_used_for_prediction": False,
        })
    return records


def _bootstrap(patient: pd.DataFrame, samples: int = 2000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    metrics = ["patient_macro_f1", "patient_nez_f1", "patient_ez_f1", "patient_balanced_accuracy", "patient_auprc_nez", "patient_auprc_ez"]
    rows = []
    for metric in metrics:
        values = patient[metric].to_numpy(dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size == 0:
            rows.append({"metric": metric, "mean": float("nan"), "ci_lower_2_5": float("nan"), "ci_upper_97_5": float("nan"), "bootstrap_samples": samples, "n_patients": 0})
            continue
        draws = np.asarray([rng.choice(values, size=len(values), replace=True).mean() for _ in range(samples)])
        rows.append({"metric": metric, "mean": float(values.mean()), "ci_lower_2_5": float(np.quantile(draws, .025)), "ci_upper_97_5": float(np.quantile(draws, .975)), "bootstrap_samples": samples, "n_patients": int(values.size)})
    return pd.DataFrame(rows)


def aggregate(run_root: str | Path, output_dir: str | Path) -> dict[str, object]:
    root, destination = Path(run_root), Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    frames = [_read_seed(root, seed) for seed in SEEDS]
    _assert_alignment(frames)
    ensemble = frames[0][KEY + ["center", "label_nez", "label_ez", "valid", *CP, "cp_feature_valid"]].copy()
    for seed, frame in zip(SEEDS, frames):
        ensemble[f"standardized_nez_logit_seed_{seed}"] = frame["standardized_nez_logit"].to_numpy(dtype=np.float64)
        ensemble[f"predicted_threshold_seed_{seed}"] = frame["predicted_patient_threshold"].to_numpy(dtype=np.float64)
    ensemble["ensemble_standardized_nez_logit"] = np.mean([frame["standardized_nez_logit"].to_numpy(dtype=np.float64) for frame in frames], axis=0)
    ensemble["ensemble_predicted_threshold"] = np.mean([frame["predicted_patient_threshold"].to_numpy(dtype=np.float64) for frame in frames], axis=0)
    ensemble["standardized_nez_logit"] = ensemble["ensemble_standardized_nez_logit"]
    ensemble["predicted_patient_threshold"] = ensemble["ensemble_predicted_threshold"]
    ensemble["predicted_nez"] = (ensemble["ensemble_standardized_nez_logit"] >= ensemble["ensemble_predicted_threshold"]).astype(int)
    ensemble["predicted_ez"] = 1 - ensemble["predicted_nez"]
    ensemble["decision_rule"] = "patient_adaptive_standardized_nez_threshold"
    ensemble["true_count_used_for_prediction"] = False
    ensemble["oracle_threshold_used_for_prediction"] = False
    records = _records(ensemble)
    summary, enriched = _summarize_prediction_records(records)
    summary.update({
        "method": "N8F_CANE_PATH_CP_NEZ_80", "analysis_status": "posthoc_sensitivity_not_primary",
        "classification_threshold": float("nan"), "threshold_source": "cross_fitted_patient_adaptive_head",
        "decision_rule": "patient_adaptive_standardized_nez_threshold", "ensemble_seeds": list(SEEDS),
        "test_based_seed_selection": False, "test_label_ensemble_weighting": False,
    })
    patient_rows = []
    for record in enriched:
        valid = np.asarray(record["channel_mask"], dtype=bool)
        labels = np.asarray(record["labels_nez"])[valid].astype(int)
        score = np.asarray(record["score_nez"])[valid]
        patient_rows.append({
            "outer_fold": record["outer_fold"], "subject_id": record["subject_id"], "center": record["center"],
            "n_channels": int(valid.sum()), "true_nez_count": int(labels.sum()), "true_ez_count": int((1-labels).sum()),
            "predicted_nez_count": int(np.asarray(record["predicted_nez_mask"])[valid].sum()),
            "predicted_ez_count": int(np.asarray(record["predicted_ez_mask"])[valid].sum()),
            "predicted_threshold": float(ensemble.loc[ensemble.subject_id == record["subject_id"], "ensemble_predicted_threshold"].iloc[0]),
            "patient_macro_f1": record["patient_macro_f1"], "patient_nez_f1": record["patient_nez_f1"],
            "patient_ez_f1": record["patient_ez_f1"], "patient_balanced_accuracy": record["patient_balanced_accuracy"],
            "patient_auprc_nez": float(average_precision_score(labels, score)) if np.unique(labels).size > 1 else float("nan"),
            "patient_auprc_ez": float(average_precision_score(1-labels, 1-score)) if np.unique(labels).size > 1 else float("nan"),
            "decision_rule": record["decision_rule"], "true_count_used_for_prediction": False,
        })
    patient = pd.DataFrame(patient_rows)
    fold = patient.groupby("outer_fold", as_index=False).mean(numeric_only=True)
    center = patient.groupby("center", as_index=False).mean(numeric_only=True)
    pd.DataFrame([summary]).to_csv(destination / "heldout_summary_cane_path_cp.csv", index=False)
    (destination / "heldout_summary_cane_path_cp.json").write_text(json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8")
    fold.to_csv(destination / "heldout_fold_summary.csv", index=False)
    center.to_csv(destination / "heldout_center_summary.csv", index=False)
    patient.to_csv(destination / "heldout_patient_predictions.csv", index=False)
    ensemble.to_csv(destination / "heldout_channel_predictions.csv", index=False)
    _bootstrap(patient).to_csv(destination / "bootstrap_ci.csv", index=False)
    for fold_idx in sorted(ensemble.outer_fold.unique()):
        audit = {"outer_fold": int(fold_idx), "seeds": list(SEEDS), "key_alignment": "passed", "test_based_seed_selection": False, "test_label_weighting": False}
        (destination / f"cane_fold_{int(fold_idx)}_ensemble_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    report_dir = destination.parent.parent / "reports"; report_dir.mkdir(parents=True, exist_ok=True)
    report = f"""# Task 1 CANE-PATH-CP Sensitivity80 Report

## Analysis status

POST-HOC 80-PATIENT SENSITIVITY ANALYSIS. NOT THE FROZEN PRIMARY OLD-90 COHORT.

## Main result

The fixed three-seed ensemble produced patient macro-F1 `{summary.get('patient_macro_f1', float('nan')):.6f}`. This value is reported as observed and is not guaranteed to exceed 0.70.

## Method and leakage controls

The model uses NEZ=1/EZ=0, bounded clean-NEZ, multi-seizure and offline ridge-VAR proxy residuals, then a cross-fitted patient-adaptive threshold. PATH targets use outer-train inner-OOF predictions only. Outer-test labels do not train ranking, PATH, or ensemble weights.

## Limitations

The 80-patient cohort is post-hoc, observed labels are not absolute pathological truth, and ridge-VAR is a directed-influence proxy rather than validated causal identification or TFCCM. The method requires confirmation on the frozen 90-patient primary cohort.
"""
    (report_dir / "TASK1_CANE_PATH_CP_SENSITIVITY80_REPORT.md").write_text(report, encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate fixed CANE-PATH-CP seeds without test-based selection.")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(aggregate(args.run_root, args.output_dir), indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
