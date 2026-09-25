from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from outcome_hifos.bootstrap import paired_patient_bootstrap
from outcome_hifos.metrics import compute_patient_metrics
from outcome_hifos.reports.shortcut import _cross_validated_metadata_probability, compute_shortcut_risk
from baseline_common.reporting import frame_to_markdown


def _macro_f1(target: np.ndarray, predicted: np.ndarray) -> float:
    return float(f1_score(target, predicted, average="macro", labels=[0, 1], zero_division=0))


def _bootstrap_summary(oof: pd.DataFrame, samples: int) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    grouped = {(str(model), int(seed)): group.sort_values("subject_id") for (model, seed), group in oof.groupby(["model", "seed"])}
    simple_names = {"elasticnet", "rbf_svm", "random_forest", "lightgbm"}
    candidate_keys = [key for key in grouped if key[0].lower() in simple_names]
    if not candidate_keys:
        candidate_keys = [key for key in grouped if key[0].lower() == "majority"]
    candidate_scores: dict[str, list[float]] = {}
    for key in candidate_keys:
        candidate_scores.setdefault(key[0], []).append(
            _macro_f1(grouped[key]["outcome"].to_numpy(), grouped[key]["predicted"].to_numpy())
        )
    reference_model = max(candidate_scores, key=lambda name: (float(np.mean(candidate_scores[name])), name)) if candidate_scores else None
    rows = []
    for (model, seed), group in grouped.items():
        y = group["outcome"].to_numpy(dtype=int)
        predicted = group["predicted"].to_numpy(dtype=int)
        estimates = []
        for _ in range(int(samples)):
            index = rng.integers(0, len(group), len(group))
            estimates.append(_macro_f1(y[index], predicted[index]))
        lower, upper = np.percentile(estimates, [2.5, 97.5])
        row = {
            "model": model, "seed": seed, "metric": "macro_f1", "estimate": _macro_f1(y, predicted),
            "ci_lower": float(lower), "ci_upper": float(upper), "bootstrap_samples": int(samples),
            "paired_reference": "", "paired_vs_best_simple_delta": np.nan,
            "paired_ci_lower": np.nan, "paired_ci_upper": np.nan,
        }
        reference_key = (reference_model, seed) if (reference_model, seed) in grouped else next(
            (key for key in grouped if key[0] == reference_model), None
        )
        if reference_key is not None:
            reference = grouped[reference_key][["subject_id", "outcome", "predicted"]].rename(columns={"predicted": "reference_predicted"})
            aligned = group[["subject_id", "outcome", "predicted"]].merge(reference, on=["subject_id", "outcome"], validate="one_to_one")
            if len(aligned) == len(group) == len(reference):
                interval = paired_patient_bootstrap(
                    aligned["outcome"].to_numpy(dtype=int), aligned["predicted"].to_numpy(dtype=int),
                    aligned["reference_predicted"].to_numpy(dtype=int), _macro_f1, samples=int(samples), seed=42,
                )
                row.update({
                    "paired_reference": reference_key[0], "paired_vs_best_simple_delta": interval.estimate,
                    "paired_ci_lower": interval.lower, "paired_ci_upper": interval.upper,
                })
        rows.append(row)
    return pd.DataFrame(rows)


def _shortcut_audit(oof: pd.DataFrame, output: Path, main_best: float) -> pd.DataFrame:
    required = {"subject_id", "center", "outcome", "outer_fold", "n_channels", "n_seizures"}
    missing = required - set(oof)
    if missing:
        raise ValueError(f"Task 2 shortcut audit is missing OOF metadata columns: {sorted(missing)}")
    base = oof.groupby("subject_id", as_index=False).agg(
        center=("center", "first"), outcome=("outcome", "first"), outer_fold=("outer_fold", "first"),
        n_channels=("n_channels", "first"), n_seizures=("n_seizures", "first"),
    )
    if base[["n_channels", "n_seizures"]].isna().any().any():
        raise ValueError("Task 2 shortcut count metadata is incomplete.")
    manifest = base[["subject_id", "center", "outcome", "n_channels", "n_seizures"]].rename(
        columns={"outcome": "outcome_label", "n_channels": "channel_count", "n_seizures": "seizure_count"}
    )
    ledger = base[["subject_id", "outer_fold"]].rename(columns={"outer_fold": "fold_idx"})
    rows = []
    definitions = {
        "center_only": ["center"], "channel_count_only": ["channel_count"],
        "seizure_count_only": ["seizure_count"],
    }
    majority_probability = np.full(len(manifest), float(manifest["outcome_label"].mean()))
    majority = compute_patient_metrics(manifest["outcome_label"].to_numpy(), majority_probability, 0.5)
    rows.append({"shortcut": "majority", "status": "evaluated", **majority.values})
    for name, columns in definitions.items():
        prediction = _cross_validated_metadata_probability(manifest, ledger, columns)
        bundle = compute_patient_metrics(prediction["outcome"].to_numpy(), prediction["probability"].to_numpy(), 0.5)
        rows.append({"shortcut": name, "status": "evaluated", **bundle.values})
    metrics = pd.DataFrame(rows)
    audit_dir = output / "shortcut_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(audit_dir / "task2_shortcut_metrics.csv", index=False)
    risk, considered = compute_shortcut_risk(main_best, metrics)
    (audit_dir / "task2_shortcut_audit.json").write_text(
        json.dumps({"shortcut_risk": risk, "main_best_macro_f1": main_best, "shortcut_models_considered": considered}, indent=2),
        encoding="utf-8",
    )
    return metrics


def summarize_task2_baselines(
    oof: pd.DataFrame,
    output_dir: str | Path,
    failures: pd.DataFrame,
    *,
    bootstrap_samples: int = 2000,
) -> pd.DataFrame:
    output = Path(output_dir)
    metrics_dir = output / "metrics"
    reports_dir = output / "reports"
    comparison_dir = output / "comparison"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    comparison_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for (model, seed), group in oof.groupby(["model", "seed"], sort=False):
        bundle = compute_patient_metrics(
            group["outcome"].to_numpy(),
            group["calibrated_probability"].to_numpy(),
            predicted=group["predicted"].to_numpy(),
        )
        tn, fp = bundle.confusion_matrix[0]
        fn, tp = bundle.confusion_matrix[1]
        rows.append({"model": model, "seed": int(seed), **bundle.values, "TN": tn, "FP": fp, "FN": fn, "TP": tp, "n_patients": len(group)})
    by_seed = pd.DataFrame(rows)
    by_seed.to_csv(metrics_dir / "task2_by_seed.csv", index=False)
    by_seed.to_csv(comparison_dir / "task2_baseline_main_table.csv", index=False)
    (comparison_dir / "task2_baseline_main_table.md").write_text(frame_to_markdown(by_seed) + "\n", encoding="utf-8")
    fold_rows = []
    center_rows = []
    for keys, destination in ((("model", "seed", "outer_fold"), fold_rows), (("model", "seed", "center"), center_rows)):
        for values, group in oof.groupby(list(keys), sort=False):
            bundle = compute_patient_metrics(group["outcome"].to_numpy(), group["calibrated_probability"].to_numpy(), predicted=group["predicted"].to_numpy())
            destination.append({**dict(zip(keys, values)), **bundle.values, "n_patients": len(group), "undefined_reasons": str(bundle.undefined_reasons)})
    pd.DataFrame(fold_rows).to_csv(metrics_dir / "task2_by_fold.csv", index=False)
    pd.DataFrame(center_rows).to_csv(metrics_dir / "task2_by_center.csv", index=False)
    bootstrap = _bootstrap_summary(oof, bootstrap_samples)
    bootstrap.to_csv(comparison_dir / "task2_bootstrap.csv", index=False)
    shortcut = _shortcut_audit(oof, output, float(by_seed["macro_f1"].max()))
    shortcut_risk = bool(json.loads((output / "shortcut_audit" / "task2_shortcut_audit.json").read_text(encoding="utf-8"))["shortcut_risk"])
    report = [
        "# Task 2 Baseline Report", "",
        "Primary cohort: frozen Task 1 old-90 success patients plus all valid failure patients.", "",
        "Positive class: success=1. Negative class: failure=0. Primary metric: patient-level macro-F1.", "",
        f"Completed model/seed rows: {len(by_seed)}. Recorded failures: {len(failures)}.", "",
        f"Patient bootstrap samples: {int(bootstrap_samples)}. Shortcut risk: {shortcut_risk}.", "",
        frame_to_markdown(by_seed),
    ]
    (reports_dir / "TASK2_BASELINE_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return by_seed


__all__ = ["summarize_task2_baselines"]
