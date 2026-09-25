"""Patient-level bootstrap confidence intervals for completed LOCO runs."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


CENTERS = ("hup", "lzu", "multicenter", "pediatric")
MODELS = ("PRQ-Net", "BCR-Net", "CDEL")
METRICS = (
    "patient_macro_f1",
    "patient_ez_f1",
    "patient_nez_f1",
    "patient_accuracy",
    "patient_ez_auroc",
)
MODEL_SCORE_COLUMNS = {
    "PRQ-Net": "p2_score_nez",
    "BCR-Net": "v3_score_nez",
    "CDEL": "fused_score_nez",
}


def _canonical_model(frame: pd.DataFrame, source: Path) -> pd.DataFrame:
    data = frame.copy()
    if "model" not in data.columns:
        if "experiment" not in data.columns:
            raise ValueError(f"LOCO patient file lacks experiment/model: {source}")
        data = data.rename(columns={"experiment": "model"})
    return data


def _assert_formal(frame: pd.DataFrame, *, source: Path) -> None:
    if "analysis_status" in frame and frame["analysis_status"].astype(str).str.contains("DIAGNOSTIC", case=False, na=False).any():
        raise ValueError(f"Diagnostic rows are forbidden in formal LOCO input: {source}")
    if "true_count_used_for_prediction" in frame:
        values = frame["true_count_used_for_prediction"].astype(str).str.lower()
        if values.isin(("true", "1", "yes")).any():
            raise ValueError(f"True-K rows are forbidden in formal LOCO input: {source}")
    if "formal_prediction" in frame and not frame["formal_prediction"].astype(bool).all():
        raise ValueError(f"Non-formal prediction rows are forbidden in LOCO input: {source}")


def _backfill_accuracy(frame: pd.DataFrame, *, run_dir: Path, source: Path) -> pd.DataFrame:
    if "patient_accuracy" in frame.columns:
        return frame
    ledger_path = run_dir / "ledgers" / "oof_channel_predictions.csv"
    if not ledger_path.is_file():
        raise ValueError(f"LOCO patient file lacks patient_accuracy and formal ledger is absent: {source}")
    ledger = pd.read_csv(ledger_path)
    required = {"subject_id", "label_nez", *MODEL_SCORE_COLUMNS.values()}
    missing = sorted(required - set(ledger.columns))
    if missing:
        raise ValueError(f"Cannot backfill LOCO patient_accuracy; ledger missing {missing}: {ledger_path}")
    if "selected_validation_threshold" not in frame.columns:
        raise ValueError(f"Cannot backfill LOCO patient_accuracy without validation threshold: {source}")
    rows = []
    for patient in frame.itertuples(index=False):
        model = str(patient.model)
        if model not in MODEL_SCORE_COLUMNS:
            continue
        subject = str(patient.subject_id)
        channels = ledger.loc[ledger["subject_id"].astype(str) == subject]
        if channels.empty:
            raise ValueError(f"No formal ledger rows for LOCO patient {subject}: {ledger_path}")
        threshold = float(getattr(patient, "selected_validation_threshold"))
        score = channels[MODEL_SCORE_COLUMNS[model]].to_numpy(float)
        predicted = score >= threshold
        rows.append({"model": model, "subject_id": subject, "patient_accuracy": float((predicted == channels["label_nez"].to_numpy(int)).mean())})
    accuracy = pd.DataFrame(rows)
    return frame.merge(accuracy, on=["model", "subject_id"], how="left", validate="one_to_one")


def load_loco_patient_metrics(root: str | Path, *, centers: Iterable[str] = CENTERS, seeds: Iterable[int] = (42, 52, 62)) -> pd.DataFrame:
    """Load only formal, complete LOCO patient metrics and validate their contract."""
    root = Path(root)
    expected_centers, expected_seeds = tuple(centers), tuple(int(seed) for seed in seeds)
    frames: list[pd.DataFrame] = []
    for center in expected_centers:
        expected_subjects: set[str] | None = None
        for seed in expected_seeds:
            run_dir = root / "loco" / f"heldout_{center}" / f"seed_{seed}"
            source = run_dir / "metrics" / "patient_level.csv"
            if not source.is_file():
                raise FileNotFoundError(f"Missing completed LOCO formal patient metrics: {source}")
            frame = _canonical_model(pd.read_csv(source), source)
            _assert_formal(frame, source=source)
            frame = _backfill_accuracy(frame, run_dir=run_dir, source=source)
            required = {"model", "subject_id", "center", "outer_fold", *METRICS}
            missing = sorted(required - set(frame.columns))
            if missing:
                raise ValueError(f"LOCO patient metrics missing {missing}: {source}")
            frame = frame.loc[frame["model"].isin(MODELS)].copy()
            found = set(frame["model"].astype(str))
            if found != set(MODELS):
                raise ValueError(f"Incomplete formal model set at {source}: expected={MODELS}, found={sorted(found)}")
            if frame.duplicated(["model", "subject_id"]).any():
                raise ValueError(f"Duplicate LOCO patient/model rows: {source}")
            if set(frame["center"].astype(str)) != {center}:
                raise ValueError(f"Cross-center patient rows in {source}; expected only {center}")
            subjects_by_model = {model: set(group["subject_id"].astype(str)) for model, group in frame.groupby("model", sort=False)}
            if len({frozenset(value) for value in subjects_by_model.values()}) != 1:
                raise ValueError(f"Formal models do not cover the same LOCO patients: {source}")
            subjects = next(iter(subjects_by_model.values()))
            if not subjects:
                raise ValueError(f"No held-out patients in {source}")
            if expected_subjects is None:
                expected_subjects = subjects
            elif subjects != expected_subjects:
                raise ValueError(f"LOCO seed patient mismatch for center={center}, seed={seed}")
            frame["training_seed"] = seed
            frame["held_out_center"] = center
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def patient_seed_average(frame: pd.DataFrame, *, seeds: Iterable[int]) -> pd.DataFrame:
    """Average each patient's metrics across seeds before any bootstrap draw."""
    expected = {int(seed) for seed in seeds}
    keys = ["held_out_center", "model", "subject_id", "center", "outer_fold"]
    counts = frame.groupby(keys, sort=False)["training_seed"].nunique()
    if not (counts == len(expected)).all():
        raise ValueError("Every LOCO patient/model must occur once for every requested seed before averaging")
    if set(frame["training_seed"].astype(int)) != expected:
        raise ValueError("LOCO input seed set is incomplete")
    return frame.groupby(keys, as_index=False, sort=False)[list(METRICS)].mean()


def patient_bootstrap_summary(frame: pd.DataFrame, *, repeats: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Percentile CIs with patient, never channel or patient-seed, resampling."""
    if repeats < 1:
        raise ValueError("bootstrap repeats must be positive")
    rows = []
    index = 0
    for (center, model), group in frame.groupby(["held_out_center", "model"], sort=True):
        if group["subject_id"].duplicated().any():
            raise ValueError("Bootstrap input contains duplicate patients")
        n_patients = len(group)
        for metric in METRICS:
            values = group[metric].to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(f"Non-finite {metric} for center={center}, model={model}")
            rng = np.random.default_rng(int(seed) + index)
            draws = values[rng.integers(0, n_patients, size=(repeats, n_patients))].mean(axis=1)
            rows.append({
                "held_out_center": center,
                "model": model,
                "metric": metric,
                "estimate": float(values.mean()),
                "ci_low": float(np.quantile(draws, 0.025)),
                "ci_high": float(np.quantile(draws, 0.975)),
                "n_unique_patients": int(n_patients),
                "n_training_seeds_averaged": 3,
                "bootstrap_repeats": int(repeats),
                "bootstrap_seed": int(seed),
                "resampling_unit": "patient_after_three_seed_mean",
                "analysis_status": "FORMAL_LOCO_PATIENT_BOOTSTRAP",
            })
            index += 1
    return pd.DataFrame(rows)


def write_loco_bootstrap_report(*, root: str | Path, output_dir: str | Path, seeds: Iterable[int], repeats: int, seed: int) -> dict:
    patients = load_loco_patient_metrics(root, seeds=seeds)
    averaged = patient_seed_average(patients, seeds=seeds)
    summary = patient_bootstrap_summary(averaged, repeats=repeats, seed=seed)
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output / "loco_patient_bootstrap_ci.csv", index=False)
    summary.sort_values(["held_out_center", "model", "metric"]).to_csv(output / "loco_patient_bootstrap_by_center.csv", index=False)
    report = [
        "# LOCO Patient Bootstrap Confidence Intervals",
        "",
        "Formal LOCO predictions were reused without retraining. Each patient's metrics were averaged across seeds 42, 52, and 62 before 2,000 patient-level bootstrap resamples.",
        "",
        "- Methods: PRQ-Net, BCR-Net, CDEL",
        "- Held-out centers: hup, lzu, multicenter, pediatric",
        "- Metrics: Macro-F1, EZ-F1, NEZ-F1, Accuracy, EZ-AUROC",
        "- True-K diagnostics, missing models, duplicate patients, cross-center rows, and incomplete seeds are rejected.",
        "",
        f"Completed rows: {len(summary)}; unique held-out patients: {averaged['subject_id'].nunique()}.",
    ]
    (output / "LOCO_PATIENT_BOOTSTRAP_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return {"status": "passed", "output_dir": str(output), "n_rows": int(len(summary)), "n_patients": int(averaged["subject_id"].nunique())}
