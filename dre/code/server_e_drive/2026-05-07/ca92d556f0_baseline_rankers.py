from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .dynamic_dataset import DynamicPatientSample
from .evaluate import (
    apply_selection_mode,
    calibrate_selection,
    raw_patient_prediction_record,
    summarize_records,
)


BASELINE_NAMES = [
    "random",
    "high_gamma_delta",
    "line_length_delta",
    "broadband_or_rms_delta",
    "combined_delta",
]


def run_all_centers_baselines(
    samples: Sequence[DynamicPatientSample],
    splits: Sequence[dict[str, Any]],
    output_dir: str | Path,
    random_repeats: int = 100,
    seed: int = 42,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_by_subject = {sample.subject_id: sample for sample in samples}
    summary_rows = []
    details: dict[str, Any] = {}
    for baseline_name in BASELINE_NAMES:
        baseline_dir = output_dir / baseline_name
        baseline_dir.mkdir(parents=True, exist_ok=True)
        repeats = random_repeats if baseline_name == "random" else 1
        repeat_summaries = []
        for repeat_idx in range(repeats):
            fold_summaries = []
            for fold in splits:
                val_samples = [sample_by_subject[s] for s in fold.get("val_subjects", []) if s in sample_by_subject]
                test_samples = [sample_by_subject[s] for s in fold.get("test_subjects", []) if s in sample_by_subject]
                val_raw = _baseline_records(val_samples, baseline_name, seed + repeat_idx)
                test_raw = _baseline_records(test_samples, baseline_name, seed + repeat_idx)
                calibration = calibrate_selection(val_raw)
                oracle = apply_selection_mode(test_raw, "oracle_true_count_topk", calibration)
                calibrated = apply_selection_mode(test_raw, "validation_calibrated_topk", calibration)
                fold_summary = {
                    "fold_idx": int(fold.get("fold_idx", 0)),
                    "oracle_true_count_topk": summarize_records(oracle),
                    "validation_calibrated_topk": summarize_records(calibrated),
                    "calibration": calibration,
                }
                fold_summaries.append(fold_summary)
            repeat_summary = _aggregate_baseline_folds(baseline_name, fold_summaries)
            repeat_summaries.append(repeat_summary)
        final = _aggregate_repeats(baseline_name, repeat_summaries)
        details[baseline_name] = {"repeats": repeat_summaries, "summary": final}
        _write_json(details[baseline_name], baseline_dir / "summary.json")
        for selection_mode, metrics in final.items():
            if not isinstance(metrics, dict):
                continue
            summary_rows.append({"baseline_name": baseline_name, "selection_mode": selection_mode, **metrics})
    _write_summary_csv(summary_rows, output_dir / "baseline_summary.csv")
    _write_json({"baselines": details, "summary_rows": summary_rows}, output_dir / "baseline_summary.json")
    return {"baselines": details, "summary_rows": summary_rows}


def _baseline_records(
    samples: Sequence[DynamicPatientSample],
    baseline_name: str,
    seed: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(int(seed))
    records = []
    for sample in samples:
        scores = _baseline_scores(sample, baseline_name, rng)
        channel_mask = _patient_valid_mask(sample)
        records.append(
            raw_patient_prediction_record(
                subject_id=sample.subject_id,
                center_id=sample.center_id,
                patient_id=sample.patient_id,
                channel_names=sample.canonical_channels,
                scores=scores,
                labels=sample.labels,
                channel_mask=channel_mask,
                predicted_count=max(1.0, float(channel_mask.sum()) * 0.05),
            )
        )
    return records


def _baseline_scores(sample: DynamicPatientSample, baseline_name: str, rng: np.random.Generator) -> np.ndarray:
    if baseline_name == "random":
        scores = rng.random(len(sample.canonical_channels)).astype(np.float32)
        scores[~_patient_valid_mask(sample)] = -np.inf
        return scores
    if baseline_name == "combined_delta":
        parts = [
            _feature_delta_score(sample, "high_gamma_power_z"),
            _feature_delta_score(sample, "line_length_z"),
            _feature_delta_score(sample, "broadband_power_z", fallback="rms_z"),
        ]
        z_parts = [_patient_robust_z(part, _patient_valid_mask(sample)) for part in parts]
        return np.nanmean(np.stack(z_parts, axis=0), axis=0).astype(np.float32)
    if baseline_name == "high_gamma_delta":
        return _feature_delta_score(sample, "high_gamma_power_z")
    if baseline_name == "line_length_delta":
        return _feature_delta_score(sample, "line_length_z")
    if baseline_name == "broadband_or_rms_delta":
        return _feature_delta_score(sample, "broadband_power_z", fallback="rms_z")
    raise ValueError(f"Unsupported baseline: {baseline_name}")


def _feature_delta_score(sample: DynamicPatientSample, feature_name: str, fallback: str | None = None) -> np.ndarray:
    feature_names = sample.feature_names or []
    idx = _feature_index(feature_names, feature_name)
    if idx is None and fallback is not None:
        idx = _feature_index(feature_names, fallback)
    if idx is None:
        return np.zeros(len(sample.canonical_channels), dtype=np.float32)
    per_seizure = []
    for seizure in sample.seizure_samples:
        onset = (seizure.time_sec >= 0.0) & (seizure.time_sec <= 2.0)
        if not np.any(onset):
            continue
        per_seizure.append(np.nanmedian(seizure.features[onset, :, idx], axis=0))
    if not per_seizure:
        return np.zeros(len(sample.canonical_channels), dtype=np.float32)
    scores = np.nanmedian(np.stack(per_seizure, axis=0), axis=0).astype(np.float32)
    scores[~_patient_valid_mask(sample)] = -np.inf
    return np.nan_to_num(scores, nan=0.0, neginf=-1e6, posinf=1e6)


def _patient_valid_mask(sample: DynamicPatientSample) -> np.ndarray:
    if not sample.seizure_samples:
        return np.zeros(len(sample.canonical_channels), dtype=bool)
    masks = np.stack([seizure.channel_mask for seizure in sample.seizure_samples], axis=0)
    return masks.any(axis=0)


def _feature_index(feature_names: Sequence[str], target: str) -> int | None:
    for idx, name in enumerate(feature_names):
        if str(name) == target:
            return idx
    return None


def _patient_robust_z(scores: np.ndarray, mask: np.ndarray) -> np.ndarray:
    result = np.asarray(scores, dtype=np.float32).copy()
    valid = result[mask & np.isfinite(result)]
    if valid.size == 0:
        return np.zeros_like(result)
    med = np.median(valid)
    iqr = np.percentile(valid, 75) - np.percentile(valid, 25)
    result = (result - med) / max(float(iqr), 1e-5)
    result[~mask] = -np.inf
    return np.nan_to_num(result, nan=0.0, neginf=-1e6, posinf=1e6).astype(np.float32)


def _aggregate_baseline_folds(baseline_name: str, fold_summaries: Sequence[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"baseline_name": baseline_name, "folds": list(fold_summaries)}
    for mode in ["oracle_true_count_topk", "validation_calibrated_topk"]:
        values = [fold[mode] for fold in fold_summaries]
        result[mode] = _mean_metric_dict(values)
    return result


def _aggregate_repeats(baseline_name: str, repeat_summaries: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if baseline_name != "random":
        return {key: value for key, value in repeat_summaries[0].items() if key != "folds"}
    result: dict[str, Any] = {"baseline_name": baseline_name}
    for mode in ["oracle_true_count_topk", "validation_calibrated_topk"]:
        metrics = [repeat[mode] for repeat in repeat_summaries]
        mean = _mean_metric_dict(metrics)
        std = _std_metric_dict(metrics)
        result[mode] = {**mean, **{f"{key}_std": value for key, value in std.items()}}
    return result


def _mean_metric_dict(items: Sequence[dict[str, Any]]) -> dict[str, float]:
    keys = sorted({key for item in items for key in item if key.startswith("macro_") or key == "selection_score"})
    result = {}
    for key in keys:
        values = np.asarray([item.get(key, np.nan) for item in items], dtype=np.float32)
        result[key] = float(np.nanmean(values)) if np.any(np.isfinite(values)) else float("nan")
    return result


def _std_metric_dict(items: Sequence[dict[str, Any]]) -> dict[str, float]:
    keys = sorted({key for item in items for key in item if key.startswith("macro_") or key == "selection_score"})
    result = {}
    for key in keys:
        values = np.asarray([item.get(key, np.nan) for item in items], dtype=np.float32)
        result[key] = float(np.nanstd(values)) if np.any(np.isfinite(values)) else float("nan")
    return result


def _write_summary_csv(rows: Sequence[dict[str, Any]], path: Path) -> None:
    fields = [
        "baseline_name",
        "selection_mode",
        "macro_AUC",
        "macro_AUC_PR",
        "macro_TOP1_HIT",
        "macro_TOP3_HIT",
        "macro_TOPK_RECALL",
        "macro_F1",
        "macro_PREC",
        "macro_REC",
        "macro_COUNT_BIAS",
        "macro_ABS_COUNT_BIAS_RATIO",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fout:
        json.dump(payload, fout, ensure_ascii=False, indent=2, default=_json_default)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


__all__ = ["BASELINE_NAMES", "run_all_centers_baselines"]
