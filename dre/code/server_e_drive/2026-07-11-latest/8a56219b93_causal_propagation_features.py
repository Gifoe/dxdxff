"""Offline ridge-VAR directed-influence proxies for CANE-PATH-CP.

These features are low-dimensional propagation proxies, not evidence of true
causality. Raw waveforms are consumed only by the offline builder.
"""

from __future__ import annotations

import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from .dual_view_data import _channel_names, _field, _record_key, _sample, _window_centers
from .raw_brainbert_data import normalize_channel_name


CAUSAL_FEATURE_NAMES = (
    "cp_preictal_suppression_rank",
    "cp_suppression_release_shift",
    "cp_causal_early_activation_rank",
    "cp_early_source_rank",
    "cp_propagation_persistence",
    "cp_cross_seizure_source_consistency",
)


def percentile_rank(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size <= 1:
        return np.full(values.shape, 0.5, dtype=np.float64)
    return (rankdata(values, method="average") - 1.0) / float(values.size - 1)


def robust_channel_normalize(window: np.ndarray) -> np.ndarray:
    signal = np.asarray(window, dtype=np.float64)
    if signal.ndim != 2:
        raise ValueError(f"raw window must be [C,T], got {signal.shape}")
    finite = np.isfinite(signal)
    safe = np.where(finite, signal, np.nan)
    median = np.nanmedian(safe, axis=1, keepdims=True)
    median = np.where(np.isfinite(median), median, 0.0)
    centered = np.where(finite, signal - median, 0.0)
    mad = np.nanmedian(np.where(finite, np.abs(signal - median), np.nan), axis=1, keepdims=True)
    scale = np.maximum(1.4826 * np.where(np.isfinite(mad), mad, 0.0), 1e-6)
    return centered / scale


def fit_ridge_var_directed_influence(window: np.ndarray, ridge_alpha: float = 0.10) -> dict[str, Any]:
    """Fit VAR(1), returning A[target,source] and numerical diagnostics."""
    signal = robust_channel_normalize(window)
    channels, samples = signal.shape
    minimum = max(64, 2 * channels)
    if samples < minimum:
        return {"valid": False, "invalid_reason": "insufficient_samples", "n_channels": channels, "n_samples": samples}
    x = signal[:, :-1].T
    y = signal[:, 1:].T
    gram = x.T @ x
    ridge_lambda = float(ridge_alpha) * float(np.trace(gram)) / max(channels, 1)
    ridge_lambda = max(ridge_lambda, 1e-8)
    regularized = gram + ridge_lambda * np.eye(channels, dtype=np.float64)
    try:
        coefficients = np.linalg.solve(regularized, x.T @ y)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.pinv(regularized) @ (x.T @ y)
    matrix = coefficients.T
    np.fill_diagonal(matrix, 0.0)
    absolute = np.abs(matrix)
    finite = bool(np.isfinite(matrix).all())
    off_diagonal = ~np.eye(channels, dtype=bool)
    stability = np.nan
    stability_reason = "insufficient_transitions"
    if x.shape[0] >= 8 and channels >= 2:
        matrices = []
        for indices in (np.arange(0, x.shape[0], 2), np.arange(1, x.shape[0], 2)):
            gx = x[indices].T @ x[indices]
            lam = max(float(ridge_alpha) * float(np.trace(gx)) / max(channels, 1), 1e-8)
            reg = gx + lam * np.eye(channels)
            try:
                b = np.linalg.solve(reg, x[indices].T @ y[indices]).T
            except np.linalg.LinAlgError:
                b = (np.linalg.pinv(reg) @ (x[indices].T @ y[indices])).T
            np.fill_diagonal(b, 0.0)
            matrices.append(np.abs(b)[off_diagonal])
        if np.std(matrices[0]) > 0 and np.std(matrices[1]) > 0:
            stability = float(spearmanr(matrices[0], matrices[1]).statistic)
            stability_reason = ""
    out_strength = absolute.sum(axis=0)
    in_strength = absolute.sum(axis=1)
    return {
        "valid": finite,
        "invalid_reason": "" if finite else "nonfinite_var_matrix",
        "A": matrix,
        "out_strength": out_strength,
        "in_strength": in_strength,
        "source_rank": percentile_rank(out_strength - in_strength),
        "suppression_rank": percentile_rank(in_strength - out_strength),
        "n_channels": channels,
        "n_samples": samples,
        "ridge_lambda": ridge_lambda,
        "condition_number": float(np.linalg.cond(regularized)),
        "matrix_finite": finite,
        "coefficient_abs_mean": float(absolute[off_diagonal].mean()) if channels > 1 else 0.0,
        "coefficient_abs_max": float(absolute.max(initial=0.0)),
        "var_stability_spearman": stability,
        "stability_invalid_reason": stability_reason,
    }


def select_causal_windows(
    centers: Sequence[float],
    *,
    max_preictal_windows: int = 4,
    max_early_windows: int = 6,
    early_ictal_seconds: float = 30.0,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(centers, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("window centers are required and must be finite")
    pre = np.flatnonzero(values < 0.0)
    pre = pre[np.argsort(values[pre])][-int(max_preictal_windows):]
    early = np.flatnonzero((values >= 0.0) & (values <= float(early_ictal_seconds)))
    early = early[np.argsort(values[early])][:int(max_early_windows)]
    return pre, early


def seizure_causal_features(
    pre_source: np.ndarray,
    pre_suppression: np.ndarray,
    early_source: np.ndarray,
    *,
    activation_cutoff: float = 0.80,
) -> np.ndarray:
    """Return five seizure-channel proxies from window-wise percentile ranks."""
    pre_source = np.asarray(pre_source, dtype=np.float64)
    pre_suppression = np.asarray(pre_suppression, dtype=np.float64)
    early_source = np.asarray(early_source, dtype=np.float64)
    if early_source.ndim != 2 or early_source.shape[0] == 0:
        raise ValueError("At least one early window is required")
    channels = early_source.shape[1]
    pre_src_mean = pre_source.mean(axis=0) if pre_source.shape[0] else np.zeros(channels)
    pre_supp_mean = pre_suppression.mean(axis=0) if pre_suppression.shape[0] else np.zeros(channels)
    early_mean = early_source.mean(axis=0)
    release = np.clip(early_mean - pre_src_mean, -1.0, 1.0)
    first = np.full(channels, early_source.shape[0], dtype=np.float64)
    for channel_idx in range(channels):
        hits = np.flatnonzero(early_source[:, channel_idx] >= float(activation_cutoff))
        if hits.size:
            first[channel_idx] = float(hits[0])
    activation = 1.0 - percentile_rank(first)
    activation[first >= early_source.shape[0]] = 0.0
    early_max = early_source.max(axis=0)
    persistence = (early_source >= float(activation_cutoff)).mean(axis=0)
    return np.stack((pre_supp_mean, release, activation, early_max, persistence), axis=1).astype(np.float32)


def aggregate_patient_causal_features(seizure_features: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    if not seizure_features:
        return np.zeros((0, 6), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    stack = np.stack(seizure_features, axis=0)
    base = np.median(stack, axis=0)
    early_enough = stack[:, :, 2] >= 0.5
    strong_source = stack[:, :, 3] >= 0.80
    consistency = (early_enough & strong_source).mean(axis=0)
    result = np.concatenate((base, consistency[:, None]), axis=1).astype(np.float32)
    return result, np.full(result.shape[0], stack.shape[0], dtype=np.int64)


def _load_cache(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict) or not isinstance(payload.get("run_records"), list):
        raise ValueError(f"Invalid window cache: {path}")
    return payload


def _slice_raw_window(raw_record: Mapping[str, Any], center: float, duration_sec: float) -> np.ndarray | None:
    sample = _sample(raw_record)
    raw = np.asarray(sample.get("raw_waveform"), dtype=np.float32)
    sfreq = float(sample.get("raw_temporal_sfreq", 0.0) or 0.0)
    total_duration = float(sample.get("raw_temporal_duration_sec", 0.0) or 0.0)
    if raw.ndim != 2 or sfreq <= 0 or total_duration <= 0 or duration_sec <= 0:
        return None
    length = int(round(duration_sec * sfreq))
    center_sample = int(round((total_duration / 2.0 + float(center)) * sfreq))
    start = center_sample - length // 2
    end = start + length
    if start < 0 or end > raw.shape[1]:
        return None
    return raw[:, start:end]


def build_causal_propagation_cache(
    raw_window_cache_path: str | Path,
    feature_window_cache_path: str | Path,
    output_path: str | Path,
    audit_output_dir: str | Path,
    retained_subjects: Sequence[str],
    *,
    ridge_alpha: float = 0.10,
    max_preictal_windows: int = 4,
    max_early_windows: int = 6,
    early_ictal_seconds: float = 30.0,
    strict: bool = True,
) -> dict[str, Any]:
    feature_payload = _load_cache(feature_window_cache_path)
    raw_payload = _load_cache(raw_window_cache_path)
    retained = set(map(str, retained_subjects))
    feature_records = [record for record in feature_payload["run_records"] if str(record.get("subject_id")) in retained]
    raw_index = {_record_key(record): record for record in raw_payload["run_records"]}
    if len(raw_index) != len(raw_payload["run_records"]):
        raise ValueError("Raw cache contains duplicate compound record keys")
    by_subject: dict[str, list[tuple[np.ndarray, list[str], dict[str, Any]]]] = defaultdict(list)
    window_audit: list[dict[str, Any]] = []
    cross_seizure_audit: list[dict[str, Any]] = []
    matched_records = matched_channels = total_channels = matched_windows = total_windows = 0
    for feature_record in feature_records:
        subject_id = str(feature_record["subject_id"])
        feature_names = _channel_names(feature_record)
        total_channels += len(feature_names)
        raw_record = raw_index.get(_record_key(feature_record))
        if raw_record is None:
            continue
        matched_records += 1
        raw_names = _channel_names(raw_record)
        raw_name_to_idx = {name: index for index, name in enumerate(raw_names)}
        common = [name for name in feature_names if name in raw_name_to_idx]
        matched_channels += len(common)
        centers = _window_centers(feature_record)
        pre_indices, early_indices = select_causal_windows(
            centers, max_preictal_windows=max_preictal_windows,
            max_early_windows=max_early_windows, early_ictal_seconds=early_ictal_seconds,
        )
        selected_indices = np.concatenate((pre_indices, early_indices))
        total_windows += len(selected_indices) * len(feature_names)
        scale = np.asarray(_field(feature_record, "feature_scale_used_secs", []), dtype=np.float64)
        if scale.size != centers.size:
            scale = np.full(centers.size, 2.0, dtype=np.float64)
        source_rows: dict[int, np.ndarray] = {}
        suppression_rows: dict[int, np.ndarray] = {}
        valid_stabilities: list[float] = []
        for index in selected_indices:
            raw_window = _slice_raw_window(raw_record, float(centers[index]), float(scale[index]))
            if raw_window is None:
                continue
            aligned = raw_window[[raw_name_to_idx[name] for name in common]]
            result = fit_ridge_var_directed_influence(aligned, ridge_alpha=ridge_alpha)
            window_audit.append({"subject_id": subject_id, "record_key": "::".join(_record_key(feature_record)), "window_center": float(centers[index]), **{k: v for k, v in result.items() if k not in {"A", "out_strength", "in_strength", "source_rank", "suppression_rank"}}})
            if result.get("valid"):
                source_rows[int(index)] = np.asarray(result["source_rank"], dtype=np.float64)
                suppression_rows[int(index)] = np.asarray(result["suppression_rank"], dtype=np.float64)
                if np.isfinite(result.get("var_stability_spearman", np.nan)):
                    valid_stabilities.append(float(result["var_stability_spearman"]))
                matched_windows += len(common)
        valid_pre = [idx for idx in pre_indices if int(idx) in source_rows]
        valid_early = [idx for idx in early_indices if int(idx) in source_rows]
        if not valid_early:
            continue
        pre_source = np.stack([source_rows[int(idx)] for idx in valid_pre]) if valid_pre else np.zeros((0, len(common)))
        pre_supp = np.stack([suppression_rows[int(idx)] for idx in valid_pre]) if valid_pre else np.zeros((0, len(common)))
        early_source = np.stack([source_rows[int(idx)] for idx in valid_early])
        local_features = seizure_causal_features(pre_source, pre_supp, early_source)
        full = np.full((len(feature_names), 5), np.nan, dtype=np.float32)
        for local_idx, name in enumerate(common):
            full[feature_names.index(name)] = local_features[local_idx]
        by_subject[subject_id].append((full, feature_names, {
            "n_pre": len(valid_pre), "n_early": len(valid_early),
            "mean_var_stability": float(np.mean(valid_stabilities)) if valid_stabilities else float("nan"),
        }))

    rows: list[dict[str, Any]] = []
    patient_index = feature_payload.get("patient_index", {})
    for subject_id in sorted(retained):
        meta = patient_index.get(subject_id, {})
        canonical = [normalize_channel_name(name) for name in meta.get("canonical_channels", [])]
        seizures = by_subject.get(subject_id, [])
        aligned_seizures: list[np.ndarray] = []
        per_channel: list[list[np.ndarray]] = [[] for _ in canonical]
        stability_by_channel: list[list[float]] = [[] for _ in canonical]
        pre_counts = np.zeros(len(canonical), dtype=np.int64)
        early_counts = np.zeros(len(canonical), dtype=np.int64)
        for features, names, quality in seizures:
            lookup = {name: index for index, name in enumerate(names)}
            aligned = np.full((len(canonical), 5), np.nan, dtype=np.float32)
            for channel_idx, name in enumerate(canonical):
                source_idx = lookup.get(name)
                if source_idx is None or not np.isfinite(features[source_idx]).all():
                    continue
                per_channel[channel_idx].append(features[source_idx])
                aligned[channel_idx] = features[source_idx]
                if np.isfinite(quality["mean_var_stability"]):
                    stability_by_channel[channel_idx].append(float(quality["mean_var_stability"]))
                pre_counts[channel_idx] += int(quality["n_pre"])
                early_counts[channel_idx] += int(quality["n_early"])
            aligned_seizures.append(aligned)
        for left_idx in range(len(aligned_seizures)):
            for right_idx in range(left_idx + 1, len(aligned_seizures)):
                left, right = aligned_seizures[left_idx], aligned_seizures[right_idx]
                for feature_idx, feature_name in enumerate(CAUSAL_FEATURE_NAMES[:5]):
                    valid_pair = np.isfinite(left[:, feature_idx]) & np.isfinite(right[:, feature_idx])
                    rho = float("nan")
                    if valid_pair.sum() >= 3 and np.std(left[valid_pair, feature_idx]) > 0 and np.std(right[valid_pair, feature_idx]) > 0:
                        rho = float(spearmanr(left[valid_pair, feature_idx], right[valid_pair, feature_idx]).statistic)
                    cross_seizure_audit.append({
                        "subject_id": subject_id, "center": subject_id.split(":", 1)[0].lower(),
                        "seizure_pair_left": left_idx, "seizure_pair_right": right_idx,
                        "causal_feature": feature_name, "n_common_channels": int(valid_pair.sum()),
                        "spearman_rho": rho,
                    })
        for channel_idx, name in enumerate(canonical):
            valid_features = per_channel[channel_idx]
            if valid_features:
                stack = np.stack(valid_features)
                base = np.median(stack, axis=0)
                consistency = float(((stack[:, 3] >= 0.80) & (stack[:, 2] >= 0.5)).mean())
                values = np.concatenate((base, [consistency])).astype(np.float32)
                valid = True
                reason = ""
            else:
                values = np.zeros(6, dtype=np.float32)
                valid = False
                reason = "no_valid_causal_seizure"
            row = {
                "subject_id": subject_id, "channel_name": name, "channel_id": channel_idx,
                **dict(zip(CAUSAL_FEATURE_NAMES, map(float, values))),
                "cp_feature_valid": valid, "cp_valid_seizure_count": len(valid_features),
                "cp_valid_preictal_window_count": int(pre_counts[channel_idx]),
                "cp_valid_early_window_count": int(early_counts[channel_idx]),
                "cp_valid_window_fraction": float((pre_counts[channel_idx] + early_counts[channel_idx]) / max(len(valid_features) * (max_preictal_windows + max_early_windows), 1)),
                "cp_mean_var_stability": (
                    float(np.mean(stability_by_channel[channel_idx]))
                    if stability_by_channel[channel_idx] else float("nan")
                ),
                "cp_invalid_reason": reason,
            }
            rows.append(row)
    frame = pd.DataFrame(rows)
    duplicate_count = int(frame.duplicated(["subject_id", "channel_name"]).sum())
    feature_values = frame[list(CAUSAL_FEATURE_NAMES)].to_numpy(dtype=np.float64)
    nonfinite_count = int((~np.isfinite(feature_values)).sum())
    valid_rate = float(frame["cp_feature_valid"].mean()) if not frame.empty else 0.0
    frame["center"] = frame["subject_id"].str.split(":", n=1).str[0].str.lower()
    center_rates = frame.groupby("center")["cp_feature_valid"].mean().to_dict()
    stability_by_center = frame.groupby("center")["cp_mean_var_stability"].mean().to_dict()
    summary = {
        "n_patients": int(frame["subject_id"].nunique()), "n_channels": int(len(frame)),
        "n_matched_patients": int(len({str(r["subject_id"]) for r in feature_records if _record_key(r) in raw_index})),
        "n_matched_records": matched_records, "n_feature_records": len(feature_records),
        "patient_match_rate": int(len({str(r["subject_id"]) for r in feature_records if _record_key(r) in raw_index})) / max(len(retained), 1),
        "channel_match_rate": matched_channels / max(total_channels, 1),
        "window_match_rate": matched_windows / max(total_windows, 1),
        "n_matched_channels": int(frame["cp_feature_valid"].sum()),
        "feature_valid_rate": valid_rate, "valid_rate_by_center": center_rates,
        "mean_var_stability_by_center": stability_by_center,
        "duplicate_key_count": duplicate_count, "nonfinite_count": nonfinite_count,
        "missing_reason_counts": dict(Counter(frame.loc[~frame["cp_feature_valid"], "cp_invalid_reason"])),
        "features": {name: {"min": float(frame[name].min()), "max": float(frame[name].max()), "mean": float(frame[name].mean()), "std": float(frame[name].std(ddof=0))} for name in CAUSAL_FEATURE_NAMES},
    }
    passed = (
        summary["n_patients"] == len(retained) == 80 and duplicate_count == 0 and nonfinite_count == 0
        and valid_rate >= 0.95 and all(float(center_rates.get(center, 0.0)) >= 0.90 for center in ("hup", "lzu", "multicenter", "pediatric"))
        and summary["patient_match_rate"] == 1.0 and summary["channel_match_rate"] >= 0.95 and summary["window_match_rate"] >= 0.90
    )
    summary["status"] = "passed" if passed else "failed"
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.drop(columns=["center"]).to_parquet(destination, index=False)
    audit_dir = Path(audit_output_dir)
    audit_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(window_audit).to_csv(audit_dir / "causal_cache_channel_audit.csv", index=False)
    pd.DataFrame(cross_seizure_audit, columns=[
        "subject_id", "center", "seizure_pair_left", "seizure_pair_right",
        "causal_feature", "n_common_channels", "spearman_rho",
    ]).to_csv(audit_dir / "causal_cache_cross_seizure_stability.csv", index=False)
    frame.groupby("subject_id", as_index=False).agg(feature_valid_rate=("cp_feature_valid", "mean"), n_channels=("channel_name", "size")).to_csv(audit_dir / "causal_cache_patient_audit.csv", index=False)
    frame.groupby("center", as_index=False).agg(feature_valid_rate=("cp_feature_valid", "mean"), n_channels=("channel_name", "size")).to_csv(audit_dir / "causal_cache_center_audit.csv", index=False)
    (audit_dir / "causal_cache_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    if strict and not passed:
        raise RuntimeError(f"Causal propagation cache audit failed: {summary}")
    return summary


class CausalPropagationFeatureStore:
    """Strict canonical-key lookup for precomputed six-dimensional features."""

    def __init__(self, cache_path: str | Path, audit_path: str | Path | None = None, *, require_passed: bool = True) -> None:
        self.frame = pd.read_parquet(cache_path)
        required = {"subject_id", "channel_name", "cp_feature_valid", *CAUSAL_FEATURE_NAMES}
        missing = required - set(self.frame.columns)
        if missing:
            raise ValueError(f"Causal cache missing columns: {sorted(missing)}")
        self.frame["subject_key"] = self.frame["subject_id"].astype(str).str.casefold()
        self.frame["channel_key"] = self.frame["channel_name"].map(normalize_channel_name)
        duplicated = self.frame.duplicated(["subject_key", "channel_key"], keep=False)
        if duplicated.any():
            raise ValueError("Causal cache contains duplicate canonical subject/channel keys")
        values = self.frame[list(CAUSAL_FEATURE_NAMES)].to_numpy(dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError("Causal cache contains non-finite model features")
        self.lookup = {(row.subject_key, row.channel_key): row for row in self.frame.itertuples(index=False)}
        self.quality_fields = (
            "cp_valid_window_fraction", "cp_valid_seizure_count", "cp_mean_var_stability"
        )
        self.has_quality_fields = all(name in self.frame.columns for name in self.quality_fields)
        self.audit = {}
        if audit_path is not None:
            self.audit = json.loads(Path(audit_path).read_text(encoding="utf-8"))
            if require_passed and self.audit.get("status") != "passed":
                raise ValueError("Formal CANE-PATH-CP requires a passed causal cache audit")

    def align(self, subject_id: str, canonical_channels: Sequence[str]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        features = np.zeros((len(canonical_channels), 6), dtype=np.float32)
        valid = np.zeros(len(canonical_channels), dtype=bool)
        quality = {name: np.zeros(len(canonical_channels), dtype=np.float32) for name in self.quality_fields}
        missing: list[str] = []
        subject_key = str(subject_id).casefold()
        for index, channel in enumerate(canonical_channels):
            key = (subject_key, normalize_channel_name(channel))
            row = self.lookup.get(key)
            if row is None:
                missing.append(str(channel))
                continue
            features[index] = np.asarray([getattr(row, name) for name in CAUSAL_FEATURE_NAMES], dtype=np.float32)
            valid[index] = bool(row.cp_feature_valid)
            if self.has_quality_fields:
                for name in self.quality_fields:
                    value = float(getattr(row, name))
                    quality[name][index] = value if np.isfinite(value) else 0.0
        return features, valid, {
            "causal_key_match_rate": float((len(canonical_channels) - len(missing)) / max(len(canonical_channels), 1)),
            "causal_missing_channels": missing,
            "causal_duplicate_keys": 0,
            "causal_nonfinite_values": 0,
            "causal_quality_fields_present": self.has_quality_fields,
            **quality,
        }


__all__ = [
    "CAUSAL_FEATURE_NAMES", "CausalPropagationFeatureStore", "aggregate_patient_causal_features",
    "build_causal_propagation_cache", "fit_ridge_var_directed_influence", "percentile_rank",
    "seizure_causal_features", "select_causal_windows",
]
