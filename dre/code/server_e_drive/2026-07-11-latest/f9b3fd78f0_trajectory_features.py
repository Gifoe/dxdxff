from __future__ import annotations

import numpy as np
import pandas as pd

from .schemas import normalize_channel_name


def _safe(values: np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype=float)
    return out[np.isfinite(out)]


def _per_seizure(series: np.ndarray) -> dict[str, float]:
    valid = _safe(series)
    if not len(valid): valid = np.array([0.0])
    mid = max(1, int(np.ceil(len(valid) * .2)))
    slope = float(np.polyfit(np.arange(len(valid)), valid, 1)[0]) if len(valid) > 1 else 0.0
    spread = float(np.ptp(valid))
    return {"mean": float(valid.mean()), "std": float(valid.std()), "median": float(np.median(valid)), "iqr": float(np.subtract(*np.percentile(valid, [75, 25]))), "max": float(valid.max()), "min": float(valid.min()), "top20": float(np.mean(np.sort(valid)[-mid:])), "early": float(valid[:mid].mean()), "late": float(valid[-mid:].mean()), "late_early": float(valid[-mid:].mean() - valid[:mid].mean()), "slope": slope, "normalized_peak": float((valid.max() - valid.mean()) / (valid.std() + 1e-8)), "total_variation": float(np.abs(np.diff(valid)).sum()) if len(valid) > 1 else 0.0, "range": spread}


def summarize_channel_trajectory(values: np.ndarray, mask: np.ndarray, *, mode: str = "trajectory_compact") -> tuple[dict[str, float], dict[str, int]]:
    """Masked within-/across-seizure summaries, never treating padding as data."""
    x, valid = np.asarray(values, dtype=float), np.asarray(mask, dtype=bool)
    if x.ndim != 3 or valid.shape != x.shape[:2]: raise ValueError("values must be [seizure, window, feature] and mask [seizure, window]")
    out: dict[str, float] = {}
    for feature_idx in range(x.shape[-1]):
        per = [_per_seizure(x[s, valid[s], feature_idx]) for s in range(x.shape[0]) if valid[s].any()]
        if not per: per = [_per_seizure(np.array([]))]
        names = tuple(per[0]) if mode == "trajectory_full" else ("mean", "std", "median", "iqr", "max", "min", "total_variation")
        for name in names:
            values_by_seizure = np.asarray([row[name] for row in per], dtype=float)
            out[f"trajectory_{name}_f{feature_idx}"] = float(values_by_seizure.mean())
            if mode == "trajectory_full":
                out[f"trajectory_seizure_std_{name}_f{feature_idx}"] = float(values_by_seizure.std())
                out[f"trajectory_seizure_max_{name}_f{feature_idx}"] = float(values_by_seizure.max())
        means = np.asarray([row["mean"] for row in per])
        out[f"trajectory_stability_f{feature_idx}"] = float(1.0 / (1.0 + means.std()))
        # Legacy aliases retained for registered compact features.
        out[f"trajectory_mean_f{feature_idx}"] = float(means.mean())
        out[f"trajectory_std_f{feature_idx}"] = float(means.std())
    return out, {"valid_windows": int(valid.sum()), "valid_seizures": int(valid.any(axis=1).sum()), "feature_dim": int(x.shape[-1]), "invalid_values": int((~np.isfinite(x[valid])).sum()), "mode": mode}


def _iqr(values: np.ndarray) -> float:
    return float(np.subtract(*np.percentile(values, [75, 25]))) if len(values) else np.nan


def build_relative_trajectory_features(ledger: pd.DataFrame, store, *, top_q: float = .20) -> tuple[pd.DataFrame, dict[str, object]]:
    """Aggregate multi-feature ranks computed within each patient/run channel set."""
    if not 0 < float(top_q) <= 1:
        raise ValueError("top_q must be in (0, 1]")
    output = ledger.copy()
    columns = ("trajectory_mean", "trajectory_std", "trajectory_iqr", "trajectory_max",
               "trajectory_total_variation", "trajectory_early_late", "trajectory_slope",
               "trajectory_stability", "trajectory_rank_median", "trajectory_rank_iqr",
               "trajectory_topq_entry_rate", "trajectory_rank_stability",
               "trajectory_leave_one_seizure_out_variance", "trajectory_seizure_coverage")
    for column in columns:
        output[column] = np.nan
    for subject_id, indices in output.groupby("subject_id", sort=True).groups.items():
        runs = store.runs(str(subject_id))
        observations: dict[str, list[dict[str, float]]] = {}
        channel_run_counts: dict[str, int] = {}
        for run_index, run in enumerate(runs):
            values = np.asarray(run.values, dtype=float)
            names = [normalize_channel_name(name) for name in run.channel_names]
            for name in set(names):
                channel_run_counts[name] = channel_run_counts.get(name, 0) + 1
            per_channel: dict[str, np.ndarray] = {}
            for channel_index, name in enumerate(names):
                valid_windows = run.channel_mask(channel_index)
                if valid_windows.any():
                    per_channel[name] = np.nanmean(values[valid_windows, channel_index, :], axis=0)
            if not per_channel:
                continue
            channel_names = list(per_channel)
            matrix = np.vstack([per_channel[name] for name in channel_names])
            for feature_index in range(matrix.shape[1]):
                scores = matrix[:, feature_index]
                finite = np.isfinite(scores)
                if not finite.any():
                    continue
                finite_scores = scores[finite]
                ranks = pd.Series(finite_scores).rank(ascending=False, method="average").to_numpy()
                n_channels = len(finite_scores)
                percentiles = 1. - (ranks - 1.) / max(n_channels - 1, 1)
                top_n = max(1, int(np.ceil(float(top_q) * n_channels)))
                finite_names = [channel_names[index] for index in np.flatnonzero(finite)]
                for name, score, rank, percentile in zip(finite_names, finite_scores, ranks, percentiles):
                    observations.setdefault(name, []).append({"run": float(run_index), "feature": float(feature_index), "score": float(score), "rank": float(rank), "rank_percentile": float(percentile), "topq": float(rank <= top_n)})
        for index, row in output.loc[indices].iterrows():
            name = str(row.get("channel_name_norm", normalize_channel_name(row["channel_name_original"])))
            records = observations.get(name)
            if not records:
                continue
            frame = pd.DataFrame(records)
            per_run = frame.groupby("run", sort=True).agg(score=("score", "mean"), rank=("rank_percentile", "mean"), topq=("topq", "mean"))
            scores = per_run["score"].to_numpy(dtype=float)
            ranks = per_run["rank"].to_numpy(dtype=float)
            topq_values = per_run["topq"].to_numpy(dtype=float)
            leave_one_out = np.asarray([np.delete(scores, i).mean() for i in range(len(scores))]) if len(scores) > 1 else np.asarray([scores.mean()])
            values = {"trajectory_mean": float(scores.mean()), "trajectory_std": float(scores.std()), "trajectory_iqr": _iqr(scores), "trajectory_max": float(scores.max()), "trajectory_total_variation": float(np.abs(np.diff(scores)).sum()) if len(scores) > 1 else 0., "trajectory_early_late": float(scores[-1] - scores[0]) if len(scores) > 1 else 0., "trajectory_slope": float(np.polyfit(np.arange(len(scores)), scores, 1)[0]) if len(scores) > 1 else 0., "trajectory_stability": float(1. / (1. + scores.std())), "trajectory_rank_median": float(np.median(ranks)), "trajectory_rank_iqr": _iqr(ranks), "trajectory_topq_entry_rate": float(topq_values.mean()), "trajectory_rank_stability": float(1. / (1. + ranks.std())), "trajectory_leave_one_seizure_out_variance": float(leave_one_out.var()), "trajectory_seizure_coverage": float(len(per_run) / max(channel_run_counts.get(name, 0), 1))}
            output.loc[index, list(values)] = list(values.values())
    feature_names = list(getattr(store, "feature_names", ()))
    if not feature_names:
        feature_dim = next((np.asarray(run.values).shape[2] for subject in store.subjects() for run in store.runs(subject)), 0)
        feature_names = [f"feature_{index}" for index in range(feature_dim)]
    return output, {"feature_names": feature_names, "aggregation": "patient_run_feature_relative_rank", "top_q": float(top_q), "summary_columns": list(columns)}
