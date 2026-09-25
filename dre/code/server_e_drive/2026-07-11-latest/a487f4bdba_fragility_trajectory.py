from __future__ import annotations

import numpy as np
import pandas as pd

from ..ngbr.neural_fragility import (ALGORITHM_VERSION as BASE_VERSION, FragilityConfig,
                                     fit_linear_dynamics, stabilize_dynamics,
                                     structured_column_fragility)
from ..ngbr.signal_quality import robust_scale_channels
from .schema import TemporalTrajectory
from .temporal_phases import phase_intervals

ALGORITHM_VERSION = f"{BASE_VERSION}+full_ranked_trajectory_v1"


def _rank_rows(values: np.ndarray) -> np.ndarray:
    output = np.full(values.shape, np.nan)
    for index, row in enumerate(values):
        valid = np.isfinite(row)
        if not valid.any(): continue
        order = np.argsort(row[valid], kind="stable")
        ranks = np.empty(order.size); ranks[order] = np.arange(1, order.size + 1)
        output[index, valid] = (ranks - .5) / order.size
    return output


def compute_fragility_trajectory(record, config: FragilityConfig | None = None, n_jobs: int = 1):
    cfg = config or FragilityConfig(); scaled, valid = robust_scale_channels(record.signal, record.valid_channel_mask)
    indices = np.where(valid)[0]; intervals = phase_intervals(record.onset_sample, record.sampling_rate, record.signal.shape[1])
    trajectories, rows = {}, []
    if indices.size < 4:
        return trajectories, pd.DataFrame(), {"fragility_valid": False, "invalid_reason": "fewer_than_four_valid_channels", "algorithm_version": ALGORITHM_VERSION}
    width = max(3, int(round(cfg.window_sec * record.sampling_rate))); step = max(1, int(round(cfg.step_sec * record.sampling_rate)))
    for phase, interval in intervals.items():
        if not interval.valid: continue
        lo = max(0, int(round(record.onset_sample + interval.start_sec * record.sampling_rate)))
        hi = min(record.signal.shape[1], int(round(record.onset_sample + interval.end_sec * record.sampling_rate)))
        starts = np.arange(lo, max(lo, hi - width + 1), step, dtype=int)
        def solve_window(item):
            wi, start = item
            values = np.full(len(record.channel_names), np.nan)
            failed, changed, rho = False, False, np.nan
            try:
                matrix = fit_linear_dynamics(scaled[indices, start:start + width], cfg.ridge_lambda)
                matrix, changed, rho = stabilize_dynamics(matrix)
                values[indices] = structured_column_fragility(matrix, cfg.angle_count, cfg.epsilon)
                failed = not np.isfinite(values[indices]).all()
            except (ValueError, np.linalg.LinAlgError, FloatingPointError): failed = True
            audit = {"patient_key": record.patient_key, "seizure_id": record.seizure_id,
                     "phase": phase, "window_index": wi, "window_start_sec": (start-record.onset_sample)/record.sampling_rate,
                     "spectral_radius_before_rescale": rho, "stability_rescaled": changed, "optimization_failed": failed}
            return wi, values, audit
        items=list(enumerate(starts))
        if int(n_jobs) > 1 and len(items) > 1:
            from joblib import Parallel, delayed
            solved=Parallel(n_jobs=min(int(n_jobs),len(items)),prefer="threads",require="sharedmem")(delayed(solve_window)(item) for item in items)
        else: solved=[solve_window(item) for item in items]
        raw = np.full((len(starts), len(record.channel_names)), np.nan)
        for wi,values,audit in solved: raw[wi]=values; rows.append(audit)
        ranked = _rank_rows(np.log1p(np.maximum(raw, 0.0)))
        times = (starts + width / 2 - record.onset_sample) / record.sampling_rate
        trajectories[phase] = TemporalTrajectory(record.patient_key, record.seizure_id, "fragility", phase,
                                                   times, ranked, np.isfinite(ranked))
    audit = pd.DataFrame(rows); failures = float(audit.optimization_failed.mean()) if len(audit) else 1.0
    quality = {"patient_key": record.patient_key, "seizure_id": record.seizure_id,
               "fragility_valid": bool(trajectories and failures <= .20), "optimization_failure_fraction": failures,
               "n_windows": len(audit), "algorithm_version": ALGORITHM_VERSION,
               **{f"{p}_actual_sec": x.actual_duration_sec for p, x in intervals.items()},
               **{f"{p}_valid": x.valid for p, x in intervals.items()}}
    return trajectories, audit, quality


def concatenate_trajectories(trajectories: dict[str, TemporalTrajectory]):
    ordered = [trajectories[p] for p in ("preictal", "onset", "spread") if p in trajectories]
    if not ordered: return np.empty(0), np.empty((0, 0)), np.empty((0, 0), dtype=bool)
    return np.concatenate([x.times_sec for x in ordered]), np.concatenate([x.values for x in ordered]), np.concatenate([x.valid_mask for x in ordered])


__all__ = ["ALGORITHM_VERSION", "FragilityConfig", "compute_fragility_trajectory", "concatenate_trajectories"]
