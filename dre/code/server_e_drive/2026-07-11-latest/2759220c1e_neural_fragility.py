from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .robust_rank import percentile_rank_channels
from .schema import RawRunRecord
from .signal_quality import robust_scale_channels, time_slice


ALGORITHM_VERSION = "structured_column_resolvent_v4"


@dataclass(frozen=True)
class FragilityConfig:
    window_sec: float = .250
    step_sec: float = .125
    ridge_lambda: float = 1e-5
    angle_count: int = 128
    epsilon: float = 1e-8


def fit_linear_dynamics(window: np.ndarray, ridge_lambda: float = 1e-5) -> np.ndarray:
    data = np.asarray(window, dtype=float)
    if data.ndim != 2 or data.shape[1] < 3:
        raise ValueError("linear dynamics requires [C,T] with T>=3")
    x, y = data[:, :-1], data[:, 1:]
    # Exact primal/dual ridge identity; solve the smaller system when a
    # 250-ms window contains fewer time points than channels.
    if x.shape[1] < x.shape[0]:
        gram = x.T @ x + float(ridge_lambda) * np.eye(x.shape[1])
        return y @ np.linalg.solve(gram, x.T)
    gram = x @ x.T + float(ridge_lambda) * np.eye(x.shape[0])
    return np.linalg.solve(gram.T, (y @ x.T).T).T


def stabilize_dynamics(matrix: np.ndarray) -> tuple[np.ndarray, bool, float]:
    eigenvalues = np.linalg.eigvals(matrix)
    rho = float(np.max(np.abs(eigenvalues))) if eigenvalues.size else 0.0
    if not np.isfinite(rho):
        raise np.linalg.LinAlgError("non-finite spectral radius")
    if rho >= 1.0:
        return matrix / (rho + 1e-3) * .99, True, rho
    return matrix, False, rho


def structured_column_fragility(matrix: np.ndarray, angle_count: int = 128, epsilon: float = 1e-8) -> np.ndarray:
    """Return reciprocal minimum single-column perturbation distance.

    For B=zI-A and a perturbation u e_c^T, the determinant lemma gives
    det(B-u e_c^T)=det(B)(1-e_c^T B^-1 u).  The minimum Euclidean-norm
    u satisfying e_c^T B^-1 u=1 has norm 1/||e_c^T B^-1||_2.  We scan a
    fixed unit-circle grid and report the reciprocal minimum distance.
    """
    a = np.asarray(matrix, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("matrix must be square")
    best_distance = np.full(a.shape[0], np.inf); angles = np.linspace(0.0, 2.0 * np.pi, int(angle_count), endpoint=False); identity = np.eye(a.shape[0], dtype=complex)
    try:
        eigenvalues, eigenvectors = np.linalg.eig(a); inverse_vectors = np.linalg.inv(eigenvectors)
        # Verify the decomposition before using the batched resolvent. Highly
        # non-normal/defective matrices fall back to direct deterministic solves.
        relative_error = np.linalg.norm(a @ eigenvectors - eigenvectors * eigenvalues[None, :]) / max(np.linalg.norm(a), epsilon)
        if not np.isfinite(relative_error) or relative_error > 1e-6: raise np.linalg.LinAlgError("unstable eigendecomposition")
        for start in range(0, len(angles), 16):
            z = np.exp(1j * angles[start:start + 16]); coefficients = eigenvectors[None, :, :] / (z[:, None, None] - eigenvalues[None, None, :]); resolvent = coefficients @ inverse_vectors
            norm = np.linalg.norm(resolvent, axis=2); distance = np.divide(1.0, norm, out=np.full_like(norm, np.inf), where=norm > epsilon); best_distance = np.minimum(best_distance, np.min(distance, axis=0))
    except np.linalg.LinAlgError:
        failures = 0
        for theta in angles:
            try: resolvent = np.linalg.solve(np.exp(1j * theta) * identity - a, identity)
            except np.linalg.LinAlgError: failures += 1; continue
            norm = np.linalg.norm(resolvent, axis=1); best_distance = np.minimum(best_distance, np.divide(1.0, norm, out=np.full_like(norm, np.inf), where=norm > epsilon))
        if failures == int(angle_count): return np.full(a.shape[0], np.nan)
    score = np.divide(1.0, best_distance + epsilon, out=np.full_like(best_distance, np.nan), where=np.isfinite(best_distance))
    return score.real


def _window_scores(data: np.ndarray, fs: float, config: FragilityConfig, phase: str, phase_start_sec: float) -> tuple[np.ndarray, int, int, list[dict]]:
    width = max(3, int(round(config.window_sec * fs))); step = max(1, int(round(config.step_sec * fs)))
    starts = np.arange(0, max(data.shape[1] - width + 1, 0), step, dtype=int)
    scores, rescaled, failures, audit = [], 0, 0, []
    for window_index, start in enumerate(starts):
        try:
            matrix = fit_linear_dynamics(data[:, start:start + width], config.ridge_lambda)
            matrix, changed, rho = stabilize_dynamics(matrix); rescaled += int(changed)
            score = structured_column_fragility(matrix, config.angle_count, config.epsilon)
            failed = not np.isfinite(score).all(); failures += int(failed); scores.append(score); audit.append({"phase": phase, "window_index": window_index, "window_start_sec": phase_start_sec + start / fs, "window_length_sec": width / fs, "spectral_radius_before_rescale": rho, "stability_rescaled": changed, "optimization_failed": failed})
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            failures += 1; scores.append(np.full(data.shape[0], np.nan)); audit.append({"phase": phase, "window_index": window_index, "window_start_sec": phase_start_sec + start / fs, "window_length_sec": width / fs, "spectral_radius_before_rescale": np.nan, "stability_rescaled": False, "optimization_failed": True})
    return np.asarray(scores, dtype=float), rescaled, failures, audit


def compute_fragility(record: RawRunRecord, config: FragilityConfig | None = None) -> tuple[np.ndarray, pd.DataFrame, dict]:
    cfg = config or FragilityConfig(); scaled, valid = robust_scale_channels(record.signal, record.valid_channel_mask)
    indices = np.where(valid)[0]; result = np.full(len(record.channel_names), np.nan)
    if indices.size < 4:
        return result, pd.DataFrame(), {"patient_key": record.patient_key, "seizure_id": record.seizure_id, "fragility_valid": False, "invalid_reason": "fewer_than_four_valid_channels"}
    pre_slice = time_slice(record.onset_sample, record.sampling_rate, scaled.shape[1], -10, -1)
    onset_slice = time_slice(record.onset_sample, record.sampling_rate, scaled.shape[1], 0, 5)
    pre, pre_rescaled, pre_failed, pre_audit = _window_scores(scaled[indices, pre_slice], record.sampling_rate, cfg, "preictal", -10.0)
    onset, onset_rescaled, onset_failed, onset_audit = _window_scores(scaled[indices, onset_slice], record.sampling_rate, cfg, "onset", 0.0)
    total = len(pre) + len(onset); failure_fraction = (pre_failed + onset_failed) / max(total, 1)
    fallback = False
    if len(onset):
        onset_value = np.nanquantile(onset, .75, axis=0)
    else:
        onset_value = np.full(indices.size, np.nan)
    pre_value = np.nanmedian(pre, axis=0) if len(pre) else np.full(indices.size, np.nan)
    change = onset_value - pre_value
    raw = change.copy()
    fallback_mask = ~np.isfinite(raw)
    if fallback_mask.any(): raw[fallback_mask] = onset_value[fallback_mask]; fallback = bool(fallback_mask.any())
    rank = percentile_rank_channels(raw, np.isfinite(raw), high_is_abnormal=True)
    valid_seizure = failure_fraction <= .20 and np.isfinite(rank).sum() >= 4
    if valid_seizure: result[indices] = rank
    channel = pd.DataFrame({
        "patient_key": record.patient_key, "seizure_id": record.seizure_id,
        "channel": [record.channel_names[i] for i in indices],
        "preictal_fragility": pre_value, "onset_fragility": onset_value,
        "fragility_change": change, "fragility": rank if valid_seizure else np.nan,
    })
    quality = {
        "patient_key": record.patient_key, "seizure_id": record.seizure_id,
        "n_windows_preictal": len(pre), "n_windows_onset": len(onset),
        "n_channels": int(indices.size),
        "n_stability_rescaled_windows": pre_rescaled + onset_rescaled,
        "optimization_failure_fraction": failure_fraction,
        "fragility_valid": valid_seizure, "fallback_used": fallback,
        "preictal_actual_sec": (pre_slice.stop - pre_slice.start) / record.sampling_rate,
        "onset_actual_sec": (onset_slice.stop - onset_slice.start) / record.sampling_rate,
        "algorithm_version": ALGORITHM_VERSION,
        "window_audit": [{"patient_key": record.patient_key, "seizure_id": record.seizure_id, **row} for row in pre_audit + onset_audit],
    }
    return result, channel, quality


__all__ = ["ALGORITHM_VERSION", "FragilityConfig", "compute_fragility", "fit_linear_dynamics", "stabilize_dynamics", "structured_column_fragility"]
