from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.signal import butter, hilbert, resample_poly, sosfiltfilt

ALGORITHM_VERSION = "discrete_phase_cmi_chunked_v1"


@dataclass(frozen=True)
class PTEConfig:
    target_fs: float = 100.0
    bins: int = 18
    window_sec: float = 2.0
    step_sec: float = 1.0
    pseudocount: float = 1e-6
    edge_chunk_size: int = 256


def robust_phase(signal: np.ndarray, fs: float, low: float, high: float, target_fs: float = 100.0) -> np.ndarray:
    x = np.asarray(signal, dtype=float); median = np.nanmedian(x, axis=1, keepdims=True)
    mad = np.nanmedian(np.abs(x - median), axis=1, keepdims=True); scale = np.where(mad > 1e-8, 1.4826 * mad, 1.0)
    x = np.clip(np.nan_to_num((x - median) / scale), -10, 10)
    filtered = sosfiltfilt(butter(4, [low, high], btype="bandpass", fs=fs, output="sos"), x, axis=1)
    phase = np.angle(hilbert(filtered, axis=1))
    if abs(fs - target_fs) > 1e-9:
        from fractions import Fraction
        ratio = Fraction(target_fs / fs).limit_denominator(1000)
        phase = np.angle(resample_poly(np.exp(1j * phase), ratio.numerator, ratio.denominator, axis=1))
    return phase


def discretize_phase(phase: np.ndarray, bins: int = 18) -> np.ndarray:
    return np.minimum((np.mod(np.asarray(phase) + np.pi, 2*np.pi) / (2*np.pi) * bins).astype(np.int16), bins - 1)


def phase_transfer_entropy_matrix(discrete_phase: np.ndarray, lag_samples: int,
                                  bins: int = 18, pseudocount: float = 1e-6,
                                  edge_chunk_size: int = 256) -> np.ndarray:
    """Directed plugin conditional MI for every edge, vectorized in edge chunks."""
    states = np.asarray(discrete_phase, dtype=np.int64)
    if states.ndim != 2 or states.shape[1] <= lag_samples: return np.full((states.shape[0], states.shape[0]), np.nan)
    c, n = states.shape; src, dst = np.where(~np.eye(c, dtype=bool)); output = np.zeros(src.size)
    future = states[:, lag_samples:]; current = states[:, :-lag_samples]
    for start in range(0, src.size, edge_chunk_size):
        s, d = src[start:start+edge_chunk_size], dst[start:start+edge_chunk_size]
        x, y, z = current[s], future[d], current[d]
        codes_xyz = (x * bins + y) * bins + z
        codes_xz = x * bins + z; codes_yz = y * bins + z
        edge_ids = np.arange(len(s), dtype=np.int64)[:, None]
        combined = (codes_xyz + edge_ids * bins**3).ravel()
        xyz = np.bincount(combined, minlength=len(s)*bins**3).reshape(len(s),bins,bins,bins).astype(float) + pseudocount
        pxyz = xyz / xyz.sum(axis=(1,2,3), keepdims=True)
        pxz = pxyz.sum(axis=2); pyz = pxyz.sum(axis=1); pz = pxyz.sum(axis=(1,2))
        ratio = pxyz * pz[:,None,None,:] / (pxz[:,:,None,:] * pyz[:,None,:,:])
        values = np.sum(pxyz * np.log(np.maximum(ratio, 1e-300)), axis=(1,2,3))
        output[start:start+len(s)] = np.maximum(values, 0.0)
    matrix = np.zeros((c,c), dtype=float); matrix[src,dst] = output
    total = matrix.sum()
    if total > 0: matrix /= total
    np.fill_diagonal(matrix, 0.0); matrix[~np.isfinite(matrix)] = np.nan
    return matrix


def compute_pte_windows(signal: np.ndarray, fs: float, onset_sample: int,
                        band: str, config: PTEConfig | None = None, n_jobs: int = 1):
    cfg = config or PTEConfig(); upper = min(100.0, .40 * fs)
    if band == "high" and upper <= 45: return np.empty(0), np.empty((0, signal.shape[0], signal.shape[0])), {"valid": False, "reason": "HIGH_BAND_UNSUPPORTED"}
    low, high, lag_sec = (0.5, 30.0, .020) if band == "low" else (30.0, upper, .005)
    phase = robust_phase(signal, fs, low, high, cfg.target_fs); states = discretize_phase(phase, cfg.bins)
    width, step = round(cfg.window_sec*cfg.target_fs), round(cfg.step_sec*cfg.target_fs)
    starts = np.arange(0, max(states.shape[1]-width+1, 0), step, dtype=int); lag = max(1, round(lag_sec*cfg.target_fs))
    if int(n_jobs)>1 and len(starts)>1:
        from joblib import Parallel,delayed
        matrices=np.asarray(Parallel(n_jobs=min(int(n_jobs),len(starts)),prefer="threads",require="sharedmem")(delayed(phase_transfer_entropy_matrix)(states[:,s:s+width],lag,cfg.bins,cfg.pseudocount,cfg.edge_chunk_size) for s in starts))
    else: matrices=np.asarray([phase_transfer_entropy_matrix(states[:, s:s+width], lag, cfg.bins, cfg.pseudocount, cfg.edge_chunk_size) for s in starts])
    onset_resampled = onset_sample / fs * cfg.target_fs
    times = (starts + width/2 - onset_resampled) / cfg.target_fs
    return times, matrices, {"valid": bool(len(starts)), "band": band, "effective_upper": high, "algorithm_version": ALGORITHM_VERSION}


__all__ = ["ALGORITHM_VERSION", "PTEConfig", "robust_phase", "discretize_phase", "phase_transfer_entropy_matrix", "compute_pte_windows"]
