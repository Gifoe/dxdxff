from __future__ import annotations

import numpy as np
from .schema import VirtualInterventionFeatures


def _row_quantile(values: np.ndarray, mask: np.ndarray, q: float) -> np.ndarray:
    return np.asarray([np.nanquantile(row[m], q) if np.any(m & np.isfinite(row)) else np.nan
                       for row, m in zip(values, mask)], dtype=float)


def virtual_intervention(values: np.ndarray, true_ez_mask: np.ndarray,
                         q_nez: np.ndarray | None = None,
                         valid_mask: np.ndarray | None = None) -> VirtualInterventionFeatures:
    x = np.asarray(values, dtype=float)
    if x.ndim == 1: x = x[None, :]
    target = np.broadcast_to(np.asarray(true_ez_mask, dtype=bool), x.shape)
    valid = np.isfinite(x) if valid_mask is None else np.broadcast_to(valid_mask, x.shape) & np.isfinite(x)
    inside, outside = target & valid, ~target & valid
    numerator = np.nansum(np.where(inside, x, 0.0), axis=1)
    denominator = np.nansum(np.where(valid, x, 0.0), axis=1)
    capture = np.divide(numerator, denominator, out=np.full(x.shape[0], np.nan), where=denominator > 0)
    outside_q90 = _row_quantile(x, outside, .90)
    gap = _row_quantile(x, inside, .75) - _row_quantile(x, outside, .75)
    outside_extent = np.asarray([np.mean(row[m] > .8) if m.any() else np.nan for row, m in zip(x, outside)])
    if q_nez is None:
        concordant = np.full(x.shape[0], np.nan)
    else:
        q = np.broadcast_to(np.asarray(q_nez, dtype=float), x.shape)
        concordant = _row_quantile(x * (1.0 - q), outside & np.isfinite(q), .90)
    return VirtualInterventionFeatures(capture, outside_q90, gap, outside_extent, concordant)


def aggregate_virtual_intervention(features: VirtualInterventionFeatures, times: np.ndarray,
                                   prefix: str = "") -> dict[str, float]:
    from .temporal_phases import phase_mask
    result = {}
    for phase in ("preictal", "onset", "spread"):
        select = phase_mask(times, phase)
        for name in features.__dataclass_fields__:
            values = np.asarray(getattr(features, name))[select]; finite = values[np.isfinite(values)]
            if finite.size:
                x = np.arange(finite.size, dtype=float)
                slope = float(np.polyfit(x, finite, 1)[0]) if finite.size > 1 else np.nan
                stats = (np.mean(finite), np.quantile(finite, .25), np.max(finite),
                         np.quantile(finite, .75) - np.quantile(finite, .25), slope)
            else: stats = (np.nan,) * 5
            for stat, value in zip(("mean", "q25", "max", "iqr", "slope"), stats):
                result[f"{prefix}{phase}__{name}__{stat}"] = float(value)
    return result


__all__ = ["virtual_intervention", "aggregate_virtual_intervention"]
