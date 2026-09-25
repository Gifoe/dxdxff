from __future__ import annotations

from collections.abc import Mapping
import numpy as np

QUANTILES = np.arange(.1, 1.01, .1)
POOL_STATS = ("mean", "q25", "max", "iqr")


def distribution_quantiles(values: np.ndarray, prefix: str = "") -> dict[str, float]:
    finite = np.asarray(values, dtype=float).ravel()
    finite = finite[np.isfinite(finite)]
    vals = np.quantile(finite, QUANTILES) if finite.size else np.full(10, np.nan)
    return {f"{prefix}q{int(q * 100):02d}": float(v) for q, v in zip(QUANTILES, vals)}


def phase_region_quantiles(values: np.ndarray, times: np.ndarray, true_ez: np.ndarray,
                           valid_mask: np.ndarray | None = None, prefix: str = "") -> dict[str, float]:
    from .temporal_phases import phase_mask
    x = np.asarray(values, dtype=float); target = np.asarray(true_ez, dtype=bool)
    valid = np.isfinite(x) if valid_mask is None else np.asarray(valid_mask, dtype=bool) & np.isfinite(x)
    result = {}
    for phase in ("preictal", "onset", "spread"):
        tm = phase_mask(times, phase)
        for region, channels in (("true_ez", target), ("outside_true_ez", ~target)):
            mask = tm[:, None] & channels[None, :] & valid
            result.update(distribution_quantiles(x[mask], f"{prefix}{phase}__{region}__"))
    return result


def pool_named_rows(rows: list[Mapping[str, float]], keys: list[str] | None = None) -> dict[str, float]:
    keys = keys or sorted({key for row in rows for key in row})
    output = {}
    for key in keys:
        # Seizure rows also carry audit identifiers (patient_key, seizure_id,
        # center). They are never patient-level numeric features.
        converted = []
        for row in rows:
            value = row.get(key, np.nan)
            try:
                converted.append(float(value))
            except (TypeError, ValueError):
                converted.append(np.nan)
        values = np.asarray(converted, dtype=float)
        finite = values[np.isfinite(values)]
        if not finite.size:
            continue
        stats = (np.mean(finite), np.quantile(finite, .25), np.max(finite),
                 np.quantile(finite, .75) - np.quantile(finite, .25))
        output.update({f"{key}__{name}": float(value) for name, value in zip(POOL_STATS, stats)})
    return output


__all__ = ["QUANTILES", "POOL_STATS", "distribution_quantiles", "phase_region_quantiles", "pool_named_rows"]
