from __future__ import annotations

import numpy as np
from .quantile_features import distribution_quantiles, pool_named_rows
from .temporal_phases import phase_mask


def cii_from_matrices(matrices: np.ndarray, true_ez_mask: np.ndarray):
    x = np.asarray(matrices, dtype=float); target = np.asarray(true_ez_mask, dtype=bool); outside = ~target
    out_strength = np.nansum(x, axis=2); in_strength = np.nansum(x, axis=1); cii = out_strength - in_strength
    ez_out = np.nansum(x[:, target][:, :, outside], axis=(1,2)); out_ez = np.nansum(x[:, outside][:, :, target], axis=(1,2))
    total = np.nansum(x, axis=(1,2)); normalized = np.divide(ez_out, total, out=np.full(len(x), np.nan), where=total > 0)
    return cii, {"ez_to_outside": ez_out, "outside_to_ez": out_ez, "net_causal_escape": ez_out-out_ez, "normalized_causal_escape": normalized}


def seizure_cii_features(times, matrices, true_ez_mask, band):
    cii, flow = cii_from_matrices(matrices, true_ez_mask); result, escape = {}, {}
    for phase in ("preictal", "onset", "spread"):
        keep = phase_mask(times, phase)
        for region, mask in (("true_ez", true_ez_mask), ("outside_true_ez", ~np.asarray(true_ez_mask, bool))):
            result.update(distribution_quantiles(cii[keep][:, mask], f"cii__{band}__{phase}__{region}__"))
        for name in ("ez_to_outside", "outside_to_ez", "net_causal_escape"):
            values = flow[name][keep]; result[f"cii__{band}__{phase}__mean_{name}"] = float(np.nanmean(values)) if np.isfinite(values).any() else np.nan
        norm = flow["normalized_causal_escape"][keep]
        result[f"cii__{band}__{phase}__q75_normalized_escape"] = float(np.nanquantile(norm,.75)) if np.isfinite(norm).any() else np.nan
        result[f"cii__{band}__{phase}__max_normalized_escape"] = float(np.nanmax(norm)) if np.isfinite(norm).any() else np.nan
        escape.update({f"{band}__{phase}__{k}": v for k,v in result.items() if f"{band}__{phase}" in k and ("escape" in k or "outside" in k)})
    return result, escape, cii


def patient_cii_features(rows): return pool_named_rows(rows)


__all__ = ["cii_from_matrices", "seizure_cii_features", "patient_cii_features"]
