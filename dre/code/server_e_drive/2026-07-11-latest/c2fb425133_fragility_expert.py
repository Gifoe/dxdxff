from __future__ import annotations

import numpy as np
from .fragility_trajectory import concatenate_trajectories
from .quantile_features import phase_region_quantiles, pool_named_rows
from .virtual_intervention import virtual_intervention, aggregate_virtual_intervention


def seizure_fragility_features(trajectories, true_ez_mask, q_nez=None):
    times, values, valid = concatenate_trajectories(trajectories)
    expert = phase_region_quantiles(values, times, true_ez_mask, valid, "fragility__")
    vi = aggregate_virtual_intervention(virtual_intervention(values, true_ez_mask, q_nez, valid), times, "fragility__")
    return expert, vi


def patient_fragility_features(rows): return pool_named_rows(rows)


__all__ = ["seizure_fragility_features", "patient_fragility_features"]
