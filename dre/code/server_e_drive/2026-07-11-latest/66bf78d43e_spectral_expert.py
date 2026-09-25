from __future__ import annotations

from .quantile_features import phase_region_quantiles, pool_named_rows
from .virtual_intervention import virtual_intervention, aggregate_virtual_intervention


def seizure_spectral_features(state, true_ez_mask, q_nez=None):
    expert, vi = {}, {}
    for marker, values in state.maps.items():
        expert.update(phase_region_quantiles(values,state.times_sec,true_ez_mask,state.valid_mask,f"spectral__{marker}__"))
        vi.update(aggregate_virtual_intervention(virtual_intervention(values,true_ez_mask,q_nez,state.valid_mask),state.times_sec,f"spectral__{marker}__"))
    return expert,vi


def patient_spectral_features(rows): return pool_named_rows(rows)


__all__=["seizure_spectral_features","patient_spectral_features"]
