from __future__ import annotations

import numpy as np
from ..ngbr.epileptogenicity_index import compute_ei
from ..ngbr.ictal_propagation import compute_propagation
from .quantile_features import pool_named_rows


def seizure_recruitment_features(record):
    ei, ei_channels, ei_audit = compute_ei(record)
    propagation, prop_channels, prop_audit = compute_propagation(record)
    target=np.asarray(record.clinical_target_mask,bool)&record.valid_channel_mask; outside=~np.asarray(record.clinical_target_mask,bool)&record.valid_channel_mask
    denom=np.nansum(ei)
    row={"ei_capture":float(np.nansum(ei[target])/denom) if denom>0 else np.nan,
         "outside_ei_q90":float(np.nanquantile(ei[outside],.9)) if np.isfinite(ei[outside]).any() else np.nan,
         "outside_ei_fraction_above_0.8":float(np.mean(ei[outside]>.8)) if outside.any() else np.nan,
         "true_ez_recruitment_median":prop_audit.get("target_recruitment_median",np.nan),
         "outside_recruitment_median":prop_audit.get("outside_recruitment_median",np.nan),
         "target_to_outside_delay":prop_audit.get("target_to_outside_delay",np.nan),
         "outside_recruited_1s_fraction":prop_audit.get("outside_recruited_1s_fraction",np.nan),
         "outside_recruited_3s_fraction":prop_audit.get("outside_recruited_3s_fraction",np.nan),
         "outside_recruited_5s_fraction":prop_audit.get("outside_recruited_5s_fraction",np.nan),
         "true_ez_recruited_3s_fraction":prop_audit.get("target_recruited_3s_fraction",np.nan),
         "earliest_outside_recruitment_time":prop_audit.get("earliest_outside_recruitment_time",np.nan),
         "propagation_outside_q90":float(np.nanquantile(propagation[outside],.9)) if np.isfinite(propagation[outside]).any() else np.nan}
    return row, ei_channels, prop_channels, {**ei_audit, **{f"prop_{k}":v for k,v in prop_audit.items()}}


def patient_recruitment_features(rows): return pool_named_rows(rows)


__all__=["seizure_recruitment_features","patient_recruitment_features"]
