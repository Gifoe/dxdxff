from __future__ import annotations

import inspect

import numpy as np
import pytest

from neuroez_c.causal_propagation_features import (
    CAUSAL_FEATURE_NAMES, aggregate_patient_causal_features, fit_ridge_var_directed_influence,
    percentile_rank, robust_channel_normalize, seizure_causal_features, select_causal_windows,
)


def _directed(seed: int = 2) -> np.ndarray:
    rng=np.random.default_rng(seed); x=np.zeros((3,800))
    for t in range(1,800):
        x[0,t]=.7*x[0,t-1]+rng.normal(scale=.2)
        x[1,t]=.85*x[0,t-1]+.1*x[1,t-1]+rng.normal(scale=.05)
        x[2,t]=rng.normal(scale=.2)
    return x


def test_known_directed_system_recovers_main_direction() -> None:
    out=fit_ridge_var_directed_influence(_directed(),.01)
    assert out["A"][1,0] > out["A"][0,1]


def test_a_target_source_orientation() -> None:
    out=fit_ridge_var_directed_influence(_directed(),.01)
    assert np.argmax(np.abs(out["A"][1])) == 0


def test_var_diagonal_zero() -> None:
    assert np.allclose(np.diag(fit_ridge_var_directed_influence(_directed())["A"]),0)


def test_ridge_outputs_finite() -> None:
    out=fit_ridge_var_directed_influence(_directed())
    assert out["valid"] and np.isfinite(out["A"]).all() and np.isfinite(out["ridge_lambda"])


def test_constant_channel_safe() -> None:
    out=fit_ridge_var_directed_influence(np.vstack([np.ones(200),np.arange(200)]))
    assert out["valid"] and np.isfinite(out["A"]).all()


def test_insufficient_samples_invalid() -> None:
    assert fit_ridge_var_directed_influence(np.ones((40,50)))["invalid_reason"] == "insufficient_samples"


def test_missing_window_center_strict_failure() -> None:
    with pytest.raises(ValueError): select_causal_windows([0,np.nan])


def test_preictal_and_early_selection() -> None:
    pre,early=select_causal_windows([-6,-5,-4,-3,-2,-1,0,1,2,31])
    assert pre.tolist()==[2,3,4,5] and early.tolist()==[6,7,8]


def test_six_feature_ranges() -> None:
    seizure=np.array([[.2,.3,1.,.9,.5],[.4,-.2,.2,.7,0]],dtype=np.float32)
    features,_=aggregate_patient_causal_features([seizure,seizure])
    assert features.shape==(2,6) and np.all(features[:,[0,2,3,4,5]]>=0) and np.all(features[:,[0,2,3,4,5]]<=1) and np.all(np.abs(features[:,1])<=1)


def test_activation_rank_earliest_is_highest() -> None:
    early=np.array([[.9,.1],[.1,.9],[.1,.1]])
    result=seizure_causal_features(np.empty((0,2)),np.empty((0,2)),early)
    assert result[0,2] > result[1,2]


def test_persistence_fraction() -> None:
    early=np.array([[.9,.1],[.9,.9],[.1,.1],[.9,.1]])
    result=seizure_causal_features(np.empty((0,2)),np.empty((0,2)),early)
    assert result[0,4] == pytest.approx(.75)


def test_cross_seizure_consistency() -> None:
    a=np.array([[.1,0,1,.9,.5],[.1,0,0,.9,.5]],np.float32)
    b=np.array([[.1,0,1,.9,.5],[.1,0,1,.7,.5]],np.float32)
    out,_=aggregate_patient_causal_features([a,b])
    assert out[0,5] == 1 and out[1,5] == 0


def test_extractor_has_no_label_argument() -> None:
    assert not any("label" in name for name in inspect.signature(fit_ridge_var_directed_influence).parameters)


def test_percentile_ties_average() -> None:
    ranks=percentile_rank(np.array([1,1,2])); assert ranks[0]==ranks[1] and ranks[2]==1


def test_robust_normalization_finite() -> None:
    x=np.array([[1,np.nan,1],[1,2,3]],float); assert np.isfinite(robust_channel_normalize(x)).all()


def test_feature_order_is_fixed() -> None:
    assert len(CAUSAL_FEATURE_NAMES)==6 and CAUSAL_FEATURE_NAMES[0]=="cp_preictal_suppression_rank" and CAUSAL_FEATURE_NAMES[-1]=="cp_cross_seizure_source_consistency"
