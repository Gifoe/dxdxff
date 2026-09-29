"""Synthetic parity with the validated historical nine-feature A1 operator."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

from prepare_ictal_train import NAMES, views_exact_a1


def test_ictal_reference_exact_historical_parity():
    source = (Path(__file__).resolve().parents[2] /
              "pareset_ez_full_original_b0_seed42_v1" / "code" / "historical" /
              "P23_TRN_NEZ_80" / "neuroez_c" / "evidence_views.py")
    spec = importlib.util.spec_from_file_location("historical_a1_evidence_views", source)
    assert spec and spec.loader
    historical = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = historical
    spec.loader.exec_module(historical)
    rng = np.random.default_rng(42)
    complete = rng.normal(size=(59, 4, len(historical.BASE_SPECTRAL_FEATURE_NAMES))).astype(np.float32)
    indices = [historical.BASE_SPECTRAL_FEATURE_NAMES.index(name) for name in NAMES]
    selected = complete[:, :, indices]
    centers = np.arange(59, dtype=np.float32) - 20
    expected = historical.b0_self_reference_features(complete, centers)
    observed = views_exact_a1(selected, centers)
    assert expected.shape == observed.shape == (59, 4, 36)
    np.testing.assert_array_equal(observed, expected)
