from __future__ import annotations

import pandas as pd
import pytest
import torch

from neuroez_c.p21_v3_anchor import V3AnchorStore, orient_v3_ledger, probability_to_logit, robust_patient_standardize_v3


def test_orientation_probability_and_robust_standardization():
    source = pd.DataFrame({"subject_id": ["a"], "channel_name": ["A1"], "a9v3_oof_score": [0.8]})
    frame, audit = orient_v3_ledger(source, "P(EZ)")
    assert frame.v3_score_nez.iloc[0] == pytest.approx(0.2)
    assert frame.v3_logit_nez.iloc[0] == pytest.approx(probability_to_logit([0.2])[0])
    assert audit["orientation_transform"] == "1-a9v3_oof_score"
    values = torch.tensor([[1.0, 2.0, 3.0, 99.0], [2.0, 2.0, 2.0, 0.0]])
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 0]], dtype=torch.bool)
    out = robust_patient_standardize_v3(values, mask, mask)
    assert out["v3_anchor_standardized"][0, 1] == 0
    assert out["v3_anchor_standardized"][0, 3] == 0
    assert torch.isfinite(out["v3_anchor_standardized"]).all()
    assert out["v3_anchor_iqr"][1] == 0


def test_screening_exact_match_duplicate_and_patient_count(tmp_path):
    path = tmp_path / "v3.csv"
    pd.DataFrame({
        "subject_id": ["A", "A", "B"], "channel_name": ["A1", "A2", "B1"],
        "v3_score_nez": [0.8, 0.2, 0.6],
    }).to_csv(path, index=False)
    store = V3AnchorStore(path, mode="precomputed_oof_screening", score_semantics="P(NEZ)", expected_patients=2)
    score, _, valid, audit = store.align("A", ["A1", "A10"])
    assert score.tolist() == pytest.approx([0.8, 0.0])
    assert valid.tolist() == [True, False]
    assert audit["formal_deployable"] is False
    with pytest.raises(ValueError, match="expected 3"):
        V3AnchorStore(path, mode="precomputed_oof_screening", score_semantics="P(NEZ)", expected_patients=3)
    duplicate = pd.read_csv(path)
    duplicate = pd.concat([duplicate, duplicate.iloc[[0]]], ignore_index=True)
    duplicate.to_csv(path, index=False)
    with pytest.raises(ValueError, match="duplicate"):
        V3AnchorStore(path, mode="precomputed_oof_screening", score_semantics="P(NEZ)", expected_patients=2)


def test_strict_provenance_fail_closed(tmp_path):
    path = tmp_path / "strict.csv"
    base = {
        "outer_fold": [1], "inner_fold": [1], "split_role": ["inner_oof"],
        "subject_id": ["a"], "channel_name": ["A1"], "v3_score_nez": [0.4],
        "v3_model_seed": [42], "v3_fit_subject_hash": ["hash"],
        "v3_heldout_subject": ["a"], "score_semantics": ["P(NEZ)"], "true_count_used": [False],
    }
    pd.DataFrame(base).to_csv(path, index=False)
    store = V3AnchorStore(path, mode="nested_fold_safe", score_semantics="P(NEZ)", expected_patients=1)
    assert store.formal_deployable is True
    pd.DataFrame({**base, "v3_heldout_subject": ["b"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="not held out"):
        V3AnchorStore(path, mode="nested_fold_safe", score_semantics="P(NEZ)", expected_patients=1)
    pd.DataFrame({**base, "true_count_used": [True]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="true count"):
        V3AnchorStore(path, mode="nested_fold_safe", score_semantics="P(NEZ)", expected_patients=1)

