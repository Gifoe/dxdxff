import pandas as pd
import pytest

from neuroez_c.p2_v3_fusion_protocol import align_fold_ledgers, canonicalize_p2_fusion_ledger, canonicalize_v3_fusion_ledger


def _raw(*, center="hup", label=1, channel="A", fold=1):
    return pd.DataFrame([
        {"subject_id": "hup:s1", "center": center, "outer_fold": fold, "channel_name": channel, "label_nez": label, "direct_nez_logit": .2},
        {"subject_id": "hup:s1", "center": center, "outer_fold": fold, "channel_name": f"{channel}2", "label_nez": 1 - label, "direct_nez_logit": -.2},
    ])


def test_strict_alignment_and_score_canonicalization():
    p2 = canonicalize_p2_fusion_ledger(_raw(), split_role="test")
    v3 = canonicalize_v3_fusion_ledger(_raw(), split_role="test")
    aligned, audit, _ = align_fold_ledgers(p2, v3, outer_fold=1, split_role="test")
    assert audit["passed"] and len(aligned) == 2
    assert 0 < aligned.p2_score_nez.iloc[0] < 1


def test_mismatch_fails_closed():
    p2 = canonicalize_p2_fusion_ledger(_raw(), split_role="test")
    v3 = canonicalize_v3_fusion_ledger(_raw(channel="B"), split_role="test")
    with pytest.raises(ValueError, match="alignment failed"):
        align_fold_ledgers(p2, v3, outer_fold=1, split_role="test")


def test_center_and_label_mismatch_fail_closed():
    p2 = canonicalize_p2_fusion_ledger(_raw(), split_role="test")
    v3 = canonicalize_v3_fusion_ledger(_raw(center="lzu", label=0), split_role="test")
    with pytest.raises(ValueError, match="alignment failed"):
        align_fold_ledgers(p2, v3, outer_fold=1, split_role="test")
