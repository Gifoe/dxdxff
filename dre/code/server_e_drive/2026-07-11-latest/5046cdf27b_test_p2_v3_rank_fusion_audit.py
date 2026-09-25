from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.audit_p2_v3_rank_fusion import (
    CANDIDATES,
    add_candidates,
    load_and_align,
    normalize_channel_name,
    paired_bootstrap,
    patient_metrics,
)


def _frame(score_name="score_ez", score=None, duplicate=False, fold=1, label_ez=None):
    score = [0.9, 0.2, 0.8, 0.1] if score is None else score
    label_ez = [1, 0, 0, 1] if label_ez is None else label_ez
    rows = []
    for subject, offset in [("hup:s1", 0), ("hup:s2", 2)]:
        for i in range(2):
            rows.append({"subject_id": subject, "channel_name": f" A{i+offset} ", "outer_fold": fold, "label_ez": label_ez[i+offset], score_name: score[i+offset]})
    if duplicate: rows.append(dict(rows[0]))
    return pd.DataFrame(rows)


def _p2():
    df = _frame()
    df["direct_nez_logit"] = -df["score_ez"]
    return df.drop(columns=["score_ez"])


def test_channel_normalization():
    assert normalize_channel_name("  a 1′  ") == "A1'"
    assert normalize_channel_name("A 1’") == "A1'"


def test_duplicate_key_rejection(tmp_path):
    p2, v3 = _p2(), _frame()
    p2 = pd.concat([p2, p2.iloc[[0]]])
    with pytest.raises(ValueError, match="duplicate"):
        load_and_align(_write(p2, tmp_path / "p2.csv"), _write(v3, tmp_path / "v3.csv"), v3_score_column="score_ez", v3_score_semantics="ez_score")


def test_fold_mismatch_rejection(tmp_path):
    p2, v3 = _p2(), _frame(fold=2)
    with pytest.raises(ValueError, match="outer_fold"):
        load_and_align(_write(p2, tmp_path / "p2.csv"), _write(v3, tmp_path / "v3.csv"), v3_score_column="score_ez", v3_score_semantics="ez_score")


def test_label_mismatch_rejection(tmp_path):
    p2, v3 = _p2(), _frame(label_ez=[0, 1, 0, 1])
    with pytest.raises(ValueError, match="label"):
        load_and_align(_write(p2, tmp_path / "p2.csv"), _write(v3, tmp_path / "v3.csv"), v3_score_column="score_ez", v3_score_semantics="ez_score")


def test_ez_score_orientation(tmp_path):
    p2, v3 = _p2(), _frame(score_name="score_nez", score=[0.1, 0.8, 0.7, 0.2])
    matched, _ = load_and_align(_write(p2, tmp_path / "p2.csv"), _write(v3, tmp_path / "v3.csv"), v3_score_column="score_nez", v3_score_semantics="nez_probability")
    assert np.allclose(matched["p2_ez_score"], p2["direct_nez_logit"] * -1)
    assert np.allclose(matched["v3_ez_score"], 1 - v3["score_nez"])


def test_percentile_rank_direction_and_candidates(tmp_path):
    matched, _ = load_and_align(_write(_p2(), tmp_path / "p2.csv"), _write(_frame(), tmp_path / "v3.csv"), v3_score_column="score_ez", v3_score_semantics="ez_score")
    out = add_candidates(matched)
    assert out.loc[out["p2_ez_score"].idxmax(), "p2_rank"] == 1.0
    assert tuple(c for c in CANDIDATES if c in out.columns) == CANDIDATES


def test_exact_true_k_selection():
    frame = pd.DataFrame({"label_ez_p2": [1, 0, 1, 0], "P2": [0.2, 0.9, 0.8, 0.1]})
    metrics = patient_metrics(frame, "P2")
    assert metrics["true_ez_count"] == 2
    assert metrics["predicted_ez_count"] == 2
    assert metrics["predicted_ez_mask"].tolist() == [0, 1, 1, 0]


def test_rrf_and_fusion_are_frozen(tmp_path):
    matched, _ = load_and_align(_write(_p2(), tmp_path / "p2.csv"), _write(_frame(), tmp_path / "v3.csv"), v3_score_column="score_ez", v3_score_semantics="ez_score")
    out = add_candidates(matched)
    assert np.allclose(out["RankAvg_P2_70"], .7 * out["p2_rank"] + .3 * out["v3_rank"])
    assert np.allclose(out["RankAvg_V3_70"], .3 * out["p2_rank"] + .7 * out["v3_rank"])
    assert np.allclose(out["RRF_k60"], 1/(60+out["p2_rank_position"]) - 1/(60+out["v3_rank_position"]))
    assert set(CANDIDATES) == {"P2", "V3", "RankAvg_50_50", "RankAvg_P2_70", "RankAvg_V3_70", "RRF_k60"}


def test_patient_macro_metric_and_bootstrap_determinism():
    patients = pd.DataFrame({"P2_patient_macro_f1": [.5, .7], "RankAvg_50_50_patient_macro_f1": [.6, .6], "RankAvg_P2_70_patient_macro_f1": [.5, .8], "RankAvg_V3_70_patient_macro_f1": [.4, .9], "RRF_k60_patient_macro_f1": [.5, .7]})
    first = paired_bootstrap(patients, 50, 42)
    second = paired_bootstrap(patients, 50, 42)
    pd.testing.assert_frame_equal(first, second)


def _write(frame, path):
    frame.to_csv(path, index=False)
    return path
