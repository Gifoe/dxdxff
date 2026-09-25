import pandas as pd

from exp_ez_hybrid import build_lcbo_target_lookup


def test_physiology_only_lookup_has_no_teacher_score() -> None:
    frame = pd.DataFrame([{
        "subject_id": "p1", "channel_name": "a", "label_ez": 0.0, "label_nez": 1.0,
        "pseudo_core_q": 0.5, "teacher_mode": "physiology_only", "teacher_score_available": False,
        "a9v3_oof_score": float("nan"), "teacher_nez_score": float("nan"),
    }])
    lookup = build_lcbo_target_lookup(frame, target_semantics="nez", teacher_mode="physiology_only")
    item = lookup[("p1", "a")]
    assert item["teacher_mode"] == "physiology_only"
    assert item["teacher_score_available"] is False
    assert item["teacher_score"] != item["teacher_score"]

