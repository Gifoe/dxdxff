from types import SimpleNamespace

import json

from exp_ez_hybrid import compute_lcbo_config_hash, load_valid_lcbo_fold_resume


def test_fold_resume_hash_changes_when_protocol_config_changes() -> None:
    base = SimpleNamespace(random_seed=42, positive_label="nez", experiment_mode="NEZ_S5_8_ASYMMETRIC", teacher_mode="physiology_only", lcbo_loss_mode="asymmetric_clean_nez_lcbo", window_cache_path="cache-a", latent_core_target_dir="targets", model_dim=32, num_heads=2, dropout=0.4, temporal_pooling="mean", record_pooling="mean", lambda_core_rank=0.2, lambda_soft_mrr=0.05, lambda_subset=0.02, lambda_core_distill=0.0, core_rank_margin=0.05, soft_mrr_tau=0.1, subset_eps=0.05, epochs=2, patience=0)
    changed_values = vars(base).copy()
    changed_values["eval_score_fusion_gamma"] = 0.05
    changed = SimpleNamespace(**changed_values)
    assert compute_lcbo_config_hash(base) != compute_lcbo_config_hash(changed)
    assert compute_lcbo_config_hash(base) == compute_lcbo_config_hash(base)


def test_fold_resume_requires_matching_hash_checkpoint_and_prediction(tmp_path) -> None:
    checkpoint = tmp_path / "model.pth"
    prediction = tmp_path / "test.csv"
    marker = tmp_path / "fold_complete.json"
    checkpoint.write_bytes(b"checkpoint")
    prediction.write_text("subject_id,channel_name\np1,a\n", encoding="utf-8")
    marker.write_text(
        json.dumps({"status": "complete", "config_hash": "abc", "checkpoint_path": str(checkpoint)}),
        encoding="utf-8",
    )
    assert load_valid_lcbo_fold_resume(marker, expected_config_hash="abc", expected_prediction_path=prediction)
    assert load_valid_lcbo_fold_resume(marker, expected_config_hash="changed", expected_prediction_path=prediction) is None
    prediction.unlink()
    assert load_valid_lcbo_fold_resume(marker, expected_config_hash="abc", expected_prediction_path=prediction) is None
