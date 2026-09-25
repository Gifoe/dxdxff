from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


STEP4B_STATIC_TOP20_PROTOCOL = "fixed_all90_step4b_static_top20_nez"
STEP4B_N6_DUALVIEW_EMA_PROTOCOL = "fixed_all90_step4b_n6_dualview_ema"
STEP4C_CANE_SET_NEZ_PROTOCOL = "fixed_all90_step4c_cane_set_nez"
SENSITIVITY80_CANE_PATH_CP_NEZ_PROTOCOL = "sensitivity80_step4d_cane_path_cp_nez"
BASE_FIXED_ALL90_PROTOCOL = "fixed_all90_patient_topk_ez"
STEP4B_STATIC_TOP20_FEATURES = (
    "early_high_gamma_slope",
    "early_line_length_slope",
    "onset_latency_high_gamma",
    "onset_latency_line_length",
    "onset_rank_high_gamma",
    "onset_rank_line_length",
    "high_gamma_top20pct_mean",
    "line_length_top20pct_mean",
)


def _contains_center_id(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return any(part.strip().lower() == "center_id" for part in value.replace(";", ",").split(","))
    if isinstance(value, (list, tuple, set)):
        return any(str(part).strip().lower() == "center_id" for part in value)
    return False


def _patient_count(patient_index: Any) -> int | None:
    if patient_index is None:
        return None
    if isinstance(patient_index, Mapping):
        return len(patient_index)
    try:
        return len(patient_index)
    except TypeError:
        return None


def _load_cache_audit(args: Any) -> dict[str, Any]:
    value = str(getattr(args, "fixed_all90_cache_audit_path", "") or "").strip()
    if not value:
        return {}
    path = Path(value)
    if not path.is_file():
        raise ValueError(f"fixed_all90_cache_audit_path does not exist: {path}")
    try:
        audit = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to read fixed All90 cache audit {path}: {exc}") from exc
    if not isinstance(audit, dict) or audit.get("status") != "passed":
        raise ValueError(f"Fixed All90 cache audit did not pass: {path}")
    return audit


def _load_dual_view_audit(args: Any) -> dict[str, Any]:
    path_value = str(getattr(args, "dual_view_cache_audit_path", "") or "").strip()
    if not path_value:
        return {}
    path = Path(path_value)
    if not path.is_file():
        raise ValueError(f"dual_view_cache_audit_path does not exist: {path}")
    try:
        audit = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to read dual-view cache audit {path}: {exc}") from exc
    if not isinstance(audit, dict) or audit.get("status") != "passed":
        raise ValueError(f"Dual-view cache audit did not pass: {path}")
    return audit


def _outer_split_errors(outer_splits: Any, expected_subjects: set[str]) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    seen_test: set[str] = set()
    duplicate_test: set[str] = set()
    fold_rows: list[dict[str, int]] = []
    for position, split in enumerate(outer_splits or [], start=1):
        fold_idx = int(split.get("fold_idx", position))
        train = set(map(str, split.get("train_subjects", [])))
        test = set(map(str, split.get("test_subjects", [])))
        overlap = sorted(train & test)
        if overlap:
            errors.append(f"fold {fold_idx} train/test patient leakage: {overlap[:5]}")
        if expected_subjects and train | test != expected_subjects:
            errors.append(f"fold {fold_idx} does not partition the fixed {len(expected_subjects)}-patient cohort")
        duplicate_test.update(seen_test & test)
        seen_test.update(test)
        fold_rows.append({"fold_idx": fold_idx, "n_train": len(train), "n_test": len(test)})
    if duplicate_test:
        errors.append(f"patients occur in multiple held-out folds: {sorted(duplicate_test)[:5]}")
    if expected_subjects and seen_test != expected_subjects:
        errors.append(
            f"held-out test union must contain all {len(expected_subjects)} patients exactly once; "
            f"missing={sorted(expected_subjects - seen_test)[:5]}, extra={sorted(seen_test - expected_subjects)[:5]}"
        )
    return errors, {
        "fold_subject_counts": fold_rows,
        "patient_disjoint_folds": not errors,
        "test_subject_union_count": len(seen_test),
        "complete_oof_coverage": bool(expected_subjects) and seen_test == expected_subjects and not duplicate_test,
    }


def assert_fixed_all90_protocol(
    args: Any,
    patient_index: Any = None,
    outer_splits: Any = None,
    cache_feature_names: Any = None,
) -> dict[str, Any]:
    errors: list[str] = []
    protocol_name = str(
        getattr(args, "fixed_all90_protocol_name", "fixed_all90_patient_topk_ez")
        or "fixed_all90_patient_topk_ez"
    ).strip()
    cache_audit = _load_cache_audit(args)
    dual_view_audit = _load_dual_view_audit(args)
    causal_audit = _load_causal_audit(args)
    positive_label = str(getattr(args, "positive_label", "")).lower()
    score_semantics = str(getattr(args, "score_semantics", "")).lower()
    split_strategy = str(getattr(args, "split_strategy", "")).lower()
    n_splits = int(getattr(args, "n_splits", -1))
    random_seed = int(getattr(args, "random_seed", -1))
    val_ratio = float(getattr(args, "val_ratio", 0.20))
    drop_high_ez_fraction_lzu = bool(getattr(args, "drop_high_ez_fraction_lzu", True))
    train_dropout_count = int(getattr(args, "train_subject_dropout_count", 0) or 0)
    train_dropout_file = str(getattr(args, "train_subject_dropout_file", "") or "").strip()

    step4b_protocols = {
        STEP4B_STATIC_TOP20_PROTOCOL,
        STEP4B_N6_DUALVIEW_EMA_PROTOCOL,
        STEP4C_CANE_SET_NEZ_PROTOCOL,
        SENSITIVITY80_CANE_PATH_CP_NEZ_PROTOCOL,
    }
    if protocol_name not in {BASE_FIXED_ALL90_PROTOCOL, *step4b_protocols}:
        errors.append(f"unsupported fixed All90 protocol_name: {protocol_name!r}")
    expected_positive_label = "nez" if protocol_name in step4b_protocols else "ez"
    if positive_label != expected_positive_label:
        errors.append(f"positive_label must be '{expected_positive_label}'")
    if split_strategy != "5fold":
        errors.append("split_strategy must be '5fold'")
    if n_splits != 5:
        errors.append("n_splits must be 5")
    if random_seed != 42:
        errors.append("random_seed must be 42")
    if abs(val_ratio - 0.20) > 1e-12:
        errors.append("val_ratio must be 0.2")
    if drop_high_ez_fraction_lzu:
        errors.append("drop_high_ez_fraction_lzu must be false")
    if train_dropout_count < 0:
        errors.append("train_subject_dropout_count must be non-negative")
    if train_dropout_count > 0 and not train_dropout_file:
        errors.append("train_subject_dropout_file is required when train_subject_dropout_count is positive")

    for attr in (
        "model_input_features",
        "input_features",
        "feature_columns",
        "b0_extra_features",
        "clinical_feature_columns",
    ):
        if _contains_center_id(getattr(args, attr, None)):
            errors.append(f"{attr} must not include center_id as a model input feature")

    n_patients = _patient_count(patient_index)
    if n_patients is None and cache_audit and protocol_name != SENSITIVITY80_CANE_PATH_CP_NEZ_PROTOCOL:
        n_patients = int(cache_audit.get("n_patients", -1))
    expected_patients = 80 if protocol_name == SENSITIVITY80_CANE_PATH_CP_NEZ_PROTOCOL else 90
    if n_patients is not None and n_patients != expected_patients:
        errors.append(f"patient_index must contain {expected_patients} patients, got {n_patients}")
    n_outer_splits = None
    if outer_splits is not None:
        n_outer_splits = len(outer_splits)
        if n_outer_splits != 5:
            errors.append(f"outer_splits must contain 5 splits, got {n_outer_splits}")

    split_audit: dict[str, Any] = {}
    if patient_index is not None and outer_splits is not None:
        split_items = list(outer_splits)
        if all(isinstance(split, Mapping) for split in split_items):
            expected_subjects = set(map(str, patient_index.keys() if isinstance(patient_index, Mapping) else patient_index))
            split_errors, split_audit = _outer_split_errors(split_items, expected_subjects)
            errors.extend(split_errors)
        elif protocol_name in step4b_protocols:
            errors.append("Step4B outer_splits must expose train_subjects and test_subjects for leakage audit")
    elif cache_audit and protocol_name != SENSITIVITY80_CANE_PATH_CP_NEZ_PROTOCOL:
        n_outer_splits = int(cache_audit.get("n_outer_splits", -1))
        if n_outer_splits != 5:
            errors.append(f"cache audit must contain 5 outer splits, got {n_outer_splits}")
        if not bool(cache_audit.get("patient_disjoint", False)):
            errors.append("cache audit reports patient leakage between folds")
        if int(cache_audit.get("test_union_count", -1)) != 90:
            errors.append("cache audit held-out test union must contain 90 patients")
        split_audit = {
            "fold_subject_counts": list(cache_audit.get("folds", [])),
            "patient_disjoint_folds": bool(cache_audit.get("patient_disjoint", False)),
            "test_subject_union_count": int(cache_audit.get("test_union_count", -1)),
            "complete_oof_coverage": bool(cache_audit.get("test_union_matches_all90", False)),
        }

    if protocol_name in step4b_protocols:
        if score_semantics != "nez_probability":
            errors.append("score_semantics must be 'nez_probability'")
        if int(getattr(args, "require_n_patients", -1) or -1) != expected_patients:
            errors.append(f"require_n_patients must be {expected_patients}")
        configured_features = tuple(
            value.strip() for value in str(getattr(args, "physics_state_features", "")).split(",") if value.strip()
        )
        if configured_features != STEP4B_STATIC_TOP20_FEATURES:
            errors.append("physics_state_features must exactly match the 8 Step4B static-top20 features")
        if str(getattr(args, "physics_feature_parts", "")).strip().lower() != "abs":
            errors.append("physics_feature_parts must be 'abs'")
        if not bool(getattr(args, "use_physics_dynamics", False)):
            errors.append("use_physics_dynamics must be true")
        if not bool(getattr(args, "use_channel_attention", False)):
            errors.append("use_channel_attention must be true")
        if not bool(getattr(args, "use_patient_relative_z", False)):
            errors.append("use_patient_relative_z must be true")
        if protocol_name != SENSITIVITY80_CANE_PATH_CP_NEZ_PROTOCOL and str(getattr(args, "group_robust_mode", "none")).lower() != "none":
            errors.append("group_robust_mode must be 'none' so center_id is not used by training")
        disabled_flags = (
            "use_diffusion_residual",
            "use_ez_ranking_loss",
            "use_hard_topk_loss",
            "use_negative_anchor_head",
            "use_two_expert_router",
            "use_feature_separated_two_expert",
            "use_broad_ez_mil_loss",
            "use_a9v8_lcbo",
            "use_teacher_anchor_eval",
            "teacher_anchor_apply_to_train_loss",
        )
        for flag in disabled_flags:
            if bool(getattr(args, flag, False)):
                errors.append(f"{flag} must be false for the Step4B single-factor experiment")
        observed_feature_names = (
            list(map(str, cache_feature_names))
            if cache_feature_names is not None
            else list(map(str, cache_audit.get("required_features_present", [])))
        )
        missing_features = sorted(set(STEP4B_STATIC_TOP20_FEATURES) - set(observed_feature_names))
        if missing_features and not bool(getattr(args, "dry_run_config_only", False)):
            errors.append(f"training cache is missing Step4B features: {missing_features}")
        if cache_audit and int(cache_audit.get("window_feature_count", -1)) != 28:
            errors.append("Step4B training cache must contain exactly 28 window features")
        use_n6 = bool(getattr(args, "use_n6_dual_view_ema", False))
        if protocol_name == STEP4B_STATIC_TOP20_PROTOCOL and use_n6:
            errors.append("use_n6_dual_view_ema must be false for the Step4B baseline")
        if protocol_name == STEP4B_N6_DUALVIEW_EMA_PROTOCOL:
            if not use_n6:
                errors.append("use_n6_dual_view_ema must be true for the N6 protocol")
            if train_dropout_count != 0 or train_dropout_file:
                errors.append("N6 fixed-All90 protocol forbids patient dropout")
            if str(getattr(args, "early_stop_metric", "")).lower() != "balanced_patient_auprc_hmean":
                errors.append("early_stop_metric must be 'balanced_patient_auprc_hmean' for N6")
            if str(getattr(args, "loss_mode", "")).lower() != "n6_dualview_ema_robust":
                errors.append("loss_mode must be 'n6_dualview_ema_robust' for N6")
            required_disabled = ("use_view_gated_fusion", "use_edf_quality_weighting")
            for flag in required_disabled:
                if bool(getattr(args, flag, False)):
                    errors.append(f"{flag} must be false for N6")
            if not dual_view_audit:
                errors.append("N6 requires dual_view_cache_audit_path")
            elif int(dual_view_audit.get("n_matched_patients", -1)) != 90:
                errors.append("N6 requires 90 matched raw patients")
            elif float(dual_view_audit.get("channel_match_rate", 0.0)) < float(getattr(args, "raw_min_channel_match_rate", 0.95)) or float(dual_view_audit.get("window_match_rate", 0.0)) < float(getattr(args, "raw_min_window_match_rate", 0.90)):
                errors.append("N6 dual-view raw coverage is below the configured formal threshold")
            expected_n6 = {
                "n6_ema_decay": 0.995,
                "n6_warmup_epochs": 5.0,
                "n6_noise_discount": 0.50,
                "n6_reliability_min": 0.50,
                "n6_feature_aux_weight": 0.15,
                "n6_raw_aux_weight": 0.15,
                "n6_rank_loss_weight": 0.05,
                "n6_rank_margin": 0.10,
                "n6_gate_anchor": 0.70,
                "n6_gate_loss_weight": 0.005,
                "n6_initial_feature_gate": 0.75,
                "raw_target_samples": 500.0,
                "raw_target_sampling_rate": 250.0,
                "raw_lr_multiplier": 2.0,
            }
            for field, expected in expected_n6.items():
                observed = float(getattr(args, field, expected))
                if abs(observed - expected) > 1e-12:
                    errors.append(f"{field} must be {expected} for the fixed N6 experiment")
        if protocol_name == STEP4C_CANE_SET_NEZ_PROTOCOL:
            if not bool(getattr(args, "use_cane_set_nez", False)):
                errors.append("use_cane_set_nez must be true for the CANE protocol")
            if use_n6:
                errors.append("use_n6_dual_view_ema must be false for the CANE protocol")
            if int(getattr(args, "split_seed", -1)) != 42:
                errors.append("split_seed must be 42 for the CANE protocol")
            if train_dropout_count != 0 or train_dropout_file:
                errors.append("CANE fixed-All90 protocol forbids patient dropout")
            if str(getattr(args, "loss_mode", "")).lower() != "cane_set_nez_countfree":
                errors.append("loss_mode must be 'cane_set_nez_countfree' for CANE")
            if str(getattr(args, "early_stop_metric", "")).lower() != "cane_predcount_patient_macro_f1":
                errors.append("early_stop_metric must be 'cane_predcount_patient_macro_f1' for CANE")
            if str(getattr(args, "class_weight_mode", "")).lower() != "none":
                errors.append("class_weight_mode must be 'none' for CANE")
            if str(getattr(args, "ez_negative_weight", "")) not in {"1", "1.0"}:
                errors.append("ez_negative_weight must be 1 for CANE")
            if int(getattr(args, "min_epochs_before_early_stop", 0)) < int(getattr(args, "cane_rank_start_epoch", 15)):
                errors.append("min_epochs_before_early_stop must be at least cane_rank_start_epoch for CANE")
            for flag in (
                "use_n6_dual_view_ema",
                "use_diffusion_residual",
                "use_ez_ranking_loss",
                "use_hard_topk_loss",
                "use_negative_anchor_head",
                "use_two_expert_router",
                "use_feature_separated_two_expert",
                "use_broad_ez_mil_loss",
                "use_a9v8_lcbo",
                "use_teacher_anchor_eval",
                "teacher_anchor_apply_to_train_loss",
                "use_view_gated_fusion",
                "use_edf_quality_weighting",
            ):
                if bool(getattr(args, flag, False)):
                    errors.append(f"{flag} must be false for CANE")
        if protocol_name == SENSITIVITY80_CANE_PATH_CP_NEZ_PROTOCOL:
            direct_outer_only = bool(getattr(args, "cane_direct_outer_only", False))
            if not bool(getattr(args, "use_cane_path_cp_nez", False)):
                errors.append("use_cane_path_cp_nez must be true")
            if str(getattr(args, "cohort_mode", "")).lower() != "sensitivity80":
                errors.append("cohort_mode must be sensitivity80")
            if int(getattr(args, "outer_split_seed", -1)) != 42:
                errors.append("outer_split_seed must be 42")
            if not direct_outer_only and int(getattr(args, "inner_split_seed", -1)) != 42:
                errors.append("nested inner_split_seed must be 42")
            if not direct_outer_only and int(getattr(args, "inner_splits", -1)) != 4:
                errors.append("nested inner_splits must be 4")
            if str(getattr(args, "loss_mode", "")).lower() != "cane_path_cp_nez":
                errors.append("loss_mode must be cane_path_cp_nez")
            expected_early_stop = "patient_macro_f1" if direct_outer_only and str(
                getattr(args, "cane_selection_objective", "auprc")
            ).lower() == "f1" else "balanced_patient_auprc_hmean"
            if str(getattr(args, "early_stop_metric", "")).lower() != expected_early_stop:
                errors.append(f"early_stop_metric must be {expected_early_stop}")
            if str(getattr(args, "class_weight_mode", "")).lower() != "none" or str(getattr(args, "ez_negative_weight", "")) not in {"1", "1.0"}:
                errors.append("CANE-PATH-CP requires no class weighting")
            if train_dropout_count != 0 or train_dropout_file:
                errors.append("Sensitivity80 protocol forbids train-only dropout")
            if not direct_outer_only and not bool(getattr(args, "center_balanced_batches", False)):
                errors.append("center_balanced_batches must be enabled")
            if not bool(getattr(args, "use_causal_propagation_residual", False)):
                errors.append("P2 requires use_causal_propagation_residual")
            if not causal_audit and not bool(getattr(args, "dry_run_config_only", False)):
                errors.append("P2 requires a passed causal cache audit")
            for flag in (
                "use_cane_set_nez", "use_n6_dual_view_ema", "use_diffusion_residual",
                "use_ez_ranking_loss", "use_hard_topk_loss", "use_negative_anchor_head",
                "use_two_expert_router", "use_feature_separated_two_expert",
                "use_broad_ez_mil_loss", "use_a9v8_lcbo", "use_teacher_anchor_eval",
                "teacher_anchor_apply_to_train_loss", "use_view_gated_fusion", "use_edf_quality_weighting",
            ):
                if bool(getattr(args, flag, False)):
                    errors.append(f"{flag} must be false for CANE-PATH-CP")

    if errors:
        raise ValueError("Fixed All90 protocol violation: " + "; ".join(errors))

    audit = {
        "protocol_name": protocol_name,
        "positive_label": positive_label,
        "score_semantics": score_semantics,
        "split_strategy": split_strategy,
        "n_splits": n_splits,
        "random_seed": random_seed,
        "val_ratio": val_ratio,
        "drop_high_ez_fraction_lzu": drop_high_ez_fraction_lzu,
        "channel_pooling_mode": str(getattr(args, "channel_pooling_mode", "mean")),
        "early_pool_frac": float(getattr(args, "early_pool_frac", 0.25)),
        "lse_pool_tau": float(getattr(args, "lse_pool_tau", 1.0)),
        "pooling_projection_dim": int(getattr(args, "model_dim", 32)),
        "dynamic_pooling_enabled": str(getattr(args, "channel_pooling_mode", "mean")).lower() != "mean",
        "loss_mode": str(getattr(args, "loss_mode", "")),
        "rank_loss_weight": float(getattr(args, "rank_loss_weight", 0.0)),
        "hard_pairwise_weight": float(getattr(args, "hard_pairwise_weight", 0.0)),
        "soft_topk_weight": float(getattr(args, "soft_topk_weight", 0.0)),
        "first_rank_weight": float(getattr(args, "first_rank_weight", 0.0)),
        "diversity_weight": float(getattr(args, "diversity_weight", 0.0)),
        "group_robust_mode": str(getattr(args, "group_robust_mode", "none")),
        "group_dro_eta": float(getattr(args, "group_dro_eta", 0.05)),
        "n_patients": n_patients,
        "n_outer_splits": n_outer_splits,
        "center_as_input_allowed": False,
        "no_center_features": True,
        "true_k_as_model_input_allowed": False,
        "true_k_usage": "training_loss_and_evaluation_only",
        "center_usage": "reporting_and_validation_diagnostics_only",
        "train_subject_dropout_mode": "fit_only_validation_and_test_unchanged",
        "train_subject_dropout_file": train_dropout_file,
        "train_subject_dropout_count": train_dropout_count,
        "train_subject_dropout_seed": int(getattr(args, "train_subject_dropout_seed", -1) or -1),
    }
    audit.update(split_audit)
    if protocol_name in step4b_protocols:
        audit.update({
            "required_step4b_features": list(STEP4B_STATIC_TOP20_FEATURES),
            "required_step4b_feature_count": len(STEP4B_STATIC_TOP20_FEATURES),
            "feature_mode": "STEP4B_STATIC_TOP20_ALL90",
            "cache_audit_path": str(getattr(args, "fixed_all90_cache_audit_path", "")),
            "center_as_input_allowed": False,
        })
        if protocol_name == STEP4B_N6_DUALVIEW_EMA_PROTOCOL:
            audit.update({
                "method": "N6F_NEZ_DualView_EMA_RobustRank",
                "feature_cache_path": str(getattr(args, "window_cache_path", "")),
                "raw_cache_path": str(getattr(args, "raw_window_cache_path", "")),
                "dual_view_cache_audit_path": str(getattr(args, "dual_view_cache_audit_path", "")),
                "raw_patient_coverage": float(dual_view_audit.get("raw_patient_coverage", 0.0)),
                "raw_channel_match_rate": float(dual_view_audit.get("channel_match_rate", 0.0)),
                "raw_window_match_rate": float(dual_view_audit.get("window_match_rate", 0.0)),
                "raw_target_samples": int(getattr(args, "raw_target_samples", 0)),
                "raw_target_sampling_rate": float(getattr(args, "raw_target_sampling_rate", 0.0)),
                "use_n6_dual_view_ema": True,
                "ema_teacher": True,
                "ema_decay": float(getattr(args, "n6_ema_decay", 0.995)),
                "teacher_receives_gradient": False,
                "teacher_uses_labels": False,
                "clean_nez_full_weight": True,
                "observed_ez_hard_relabeling": False,
                "learned_class_prior": False,
                "propensity_head": False,
                "nnpu": False,
                "posthoc_calibration": "none",
                "center_bias": False,
                "threshold_source": "best_ema_validation_only",
                "early_stop_metric": "balanced_patient_auprc_hmean",
                "test_used_for_training": False,
                "test_used_for_model_selection": False,
                "test_used_for_threshold": False,
                "center_usage": "reporting_only",
                "true_ez_count_used_for_prediction": False,
            })
        elif protocol_name == STEP4C_CANE_SET_NEZ_PROTOCOL:
            audit.update({
                "method": "N7F_CANE_Set_NEZ_CountFree",
                "split_seed": int(getattr(args, "split_seed", 42)),
                "model_seed": int(getattr(args, "model_seed", 42)),
                "label_semantics": "NEZ=1,EZ=0",
                "score_semantics": "P(NEZ)",
                "decision_rule": "predicted_nez_cardinality_topk",
                "true_nez_count_used_for_prediction": False,
                "true_ez_count_used_for_prediction": False,
                "classification_threshold_used": False,
                "predicted_cardinality_used": True,
                "cardinality_target_used_for_training_only": True,
                "center_used_as_model_input": False,
                "test_used_for_training": False,
                "test_used_for_model_selection": False,
                "test_used_for_count_selection": False,
                "test_used_for_ensemble_weighting": False,
            })
        elif protocol_name == SENSITIVITY80_CANE_PATH_CP_NEZ_PROTOCOL:
            direct_outer_only = bool(getattr(args, "cane_direct_outer_only", False))
            audit.update({
                "method": "N8F_CANE_PATH_CP_NEZ_80",
                "analysis_status": "posthoc_sensitivity_not_primary",
                "cohort_status": "posthoc_sensitivity",
                "outer_split_seed": int(getattr(args, "outer_split_seed", 42)),
                "inner_split_seed": None if direct_outer_only else int(getattr(args, "inner_split_seed", 42)),
                "inner_splits": 0 if direct_outer_only else int(getattr(args, "inner_splits", 4)),
                "inner_crossfit_used": not direct_outer_only,
                "model_seed": int(getattr(args, "model_seed", 42)),
                "center_counts": {"hup": 36, "lzu": 21, "multicenter": 15, "pediatric": 8},
                "label_semantics": "NEZ=1,EZ=0",
                "score_semantics": "P(NEZ)",
                "training_mode": "direct_outer_only" if direct_outer_only else "nested_crossfit",
                "selection_objective": str(getattr(args, "cane_selection_objective", "auprc")),
                "decision_rule": "fixed_nez_probability_threshold" if direct_outer_only else "patient_adaptive_standardized_nez_threshold",
                "true_ez_count_used_for_prediction": False,
                "true_nez_count_used_for_prediction": False,
                "oracle_threshold_used_for_prediction": False,
                "global_threshold_used_for_prediction": direct_outer_only,
                "predicted_cardinality_used": False,
                "patient_adaptive_threshold_used": not direct_outer_only,
                "threshold_head_training_source": "disabled" if direct_outer_only else "outer_train_inner_oof_only",
                "threshold_source": (
                    "fold_validation_macro_f1"
                    if direct_outer_only and str(getattr(args, "cane_selection_objective", "auprc")) == "f1"
                    else "fixed_predeclared_probability_threshold"
                    if direct_outer_only
                    else "cross_fitted_patient_adaptive_head"
                ),
                "center_used_as_model_input": False,
                "center_used_for_training_sampler": True,
                "center_used_for_group_loss": True,
                "test_used_for_training": False,
                "test_used_for_model_selection": False,
                "test_used_for_threshold_training": False,
                "test_used_for_ensemble_weighting": False,
                "causal_cache_audit_path": str(getattr(args, "causal_cache_audit_path", "")),
                "causal_cache_status": causal_audit.get("status", "dry_run_not_checked"),
            })
        for key in (
            "target_cache_path",
            "source_cache_path",
            "n_runs",
            "center_distribution",
            "legacy_high_ez_lzu_actual_count",
            "legacy_high_ez_lzu_subjects",
            "legacy_missing_lzu_claimed_count",
            "legacy_missing_lzu_count_matches_claim",
            "required_features_present",
            "required_features_missing",
            "window_feature_count",
            "raw_cache_used",
        ):
            if key in cache_audit:
                audit[key] = cache_audit[key]
    return audit


def _load_causal_audit(args: Any) -> dict[str, Any]:
    path_value = str(getattr(args, "causal_cache_audit_path", "") or "").strip()
    if not path_value:
        return {}
    path = Path(path_value)
    if not path.is_file():
        raise ValueError(f"causal_cache_audit_path does not exist: {path}")
    audit = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(audit, dict) or audit.get("status") != "passed":
        raise ValueError(f"Causal propagation cache audit did not pass: {path}")
    required = {
        "n_patients", "patient_match_rate", "channel_match_rate", "window_match_rate",
        "feature_valid_rate", "valid_rate_by_center", "duplicate_key_count", "nonfinite_count",
    }
    missing = sorted(required - set(audit))
    if missing:
        raise ValueError(f"Causal propagation cache audit is incomplete: {missing}")
    center_rates = audit.get("valid_rate_by_center", {})
    violations = []
    if int(audit["n_patients"]) != 80:
        violations.append("n_patients must be 80")
    if float(audit["patient_match_rate"]) < 1.0:
        violations.append("patient_match_rate must be 1.0")
    if float(audit["channel_match_rate"]) < 0.95:
        violations.append("channel_match_rate must be >=0.95")
    if float(audit["window_match_rate"]) < 0.90:
        violations.append("window_match_rate must be >=0.90")
    if float(audit["feature_valid_rate"]) < 0.95:
        violations.append("feature_valid_rate must be >=0.95")
    for center in ("hup", "lzu", "multicenter", "pediatric"):
        if float(center_rates.get(center, 0.0)) < 0.90:
            violations.append(f"{center} feature_valid_rate must be >=0.90")
    if int(audit["duplicate_key_count"]) != 0 or int(audit["nonfinite_count"]) != 0:
        violations.append("duplicate_key_count and nonfinite_count must be zero")
    if violations:
        raise ValueError("Causal propagation cache audit coverage failed: " + "; ".join(violations))
    return audit


__all__ = [
    "BASE_FIXED_ALL90_PROTOCOL",
    "STEP4B_N6_DUALVIEW_EMA_PROTOCOL",
    "STEP4B_STATIC_TOP20_FEATURES",
    "STEP4B_STATIC_TOP20_PROTOCOL",
    "STEP4C_CANE_SET_NEZ_PROTOCOL",
    "SENSITIVITY80_CANE_PATH_CP_NEZ_PROTOCOL",
    "assert_fixed_all90_protocol",
]
