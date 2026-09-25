from __future__ import annotations

from typing import Any, Mapping


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


def assert_fixed_all90_protocol(args: Any, patient_index: Any = None, outer_splits: Any = None) -> dict[str, Any]:
    errors: list[str] = []
    positive_label = str(getattr(args, "positive_label", "")).lower()
    split_strategy = str(getattr(args, "split_strategy", "")).lower()
    n_splits = int(getattr(args, "n_splits", -1))
    random_seed = int(getattr(args, "random_seed", -1))
    drop_high_ez_fraction_lzu = bool(getattr(args, "drop_high_ez_fraction_lzu", True))

    if positive_label not in {"ez", "nez"}:
        errors.append("positive_label must be 'ez' or 'nez'")
    if split_strategy != "5fold":
        errors.append("split_strategy must be '5fold'")
    if n_splits != 5:
        errors.append("n_splits must be 5")
    if random_seed != 42:
        errors.append("random_seed must be 42")
    if drop_high_ez_fraction_lzu:
        errors.append("drop_high_ez_fraction_lzu must be false")

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
    if n_patients is not None and n_patients != 90:
        errors.append(f"patient_index must contain 90 patients, got {n_patients}")
    n_outer_splits = None
    if outer_splits is not None:
        n_outer_splits = len(outer_splits)
        if n_outer_splits != 5:
            errors.append(f"outer_splits must contain 5 splits, got {n_outer_splits}")

    if errors:
        raise ValueError("A10 fixed all90 protocol violation: " + "; ".join(errors))

    return {
        "protocol_name": "fixed_all90_validation_threshold",
        "positive_label": positive_label,
        "split_strategy": split_strategy,
        "n_splits": n_splits,
        "random_seed": random_seed,
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
        "true_k_usage": "ranking_diagnostic_only",
        "primary_hard_label_rule": "validation_only_threshold",
        "center_usage": "reporting_and_validation_diagnostics_only",
    }
