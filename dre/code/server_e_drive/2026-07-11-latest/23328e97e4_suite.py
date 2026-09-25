from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import json
from itertools import combinations

import numpy as np
import pandas as pd

from .audit import audit_anchor, candidate_ceiling_audit
from .anchor_features import build_patient_nez_anchor_features
from .action_simulation import recompute_utility, simulate_patient_policy
from .calibration import ProbabilityCalibrator
from .candidate_pool import CandidateConfig, build_candidate_pools
from .config import A12Config
from .cv import inner_subject_splits, outer_subject_folds
from .evaluation import evaluate_patient_predictions
from .feature_registry import FeatureRegistry
from .matching import select_max_weight_swaps
from .models import CatBoostUtilityModel, EnsembleUtilityModel, LearnedGateModel, SiameseUtilityMLP, TrajectoryTCNUtilityModel
from .pair_dataset import build_pair_dataset, write_pair_cache
from .patient_gate import (LearnedPatientGate, RuleGate, apply_patient_gate,
                           build_patient_action_summary, cross_fit_patient_gate)
from .protocol import ProtocolError, assert_no_leakage
from .provenance import assert_resume_compatible, execution_fingerprint, fingerprint_hash
from .reporting import write_comparison, write_variant_artifacts
from .schemas import build_canonical_ledger, normalize_channel_name
from .thresholds import SelectedThresholds, select_thresholds
from .trajectory_features import build_relative_trajectory_features
from .trajectory_store import PatientTrajectoryStore, RunTrajectory
from .topology_features import parse_channel_topology
from .utils import environment_snapshot, sha256_file, stable_hash, write_json


ALL_VARIANTS = ("A12-V0", "A12-D0", "A12-V1", "A12-V2", "A12-V3", "A12-V4", "A12-V5", "A12-V6", "A12-V7", "A12-V8", "A12-V9", "A12-V10")
VARIANT_FEATURE_GROUPS = {
    "A12-V1": {"v3", "patient_context"},
    "A12-V2": {"v3", "patient_context", "anchor"},
    "A12-V3": {"v3", "patient_context", "anchor", "trajectory"},
    "A12-V4": {"v3", "patient_context", "anchor", "trajectory", "topology"},
    "A12-V5": {"v3", "patient_context", "anchor", "trajectory", "topology", "hnc"},
    "A12-V6": {"v3", "patient_context", "anchor", "trajectory", "topology"},
    "A12-V7": {"v3", "patient_context", "anchor", "trajectory", "topology"},
    "A12-V8": {"v3", "patient_context", "anchor", "trajectory", "topology"},
    "A12-V9": {"v3", "patient_context", "anchor", "trajectory", "topology"},
    "A12-V10": {"v3", "patient_context", "anchor", "trajectory", "topology"},
}


def _result_config_hash(config: A12Config) -> str:
    """Hash result-affecting settings; `resume` only controls execution flow."""
    payload = config.to_dict()
    payload.pop("resume", None)
    return stable_hash(payload)


def resolve_variant_feature_groups(variant: str, available_groups: set[str]) -> tuple[str, ...]:
    """Freeze the pre-registered incremental feature set for one variant."""
    if variant not in VARIANT_FEATURE_GROUPS:
        return tuple()
    return tuple(sorted(VARIANT_FEATURE_GROUPS[variant] & set(available_groups)))


def make_synthetic_ledger(*, n_subjects: int = 10, n_folds: int = 2) -> pd.DataFrame:
    rows = []
    for subject_index in range(n_subjects):
        subject = f"synthetic:{subject_index:03d}"
        labels = [1, 1, 0, 0, 0, 0]
        predicted = [1, 0, 1, 0, 0, 0]
        scores = [0.95, 0.72, 0.82, 0.40, 0.30, 0.20]
        for channel_index, (label, pred, score) in enumerate(zip(labels, predicted, scores), start=1):
            rows.append({"fold_idx": subject_index % n_folds + 1, "subject_id": subject, "center": ("hup", "lzu")[subject_index % 2], "channel_name": f"S{subject_index:02d}{channel_index:02d}", "true_ez": label, "true_nez": 1 - label, "score_ez_probability": score + subject_index * 1e-5, "rank_ez_desc": int(np.argsort(np.argsort(-np.asarray(scores)))[channel_index - 1] + 1), "predicted_ez": pred})
    return build_canonical_ledger(pd.DataFrame(rows), strict=True)[0]


def make_synthetic_capabilities(ledger: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict[str, list[float]]], PatientTrajectoryStore]:
    """Create deterministic 20-feature, multi-seizure capabilities for all variants."""
    store = PatientTrajectoryStore(feature_names=tuple(f"feature_{index}" for index in range(20)))
    for subject_index, (subject_id, group) in enumerate(ledger.groupby("subject_id", sort=True)):
        names = tuple(group.sort_values("old_v3_rank")["channel_name_original"].astype(str))
        scores = group.set_index("channel_name_original").loc[list(names), "old_v3_score_ez"].to_numpy(dtype=float)
        for run_index, windows in enumerate((3, 4, 5)):
            time = np.linspace(0., 1., windows)[:, None, None]
            channel_signal = scores[None, :, None]
            feature_scale = np.linspace(.5, 1.5, 20)[None, None, :]
            values = channel_signal * feature_scale + time * (.05 + run_index * .01) + subject_index * 1e-3
            window_mask = np.ones((windows, len(names)), dtype=bool)
            if run_index == 2:
                window_mask[-1, -1] = False
                values[-1, -1, 0] = np.nan
            store.add_run(str(subject_id), RunTrajectory(f"seizure-{run_index + 1}", names, values, window_mask=window_mask, finite_mask=np.isfinite(values)))
    enriched = ledger.copy()
    enriched["optional_hnc_score"] = 1. - enriched["old_v3_score_ez"]
    enriched["hnc_score_semantics"] = "p_nez"
    enriched["hnc_eject_priority"] = enriched["optional_hnc_score"]
    enriched["hnc_add_priority"] = -enriched["optional_hnc_score"]
    return enriched, store.static_channel_features(), store


def _pair_features(pairs: pd.DataFrame, ledger: pd.DataFrame, feature_groups: set[str]) -> tuple[pd.DataFrame, list[str], dict[str, object]]:
    out = pairs.copy()
    out["score_delta"] = out["add_score_ez"] - out["eject_score_ez"]
    out["score_abs_delta"] = out["score_delta"].abs()
    out["rank_distance"] = out["add_rank"] - out["eject_rank"]
    feature_columns = ["eject_score_ez", "add_score_ez", "score_delta", "score_abs_delta", "eject_rank", "add_rank", "rank_distance"]
    if "patient_context" in feature_groups:
        context = ledger.groupby("subject_id", as_index=True)[["old_v3_k", "n_patient_channels"]].first()
        out["patient_k"] = out["subject_id"].map(context["old_v3_k"]).astype(float)
        out["patient_n_channels"] = out["subject_id"].map(context["n_patient_channels"]).astype(float)
        out["patient_k_fraction"] = out["patient_k"] / out["patient_n_channels"].clip(lower=1.0)
        feature_columns.extend(["patient_k", "patient_n_channels", "patient_k_fraction"])
    optional_by_group = {
        "anchor": ("distance_to_patient_nez_anchor_l2", "robust_anchor_z_mean"),
        "trajectory": ("trajectory_mean", "trajectory_std"),
        "hnc": ("optional_hnc_score",),
    }
    optional = [name for group, names in optional_by_group.items() if group in feature_groups for name in names if name in ledger.columns]
    if "trajectory" in feature_groups:
        optional.extend(name for name in ledger.columns if name.startswith("trajectory_") and name not in optional)
    if optional:
        lookup = ledger.set_index(["subject_id", "channel_name_norm"])[optional]
        eject = lookup.reindex(list(zip(out["subject_id"].astype(str), out["eject_channel_norm"].astype(str))))
        add = lookup.reindex(list(zip(out["subject_id"].astype(str), out["add_channel_norm"].astype(str))))
        for name in optional:
            eject_name, add_name = f"eject_{name}", f"add_{name}"
            out[eject_name] = eject[name].to_numpy()
            out[add_name] = add[name].to_numpy()
            out[f"{name}_delta"] = out[add_name] - out[eject_name]
            feature_columns.extend([eject_name, add_name, f"{name}_delta"])
    if "topology" in feature_groups:
        eject_topology = out["eject_channel"].map(parse_channel_topology)
        add_topology = out["add_channel"].map(parse_channel_topology)
        out["same_shaft"] = [int(e[2] and a[2] and e[0] == a[0]) for e, a in zip(eject_topology, add_topology)]
        out["contact_distance"] = [abs(e[1] - a[1]) if e[2] and a[2] and e[0] == a[0] else -1.0 for e, a in zip(eject_topology, add_topology)]
        out["is_adjacent_contact"] = (out["contact_distance"] == 1).astype(int)
        feature_columns.extend(["same_shaft", "contact_distance", "is_adjacent_contact"])
    registry = FeatureRegistry()
    for name in feature_columns:
        if name.startswith(("eject_score", "add_score", "score_", "eject_rank", "add_rank", "rank_")):
            group = "v3"
        elif "anchor" in name:
            group = "anchor"
        elif "trajectory" in name:
            group = "trajectory"
        elif "hnc" in name:
            group = "hnc"
        elif name in {"same_shaft", "contact_distance", "is_adjacent_contact"}:
            group = "topology"
        else:
            group = "patient_context"
        registry.add(name, group=group, source="derived_from_frozen_v3_or_cache", uses_label=False)
    registry_payload = registry.to_dict()
    registry_payload["feature_groups"] = sorted(feature_groups)
    return out, feature_columns, registry_payload


def _model_for_variant(variant: str, seed: int, config: A12Config | None = None,
                       trajectories: PatientTrajectoryStore | None = None,
                       output_dir: Path | None = None):
    config = config or A12Config()
    if variant in {"A12-V1", "A12-V2", "A12-V3", "A12-V4", "A12-V5"}:
        return CatBoostUtilityModel(iterations=config.catboost_max_iterations, depth=config.catboost_depth, learning_rate=config.catboost_learning_rate, l2_leaf_reg=config.catboost_l2_leaf_reg, early_stopping_rounds=config.catboost_early_stopping_rounds, random_seed=seed, risk_lambda=config.risk_lambda)
    if variant in {"A12-V6", "A12-V7"}:
        return SiameseUtilityMLP(epochs=config.max_epochs, hidden_dim=config.hidden_dim, dropout=config.dropout, learning_rate=config.learning_rate, weight_decay=config.weight_decay, batch_size=config.batch_size, num_workers=config.num_workers, patience=config.patience, gradient_clip=config.gradient_clip, strict_device=config.strict_device, output_dir=output_dir, random_seed=seed, risk_lambda=config.risk_lambda, device=config.device)
    if variant in {"A12-V8", "A12-V9"}:
        return TrajectoryTCNUtilityModel(epochs=config.max_epochs, hidden_dim=config.hidden_dim, dropout=config.dropout, learning_rate=config.learning_rate, weight_decay=config.weight_decay, batch_size=config.batch_size, num_workers=config.num_workers, patience=config.patience, gradient_clip=config.gradient_clip, strict_device=config.strict_device, output_dir=output_dir, random_seed=seed, risk_lambda=config.risk_lambda, device=config.device, trajectory_store=trajectories)
    if variant == "A12-V10":
        return EnsembleUtilityModel(random_seed=seed, cat_iterations=config.catboost_max_iterations, cat_depth=config.catboost_depth, cat_learning_rate=config.catboost_learning_rate, cat_l2_leaf_reg=config.catboost_l2_leaf_reg, cat_early_stopping_rounds=config.catboost_early_stopping_rounds, neural_epochs=config.max_epochs, hidden_dim=config.hidden_dim, dropout=config.dropout, learning_rate=config.learning_rate, weight_decay=config.weight_decay, batch_size=config.batch_size, num_workers=config.num_workers, patience=config.patience, gradient_clip=config.gradient_clip, strict_device=config.strict_device, output_dir=output_dir, trajectory_store=trajectories, device=config.device, risk_lambda=config.risk_lambda)
    raise ValueError(variant)


def _seed_aggregate(predictions: list[pd.DataFrame], *, risk_lambda: float, uncertainty_lambda: float) -> pd.DataFrame:
    stacked = pd.concat([item.reset_index(drop=True) for item in predictions], keys=range(len(predictions)), names=["seed", "row"])
    mean = stacked.groupby("row")[["p_benefit", "p_harm", "pred_delta"]].mean()
    std = stacked.groupby("row")[["p_benefit", "p_harm", "pred_delta"]].std(ddof=0).fillna(0.0).add_suffix("_std")
    positive = ((stacked["p_benefit"] >= .5) & (stacked["p_harm"] <= .5) & (stacked["pred_delta"] > 0)).groupby("row").mean()
    out = mean.join(std)
    out["seed_positive_fraction"] = positive.to_numpy()
    out["seed_negative_fraction"] = 1.0 - positive.to_numpy()
    out["seed_consensus"] = np.maximum(positive.to_numpy(), 1.0 - positive.to_numpy())
    out["seed_prediction_std"] = std[["p_benefit_std", "p_harm_std", "pred_delta_std"]].mean(axis=1).to_numpy()
    if "member_variance" in stacked:
        out["member_variance"] = stacked.groupby("row")["member_variance"].mean().to_numpy()
    else:
        out["member_variance"] = 0.
    out["seed_variance"] = out["pred_delta_std"].to_numpy() ** 2
    out["total_uncertainty"] = np.sqrt(out["seed_variance"] + out["member_variance"])
    out["pred_delta_std"] = out["total_uncertainty"]
    out = out.rename(columns={"pred_delta_std": "pred_delta_std"})
    return recompute_utility(out.reset_index(drop=True), risk_lambda=risk_lambda, uncertainty_lambda=uncertainty_lambda)


def _cross_fitted_predictions(pairs: pd.DataFrame, columns: list[str], model_factory: Callable[[int], Any], config: A12Config) -> pd.DataFrame:
    output = []
    for fold_index, (fit_subjects, holdout_subjects) in enumerate(inner_subject_splits(pairs["subject_id"], config.inner_folds, config.seeds[0]), start=1):
        train = pairs[pairs["subject_id"].astype(str).isin(fit_subjects)]
        holdout = pairs[pairs["subject_id"].astype(str).isin(holdout_subjects)]
        if train.empty or holdout.empty:
            continue
        seed_predictions = []
        seed_member_predictions: dict[str, list[pd.DataFrame]] = {}
        for seed in config.seeds:
            model = model_factory(seed + fold_index)
            model.fit(train, columns)
            seed_predictions.append(model.predict(holdout))
            if hasattr(model, "predict_members"):
                for name, prediction in model.predict_members(holdout).items():
                    seed_member_predictions.setdefault(name, []).append(prediction)
        pred = _seed_aggregate(seed_predictions, risk_lambda=config.risk_lambda, uncertainty_lambda=config.uncertainty_lambda)
        block = holdout.copy()
        for column in pred.columns:
            block[column] = pred[column].to_numpy()
        block["seed_count"] = len(config.seeds)
        for name, member_predictions in seed_member_predictions.items():
            member = _seed_aggregate(member_predictions, risk_lambda=config.risk_lambda, uncertainty_lambda=config.uncertainty_lambda)
            for column in ("p_benefit", "p_harm", "pred_delta"):
                block[f"{name}_{column}"] = member[column].to_numpy()
        output.append(block)
    return pd.concat(output, ignore_index=True) if output else pd.DataFrame()


def _select_catboost_config(train_pairs: pd.DataFrame, columns: list[str], train_ledger: pd.DataFrame,
                            config: A12Config, output_dir: Path, outer_fold: int) -> A12Config:
    """Select CatBoost parameters with outer-train inner-OOF policy simulation."""
    if config.smoke_mode:
        grid = [(config.catboost_depth, config.catboost_learning_rate, config.catboost_l2_leaf_reg)]
    else:
        grid = [(depth, learning_rate, l2) for depth in (2, 3, 4)
                for learning_rate in (.02, .05) for l2 in (5., 20.)]
    rows: list[dict[str, object]] = []
    candidates: list[tuple[tuple[float, float, int, float, float], A12Config]] = []
    for depth, learning_rate, l2 in grid:
        candidate_config = replace(config, catboost_depth=depth,
                                   catboost_learning_rate=learning_rate,
                                   catboost_l2_leaf_reg=l2)
        def factory(seed: int):
            return _model_for_variant("A12-V1", seed, candidate_config)
        oof = _cross_fitted_predictions(train_pairs, columns, factory, candidate_config)
        if oof.empty:
            continue
        benefit = ProbabilityCalibrator().fit(oof["p_benefit"], oof["beneficial_label"], subject_ids=oof["subject_id"])
        harm = ProbabilityCalibrator().fit(oof["p_harm"], oof["harmful_label"], subject_ids=oof["subject_id"])
        oof["p_benefit"] = benefit.predict(oof["p_benefit"]); oof["p_harm"] = harm.predict(oof["p_harm"])
        oof = recompute_utility(oof, risk_lambda=config.risk_lambda, uncertainty_lambda=config.uncertainty_lambda)
        policy = select_thresholds(oof, ledger=train_ledger,
                                   minimum_action_coverage=config.minimum_action_coverage,
                                   harm_limit=config.patient_harm_limit,
                                   max_swaps_limit=config.max_swaps, smoke_mode=config.smoke_mode)
        row = {"outer_fold": outer_fold, "depth": depth, "learning_rate": learning_rate,
               "l2_leaf_reg": l2, "max_iterations": config.catboost_max_iterations,
               "early_stopping_rounds": config.catboost_early_stopping_rounds,
               "inner_patient_macro_f1": policy.objective,
               "patient_harm_rate": policy.patient_harm_rate,
               "action_harm_rate": policy.action_harm_rate,
               "action_coverage": policy.action_coverage}
        rows.append(row)
        candidates.append(((policy.objective, -policy.patient_harm_rate, -depth, -learning_rate, -l2), candidate_config))
    if not candidates:
        raise ProtocolError(f"CatBoost inner search produced no valid outer-train OOF predictions for fold {outer_fold}")
    selected = max(candidates, key=lambda item: item[0])[1]
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_dir / f"catboost_inner_search_fold_{outer_fold}.csv", index=False)
    write_json(output_dir / f"catboost_selected_params_fold_{outer_fold}.json", {"depth": selected.catboost_depth, "learning_rate": selected.catboost_learning_rate, "l2_leaf_reg": selected.catboost_l2_leaf_reg, "max_iterations": selected.catboost_max_iterations, "early_stopping_rounds": selected.catboost_early_stopping_rounds, "selection_subjects": sorted(train_pairs["subject_id"].astype(str).unique())})
    return selected


def _actions_to_ledger(test_ledger: pd.DataFrame, actions: pd.DataFrame) -> pd.DataFrame:
    output = test_ledger.copy()
    output["pred_ez"] = output["old_v3_selected"].astype(int)
    if actions.empty:
        return output
    normalized = output["channel_name_norm"].astype(str)
    for _, action in actions.iterrows():
        subject = str(action["subject_id"])
        output.loc[(output["subject_id"].astype(str) == subject) & (normalized == normalize_channel_name(action.get("eject_channel_norm", action["eject_channel"]))), "pred_ez"] = 0
        output.loc[(output["subject_id"].astype(str) == subject) & (normalized == normalize_channel_name(action.get("add_channel_norm", action["add_channel"]))), "pred_ez"] = 1
    return output


def _run_learned_gate(actions: pd.DataFrame, inner: pd.DataFrame, train_ledger: pd.DataFrame, test_ledger: pd.DataFrame, test_scored: pd.DataFrame, policy: SelectedThresholds, *, output_dir: Path | None = None, outer_fold: int | None = None) -> pd.DataFrame:
    if inner.empty:
        return actions
    rows = []
    for subject, pairs in inner.groupby("subject_id", sort=True):
        patient = train_ledger[train_ledger["subject_id"].astype(str) == str(subject)]
        eligible = pairs[(pairs["p_benefit"] >= policy.tau_benefit) & (pairs["p_harm"] <= policy.tau_harm) & (pairs["utility"] >= policy.tau_utility) & (pairs["pred_delta"] >= policy.tau_delta) & (pairs["seed_positive_fraction"] >= policy.tau_positive_seed_fraction)]
        matched = select_max_weight_swaps(eligible, max_swaps=policy.max_swaps, utility_threshold=policy.tau_utility)
        summary = build_patient_action_summary(patient, pairs, eligible, matched, policy)
        final = _actions_to_ledger(patient, matched)
        anchor_metric = evaluate_patient_predictions(patient, "old_v3_selected")["patient_macro_f1"]
        summary["anchor_patient_macro_f1"] = anchor_metric
        summary["net_delta_patient_macro_f1"] = evaluate_patient_predictions(final, "pred_ez")["patient_macro_f1"] - anchor_metric
        summary["net_beneficial"] = (summary["net_delta_patient_macro_f1"] > 0).astype(int)
        rows.append(summary)
    rows = pd.concat(rows, ignore_index=True)
    fields = [name for name in rows.columns if name not in {"subject_id", "net_beneficial", "net_delta_patient_macro_f1", "anchor_patient_macro_f1", "selected_policy"} and pd.api.types.is_numeric_dtype(rows[name])]
    cross_fitted = cross_fit_patient_gate(rows, fields, n_folds=min(5, max(2, rows["subject_id"].nunique())), seed=0)
    threshold = float(cross_fitted["selected_gate_threshold"].iloc[0])
    gate = LearnedPatientGate(threshold=threshold).fit(rows, fields)
    output = actions.copy()
    test_rows = []
    for subject, patient in test_ledger.groupby("subject_id", sort=True):
        patient_actions = output[output["subject_id"].astype(str) == str(subject)]
        patient_pairs = test_scored[test_scored["subject_id"].astype(str) == str(subject)]
        eligible = patient_pairs[patient_pairs["gate_pass"]] if "gate_pass" in patient_pairs else patient_pairs.iloc[0:0]
        test_rows.append(build_patient_action_summary(patient, patient_pairs, eligible, patient_actions, policy))
    summaries = pd.concat(test_rows, ignore_index=True)
    summaries["gate_probability"] = gate.predict_proba(summaries)
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        cross_fitted.to_csv(output_dir / f"gate_crossfit_fold_{outer_fold}.csv", index=False)
        summaries.to_csv(output_dir / f"gate_test_summary_fold_{outer_fold}.csv", index=False)
        write_json(output_dir / f"gate_manifest_fold_{outer_fold}.json", {"fit_subjects": sorted(gate.fit_subjects), "scaler_fit_subjects": sorted(gate.scaler_fit_subjects), "gate_threshold_subjects": sorted(rows["subject_id"].astype(str).unique()), "selected_threshold": threshold, "feature_columns": fields, "single_class_constant": gate.constant})
    kept = apply_patient_gate(output, summaries[["subject_id", "gate_probability"]], threshold=threshold) if not output.empty else output.copy()
    if not kept.empty:
        kept["learned_gate_probability"] = kept["gate_probability"]
        kept["learned_gate_threshold"] = threshold
    return kept


def _run_model_variant(variant: str, ledger: pd.DataFrame, root: Path, config: A12Config, evaluation_folds: set[int] | None = None, trajectories: PatientTrajectoryStore | None = None) -> dict[str, Any]:
    all_candidates = build_candidate_pools(ledger, CandidateConfig(config.selected_tail_frac, config.selected_tail_min, config.selected_tail_max, config.boundary_width, config.add_pool_max, config.max_eject_candidates, config.max_add_candidates, config.max_pairs_per_patient))
    final_parts, all_actions, selections, audit_rows, all_inner, all_test_scored, training_rows, fitted_models = [], [], {}, [], [], [], [], []
    available_groups = {"v3", "patient_context", "topology"}
    if "distance_to_patient_nez_anchor_l2" in ledger.columns:
        available_groups.add("anchor")
    if any(name.startswith("trajectory_") for name in ledger.columns):
        available_groups.add("trajectory")
    if "optional_hnc_score" in ledger.columns:
        available_groups.add("hnc")
    feature_groups = set(resolve_variant_feature_groups(variant, available_groups))
    if variant != "A12-V1" and not feature_groups.issuperset({"v3", "patient_context"}):
        raise ProtocolError(f"variant {variant} resolved no valid base feature groups")
    registry_payload: dict[str, object] = {"feature_groups": sorted(feature_groups), "features": []}
    for outer_fold, train_subjects, test_subjects in outer_subject_folds(ledger):
        if evaluation_folds is not None and outer_fold not in evaluation_folds:
            continue
        train_ledger = ledger[ledger["subject_id"].astype(str).isin(train_subjects)]
        test_ledger = ledger[ledger["subject_id"].astype(str).isin(test_subjects)]
        train_candidates = all_candidates[all_candidates["subject_id"].astype(str).isin(train_subjects)]
        test_candidates = all_candidates[all_candidates["subject_id"].astype(str).isin(test_subjects)]
        train_pairs, test_pairs = build_pair_dataset(train_ledger, train_candidates, include_labels=True), build_pair_dataset(test_ledger, test_candidates, include_labels=False)
        train_pairs, columns, registry_payload = _pair_features(train_pairs, train_ledger, feature_groups)
        test_pairs, _, _ = _pair_features(test_pairs, test_ledger, feature_groups)
        write_pair_cache(train_pairs, root / "pair_cache" / variant.replace("-", "_") / f"fold_{outer_fold}_train_pairs", config={"variant": variant, "outer_fold": outer_fold, "config_hash": _result_config_hash(config)}, force_rebuild=config.force_rebuild_pair_cache)
        gate_subjects = train_subjects if variant in {"A12-V7", "A12-V9"} else ()
        leakage = assert_no_leakage(train_subjects=train_subjects, test_subjects=test_subjects, feature_names=columns, calibration_subjects=train_subjects, gate_subjects=gate_subjects, scaler_subjects=train_subjects, gate_threshold_subjects=gate_subjects, hyperparameter_search_subjects=train_subjects if variant in {"A12-V1", "A12-V2", "A12-V3", "A12-V4", "A12-V5", "A12-V10"} else (), checkpoint_selection_subjects=train_subjects if variant in {"A12-V6", "A12-V7", "A12-V8", "A12-V9", "A12-V10"} else (), ensemble_selection_subjects=train_subjects if variant == "A12-V10" else ())
        audit_rows.append({"outer_fold": outer_fold, **leakage})
        write_json(root / "audit" / f"no_leakage_{variant}_fold_{outer_fold}.json", {"variant": variant, "outer_fold": outer_fold, "outer_test_subjects": sorted(test_subjects), **leakage})
        model_config = config
        if variant in {"A12-V1", "A12-V2", "A12-V3", "A12-V4", "A12-V5", "A12-V10"}:
            model_config = _select_catboost_config(train_pairs, columns, train_ledger, config, root / "variants" / variant.replace("-", "_"), outer_fold)
        def factory(seed: int):
            model = _model_for_variant(variant, seed, model_config, trajectories, root / "variants" / variant.replace("-", "_") / f"fold_{outer_fold}")
            if hasattr(model, "trajectory_store"):
                model.trajectory_store = trajectories
            return model
        inner = _cross_fitted_predictions(train_pairs, columns, factory, config)
        if inner.empty:
            raise ProtocolError(f"outer fold {outer_fold} has no inner OOF pair predictions")
        raw_inner_dir = root / "variants" / variant.replace("-", "_")
        raw_inner_dir.mkdir(parents=True, exist_ok=True)
        inner.to_csv(raw_inner_dir / f"inner_oof_raw_fold_{outer_fold}.csv", index=False)
        selected_members: tuple[str, ...] | None = None
        if variant == "A12-V10":
            available_members = ("catboost", "siamese", "tcn")
            choices = [combo for size in range(1, 4) for combo in combinations(available_members, size)]
            candidate_models = []
            for combo in choices:
                candidate = inner.copy()
                for field in ("p_benefit", "p_harm", "pred_delta"):
                    candidate[field] = candidate[[f"{name}_{field}" for name in combo]].mean(axis=1)
                candidate["member_variance"] = candidate[[f"{name}_pred_delta" for name in combo]].var(axis=1, ddof=0)
                candidate["seed_variance"] = pd.to_numeric(candidate.get("pred_delta_std", 0.), errors="coerce").fillna(0.) ** 2
                candidate["total_uncertainty"] = np.sqrt(candidate["member_variance"] + candidate["seed_variance"])
                candidate["pred_delta_std"] = candidate["total_uncertainty"]
                benefit_candidate = ProbabilityCalibrator().fit(candidate["p_benefit"], candidate["beneficial_label"], subject_ids=candidate["subject_id"])
                harm_candidate = ProbabilityCalibrator().fit(candidate["p_harm"], candidate["harmful_label"], subject_ids=candidate["subject_id"])
                candidate["p_benefit"] = benefit_candidate.predict(candidate["p_benefit"]); candidate["p_harm"] = harm_candidate.predict(candidate["p_harm"])
                candidate = recompute_utility(candidate, risk_lambda=config.risk_lambda, uncertainty_lambda=config.uncertainty_lambda)
                policy = select_thresholds(candidate, ledger=train_ledger, minimum_action_coverage=config.minimum_action_coverage, harm_limit=config.patient_harm_limit, max_swaps_limit=config.max_swaps, smoke_mode=config.smoke_mode)
                candidate_models.append((policy.objective, combo, candidate, benefit_candidate, harm_candidate))
            _, selected_members, inner, prefit_benefit, prefit_harm = max(candidate_models, key=lambda item: (item[0], -len(item[1]), item[1]))
        if selected_members is None:
            benefit_calibrator = ProbabilityCalibrator().fit(inner["p_benefit"], inner["beneficial_label"], subject_ids=inner["subject_id"])
            harm_calibrator = ProbabilityCalibrator().fit(inner["p_harm"], inner["harmful_label"], subject_ids=inner["subject_id"])
        else:
            benefit_calibrator, harm_calibrator = prefit_benefit, prefit_harm
        if selected_members is None:
            inner["p_benefit"] = benefit_calibrator.predict(inner["p_benefit"])
            inner["p_harm"] = harm_calibrator.predict(inner["p_harm"])
        inner = recompute_utility(inner, risk_lambda=config.risk_lambda, uncertainty_lambda=config.uncertainty_lambda)
        inner["outer_fold"] = outer_fold
        all_inner.append(inner)
        thresholds, policy_grid = select_thresholds(inner, ledger=train_ledger, minimum_action_coverage=config.minimum_action_coverage, harm_limit=config.patient_harm_limit, max_swaps_limit=config.max_swaps, return_grid=True, smoke_mode=config.smoke_mode)
        selections[str(outer_fold)] = thresholds.to_dict()
        if selected_members is not None:
            selections[str(outer_fold)]["selected_ensemble_members"] = list(selected_members)
        policy_dir = root / "variants" / variant.replace("-", "_"); policy_dir.mkdir(parents=True, exist_ok=True)
        policy_grid.to_csv(policy_dir / f"inner_policy_grid_fold_{outer_fold}.csv", index=False)
        seed_predictions = []
        for seed in config.seeds:
            model = factory(seed)
            if selected_members is not None:
                model.selected_member_names = selected_members
            model.fit(train_pairs, columns)
            fitted_models.append(model)
            seed_predictions.append(model.predict(test_pairs))
        prediction = _seed_aggregate(seed_predictions, risk_lambda=config.risk_lambda, uncertainty_lambda=config.uncertainty_lambda)
        prediction["seed_count"] = len(config.seeds)
        prediction["p_benefit"] = benefit_calibrator.predict(prediction["p_benefit"])
        prediction["p_harm"] = harm_calibrator.predict(prediction["p_harm"])
        prediction = recompute_utility(prediction, risk_lambda=config.risk_lambda, uncertainty_lambda=config.uncertainty_lambda)
        test_scored = test_pairs.copy()
        test_scored[["p_benefit", "p_harm", "pred_delta", "utility", "seed_positive_fraction", "seed_negative_fraction", "seed_consensus", "seed_prediction_std"]] = prediction[["p_benefit", "p_harm", "pred_delta", "utility", "seed_positive_fraction", "seed_negative_fraction", "seed_consensus", "seed_prediction_std"]]
        gate = RuleGate(thresholds.tau_benefit, thresholds.tau_harm, thresholds.tau_utility, thresholds.tau_delta, thresholds.tau_positive_seed_fraction)
        test_scored["gate_pass"] = test_scored.apply(lambda row: gate.accept(row), axis=1)
        test_scored["outer_fold"] = outer_fold
        all_test_scored.append(test_scored)
        selected_by_subject = []
        for subject_id, edges in test_scored[test_scored["gate_pass"]].groupby("subject_id", sort=True):
            chosen = select_max_weight_swaps(edges, max_swaps=thresholds.max_swaps, utility_threshold=thresholds.tau_utility)
            if not chosen.empty:
                selected_by_subject.append(chosen)
        actions = pd.concat(selected_by_subject, ignore_index=True) if selected_by_subject else test_scored.iloc[0:0].copy()
        if variant in {"A12-V7", "A12-V9"}:
            actions = _run_learned_gate(actions, inner, train_ledger, test_ledger, test_scored, thresholds, output_dir=root / "variants" / variant.replace("-", "_") / "gate", outer_fold=outer_fold)
        actions["variant"] = variant
        final_parts.append(_actions_to_ledger(test_ledger, actions))
        all_actions.append(actions)
        training_rows.append({"outer_fold": outer_fold, "variant": variant, "n_train_subjects": len(train_subjects), "n_test_subjects": len(test_subjects), "n_train_pairs": len(train_pairs), "n_test_pairs": len(test_pairs), "inner_oof_pairs": len(inner), "seeds_trained": list(config.seeds), "seed_count": len(config.seeds), "inner_selected_ensemble_members": list(selected_members or ()), "fit_subjects": sorted(set().union(*(getattr(model, "fit_subjects", set()) for model in fitted_models[-len(config.seeds):]))), "validation_subjects": sorted(set().union(*(getattr(model, "validation_subjects", set()) for model in fitted_models[-len(config.seeds):]))), "best_epochs": [getattr(model, "best_epoch", getattr(model, "best_iteration", None)) for model in fitted_models[-len(config.seeds):]]})
    final = pd.concat(final_parts, ignore_index=True)
    actions = pd.concat(all_actions, ignore_index=True) if all_actions else pd.DataFrame()
    metrics = evaluate_patient_predictions(final, "pred_ez")
    metrics["patient_rows"]["prediction_variant"] = variant
    action_coverage = float(actions["subject_id"].nunique() / max(final["subject_id"].nunique(), 1)) if not actions.empty else 0.0
    variant_root = root / "variants" / variant.replace("-", "_")
    device_resolutions = [getattr(model, "device_resolution", None) for model in fitted_models]
    device_payload = next((item.to_dict() for item in device_resolutions if item is not None), {"cuda_available": False, "requested_device": config.device, "resolved_device": "cpu", "fallback_reason": None})
    parameter_count = 0
    for model in fitted_models:
        for member in getattr(model, "members", [model]):
            network = getattr(member, "network", None)
            if network is not None:
                parameter_count += sum(parameter.numel() for parameter in network.module.parameters())
            else:
                parameter_count += sum(max(0, int(getattr(tree, "tree_count_", 0))) for tree in (getattr(member, "benefit_model", None), getattr(member, "harm_model", None), getattr(member, "delta_model", None)) if tree is not None)
    manifest = {
        "variant": variant,
        "feature_columns": columns,
        "feature_groups": sorted(feature_groups),
        "source_outer_folds": sorted(final["outer_fold"].unique().tolist()),
        "config_hash": _result_config_hash(config),
        "fingerprint": stable_hash({"run_fingerprint": config.input_fingerprint, "variant": variant, "outer_folds": sorted(final["outer_fold"].unique().tolist())}),
        **device_payload,
        "training_params": {key: config.to_dict()[key] for key in (
            "batch_size", "max_epochs", "patience", "learning_rate", "weight_decay",
            "hidden_dim", "dropout", "gradient_clip", "risk_lambda", "uncertainty_lambda",
            "catboost_depth", "catboost_learning_rate", "catboost_l2_leaf_reg",
            "catboost_max_iterations", "catboost_early_stopping_rounds")},
        "parameter_count": parameter_count,
        "fit_subjects": sorted(set().union(*(getattr(model, "fit_subjects", set()) for model in fitted_models))),
        "validation_subjects": sorted(set().union(*(getattr(model, "validation_subjects", set()) for model in fitted_models))),
        "best_epochs": [getattr(model, "best_epoch", getattr(model, "best_iteration", None)) for model in fitted_models],
    }
    write_variant_artifacts(variant_root, config=config.to_dict(), final_ledger=final, actions=actions, metrics=metrics, inner_selection=selections, manifest=manifest, pair_oof=pd.concat(all_inner, ignore_index=True) if all_inner else pd.DataFrame(), test_candidates=pd.concat(all_test_scored, ignore_index=True) if all_test_scored else pd.DataFrame(), training_log=pd.DataFrame(training_rows))
    write_json(variant_root / "feature_registry.json", registry_payload)
    write_json(root / "audit" / f"no_leakage_{variant}.json", {"variant": variant, "folds": audit_rows})
    return {"status": "success", "metrics": metrics, "actions": actions, "action_coverage": action_coverage}


def _load_resumable_variant(variant: str, root: Path, config: A12Config) -> dict[str, Any] | None:
    variant_root = root / "variants" / variant.replace("-", "_")
    completion, final_path, action_path = variant_root / "completion.json", variant_root / "final_channel_ledger.csv", variant_root / "repair_actions.csv"
    if not completion.exists():
        return None
    state = json.loads(completion.read_text(encoding="utf-8"))
    current_hash = _result_config_hash(config)
    if state.get("config_hash") != current_hash:
        raise ProtocolError(f"resume refused for {variant}: config hash changed")
    if not final_path.exists() or not action_path.exists():
        raise ProtocolError(f"resume refused for {variant}: completion marker is incomplete")
    final = pd.read_csv(final_path)
    try:
        actions = pd.read_csv(action_path)
    except pd.errors.EmptyDataError:
        actions = pd.DataFrame(columns=["subject_id"])
    metrics = evaluate_patient_predictions(final, "pred_ez")
    coverage = float(actions["subject_id"].nunique() / max(final["subject_id"].nunique(), 1)) if not actions.empty and "subject_id" in actions else 0.0
    return {"status": "success", "resumed": True, "metrics": metrics, "actions": actions, "action_coverage": coverage}


def _add_cache_features(ledger: pd.DataFrame, channel_features: dict[str, dict[str, list[float]]] | None, trajectories: PatientTrajectoryStore | None, config: A12Config) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    if not channel_features:
        return ledger, pd.DataFrame(), {}
    output, coverage = build_patient_nez_anchor_features(
        ledger, channel_features, quantile=config.anchor_quantile,
        min_channels=config.anchor_min_channels,
        minimum_channel_coverage=config.minimum_anchor_channel_coverage,
        minimum_feature_finite_ratio=config.minimum_anchor_feature_finite_ratio,
        strict=config.strict)
    if trajectories:
        output, mapping = build_relative_trajectory_features(output, trajectories)
    else:
        mapping = {}
    return output, coverage, mapping


def run_a12_suite(ledger: pd.DataFrame, *, output_dir: str | Path, config: A12Config = A12Config(), synthetic: bool = False, run_classification: str | None = None, cache_audit: dict[str, Any] | None = None, hnc_available: bool = False, hnc_audit: dict[str, Any] | None = None, resolved_schema: dict[str, Any] | None = None, channel_features: dict[str, dict[str, list[float]]] | None = None, trajectories: PatientTrajectoryStore | None = None, evaluation_folds: set[int] | None = None) -> dict[str, Any]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    if synthetic and (channel_features is None or trajectories is None):
        ledger, synthetic_features, synthetic_trajectories = make_synthetic_capabilities(ledger)
        channel_features = channel_features or synthetic_features
        trajectories = trajectories or synthetic_trajectories
        hnc_available = True
    fingerprint_config = config.to_dict()
    fingerprint_config.pop("resume", None); fingerprint_config.pop("input_fingerprint", None)
    old_v3_summary_hash = sha256_file(config.old_v3_summary) if config.old_v3_summary and Path(config.old_v3_summary).exists() else None
    fingerprint_payload = execution_fingerprint(ledger=ledger, cache_audit=cache_audit, hnc_audit=hnc_audit, resolved_schema=resolved_schema, feature_registry={"registered_variants": ALL_VARIANTS, "trajectory_mode": config.trajectory_mode}, config=fingerprint_config, old_v3_summary_hash=old_v3_summary_hash)
    fingerprint_path = root / "run_fingerprint.json"
    if config.resume and fingerprint_path.exists():
        prior = json.loads(fingerprint_path.read_text(encoding="utf-8"))
        assert_resume_compatible(prior.get("payload", {}), fingerprint_payload)
    if config.input_fingerprint is None:
        config = replace(config, input_fingerprint=fingerprint_hash(fingerprint_payload))
    expected_patients = None if synthetic else 90
    expected_folds = int(ledger["outer_fold"].nunique()) if synthetic else 5
    ledger, anchor_coverage, trajectory_mapping = _add_cache_features(ledger, channel_features, trajectories, config)
    if not synthetic and config.expected_old_v3_macro_f1 is None:
        raise ProtocolError("real A12 runs require --expected-old-v3-macro-f1 or an independently authored old-V3 summary")
    anchor = audit_anchor(ledger, expected_macro_f1=config.expected_old_v3_macro_f1, expected_patients=expected_patients, expected_folds=expected_folds, tolerance=config.anchor_parity_tolerance)
    if not anchor["passed"]:
        raise ProtocolError("frozen V3 anchor parity failed")
    write_json(root / "audit" / "anchor_parity.json", anchor)
    write_json(root / "audit" / "anchor_contract.json", anchor["contract"])
    write_json(root / "audit" / "anchor_metric_parity.json", anchor.get("metric_parity", anchor))
    if not anchor_coverage.empty:
        anchor_coverage.to_csv(root / "audit" / "anchor_feature_coverage.csv", index=False)
        write_json(root / "audit" / "anchor_feature_coverage.json", anchor_coverage.to_dict("records"))
    if trajectory_mapping:
        write_json(root / "audit" / "trajectory_feature_mapping.json", trajectory_mapping)
    anchor_rows = evaluate_patient_predictions(ledger, "old_v3_selected")["patient_rows"]
    anchor_rows.to_csv(root / "audit" / "anchor_metrics_by_patient.csv", index=False)
    anchor_rows.groupby("outer_fold", dropna=False).mean(numeric_only=True).reset_index().to_csv(root / "audit" / "anchor_metrics_by_fold.csv", index=False)
    anchor_rows.groupby("center", dropna=False).mean(numeric_only=True).reset_index().to_csv(root / "audit" / "anchor_metrics_by_center.csv", index=False)
    write_json(root / "run_config.json", config.to_dict())
    write_json(root / "environment.json", environment_snapshot())
    write_json(root / "run_fingerprint.json", {"fingerprint": config.input_fingerprint, "payload": fingerprint_payload, "config": config.to_dict()})
    write_json(root / "audit" / "resolved_schema.json", {"synthetic": synthetic, "cache": cache_audit})
    requested = ALL_VARIANTS if "all" in config.variants else config.variants
    available_groups = {"v3", "patient_context", "topology"}
    if "distance_to_patient_nez_anchor_l2" in ledger: available_groups.add("anchor")
    if any(name.startswith("trajectory_") for name in ledger): available_groups.add("trajectory")
    if hnc_available and "optional_hnc_score" in ledger: available_groups.add("hnc")
    variants: dict[str, dict[str, Any]] = {}
    for variant in requested:
        if config.resume and variant not in {"A12-D0"}:
            resumed = _load_resumable_variant(variant, root, config)
            if resumed is not None:
                variants[variant] = resumed
                continue
        if variant == "A12-V0":
            final = ledger.copy() if evaluation_folds is None else ledger[ledger["outer_fold"].astype(int).isin(evaluation_folds)].copy()
            final["pred_ez"] = final["old_v3_selected"].astype(int)
            metrics = evaluate_patient_predictions(final, "pred_ez")
            write_variant_artifacts(root / "variants" / "A12_V0", config=config.to_dict(), final_ledger=final, actions=pd.DataFrame(), metrics=metrics, inner_selection={}, manifest={"variant": variant, "frozen_anchor": True, "config_hash": _result_config_hash(config), "fingerprint": stable_hash({"run_fingerprint": config.input_fingerprint, "variant": variant, "outer_folds": sorted(final["outer_fold"].unique().tolist())})})
            variants[variant] = {"status": "success", "metrics": metrics, "actions": pd.DataFrame(), "action_coverage": 0.0}
        elif variant == "A12-D0":
            summary, by_patient, pairs = candidate_ceiling_audit(ledger)
            write_json(root / "audit" / "candidate_ceiling_summary.json", summary)
            by_patient.to_csv(root / "audit" / "candidate_ceiling_by_patient.csv", index=False)
            pairs.to_csv(root / "audit" / "candidate_source_contribution.csv", index=False)
            variants[variant] = {"status": "diagnostic", "metrics": {}, "actions": pd.DataFrame(), "action_coverage": 0.0}
        elif variant == "A12-V5" and not hnc_available:
            if config.strict:
                raise ProtocolError("A12-V5 requires a valid normalized OOF HNC ledger with explicit semantics")
            variants[variant] = {"status": "skipped", "reason": "valid HNC OOF ledger unavailable", "metrics": {}, "actions": pd.DataFrame()}
        elif variant in VARIANT_FEATURE_GROUPS and not VARIANT_FEATURE_GROUPS[variant].issubset(available_groups):
            missing_groups = sorted(VARIANT_FEATURE_GROUPS[variant] - available_groups)
            if config.strict:
                raise ProtocolError(f"{variant} missing required capability groups: {missing_groups}")
            variants[variant] = {"status": "skipped", "reason": f"missing required capability groups: {missing_groups}", "metrics": {}, "actions": pd.DataFrame()}
        elif variant in {"A12-V8", "A12-V9", "A12-V10"} and trajectories is None:
            if config.strict:
                raise ProtocolError(f"{variant} requires a PatientTrajectoryStore")
            variants[variant] = {"status": "skipped", "reason": "trajectory store unavailable", "metrics": {}, "actions": pd.DataFrame()}
        elif variant in {"A12-V2", "A12-V3", "A12-V4", "A12-V6", "A12-V7", "A12-V8", "A12-V9", "A12-V10"} and not channel_features and not synthetic:
            if config.strict:
                raise ProtocolError(f"{variant} requires a validated window feature cache")
            variants[variant] = {"status": "skipped", "reason": "validated window feature cache unavailable", "metrics": {}, "actions": pd.DataFrame()}
        else:
            try:
                variants[variant] = _run_model_variant(variant, ledger, root, config, evaluation_folds, trajectories)
            except Exception as exc:
                if config.strict:
                    raise
                variants[variant] = {"status": "failed", "reason": f"{type(exc).__name__}: {exc}", "metrics": {}, "actions": pd.DataFrame()}
    run_classification = run_classification or ("synthetic" if synthetic else "partial real fold" if evaluation_folds is not None else "full real 5-fold")
    comparison = write_comparison(root, variants, run_classification=run_classification)
    return {"output_dir": str(root), "variants": variants, "comparison": comparison, "config_hash": _result_config_hash(config)}
