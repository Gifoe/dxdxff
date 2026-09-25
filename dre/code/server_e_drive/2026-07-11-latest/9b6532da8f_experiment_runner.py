from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
import hashlib
import json
import shutil
from typing import Any, Callable, Sequence

import joblib
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from outcome_hifos.cache_schema import load_cache_contract
from outcome_hifos.checkpoint import load_checkpoint
from outcome_hifos.calibration import PlattCalibrator, select_macro_f1_threshold
from outcome_hifos.collate import collate_outcome_patients
from outcome_hifos.dataset import OutcomePatientExample, build_outcome_patient_examples
from outcome_hifos.feature_views import fit_normalizer_for_examples, transform_examples
from outcome_hifos.folds import build_composite_fold_ledger, build_inner_ledgers, iter_outer_partitions, ledger_hash
from outcome_hifos.fm.embedding_cache import load_fm_embedding_as_feature_cache
from outcome_hifos.logging_utils import environment_manifest, write_environment_manifest
from outcome_hifos.metrics import compute_patient_metrics
from outcome_hifos.models.baselines import SummaryMLModel, patient_summary_features
from outcome_hifos.models.model_registry import build_model
from outcome_hifos.reports.summarize import summarize_experiment
from outcome_hifos.reports.shortcut import replay_model_perturbations, run_shortcut_audit
from outcome_hifos.reports.exports import aggregate_interpretability_artifacts, export_interpretability
from outcome_hifos.training.outer_cv import build_loco_partitions, select_best_profile
from outcome_hifos.training.evaluator import evaluate_model
from outcome_hifos.training.late_fusion import fit_cross_fitted_stacker
from outcome_hifos.training.fusion_protocol import build_fusion_cohort
from outcome_hifos.training.trainer import OutcomeTrainer


@dataclass(frozen=True)
class ExperimentSummary:
    predictions: pd.DataFrame
    failures: pd.DataFrame


def _stable_config_hash(config: dict[str, Any]) -> str:
    encoded = json.dumps(config, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _subject_set_hash(subject_ids: Sequence[str]) -> str:
    encoded = "\n".join(sorted(map(str, subject_ids))).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_resume_predictions(
    predictions: pd.DataFrame,
    *,
    config_hash: str,
    fold_hash: str,
    cohort_id: str,
    subject_set_hash: str,
) -> None:
    expected = {
        "config_hash": str(config_hash),
        "fold_ledger_hash": str(fold_hash),
        "cohort_id": str(cohort_id),
        "subject_set_hash": str(subject_set_hash),
    }
    missing = sorted(set(expected) - set(predictions.columns))
    if missing:
        raise ValueError(f"Stale resume artifact is missing provenance columns: {missing}")
    for column, value in expected.items():
        observed = set(predictions[column].astype(str))
        if observed != {value}:
            raise ValueError(f"Stale resume artifact {column} mismatch: expected {value}, observed {sorted(observed)}")


def _attach_prediction_provenance(
    frame: pd.DataFrame,
    *,
    checkpoint_path: str | Path,
    fold_hash: str,
    config_hash: str,
    cohort_id: str,
    subject_set_hash: str,
    calibration_method: str = "platt_inner_oof",
    threshold_source: str = "inner_oof",
) -> pd.DataFrame:
    output = frame.copy()
    output["raw_logit"] = output["logit"]
    output["calibrated_probability"] = output["probability"]
    output["selected_threshold"] = output["threshold"]
    output["calibration_method"] = str(calibration_method)
    output["threshold_source"] = str(threshold_source)
    output["checkpoint_path"] = str(checkpoint_path)
    output["fold_ledger_hash"] = str(fold_hash)
    output["config_hash"] = str(config_hash)
    output["cohort_id"] = str(cohort_id)
    output["subject_set_hash"] = str(subject_set_hash)
    return output


def run_partitioned_experiment(
    manifest: pd.DataFrame,
    ledger: pd.DataFrame,
    variants: Sequence[str],
    seeds: Sequence[int],
    fit_predict: Callable[[str, int, tuple[str, ...], tuple[str, ...], int], pd.DataFrame],
    *,
    max_outer_folds: int = 0,
) -> ExperimentSummary:
    predictions: list[pd.DataFrame] = []
    failures: list[dict[str, object]] = []
    partitions = list(iter_outer_partitions(manifest, ledger))
    if max_outer_folds > 0:
        partitions = partitions[: int(max_outer_folds)]
    for partition in partitions:
        for variant in variants:
            for seed in seeds:
                try:
                    frame = fit_predict(variant, int(seed), partition.train_subjects, partition.test_subjects, partition.fold_idx).copy()
                    frame["variant"] = str(variant)
                    frame["seed"] = int(seed)
                    frame["outer_fold_idx"] = int(partition.fold_idx)
                    frame["role"] = "outer_test"
                    predictions.append(frame)
                except Exception as exc:
                    failures.append(
                        {
                            "variant": str(variant),
                            "seed": int(seed),
                            "outer_fold_idx": int(partition.fold_idx),
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                    )
    return ExperimentSummary(
        pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame(),
        pd.DataFrame(failures),
    )


def _example_manifest(examples: Sequence[OutcomePatientExample]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "subject_id": example.subject_id,
                "center": example.center,
                "outcome_label": int(example.target),
                "channel_count": len(example.canonical_channels),
                "seizure_count": len(example.model_input["feature_runs"]),
                "window_count": sum(run.shape[0] for run in example.model_input["feature_runs"]),
            }
            for example in examples
        ]
    )


def _limit_examples(examples: list[OutcomePatientExample], maximum: int) -> list[OutcomePatientExample]:
    if maximum <= 0 or len(examples) <= maximum:
        return sorted(examples, key=lambda example: example.subject_id)
    by_target = {
        target: sorted([example for example in examples if int(example.target) == target], key=lambda example: example.subject_id)
        for target in (0, 1)
    }
    selected = []
    while len(selected) < maximum and (by_target[0] or by_target[1]):
        for target in (0, 1):
            if by_target[target] and len(selected) < maximum:
                selected.append(by_target[target].pop(0))
    return selected


def _by_subject(examples: Sequence[OutcomePatientExample], subjects: Sequence[str]) -> list[OutcomePatientExample]:
    selected = set(str(subject) for subject in subjects)
    return [example for example in examples if example.subject_id in selected]


def _loader(examples: Sequence[OutcomePatientExample], batch_size: int) -> DataLoader:
    return DataLoader(list(examples), batch_size=max(1, int(batch_size)), shuffle=False, collate_fn=partial(collate_outcome_patients, padding_value=0.0))


def _summary_matrix(examples: list[OutcomePatientExample]) -> tuple[np.ndarray, np.ndarray]:
    batch = collate_outcome_patients(examples)
    features = patient_summary_features(batch["model_input"]).detach().cpu().numpy()
    targets = batch["outcome"].detach().cpu().numpy().astype(np.int64)
    return features, targets


def _calibrate_inner_oof(predictions: pd.DataFrame, run_dir: Path) -> tuple[PlattCalibrator, float]:
    inner = predictions.copy()
    inner["role"] = "inner_oof"
    calibrator = PlattCalibrator().fit(inner["logit"].to_numpy(), inner["outcome"].to_numpy(), roles=inner["role"].tolist())
    inner["probability"] = calibrator.predict(inner["logit"].to_numpy())
    threshold = select_macro_f1_threshold(inner)
    run_dir.mkdir(parents=True, exist_ok=True)
    threshold.curve.to_csv(run_dir / "threshold_curve.csv", index=False)
    joblib.dump(calibrator, run_dir / "platt_calibrator.joblib")
    pd.Series({"threshold": threshold.threshold, "inner_oof_macro_f1": threshold.macro_f1}).to_json(run_dir / "threshold.json", indent=2)
    return calibrator, threshold.threshold


def _predict_summary_model(model: SummaryMLModel, examples: list[OutcomePatientExample]) -> pd.DataFrame:
    features, targets = _summary_matrix(examples)
    probability = model.predict_proba(features)
    logits = np.log(np.clip(probability, 1e-6, 1 - 1e-6) / np.clip(1 - probability, 1e-6, 1 - 1e-6))
    return pd.DataFrame(
        {
            "subject_id": [example.subject_id for example in examples],
            "center": [example.center for example in examples],
            "outcome": targets,
            "logit": logits,
            "probability": probability,
        }
    )


def _run_h1_fold(
    examples: list[OutcomePatientExample],
    train_manifest: pd.DataFrame,
    test_subjects: Sequence[str],
    protocol: str,
    protocol_config: dict[str, Any],
    seed: int,
    run_dir: Path,
) -> pd.DataFrame:
    inner_folds = min(int(protocol_config.get("inner_folds", 4)), len(train_manifest))
    inner_ledger = build_inner_ledgers(train_manifest, n_splits=max(2, inner_folds), seed=seed)
    inner_predictions = []
    partitions = list(iter_outer_partitions(train_manifest, inner_ledger))
    if protocol == "screening":
        partitions = partitions[:1]
    for partition in partitions:
        fit_raw = _by_subject(examples, partition.train_subjects)
        validation_raw = _by_subject(examples, partition.test_subjects)
        normalizer = fit_normalizer_for_examples(fit_raw)
        fit_examples = transform_examples(fit_raw, normalizer)
        validation_examples = transform_examples(validation_raw, normalizer)
        train_x, train_y = _summary_matrix(fit_examples)
        model = SummaryMLModel(random_seed=seed).fit(train_x, train_y)
        inner_predictions.append(_predict_summary_model(model, validation_examples))
    inner_oof = pd.concat(inner_predictions, ignore_index=True)
    calibrator, threshold = _calibrate_inner_oof(inner_oof, run_dir)
    outer_train_raw = _by_subject(examples, train_manifest["subject_id"].astype(str).tolist())
    test_raw = _by_subject(examples, test_subjects)
    normalizer = fit_normalizer_for_examples(outer_train_raw)
    normalizer.save(run_dir / "normalizer.joblib", run_dir / "normalizer_audit.json")
    outer_train = transform_examples(outer_train_raw, normalizer)
    test = transform_examples(test_raw, normalizer)
    train_x, train_y = _summary_matrix(outer_train)
    model = SummaryMLModel(random_seed=seed).fit(train_x, train_y)
    prediction = _predict_summary_model(model, test)
    prediction["probability"] = calibrator.predict(prediction["logit"].to_numpy())
    prediction["threshold"] = threshold
    prediction["predicted"] = (prediction["probability"] >= threshold).astype(int)
    inner_export = inner_oof.assign(
        role="inner_oof",
        raw_logit=inner_oof["logit"],
        raw_probability=inner_oof["probability"],
        calibrated_probability=calibrator.predict(inner_oof["logit"].to_numpy()),
        input_representation="raw_logit",
    )
    inner_export.to_csv(run_dir / "inner_oof_predictions.csv", index=False)
    return prediction


def _train_neural_inner(
    examples: list[OutcomePatientExample],
    train_subjects: Sequence[str],
    validation_subjects: Sequence[str],
    variant: str,
    model_config: dict[str, Any],
    training_config: dict[str, Any],
    seed: int,
    run_dir: Path,
    device: torch.device,
    resume: bool = False,
) -> tuple[pd.DataFrame, int, int]:
    fit_raw = _by_subject(examples, train_subjects)
    validation_raw = _by_subject(examples, validation_subjects)
    if bool(model_config.get("input_pre_normalized", False)):
        fit_examples = fit_raw
        validation_examples = validation_raw
    else:
        normalizer = fit_normalizer_for_examples(fit_raw)
        fit_examples = transform_examples(fit_raw, normalizer)
        validation_examples = transform_examples(validation_raw, normalizer)
    input_dim = int(fit_examples[0].model_input["feature_runs"][0].shape[-1])
    torch.manual_seed(seed)
    model = build_model(variant, {**model_config, "random_seed": seed}, input_dim=input_dim)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainer = OutcomeTrainer({**training_config, "random_seed": seed}, device=device)
    result = trainer.fit(model, fit_examples, validation_examples, run_dir, resume=resume)
    return result.validation_predictions, result.best_epoch, int(parameter_count)


def _run_neural_fold(
    examples: list[OutcomePatientExample],
    train_manifest: pd.DataFrame,
    test_subjects: Sequence[str],
    variant: str,
    protocol: str,
    model_config: dict[str, Any],
    training_config: dict[str, Any],
    protocol_config: dict[str, Any],
    candidate_profiles: dict[str, Any],
    outer_fold_idx: int,
    seed: int,
    run_dir: Path,
    device: torch.device,
    resume: bool = False,
) -> pd.DataFrame:
    inner_folds = min(int(protocol_config.get("inner_folds", 4)), len(train_manifest))
    inner_ledger = build_inner_ledgers(train_manifest, n_splits=max(2, inner_folds), seed=seed)
    partitions = list(iter_outer_partitions(train_manifest, inner_ledger))
    if protocol == "screening" and not bool(protocol_config.get("full_inner_oof", False)):
        partitions = partitions[:1]
    if protocol == "final_nested":
        if not candidate_profiles:
            raise ValueError("final_nested requires non-empty candidate_profiles; fixed configuration is not full nested selection.")
        profiles = candidate_profiles
    else:
        profiles = {"screening_locked": {}}
    candidate_states: dict[str, dict[str, Any]] = {}
    candidate_metrics = []
    for profile_name, profile_payload in profiles.items():
        payload = dict(profile_payload or {})
        profile_model = {**model_config, **dict(payload.get("model") or {})}
        profile_training = {**training_config, **dict(payload.get("training") or {})}
        inner_rows = []
        best_epochs = []
        parameter_count = 0
        for partition in partitions:
            prediction, best_epoch, parameter_count = _train_neural_inner(
                examples,
                partition.train_subjects,
                partition.test_subjects,
                variant,
                profile_model,
                profile_training,
                seed + partition.fold_idx,
                run_dir / "candidates" / str(profile_name) / "inner" / f"fold_{partition.fold_idx}",
                device,
                resume,
            )
            prediction["inner_fold_idx"] = partition.fold_idx
            prediction["role"] = "inner_oof"
            inner_rows.append(prediction)
            best_epochs.append(best_epoch)
        candidate_inner = pd.concat(inner_rows, ignore_index=True)
        candidate_calibrator, candidate_threshold = _calibrate_inner_oof(
            candidate_inner,
            run_dir / "candidates" / str(profile_name),
        )
        calibrated = candidate_calibrator.predict(candidate_inner["logit"].to_numpy())
        metrics = compute_patient_metrics(candidate_inner["outcome"].to_numpy(), calibrated, candidate_threshold)
        candidate_metrics.append(
            {
                "candidate_profile": str(profile_name),
                "role": "inner_oof",
                "macro_f1": metrics.values["macro_f1"],
                "auroc": metrics.values["auroc"],
                "brier": metrics.values["brier"],
                "parameter_count": int(parameter_count),
            }
        )
        candidate_states[str(profile_name)] = {
            "inner_oof": candidate_inner,
            "best_epochs": best_epochs,
            "model_config": profile_model,
            "training_config": profile_training,
        }
    candidate_table = pd.DataFrame(candidate_metrics)
    selected_row = select_best_profile(candidate_table)
    selected_name = str(selected_row["candidate_profile"])
    selected_state = candidate_states[selected_name]
    if protocol == "final_nested":
        candidate_table.to_csv(run_dir / "inner_candidate_metrics.csv", index=False)
        (run_dir / "selected_candidate.json").write_text(
            json.dumps(
                {
                    **{key: value.item() if hasattr(value, "item") else value for key, value in selected_row.items()},
                    "model": selected_state["model_config"],
                    "training": selected_state["training_config"],
                    "selection_source": "inner_oof",
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    inner_oof = selected_state["inner_oof"]
    best_epochs = selected_state["best_epochs"]
    selected_model_config = selected_state["model_config"]
    selected_training_config = selected_state["training_config"]
    calibrator, threshold = _calibrate_inner_oof(inner_oof, run_dir)

    outer_train_raw = _by_subject(examples, train_manifest["subject_id"].astype(str).tolist())
    test_raw = _by_subject(examples, test_subjects)
    if bool(selected_model_config.get("input_pre_normalized", False)):
        outer_train = outer_train_raw
        test_examples = test_raw
        (run_dir / "normalizer_audit.json").write_text(json.dumps({"mode": "pre_normalized_token_input", "fit_subjects": sorted(train_manifest["subject_id"].astype(str).tolist())}, indent=2), encoding="utf-8")
    else:
        normalizer = fit_normalizer_for_examples(outer_train_raw)
        normalizer.save(run_dir / "normalizer.joblib", run_dir / "normalizer_audit.json")
        outer_train = transform_examples(outer_train_raw, normalizer)
        test_examples = transform_examples(test_raw, normalizer)
    input_dim = int(outer_train[0].model_input["feature_runs"][0].shape[-1])
    torch.manual_seed(seed)
    model = build_model(variant, {**selected_model_config, "random_seed": seed}, input_dim=input_dim)
    trainer = OutcomeTrainer({**selected_training_config, "random_seed": seed}, device=device)
    selected_epochs = max(1, int(round(float(np.median(best_epochs)))))
    outer_checkpoint = trainer.fit_fixed_epochs(model, outer_train, run_dir / "outer_train", epochs=selected_epochs)
    load_checkpoint(outer_checkpoint, model=model, map_location=device)
    _, prediction = evaluate_model(model, _loader(test_examples, int(selected_training_config.get("batch_size", 1))), device)
    replay_model_perturbations(
        model,
        test_examples,
        run_dir,
        device,
        calibrator=calibrator,
        threshold=threshold,
        seed=seed,
        outer_fold_idx=int(outer_fold_idx),
        checkpoint_path=outer_checkpoint,
    )
    if bool(selected_training_config.get("save_diagnostics", True)):
        export_interpretability(model, test_examples, run_dir, device)
    prediction["probability"] = calibrator.predict(prediction["logit"].to_numpy())
    prediction["threshold"] = threshold
    prediction["predicted"] = (prediction["probability"] >= threshold).astype(int)
    inner_export = inner_oof.assign(
        role="inner_oof",
        raw_logit=inner_oof["logit"],
        raw_probability=inner_oof["probability"],
        calibrated_probability=calibrator.predict(inner_oof["logit"].to_numpy()),
        input_representation="raw_logit",
    )
    inner_export.to_csv(run_dir / "inner_oof_predictions.csv", index=False)
    return prediction


def _resolve_device(value: str) -> torch.device:
    requested = str(value).lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _configured_cohort_subjects(config: dict[str, Any]) -> list[str] | None:
    values = (config.get("cohort") or {}).get("primary_subject_ids")
    if values is None:
        return None
    if not isinstance(values, (list, tuple, set)):
        raise ValueError("cohort.primary_subject_ids must be a sequence of explicit patient identifiers.")
    subjects = sorted({str(value) for value in values if str(value).strip()})
    if not subjects:
        raise ValueError("cohort.primary_subject_ids cannot be empty when configured.")
    return subjects


def run_outcome_experiment(
    config: dict[str, Any],
    *,
    protocol: str,
    variants: Sequence[str],
    seeds: Sequence[int],
    max_outer_folds: int = 0,
    max_patients: int = 0,
    device: str = "auto",
    resume: bool = False,
    overwrite: bool = False,
    dry_run: bool = False,
) -> ExperimentSummary:
    if protocol not in {"screening", "final_nested"}:
        raise ValueError("protocol must be screening or final_nested.")
    paths = dict(config.get("paths") or {})
    output_root = Path(paths["output_dir"])
    output_root.mkdir(parents=True, exist_ok=True)
    feature_cache = load_cache_contract(paths["feature_cache_path"])
    exclusion_rows: list[dict[str, Any]] = []
    configured_subjects = _configured_cohort_subjects(config)
    feature_examples = _limit_examples(
        build_outcome_patient_examples(feature_cache, subject_ids=configured_subjects, exclusion_audit=exclusion_rows),
        int(max_patients),
    )
    if configured_subjects is not None and set(configured_subjects) != {example.subject_id for example in feature_examples}:
        missing = sorted(set(configured_subjects) - {example.subject_id for example in feature_examples})
        raise ValueError(f"Configured Task 2 cohort is not fully available after outcome resolution: {missing[:20]}")
    if len({int(example.target) for example in feature_examples}) < 2:
        raise ValueError("Selected outcome cohort must contain both success and failure patients.")
    protocol_config = dict(config.get("protocol") or {})
    model_config = dict(config.get("model") or {})
    training_config = dict(config.get("training") or {})
    candidate_profiles = dict(config.get("candidate_profiles") or {})
    report_config = dict(config.get("report") or {})
    resolved_config_hash = _stable_config_hash(config)
    outer_folds = min(int(protocol_config.get("outer_folds", 5)), len(feature_examples))
    manifest = _example_manifest(feature_examples)
    ledger = build_composite_fold_ledger(manifest, n_splits=max(2, outer_folds), seed=int(protocol_config.get("random_seed", 42)), cohort="feature")
    folds_dir = output_root / "folds"
    folds_dir.mkdir(parents=True, exist_ok=True)
    ledger.to_csv(folds_dir / "outcome_fold_assignments_feature.csv", index=False)
    feature_fold_hash = ledger_hash(ledger)
    feature_subject_hash = _subject_set_hash(manifest["subject_id"].astype(str).tolist())
    manifests_dir = output_root / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifests_dir / "outcome_training_patient_manifest.csv", index=False)
    pd.DataFrame(
        exclusion_rows,
        columns=["subject_id", "run_id", "sample_id", "reason", "duplicate_count"],
    ).to_csv(manifests_dir / "outcome_training_exclusions.csv", index=False)
    environment = environment_manifest()
    (output_root / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    (output_root / "config_hash.txt").write_text(resolved_config_hash + "\n", encoding="utf-8")
    (output_root / "fold_ledger_hash.txt").write_text(feature_fold_hash + "\n", encoding="utf-8")
    manifest_hash = hashlib.sha256(manifest.to_csv(index=False).encode("utf-8")).hexdigest()
    (output_root / "data_manifest_hash.txt").write_text(manifest_hash + "\n", encoding="utf-8")
    (output_root / "git_commit.txt").write_text(str(environment.get("git_commit", "unknown")) + "\n", encoding="utf-8")
    (output_root / "environment.json").write_text(json.dumps(environment, indent=2, sort_keys=True), encoding="utf-8")
    if dry_run:
        return ExperimentSummary(pd.DataFrame(), pd.DataFrame())

    predictions = []
    failures = []
    active_device = _resolve_device(device)
    selected_variants = list(variants)
    feature_variants = [variant for variant in selected_variants if variant not in {"H9_FM_RECURRENCE", "H10_LATE_FUSION"}]
    partitions = list(iter_outer_partitions(manifest, ledger))
    if max_outer_folds > 0:
        partitions = partitions[: int(max_outer_folds)]
    for partition in partitions:
        for variant in feature_variants:
            for seed in seeds:
                run_dir = output_root / protocol / variant / f"outer_fold_{partition.fold_idx}" / f"seed_{int(seed)}"
                try:
                    completed_path = run_dir / "outer_test_predictions.csv"
                    if completed_path.exists() and resume:
                        resumed = pd.read_csv(completed_path)
                        _validate_resume_predictions(resumed, config_hash=resolved_config_hash, fold_hash=feature_fold_hash, cohort_id="feature_full", subject_set_hash=feature_subject_hash)
                        predictions.append(resumed)
                        continue
                    if run_dir.exists() and overwrite:
                        shutil.rmtree(run_dir)
                    elif run_dir.exists() and any(run_dir.iterdir()) and not resume:
                        raise FileExistsError(f"Run directory already exists; use --resume or --overwrite: {run_dir}")
                    run_dir.mkdir(parents=True, exist_ok=True)
                    if variant == "H1_SUMMARY_ML":
                        frame = _run_h1_fold(feature_examples, partition.train_manifest, partition.test_subjects, protocol, protocol_config, int(seed), run_dir)
                    else:
                        frame = _run_neural_fold(feature_examples, partition.train_manifest, partition.test_subjects, variant, protocol, model_config, training_config, protocol_config, candidate_profiles, int(partition.fold_idx), int(seed), run_dir, active_device, resume)
                    checkpoint_path = "" if variant == "H1_SUMMARY_ML" else run_dir / "outer_train" / "checkpoint_outer_train_final.pt"
                    frame = _attach_prediction_provenance(
                        frame,
                        checkpoint_path=checkpoint_path,
                        fold_hash=feature_fold_hash,
                        config_hash=resolved_config_hash,
                        cohort_id="feature_full",
                        subject_set_hash=feature_subject_hash,
                    )
                    frame["variant"] = variant
                    frame["seed"] = int(seed)
                    frame["outer_fold_idx"] = int(partition.fold_idx)
                    frame["role"] = "outer_test"
                    frame.to_csv(run_dir / "outer_test_predictions.csv", index=False)
                    predictions.append(frame)
                    write_environment_manifest(run_dir / "environment.json")
                except Exception as exc:
                    failures.append({"variant": variant, "seed": int(seed), "outer_fold_idx": int(partition.fold_idx), "error_type": type(exc).__name__, "error": str(exc)})

    if "H9_FM_RECURRENCE" in selected_variants:
        fm_cache_path = paths.get("fm_embedding_cache_path")
        if not fm_cache_path:
            failures.append({"variant": "H9_FM_RECURRENCE", "seed": -1, "outer_fold_idx": -1, "error_type": "MissingDependency", "error": "paths.fm_embedding_cache_path is required for H9."})
        else:
            fm_cache = load_fm_embedding_as_feature_cache(fm_cache_path)
            fm_exclusions: list[dict[str, Any]] = []
            fm_examples = _limit_examples(build_outcome_patient_examples(fm_cache, exclusion_audit=fm_exclusions), int(max_patients))
            fm_manifest = _example_manifest(fm_examples)
            fm_outer_folds = min(int(protocol_config.get("outer_folds", 5)), len(fm_examples))
            fm_ledger = build_composite_fold_ledger(
                fm_manifest,
                n_splits=max(2, fm_outer_folds),
                seed=int(protocol_config.get("random_seed", 42)),
                cohort="raw",
            )
            fm_ledger.to_csv(folds_dir / "outcome_fold_assignments_raw.csv", index=False)
            fm_fold_hash = ledger_hash(fm_ledger)
            fm_subject_hash = _subject_set_hash(fm_manifest["subject_id"].astype(str).tolist())
            fm_partitions = list(iter_outer_partitions(fm_manifest, fm_ledger))
            if max_outer_folds > 0:
                fm_partitions = fm_partitions[: int(max_outer_folds)]
            for partition in fm_partitions:
                for seed in seeds:
                    run_dir = output_root / protocol / "H9_FM_RECURRENCE" / f"outer_fold_{partition.fold_idx}" / f"seed_{int(seed)}"
                    try:
                        completed_path = run_dir / "outer_test_predictions.csv"
                        if completed_path.exists() and resume:
                            resumed = pd.read_csv(completed_path)
                            _validate_resume_predictions(resumed, config_hash=resolved_config_hash, fold_hash=fm_fold_hash, cohort_id="fm_full", subject_set_hash=fm_subject_hash)
                            predictions.append(resumed)
                            continue
                        if run_dir.exists() and overwrite:
                            shutil.rmtree(run_dir)
                        elif run_dir.exists() and any(run_dir.iterdir()) and not resume:
                            raise FileExistsError(f"Run directory already exists; use --resume or --overwrite: {run_dir}")
                        run_dir.mkdir(parents=True, exist_ok=True)
                        frame = _run_neural_fold(
                            fm_examples,
                            partition.train_manifest,
                            partition.test_subjects,
                            "H9_FM_RECURRENCE",
                            protocol,
                            model_config,
                            training_config,
                            protocol_config,
                            candidate_profiles,
                            int(partition.fold_idx),
                            int(seed),
                            run_dir,
                            active_device,
                            resume,
                        )
                        frame = _attach_prediction_provenance(
                            frame,
                            checkpoint_path=run_dir / "outer_train" / "checkpoint_outer_train_final.pt",
                            fold_hash=fm_fold_hash,
                            config_hash=resolved_config_hash,
                            cohort_id="fm_full",
                            subject_set_hash=fm_subject_hash,
                        )
                        frame["variant"] = "H9_FM_RECURRENCE"
                        frame["seed"] = int(seed)
                        frame["outer_fold_idx"] = int(partition.fold_idx)
                        frame["role"] = "outer_test"
                        frame.to_csv(completed_path, index=False)
                        predictions.append(frame)
                    except Exception as exc:
                        failures.append({"variant": "H9_FM_RECURRENCE", "seed": int(seed), "outer_fold_idx": int(partition.fold_idx), "error_type": type(exc).__name__, "error": str(exc)})

    if "H10_LATE_FUSION" in selected_variants:
        fm_cache_path = paths.get("fm_embedding_cache_path")
        if not fm_cache_path:
            failures.append({"variant": "H10_LATE_FUSION", "seed": -1, "outer_fold_idx": -1, "error_type": "MissingDependency", "error": "paths.fm_embedding_cache_path is required for automatic H10 fusion."})
        else:
            try:
                fm_cache = load_fm_embedding_as_feature_cache(fm_cache_path)
                fm_examples_for_fusion = _limit_examples(build_outcome_patient_examples(fm_cache), int(max_patients))
                fusion_config = dict(config.get("fusion") or {})
                fusion_cohort = build_fusion_cohort(
                    feature_examples,
                    fm_examples_for_fusion,
                    minimum_run_alignment_ratio=float(fusion_config.get("minimum_run_alignment_ratio", 1.0)),
                )
                fusion_manifest = fusion_cohort.manifest.copy()
                if len({int(value) for value in fusion_manifest["outcome_label"]}) < 2:
                    raise ValueError("Fusion cohort must contain both outcome classes.")
                fusion_manifest.to_csv(manifests_dir / "outcome_fusion_patient_manifest.csv", index=False)
                fusion_cohort.alignment_audit.to_csv(manifests_dir / "outcome_fusion_run_alignment_audit.csv", index=False)
                fusion_outer_folds = min(int(protocol_config.get("outer_folds", 5)), len(fusion_manifest))
                fusion_ledger = build_composite_fold_ledger(
                    fusion_manifest,
                    n_splits=max(2, fusion_outer_folds),
                    seed=int(protocol_config.get("random_seed", 42)),
                    cohort="fusion",
                )
                fusion_ledger.to_csv(folds_dir / "outcome_fold_assignments_fusion.csv", index=False)
                fusion_fold_hash = ledger_hash(fusion_ledger)
                fusion_subject_hash = _subject_set_hash(fusion_manifest["subject_id"].astype(str).tolist())
                fusion_partitions = list(iter_outer_partitions(fusion_manifest, fusion_ledger))
                if max_outer_folds > 0:
                    fusion_partitions = fusion_partitions[: int(max_outer_folds)]
                feature_branch_variant = str(fusion_config.get("feature_variant", "H8_RECURRENCE"))
                for partition in fusion_partitions:
                    for seed in seeds:
                        fusion_dir = output_root / protocol / "H10_LATE_FUSION" / f"outer_fold_{partition.fold_idx}" / f"seed_{int(seed)}"
                        try:
                            if fusion_dir.exists() and overwrite:
                                shutil.rmtree(fusion_dir)
                            fusion_dir.mkdir(parents=True, exist_ok=True)
                            feature_dir = fusion_dir / "feature_branch"
                            fm_dir = fusion_dir / "fm_branch"
                            feature_outer = _run_neural_fold(
                                list(fusion_cohort.feature_examples), partition.train_manifest, partition.test_subjects,
                                feature_branch_variant, protocol, model_config, training_config, protocol_config,
                                candidate_profiles, int(partition.fold_idx), int(seed), feature_dir, active_device, resume,
                            )
                            fm_outer = _run_neural_fold(
                                list(fusion_cohort.fm_examples), partition.train_manifest, partition.test_subjects,
                                "H9_FM_RECURRENCE", protocol, model_config, training_config, protocol_config,
                                candidate_profiles, int(partition.fold_idx), int(seed), fm_dir, active_device, resume,
                            )
                            feature_intersection_name = "H8_FEATURE_INTERSECTION" if feature_branch_variant == "H8_RECURRENCE" else f"{feature_branch_variant}_FEATURE_INTERSECTION"
                            feature_intersection = _attach_prediction_provenance(
                                feature_outer,
                                checkpoint_path=feature_dir / "outer_train" / "checkpoint_outer_train_final.pt",
                                fold_hash=fusion_fold_hash,
                                config_hash=resolved_config_hash,
                                cohort_id="fusion_intersection",
                                subject_set_hash=fusion_subject_hash,
                            )
                            feature_intersection["variant"] = feature_intersection_name
                            feature_intersection["seed"] = int(seed)
                            feature_intersection["outer_fold_idx"] = int(partition.fold_idx)
                            feature_intersection["role"] = "outer_test"
                            fm_intersection = _attach_prediction_provenance(
                                fm_outer,
                                checkpoint_path=fm_dir / "outer_train" / "checkpoint_outer_train_final.pt",
                                fold_hash=fusion_fold_hash,
                                config_hash=resolved_config_hash,
                                cohort_id="fusion_intersection",
                                subject_set_hash=fusion_subject_hash,
                            )
                            fm_intersection["variant"] = "H9_FM_INTERSECTION"
                            fm_intersection["seed"] = int(seed)
                            fm_intersection["outer_fold_idx"] = int(partition.fold_idx)
                            fm_intersection["role"] = "outer_test"
                            feature_intersection.to_csv(fusion_dir / "feature_intersection_outer_predictions.csv", index=False)
                            fm_intersection.to_csv(fusion_dir / "fm_intersection_outer_predictions.csv", index=False)
                            predictions.extend([feature_intersection, fm_intersection])
                            feature_inner = pd.read_csv(feature_dir / "inner_oof_predictions.csv")
                            fm_inner = pd.read_csv(fm_dir / "inner_oof_predictions.csv")
                            for inner in (feature_inner, fm_inner):
                                inner["outer_fold_idx"] = int(partition.fold_idx)
                                inner["seed"] = int(seed)
                                inner["role"] = "inner_oof"
                            outer_table = feature_outer[["subject_id", "center", "outcome", "logit"]].rename(columns={"logit": "feature_raw_logit"})
                            outer_table = outer_table.merge(
                                fm_outer[["subject_id", "outcome", "logit"]].rename(columns={"outcome": "fm_outcome", "logit": "fm_raw_logit"}),
                                on="subject_id", how="inner", validate="one_to_one",
                            )
                            if len(outer_table) != len(partition.test_subjects) or not np.array_equal(outer_table["outcome"], outer_table["fm_outcome"]):
                                raise ValueError("Feature and FM outer fusion rows do not exactly align on cohort and outcome.")
                            outer_table = outer_table.drop(columns=["fm_outcome"])
                            outer_table["outer_fold_idx"] = int(partition.fold_idx)
                            outer_table["seed"] = int(seed)
                            outer_table["role"] = "outer_test"
                            outer_table["feature_input_representation"] = "raw_logit"
                            outer_table["fm_input_representation"] = "raw_logit"
                            fusion = fit_cross_fitted_stacker(feature_inner, fm_inner, outer_table)
                            inner_for_threshold = pd.DataFrame(
                                {
                                    "subject_id": fusion.inner_oof["subject_id"],
                                    "outcome": fusion.inner_oof["outcome"],
                                    "probability": fusion.inner_oof["fusion_probability"],
                                    "role": "inner_oof",
                                }
                            )
                            threshold_selection = select_macro_f1_threshold(inner_for_threshold)
                            fused = fusion.outer_predictions.copy()
                            fused["logit"] = np.log(np.clip(fused["fusion_probability"], 1e-6, 1 - 1e-6) / np.clip(1 - fused["fusion_probability"], 1e-6, 1 - 1e-6))
                            fused["probability"] = fused["fusion_probability"]
                            fused["threshold"] = float(threshold_selection.threshold)
                            fused["predicted"] = (fused["probability"] >= float(threshold_selection.threshold)).astype(int)
                            fused = _attach_prediction_provenance(
                                fused,
                                checkpoint_path=fusion_dir / "late_fusion_coefficients.csv",
                                fold_hash=fusion_fold_hash,
                                config_hash=resolved_config_hash,
                                cohort_id="fusion_intersection",
                                subject_set_hash=fusion_subject_hash,
                                calibration_method="inner_oof_logistic_stacker",
                            )
                            fused["variant"] = "H10_LATE_FUSION"
                            fused.to_csv(fusion_dir / "late_fusion_outer_predictions.csv", index=False)
                            fusion.inner_oof.to_csv(fusion_dir / "late_fusion_inner_oof.csv", index=False)
                            pd.DataFrame([fusion.coefficients]).to_csv(fusion_dir / "late_fusion_coefficients.csv", index=False)
                            threshold_selection.curve.to_csv(fusion_dir / "late_fusion_threshold_curve.csv", index=False)
                            (fusion_dir / "late_fusion_protocol_audit.json").write_text(
                                json.dumps({"outer_fold_idx": int(partition.fold_idx), "seed": int(seed), "fold_ledger_hash": fusion_fold_hash, "feature_variant": feature_branch_variant, "fm_variant": "H9_FM_RECURRENCE", "feature_input_representation": "raw_logit", "fm_input_representation": "raw_logit", "stacker_fit_source": "inner_oof", "threshold_source": "fusion_inner_oof"}, indent=2, sort_keys=True),
                                encoding="utf-8",
                            )
                            predictions.append(fused)
                        except Exception as exc:
                            failures.append({"variant": "H10_LATE_FUSION", "seed": int(seed), "outer_fold_idx": int(partition.fold_idx), "error_type": type(exc).__name__, "error": str(exc)})
            except Exception as exc:
                failures.append({"variant": "H10_LATE_FUSION", "seed": -1, "outer_fold_idx": -1, "error_type": type(exc).__name__, "error": str(exc)})

    prediction_frame = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    failure_frame = pd.DataFrame(failures)
    run_manifest = {
        "protocol": protocol,
        "variants": selected_variants,
        "seeds": [int(seed) for seed in seeds],
        "random_seed": int(protocol_config.get("random_seed", 42)),
        "patient_count": len(feature_examples),
        "fold_ledger_hash": feature_fold_hash,
        "max_outer_folds": int(max_outer_folds),
        "max_patients": int(max_patients),
        "device": str(active_device),
        "environment": environment,
        "bootstrap_reference": str(report_config.get("bootstrap_reference", selected_variants[0] if selected_variants else "H2_HIER_POOL")),
        "bootstrap_pairs": list(report_config.get("bootstrap_pairs") or []),
        "bootstrap_samples": int(protocol_config.get("bootstrap_samples", 500)),
    }
    if not prediction_frame.empty:
        summarize_experiment(prediction_frame, failure_frame, output_root / "reports", run_manifest=run_manifest)
        aggregate_interpretability_artifacts(output_root / protocol, output_root / "reports")
        run_shortcut_audit(output_root)
        summarize_experiment(prediction_frame, failure_frame, output_root / "reports", run_manifest=run_manifest)
    return ExperimentSummary(prediction_frame, failure_frame)


def run_loco_experiment(
    config: dict[str, Any],
    *,
    variants: Sequence[str],
    seeds: Sequence[int],
    device: str = "auto",
) -> ExperimentSummary:
    paths = dict(config.get("paths") or {})
    output_root = Path(paths["output_dir"]) / "loco"
    exclusion_rows: list[dict[str, Any]] = []
    examples = build_outcome_patient_examples(load_cache_contract(paths["feature_cache_path"]), exclusion_audit=exclusion_rows)
    manifest = _example_manifest(examples)
    model_config = dict(config.get("model") or {})
    training_config = dict(config.get("training") or {})
    protocol_config = dict(config.get("protocol") or {})
    active_device = _resolve_device(device)
    predictions = []
    failures = []
    loco_partitions = build_loco_partitions(manifest)
    loco_assignments = pd.concat(
        [partition.test_manifest[["subject_id"]].assign(fold_idx=index) for index, partition in enumerate(loco_partitions, start=1)],
        ignore_index=True,
    )
    loco_fold_hash = hashlib.sha256(
        loco_assignments.sort_values("subject_id", kind="stable").to_csv(index=False).encode("utf-8")
    ).hexdigest()
    loco_config_hash = _stable_config_hash(config)
    loco_subject_hash = _subject_set_hash(manifest["subject_id"].astype(str).tolist())
    for partition_index, partition in enumerate(loco_partitions, start=1):
        test_subjects = partition.test_manifest["subject_id"].astype(str).tolist()
        for variant in variants:
            for seed in seeds:
                run_dir = output_root / variant / f"held_out_{partition.held_out_center}" / f"seed_{int(seed)}"
                try:
                    if variant == "H1_SUMMARY_ML":
                        frame = _run_h1_fold(examples, partition.train_manifest, test_subjects, "final_nested", protocol_config, int(seed), run_dir)
                    else:
                        frame = _run_neural_fold(examples, partition.train_manifest, test_subjects, variant, "loco_fixed_config", model_config, training_config, protocol_config, {}, int(partition_index), int(seed), run_dir, active_device)
                    frame = _attach_prediction_provenance(
                        frame,
                        checkpoint_path="" if variant == "H1_SUMMARY_ML" else run_dir / "outer_train" / "checkpoint_outer_train_final.pt",
                        fold_hash=loco_fold_hash,
                        config_hash=loco_config_hash,
                        cohort_id="loco_all_centers",
                        subject_set_hash=loco_subject_hash,
                    )
                    frame["variant"] = variant
                    frame["seed"] = int(seed)
                    frame["outer_fold_idx"] = partition_index
                    frame["held_out_center"] = partition.held_out_center
                    frame["role"] = "loco_test"
                    predictions.append(frame)
                except Exception as exc:
                    failures.append({"variant": variant, "seed": int(seed), "outer_fold_idx": partition_index, "held_out_center": partition.held_out_center, "error_type": type(exc).__name__, "error": str(exc)})
    prediction_frame = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    failure_frame = pd.DataFrame(failures)
    if not prediction_frame.empty:
        summarize_experiment(
            prediction_frame,
            failure_frame,
            output_root / "reports",
            run_manifest={"protocol": "loco", "random_seed": int(protocol_config.get("random_seed", 42)), "seeds": list(seeds), "variants": list(variants)},
            evaluation_roles=("loco_test",),
        )
    return ExperimentSummary(prediction_frame, failure_frame)


__all__ = ["ExperimentSummary", "run_loco_experiment", "run_outcome_experiment", "run_partitioned_experiment"]
