from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch

from outcome_hifos.cache_schema import file_sha256, load_cache_contract
from outcome_hifos.dataset import build_outcome_patient_examples
from outcome_hifos.fm.embedding_cache import load_fm_embedding_as_feature_cache, validate_official_fm_embedding_cache
from outcome_hifos.folds import build_composite_fold_ledger, ledger_hash
from outcome_hifos.training.experiment_runner import run_outcome_experiment
from .patient_summary import build_patient_summary_matrix
from .cohort_manifests import load_success_subject_manifest
from .protocol import build_task2_primary_cohort
from .reporting import summarize_task2_baselines
from .token_data import build_embedding_patient_examples, build_raw_patient_examples
from .training import run_hierarchical_token_oof, run_simple_baseline_loco, run_simple_baseline_oof
from task1_baselines.raw_preprocessing import RawPreprocessingConfig


NEURAL_ALIASES = {"h2_hier_pool": "H2_HIER_POOL", "h3_attention_mil": "H3_ATTENTION_MIL"}
RAW_ALIASES = {
    "eegnet_hier": "T2_EEGNET_HIERPOOL", "shallow_fbcsp_hier": "T2_SHALLOWFBCSPNET_HIERPOOL",
    "deep4_hier": "T2_DEEP4NET_HIERPOOL", "eegconformer_hier": "T2_EEGCONFORMER_HIERPOOL",
}
FM_ALIASES = {"brainbert_frozen_hier": "T2_BRAINBERT_FROZEN_HIERPOOL", "cbramod_frozen_hier": "T2_CBRAMOD_FROZEN_HIERPOOL"}
SIMPLE_MODELS = {"majority", "elasticnet", "rbf_svm", "random_forest", "lightgbm"}


def run_task2_baseline_suite(
    *,
    feature_cache_path: str | Path,
    raw_cache_path: str | Path | None = None,
    brainbert_embedding_cache_path: str | Path | None = None,
    cbramod_embedding_cache_path: str | Path | None = None,
    old90_manifest_path: str | Path,
    cohort_name: str = "task2_primary",
    positive_class: str = "success",
    output_dir: str | Path,
    models: Sequence[str],
    seeds: Sequence[int],
    inner_folds: int = 4,
    fixed_baseline: bool = False,
    run_loco: bool = False,
    compact: bool = False,
    strict: bool = False,
    device: str = "auto",
    epochs: int | None = None,
    max_outer_folds: int = 0,
) -> int:
    started_at = time.perf_counter()

    def progress(message: str) -> None:
        print(f"[Task2 {cohort_name} +{time.perf_counter() - started_at:8.1f}s] {message}", flush=True)

    output = Path(output_dir)
    for name in ("audit", "cohorts", "folds", "configs", "provenance", "checkpoints", "embedding_cache", "oof_predictions", "metrics", "logs", "comparison", "shortcut_audit", "loco", "reports"):
        (output / name).mkdir(parents=True, exist_ok=True)
    progress(
        f"starting models={list(models)}, seeds={list(seeds)}, positive_class={positive_class}, "
        f"protocol={'fixed outer-5-fold' if fixed_baseline else f'nested outer-5/inner-{inner_folds}-fold'}"
    )
    progress(f"loading feature cache: {feature_cache_path}")
    cache = load_cache_contract(feature_cache_path)
    progress(f"feature cache ready: run_records={len(cache.run_records)}, patients={len(cache.patient_index)}")
    all_examples = build_outcome_patient_examples(cache)
    outcome_manifest = pd.DataFrame([{"subject_id": example.subject_id, "center": example.center, "outcome_group": "success" if example.target == 1 else "failure", "outcome_label": int(example.target)} for example in all_examples])
    old90 = load_success_subject_manifest(old90_manifest_path)
    cohort, cohort_audit = build_task2_primary_cohort(
        outcome_manifest, old90_subjects=old90, positive_class=positive_class,
    )
    cohort_audit["cohort_name"] = cohort_name
    progress(
        f"cohort ready: success={cohort_audit['old90_success']}, failure={cohort_audit['failure']}, "
        f"total={cohort_audit['cohort_size']}"
    )
    cohort.to_csv(output / "cohorts" / "task2_primary_cohort.csv", index=False)
    (output / "audit" / "task2_cohort_audit.json").write_text(json.dumps(cohort_audit, indent=2, sort_keys=True), encoding="utf-8")
    cohort_targets = dict(zip(cohort["subject_id"].astype(str), cohort["outcome_label"].astype(int)))
    selected_examples = [
        replace(example, target=float(cohort_targets[example.subject_id]))
        for example in all_examples if example.subject_id in cohort_targets
    ]
    summary, summary_manifest = build_patient_summary_matrix(selected_examples)
    summary["target"] = summary["subject_id"].map(cohort_targets).astype(int)
    size_metadata = pd.DataFrame(
        [
            {
                "subject_id": example.subject_id,
                "n_seizures": len(example.model_input["feature_runs"]),
                "n_channels": len(example.canonical_channels),
                "n_windows": int(sum(run.shape[0] for run in example.model_input["feature_runs"])),
            }
            for example in selected_examples
        ]
    )
    (output / "configs" / "task2_patient_summary_feature_manifest.json").write_text(json.dumps(summary_manifest, indent=2, sort_keys=True), encoding="utf-8")
    folds = build_composite_fold_ledger(cohort, n_splits=5, seed=42, cohort=cohort_name)
    for name in ("task2_fold_assignments_feature.csv", "task2_fold_assignments_raw.csv", "task2_fold_assignments_fusion.csv"):
        folds.to_csv(output / "folds" / name, index=False)
    provenance = {
        "task": "task2_surgery_outcome",
        "label_semantics": {positive_class: 1, ("failure" if positive_class == "success" else "success"): 0},
        "feature_cache_path": str(Path(feature_cache_path).resolve()),
        "feature_cache_sha256": file_sha256(feature_cache_path),
        "old90_manifest_path": str(Path(old90_manifest_path).resolve()),
        "old90_manifest_sha256": file_sha256(old90_manifest_path),
        "cohort_hash": cohort_audit["cohort_hash"],
        "fold_ledger_hash": ledger_hash(folds),
    }
    (output / "provenance" / "task2_protocol.json").write_text(json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8")
    predictions = []
    failures: list[dict[str, Any]] = []
    requested = [str(model).lower() for model in models]
    config_hash = hashlib.sha256(json.dumps({"models": requested, "seeds": list(seeds), "inner_folds": inner_folds, "fixed_baseline": fixed_baseline, "compact": compact}, sort_keys=True).encode("utf-8")).hexdigest()
    for model in requested:
        if model not in SIMPLE_MODELS:
            continue
        for seed in seeds:
            try:
                progress(f"START model={model} seed={seed}: outer-CV")
                result = run_simple_baseline_oof(summary, folds, model_name=model, seed=int(seed), inner_folds=inner_folds, compact=compact, fixed_baseline=fixed_baseline)
                result.oof = result.oof.copy()
                result.oof["feature_cache_hash"] = provenance["feature_cache_sha256"]
                result.oof["raw_cache_hash"] = ""
                result.oof["fold_ledger_hash"] = provenance["fold_ledger_hash"]
                result.oof["config_hash"] = config_hash
                result.oof["checkpoint_path"] = ""
                result.oof["cohort_name"] = cohort_name
                result.oof = result.oof.merge(size_metadata, on="subject_id", validate="one_to_one")
                model_dir = output / "oof_predictions" / model
                model_dir.mkdir(parents=True, exist_ok=True)
                result.oof.to_csv(model_dir / f"seed_{int(seed)}_patient_oof.csv", index=False)
                result.selected_parameters.to_csv(output / "configs" / f"{model}_seed_{int(seed)}_selected_parameters.csv", index=False)
                predictions.append(result.oof)
                progress(f"DONE model={model} seed={seed}: oof_patients={len(result.oof)}")
            except Exception as exc:
                progress(f"FAIL model={model} seed={seed}: {type(exc).__name__}: {exc}")
                failures.append({"model": model, "seed": int(seed), "error_type": type(exc).__name__, "error": str(exc)})
    loco_predictions = []
    for model in requested if run_loco else []:
        if model not in SIMPLE_MODELS:
            continue
        for seed in seeds:
            try:
                progress(f"START LOCO model={model} seed={seed}")
                loco = run_simple_baseline_loco(
                    summary, model_name=model, seed=int(seed), inner_folds=inner_folds, compact=compact, fixed_baseline=fixed_baseline
                )
                loco.oof["protocol"] = "leave_one_center_out"
                loco_predictions.append(loco.oof)
                model_dir = output / "loco" / model
                model_dir.mkdir(parents=True, exist_ok=True)
                loco.oof.to_csv(model_dir / f"seed_{int(seed)}_patient_predictions.csv", index=False)
                progress(f"DONE LOCO model={model} seed={seed}: oof_patients={len(loco.oof)}")
            except Exception as exc:
                progress(f"FAIL LOCO model={model} seed={seed}: {type(exc).__name__}: {exc}")
                failures.append({"model": f"{model}:loco", "seed": int(seed), "error_type": type(exc).__name__, "error": str(exc)})
    if loco_predictions:
        loco_table = pd.concat(loco_predictions, ignore_index=True)
        loco_rows = []
        from outcome_hifos.metrics import compute_patient_metrics
        for (model, seed, center), group in loco_table.groupby(["model", "seed", "held_out_center"]):
            bundle = compute_patient_metrics(
                group["outcome"].to_numpy(), group["calibrated_probability"].to_numpy(),
                predicted=group["predicted"].to_numpy(),
            )
            loco_rows.append({"model": model, "seed": seed, "held_out_center": center, "n_patients": len(group), **bundle.values})
        pd.DataFrame(loco_rows).to_csv(output / "loco" / "task2_loco.csv", index=False)
    neural = [NEURAL_ALIASES[model] for model in requested if model in NEURAL_ALIASES]
    if neural:
        neural_config = {
            "paths": {"feature_cache_path": str(feature_cache_path), "output_dir": str(output / "neural")},
            "cohort": {"primary_subject_ids": sorted(cohort["subject_id"].astype(str).tolist())},
            "model": {"model_dim": 64, "num_cores": 4, "dropout": 0.15, "head_dropout": 0.2},
            "training": {"epochs": int(epochs or (2 if compact else 40)), "patience": 2 if compact else 8, "batch_size": 1, "num_workers": 0 if compact else 4, "amp": not compact, "learning_rate": 1e-3, "weight_decay": 1e-4, "grad_clip": 1.0},
            "protocol": {"outer_folds": 5, "inner_folds": inner_folds, "random_seed": 42, "bootstrap_samples": 50 if compact else 2000, "full_inner_oof": True},
        }
        neural_result = run_outcome_experiment(neural_config, protocol="screening", variants=neural, seeds=seeds, max_outer_folds=max_outer_folds, device=device, overwrite=True)
        if not neural_result.predictions.empty:
            for (variant, current_seed), frame in neural_result.predictions.groupby(["variant", "seed"]):
                normalized = frame.rename(columns={"variant": "model", "probability": "calibrated_probability", "threshold": "selected_threshold", "outer_fold_idx": "outer_fold"}).copy()
                normalized["raw_logit"] = normalized["logit"]
                normalized["raw_probability"] = 1.0 / (1.0 + np.exp(-normalized["logit"].to_numpy(dtype=float)))
                normalized["calibration_method"] = "platt_inner_oof"
                normalized["threshold_source"] = "inner_oof_patient_macro_f1"
                normalized["feature_cache_hash"] = provenance["feature_cache_sha256"]
                normalized["fold_ledger_hash"] = provenance["fold_ledger_hash"]
                normalized["config_hash"] = config_hash
                normalized["cohort_name"] = cohort_name
                normalized = normalized.merge(size_metadata, on="subject_id", how="left", validate="one_to_one")
                model_dir = output / "oof_predictions" / str(variant).lower()
                model_dir.mkdir(parents=True, exist_ok=True)
                normalized.to_csv(model_dir / f"seed_{int(current_seed)}_patient_oof.csv", index=False)
                predictions.append(normalized)
        if not neural_result.failures.empty:
            failures.extend(neural_result.failures.rename(columns={"variant": "model"}).to_dict("records"))

    active_device = "cuda" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device
    raw_examples = None
    raw_hash = ""
    if set(requested) & set(RAW_ALIASES):
        if raw_cache_path is None:
            failures.append({"model": "raw_hierarchical", "seed": -1, "error_type": "MissingDependency", "error": "Task 2 raw cache path is required."})
        else:
            try:
                raw_cache = load_cache_contract(raw_cache_path)
                raw_examples, raw_audit = build_raw_patient_examples(raw_cache, selected_examples, RawPreprocessingConfig())
                raw_hash = file_sha256(raw_cache_path)
                pd.DataFrame(raw_audit).to_csv(output / "audit" / "task2_raw_preprocessing.csv", index=False)
            except Exception as exc:
                failures.append({"model": "raw_hierarchical", "seed": -1, "error_type": type(exc).__name__, "error": str(exc)})
    for alias, variant in RAW_ALIASES.items():
        if alias not in requested or raw_examples is None:
            continue
        for seed in seeds:
            try:
                frame = run_hierarchical_token_oof(raw_examples, folds, variant=variant, seed=int(seed), output_dir=output / "token_runs", device=active_device, inner_folds=inner_folds, epochs=int(epochs or (2 if compact else 40)), max_outer_folds=max_outer_folds, model_config={"raw_sfreq": 200.0, "token_embedding_dim": 128})
                frame["feature_cache_hash"] = provenance["feature_cache_sha256"]
                frame["raw_cache_hash"] = raw_hash
                frame["fold_ledger_hash"] = provenance["fold_ledger_hash"]
                frame["config_hash"] = config_hash
                frame["cohort_name"] = cohort_name
                frame = frame.merge(size_metadata, on="subject_id", how="left", validate="one_to_one")
                model_dir = output / "oof_predictions" / alias
                model_dir.mkdir(parents=True, exist_ok=True)
                frame.to_csv(model_dir / f"seed_{int(seed)}_patient_oof.csv", index=False)
                predictions.append(frame)
            except Exception as exc:
                failures.append({"model": alias, "seed": int(seed), "error_type": type(exc).__name__, "error": str(exc)})

    fm_paths = {"brainbert_frozen_hier": brainbert_embedding_cache_path, "cbramod_frozen_hier": cbramod_embedding_cache_path}
    for alias, variant in FM_ALIASES.items():
        if alias not in requested:
            continue
        fm_path = fm_paths[alias]
        if fm_path is None:
            failures.append({"model": alias, "seed": -1, "error_type": "MissingDependency", "error": f"{alias} embedding cache path is required."})
            continue
        try:
            fm_name = "brainbert" if alias.startswith("brainbert") else "cbramod"
            fm_provenance = validate_official_fm_embedding_cache(fm_path, fm_name)
            (output / "provenance" / f"{fm_name}_official_checkpoint.json").write_text(
                json.dumps(fm_provenance, indent=2, sort_keys=True), encoding="utf-8"
            )
            fm_cache = load_fm_embedding_as_feature_cache(fm_path)
            fm_examples, fm_audit = build_embedding_patient_examples(fm_cache, selected_examples)
            pd.DataFrame(fm_audit).to_csv(output / "audit" / f"{alias}_alignment.csv", index=False)
            fm_hash = file_sha256(Path(fm_path) / "all_windows_embeddings.pkl" if Path(fm_path).is_dir() else fm_path)
            for seed in seeds:
                frame = run_hierarchical_token_oof(fm_examples, folds, variant=variant, seed=int(seed), output_dir=output / "token_runs", device=active_device, inner_folds=inner_folds, epochs=int(epochs or (2 if compact else 40)), max_outer_folds=max_outer_folds)
                frame["feature_cache_hash"] = provenance["feature_cache_sha256"]
                frame["raw_cache_hash"] = fm_hash
                frame["fold_ledger_hash"] = provenance["fold_ledger_hash"]
                frame["config_hash"] = config_hash
                frame["cohort_name"] = cohort_name
                frame = frame.merge(size_metadata, on="subject_id", how="left", validate="one_to_one")
                model_dir = output / "oof_predictions" / alias
                model_dir.mkdir(parents=True, exist_ok=True)
                frame.to_csv(model_dir / f"seed_{int(seed)}_patient_oof.csv", index=False)
                predictions.append(frame)
        except Exception as exc:
            failures.append({"model": alias, "seed": -1, "error_type": type(exc).__name__, "error": str(exc)})

    supported = SIMPLE_MODELS | set(NEURAL_ALIASES) | set(RAW_ALIASES) | set(FM_ALIASES)
    unsupported = [model for model in requested if model not in supported]
    for model in unsupported:
        failures.append({"model": model, "seed": -1, "error_type": "UnsupportedBaseline", "error": "Unknown Task 2 baseline name."})
    failure_frame = pd.DataFrame(failures)
    failure_frame.to_csv(output / "logs" / "task2_failures.csv", index=False)
    if predictions:
        progress("writing metrics, bootstrap, shortcut audit, and report")
        summarize_task2_baselines(
            pd.concat(predictions, ignore_index=True), output, failure_frame,
            bootstrap_samples=50 if compact else 2000,
            positive_class=positive_class,
        )
    progress(f"finished successful_runs={len(predictions)}, failures={len(failures)}")
    return 2 if strict and failures else 0 if predictions or neural else 2


__all__ = ["run_task2_baseline_suite"]
