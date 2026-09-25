from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import time
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import pandas as pd

from outcome_hifos.cache_schema import file_sha256, load_cache_contract
from outcome_hifos.fm.embedding_cache import load_fm_embedding_as_feature_cache, validate_official_fm_embedding_cache
from task1_baselines.cache_io import task1_feature_records
from task1_baselines.feature_aggregation import build_channel_feature_table
from task1_baselines.fold_protocol import fold_manifest_hash, freeze_v3_fold_manifest
from task1_baselines.reporting import summarize_task1
from task1_baselines.raw_preprocessing import RawPreprocessingConfig
from task1_baselines.models.raw_backbones import FrozenEmbeddingEncoder, build_raw_token_encoder
from task1_baselines.token_data import build_task1_embedding_tokens, build_task1_raw_tokens
from task1_baselines.training.feature_runner import run_feature_oof
from task1_baselines.training.token_runner import run_task1_token_oof


FEATURE_MODELS = {"rbf_svm", "random_forest", "lightgbm"}
RAW_MODELS = {"eegnet", "shallowfbcspnet", "deep4net", "eegconformer"}
FM_MODELS = {"brainbert_frozen", "cbramod_frozen"}


def _value(cli: str | None, env: str) -> str:
    value = cli or os.environ.get(env, "")
    if not value:
        raise ValueError(f"Path is required via CLI or {env}.")
    return value


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _load_existing_task1_oof(output: Path) -> list[pd.DataFrame]:
    existing: list[pd.DataFrame] = []
    root = output / "oof_ledgers"
    if not root.exists():
        return existing
    for path in sorted(root.glob("*/seed_*_channel_oof.csv")):
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        required = {"model", "seed", "subject_id", "channel_name"}
        if required.issubset(frame.columns) and not frame.empty:
            existing.append(frame)
    return existing


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the independent Task 1 old-90 baseline suite.")
    parser.add_argument("--feature-cache")
    parser.add_argument("--raw-cache")
    parser.add_argument("--brainbert-embedding-cache")
    parser.add_argument("--cbramod-embedding-cache")
    parser.add_argument("--v3-ledger")
    parser.add_argument("--output-dir")
    parser.add_argument("--models", default="rbf_svm,random_forest,lightgbm")
    parser.add_argument("--seeds", default="42,52,62")
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--compact", action="store_true")
    parser.add_argument("--fixed-baseline", action="store_true", help="Use one fixed configuration per model; skip inner hyperparameter tuning.")
    parser.add_argument("--strict", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    requested_models = [value.lower() for value in _csv(args.models)]
    started_at = time.perf_counter()

    def progress(message: str) -> None:
        print(f"[Task1 +{time.perf_counter() - started_at:8.1f}s] {message}", flush=True)

    progress(f"starting models={requested_models}, seeds={_csv(args.seeds)}")
    cache_path = Path(_value(args.feature_cache, "DRE_TASK1_FEATURE_CACHE_PATH"))
    ledger_path = Path(_value(args.v3_ledger, "DRE_TASK1_V3_LEDGER_PATH"))
    output = Path(_value(args.output_dir, "DRE_TASK1_OUTPUT_DIR"))
    for name in ("audit", "configs", "provenance", "checkpoints", "embedding_cache", "oof_ledgers", "metrics", "logs", "comparison", "reports"):
        (output / name).mkdir(parents=True, exist_ok=True)
    progress(f"loading V3 ledger: {ledger_path}")
    v3 = pd.read_csv(ledger_path)
    folds = freeze_v3_fold_manifest(v3)
    progress(f"V3 fixed folds ready: patients={len(folds)}, folds={sorted(folds['outer_fold'].unique().tolist())}")
    folds.to_csv(output / "configs" / "task1_fixed_old90_fold_manifest.csv", index=False)
    progress(f"loading feature cache: {cache_path}")
    cache = load_cache_contract(cache_path)
    progress(f"feature cache ready: run_records={len(cache.run_records)}, patients={len(cache.patient_index)}")
    features = None
    if FEATURE_MODELS & set(requested_models):
        records = task1_feature_records(cache, set(folds["subject_id"]))
        features, feature_manifest = build_channel_feature_table(records)
        (output / "configs" / "task1_channel_feature_manifest.json").write_text(json.dumps(feature_manifest, indent=2, sort_keys=True), encoding="utf-8")
    cohort_hash = hashlib.sha256("\n".join(sorted(folds["subject_id"])).encode("utf-8")).hexdigest()
    provenance = {
        "task": "task1_ez_nez_channel_localization",
        "label_semantics": {"NEZ": 1, "EZ": 0},
        "feature_cache_path": str(cache_path.resolve()),
        "feature_cache_sha256": file_sha256(cache_path),
        "v3_ledger_path": str(ledger_path.resolve()),
        "v3_ledger_sha256": file_sha256(ledger_path),
        "cohort_hash": cohort_hash,
        "fold_ledger_hash": fold_manifest_hash(folds),
    }
    config_hash = hashlib.sha256(json.dumps({"models": requested_models, "seeds": _csv(args.seeds), "inner_folds": args.inner_folds, "compact": args.compact}, sort_keys=True).encode("utf-8")).hexdigest()
    (output / "provenance" / "task1_protocol.json").write_text(json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8")
    predictions = []
    failures = []
    raw_tokens = None
    raw_cache_path = None
    raw_dependency_error = None
    if RAW_MODELS & set(requested_models) and importlib.util.find_spec("braindecode") is None:
        raw_dependency_error = "braindecode is required for raw supervised baselines; install it in the active Python environment."
        progress(f"RAW MODELS UNAVAILABLE: {raw_dependency_error}")
    fm_tokens: dict[str, object] = {}
    for model in requested_models:
        for seed in [int(value) for value in _csv(args.seeds)]:
            try:
                source_hash = provenance["feature_cache_sha256"]
                if model in FEATURE_MODELS:
                    progress(f"START model={model} seed={seed}: feature aggregation/training")
                    assert features is not None
                    result = run_feature_oof(
                        features, folds, model_name=model, seed=seed, inner_folds=args.inner_folds,
                        compact=args.compact, fixed_baseline=args.fixed_baseline,
                    )
                    selected = result.selected_parameters
                elif model in RAW_MODELS:
                    if raw_dependency_error is not None:
                        raise RuntimeError(raw_dependency_error)
                    if raw_tokens is None:
                        progress("loading raw cache and expanding channel-window tokens; this can take a while")
                        raw_cache_path = Path(_value(args.raw_cache, "DRE_TASK1_RAW_CACHE_PATH"))
                        raw_cache = load_cache_contract(raw_cache_path)
                        raw_config = RawPreprocessingConfig()
                        progress(f"raw cache ready: run_records={len(raw_cache.run_records)}, patients={len(raw_cache.patient_index)}; preprocessing tokens")
                        raw_tokens = build_task1_raw_tokens(raw_cache, set(folds["subject_id"]), raw_config)
                        progress(f"raw tokens ready: tokens={len(raw_tokens.rows)}, waveform_length={raw_tokens.values.shape[-1]}")
                        (output / "audit" / "task1_raw_preprocessing.json").write_text(json.dumps(list(raw_tokens.preprocessing_audit), indent=2, default=str), encoding="utf-8")
                    progress(f"START model={model} seed={seed}: raw training")
                    source_hash = file_sha256(raw_cache_path)
                    epochs = int(args.epochs or (2 if args.compact else 40 if model == "eegconformer" else 35 if model == "deep4net" else 30))
                    batch = 32 if args.compact else 64 if model == "eegconformer" else 128 if model == "deep4net" else 256
                    result = run_task1_token_oof(
                        raw_tokens, folds, model_name=model, seed=seed,
                        encoder_factory=lambda name=model, n_times=int(raw_tokens.values.shape[-1]), sfreq=float(raw_config.target_sfreq): build_raw_token_encoder(
                            name, n_times=n_times, sfreq=sfreq, embedding_dim=128
                        ),
                        frozen_backbone=False, inner_folds=args.inner_folds, max_epochs=epochs, batch_size=batch, device=args.device,
                        fixed_baseline=args.fixed_baseline,
                    )
                    selected = result.training_audit
                elif model in FM_MODELS:
                    if model not in fm_tokens:
                        cli_path = args.brainbert_embedding_cache if model == "brainbert_frozen" else args.cbramod_embedding_cache
                        env_name = "DRE_TASK1_BRAINBERT_EMBEDDING_CACHE" if model == "brainbert_frozen" else "DRE_TASK1_CBRAMOD_EMBEDDING_CACHE"
                        fm_path = Path(_value(cli_path, env_name))
                        progress(f"loading {model} embedding cache: {fm_path}")
                        fm_name = "brainbert" if model == "brainbert_frozen" else "cbramod"
                        fm_audit = validate_official_fm_embedding_cache(fm_path, fm_name)
                        (output / "provenance" / f"{fm_name}_official_checkpoint.json").write_text(
                            json.dumps(fm_audit, indent=2, sort_keys=True), encoding="utf-8"
                        )
                        fm_cache = load_fm_embedding_as_feature_cache(fm_path)
                        fm_tokens[model] = (build_task1_embedding_tokens(fm_cache, set(folds["subject_id"])), fm_path)
                        progress(f"{model} embeddings ready: tokens={len(fm_tokens[model][0].rows)}")
                    token_data, fm_path = fm_tokens[model]
                    progress(f"START model={model} seed={seed}: frozen embedding head training")
                    source_hash = file_sha256(fm_path / "all_windows_embeddings.pkl" if fm_path.is_dir() else fm_path)
                    dimension = int(token_data.values.shape[1])
                    result = run_task1_token_oof(
                        token_data, folds, model_name=model, seed=seed,
                        encoder_factory=lambda dim=dimension: FrozenEmbeddingEncoder(dim), frozen_backbone=True,
                        inner_folds=args.inner_folds, max_epochs=int(args.epochs or (2 if args.compact else 30)), batch_size=256, device=args.device,
                        fixed_baseline=args.fixed_baseline,
                    )
                    selected = result.training_audit
                else:
                    raise ValueError(f"Unknown Task 1 model {model!r}.")
                result.oof["feature_cache_hash"] = provenance["feature_cache_sha256"]
                result.oof["input_cache_hash"] = source_hash
                result.oof["fold_ledger_hash"] = provenance["fold_ledger_hash"]
                result.oof["cohort_hash"] = provenance["cohort_hash"]
                result.oof["config_hash"] = config_hash
                result.oof["checkpoint_path"] = ""
                result.oof["cohort_name"] = "task1_old90_frozen_v3"
                model_dir = output / "oof_ledgers" / model
                model_dir.mkdir(parents=True, exist_ok=True)
                result.oof.to_csv(model_dir / f"seed_{seed}_channel_oof.csv", index=False)
                selected.to_csv(output / "configs" / f"{model}_seed_{seed}_selected_parameters.csv", index=False)
                predictions.append(result.oof)
                progress(f"DONE model={model} seed={seed}: oof_rows={len(result.oof)}")
            except Exception as exc:
                progress(f"FAIL model={model} seed={seed}: {type(exc).__name__}: {exc}")
                failures.append({"model": model, "seed": seed, "error_type": type(exc).__name__, "error": str(exc)})
    # Keep successful OOF ledgers from earlier invocations so failed models can be
    # repaired incrementally without retraining completed models.
    existing_oof = _load_existing_task1_oof(output)
    current_oof = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    all_oof_parts = existing_oof + ([current_oof] if not current_oof.empty else [])
    all_oof = pd.concat(all_oof_parts, ignore_index=True) if all_oof_parts else pd.DataFrame()
    if not all_oof.empty:
        all_oof = all_oof.drop_duplicates(["model", "seed", "subject_id", "channel_name"], keep="last")
    failure_frame = pd.DataFrame(failures)
    previous_failure_path = output / "logs" / "task1_failures.csv"
    if previous_failure_path.exists():
        previous_failures = pd.read_csv(previous_failure_path)
        if not previous_failures.empty:
            failure_frame = pd.concat([previous_failures, failure_frame], ignore_index=True)
            if {"model", "seed"}.issubset(failure_frame.columns):
                failure_frame = failure_frame.drop_duplicates(["model", "seed"], keep="last")
    if not all_oof.empty and {"model", "seed"}.issubset(failure_frame.columns):
        completed = all_oof[["model", "seed"]].drop_duplicates()
        failure_frame = failure_frame.merge(completed.assign(_completed=True), on=["model", "seed"], how="left")
        failure_frame = failure_frame[failure_frame["_completed"].isna()].drop(columns=["_completed"])
    failure_frame.to_csv(output / "logs" / "task1_failures.csv", index=False)
    if not all_oof.empty:
        summarize_task1(
            all_oof, output, failures=failure_frame, v3_reference=v3,
            bootstrap_samples=50 if args.compact else 2000,
        )
    if failures and args.strict:
        progress(f"finished with failures={len(failures)}")
        return 2
    progress(f"finished successful_runs={len(predictions)}, failures={len(failures)}")
    return 0 if predictions else 2


if __name__ == "__main__":
    raise SystemExit(main())
