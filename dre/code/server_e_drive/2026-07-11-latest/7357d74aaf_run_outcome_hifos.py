from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

import pandas as pd

from outcome_hifos.cache_audit import audit_caches
from outcome_hifos.cache_schema import load_cache_contract
from outcome_hifos.code_audit import write_code_audit
from outcome_hifos.config import resolve_config
from outcome_hifos.fm.embedding_cache import extract_embedding_cache
from outcome_hifos.fm.registry import build_fm_adapter
from outcome_hifos.reports.summarize import summarize_experiment
from outcome_hifos.reports.shortcut import run_shortcut_audit
from outcome_hifos.training.experiment_runner import run_loco_experiment, run_outcome_experiment
from outcome_hifos.baselines.suite import run_task2_baseline_suite


def _csv_list(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _int_list(value: str) -> list[int]:
    return [int(item) for item in _csv_list(value)]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run label-blind HiFOS-PACT patient outcome experiments.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit")
    audit.add_argument("--config")
    audit.add_argument("--feature-cache-path")
    audit.add_argument("--raw-cache-path")
    audit.add_argument("--output-dir")
    audit.add_argument("--fail-on-conflict", action=argparse.BooleanOptionalAction, default=False)

    run = subparsers.add_parser("run")
    run.add_argument("--config", required=True)
    run.add_argument("--protocol", choices=["screening", "final_nested"], required=True)
    run.add_argument("--variants", default="all")
    run.add_argument("--seeds", default="42,52,62")
    run.add_argument("--max-outer-folds", type=int, default=0)
    run.add_argument("--max-inner-folds", type=int, default=0)
    run.add_argument("--max-patients", type=int, default=0)
    run.add_argument("--epochs", type=int)
    run.add_argument("--device", default="auto")
    run.add_argument("--num-workers", type=int)
    run.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    run.add_argument("--early-stop-metric", choices=["validation_loss", "auroc", "auprc", "macro_f1_at_0_5"])
    run.add_argument("--candidate-profiles", help="Comma-separated configured candidate profile names for final_nested.")
    run.add_argument("--save-diagnostics", action=argparse.BooleanOptionalAction, default=None)
    run.add_argument("--dry-run", action="store_true", help="Validate cache, cohort, folds, and provenance without training.")
    run.add_argument("--resume", action="store_true")
    run.add_argument("--overwrite", action="store_true")

    embed = subparsers.add_parser("embed-fm")
    embed.add_argument("--config", required=True)
    embed.add_argument("--model", required=True, choices=["biot", "cbramod", "labram", "brainbert"])
    embed.add_argument("--checkpoint-path", required=True)
    embed.add_argument("--external-repo-path")
    embed.add_argument("--output-dir", required=True)
    embed.add_argument("--device", default="auto")
    embed.add_argument("--streaming-embedding-format", choices=["chunked_npy"], default="chunked_npy")
    embed.add_argument(
        "--raw-time-origin",
        choices=["unknown", "seizure_onset_at_raw_target_midpoint"],
        default=None,
    )

    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--input-dir", required=True)
    summarize.add_argument("--output-dir")

    loco = subparsers.add_parser("loco")
    loco.add_argument("--config", required=True)
    loco.add_argument("--variants", default="H6_ANCHORED_UOT_DESC,H8_RECURRENCE")
    loco.add_argument("--seeds", default="42")
    loco.add_argument("--device", default="auto")

    shortcut = subparsers.add_parser("shortcut-audit")
    shortcut.add_argument("--input-dir", required=True)

    baselines = subparsers.add_parser("run-baselines")
    baselines.add_argument("--config", required=True)
    baselines.add_argument("--models", default="majority,elasticnet,rbf_svm,random_forest,lightgbm,h2_hier_pool,h3_attention_mil")
    baselines.add_argument("--seeds", default="42,52,62")
    baselines.add_argument("--device", default="auto")
    baselines.add_argument("--epochs", type=int)
    baselines.add_argument("--max-outer-folds", type=int, default=0)
    baselines.add_argument("--compact", action="store_true")
    baselines.add_argument("--strict", action="store_true")
    return parser


def _run_audit(args: argparse.Namespace) -> int:
    config = resolve_config(
        args.config,
        {
            "feature_cache_path": args.feature_cache_path,
            "raw_cache_path": args.raw_cache_path,
            "output_dir": args.output_dir,
        },
        os.environ,
        require_paths=("feature_cache_path", "raw_cache_path", "output_dir"),
    )
    paths = config["paths"]
    output = Path(paths["output_dir"])
    summary = audit_caches(
        paths["feature_cache_path"],
        paths["raw_cache_path"],
        output,
        fail_on_conflict=bool(args.fail_on_conflict),
    )
    write_code_audit(
        output / "code_audit.md",
        feature_cache_path=paths["feature_cache_path"],
        raw_cache_path=paths["raw_cache_path"],
    )
    print(json.dumps({"patient_count": summary.patient_count, "outcome_counts": summary.outcome_counts}, ensure_ascii=False))
    return 0


def _run_experiment(args: argparse.Namespace) -> int:
    config = resolve_config(args.config, {}, os.environ, require_paths=("feature_cache_path", "output_dir"))
    if args.epochs is not None:
        config.setdefault("training", {})["epochs"] = int(args.epochs)
    if args.num_workers is not None:
        config.setdefault("training", {})["num_workers"] = int(args.num_workers)
    if args.amp is not None:
        config.setdefault("training", {})["amp"] = bool(args.amp)
    if args.early_stop_metric is not None:
        config.setdefault("training", {})["early_stop_metric"] = str(args.early_stop_metric)
    if args.save_diagnostics is not None:
        config.setdefault("training", {})["save_diagnostics"] = bool(args.save_diagnostics)
    if args.candidate_profiles:
        requested_profiles = _csv_list(args.candidate_profiles)
        configured_profiles = dict(config.get("candidate_profiles") or {})
        missing_profiles = sorted(set(requested_profiles) - set(configured_profiles))
        if missing_profiles:
            raise ValueError(f"Unknown candidate profile(s): {missing_profiles}")
        config["candidate_profiles"] = {name: configured_profiles[name] for name in requested_profiles}
    if args.max_inner_folds > 0:
        config.setdefault("protocol", {})["inner_folds"] = int(args.max_inner_folds)
    variants = _csv_list(args.variants)
    if variants == ["all"]:
        variants = [
            "H1_SUMMARY_ML",
            "H2_HIER_POOL",
            "H3_ATTENTION_MIL",
            "H4_MULTI_CORE",
            "H5_ANCHORED_CORE",
            "H6_ANCHORED_UOT_DESC",
            "H7_TRANSPORT_GRAPH",
            "H8_RECURRENCE",
            "H9_FM_RECURRENCE",
        ]
    summary = run_outcome_experiment(
        config,
        protocol=args.protocol,
        variants=variants,
        seeds=_int_list(args.seeds),
        max_outer_folds=int(args.max_outer_folds),
        max_patients=int(args.max_patients),
        device=args.device,
        resume=bool(args.resume),
        overwrite=bool(args.overwrite),
        dry_run=bool(args.dry_run),
    )
    if args.dry_run:
        return 0
    if summary.predictions.empty:
        print(summary.failures.to_string(index=False))
        return 2
    if not summary.failures.empty:
        print(summary.failures.to_string(index=False))
        return 3
    return 0


def _run_embed(args: argparse.Namespace) -> int:
    config = resolve_config(args.config, {}, os.environ, require_paths=("raw_cache_path",))
    fm_config = dict(config.get("fm") or {})
    adapter = build_fm_adapter(args.model, fm_config)
    device = "cuda" if args.device == "auto" and __import__("torch").cuda.is_available() else "cpu" if args.device == "auto" else args.device
    adapter.load(checkpoint_path=args.checkpoint_path, external_repo_path=args.external_repo_path, device=device)
    raw_time_origin = args.raw_time_origin or str(fm_config.get("raw_time_origin", "unknown"))
    extract_embedding_cache(
        load_cache_contract(config["paths"]["raw_cache_path"]),
        adapter,
        args.output_dir,
        raw_window_key=str(fm_config.get("raw_window_key", "raw_window_waveforms")),
        batch_size=int(fm_config.get("batch_size", 64)),
        raw_time_origin=raw_time_origin,
    )
    return 0


def _run_summarize(args: argparse.Namespace) -> int:
    input_dir = Path(args.input_dir)
    prediction_path = input_dir / "outcome_oof_predictions.csv"
    if not prediction_path.exists():
        candidates = list(input_dir.rglob("outer_test_predictions.csv"))
        if not candidates:
            raise FileNotFoundError(f"No OOF or outer-test predictions found under {input_dir}")
        predictions = pd.concat([pd.read_csv(path) for path in candidates], ignore_index=True)
    else:
        predictions = pd.read_csv(prediction_path)
    failure_path = input_dir / "outcome_training_failures.csv"
    failures = pd.read_csv(failure_path) if failure_path.exists() else pd.DataFrame()
    output = Path(args.output_dir) if args.output_dir else input_dir
    manifest_path = input_dir / "outcome_run_manifest.json"
    if manifest_path.exists():
        run_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(run_manifest, dict):
            raise TypeError(f"Run manifest must contain a JSON object: {manifest_path}")
    else:
        run_manifest = {"protocol": "summarize", "random_seed": 42}
    summarize_experiment(predictions, failures, output, run_manifest=run_manifest)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "audit":
        return _run_audit(args)
    if args.command == "run":
        return _run_experiment(args)
    if args.command == "embed-fm":
        return _run_embed(args)
    if args.command == "summarize":
        return _run_summarize(args)
    if args.command == "loco":
        config = resolve_config(args.config, {}, os.environ, require_paths=("feature_cache_path", "output_dir"))
        result = run_loco_experiment(config, variants=_csv_list(args.variants), seeds=_int_list(args.seeds), device=args.device)
        return 0 if not result.predictions.empty else 2
    if args.command == "shortcut-audit":
        run_shortcut_audit(args.input_dir)
        return 0
    if args.command == "run-baselines":
        config = resolve_config(
            args.config,
            {},
            os.environ,
            require_paths=("feature_cache_path", "old90_manifest_path", "output_dir"),
        )
        paths = config["paths"]
        return run_task2_baseline_suite(
            feature_cache_path=paths["feature_cache_path"],
            raw_cache_path=paths.get("raw_cache_path"),
            brainbert_embedding_cache_path=paths.get("brainbert_embedding_cache_path"),
            cbramod_embedding_cache_path=paths.get("cbramod_embedding_cache_path"),
            old90_manifest_path=paths["old90_manifest_path"],
            output_dir=paths["output_dir"],
            models=_csv_list(args.models),
            seeds=_int_list(args.seeds),
            inner_folds=int((config.get("protocol") or {}).get("inner_folds", 4)),
            compact=bool(args.compact),
            strict=bool(args.strict),
            device=args.device,
            epochs=args.epochs,
            max_outer_folds=int(args.max_outer_folds),
        )
    raise RuntimeError(f"Unsupported command {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
