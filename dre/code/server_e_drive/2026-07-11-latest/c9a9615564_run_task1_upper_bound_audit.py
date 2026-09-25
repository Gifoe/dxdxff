from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from task1_baselines.upper_bound import UpperBoundAuditError, run_upper_bound_audit


def _path_value(cli: str | None, environment: str, *, required: bool = True) -> str | None:
    value = cli or os.environ.get(environment)
    if required and not value:
        raise UpperBoundAuditError(f"Path is required via --{environment.lower().replace('_', '-')} or {environment}.")
    return value


def _models(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [item.strip().lower() for item in value.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit strict Task 1 count-free OOF performance and diagnostic ranking ceilings."
    )
    parser.add_argument("--task1-output-dir")
    parser.add_argument("--v3-ledger")
    parser.add_argument("--output-dir")
    parser.add_argument("--models", help="Optional comma-separated allow-list of model names.")
    parser.add_argument("--extra-ledger", action="append", default=[], help="Additional patient-channel OOF CSV; repeatable.")
    parser.add_argument("--extra-ledger-glob", action="append", default=[], help="Glob for additional OOF CSVs; repeatable.")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--ensemble-random-candidates", type=int, default=5000)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--clean-labels-csv")
    parser.add_argument("--strict", action="store_true")
    return parser


def _training_command(task1_output: str, v3_ledger: str) -> str:
    feature_cache = os.environ.get("DRE_TASK1_FEATURE_CACHE_PATH", "$DRE_TASK1_FEATURE_CACHE_PATH")
    return (
        "python scripts/task1_baselines/run_task1_baseline_suite.py "
        f'--feature-cache "{feature_cache}" --v3-ledger "{v3_ledger}" '
        f'--output-dir "{task1_output}" --models rbf_svm,random_forest,lightgbm '
        "--seeds 42,52,62 --inner-folds 4"
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started = time.perf_counter()

    def progress(message: str) -> None:
        print(f"[Task1 upper-bound +{time.perf_counter() - started:8.1f}s] {message}", flush=True)

    try:
        task1_output = _path_value(args.task1_output_dir, "DRE_TASK1_OUTPUT_DIR")
        v3_ledger = _path_value(args.v3_ledger, "DRE_TASK1_V3_LEDGER_PATH")
        output = (
            args.output_dir
            or os.environ.get("DRE_TASK1_UPPER_BOUND_OUTPUT_DIR")
            or str(Path(task1_output) / "upper_bound_audit")
        )
        progress(f"input discovery under {Path(task1_output) / 'oof_ledgers'}")
        discovered = sorted((Path(task1_output) / "oof_ledgers").glob("*/seed_*_channel_oof.csv"))
        for path in discovered:
            progress(f"discovered OOF: {path}")
        progress(f"V3 canonical/reference OOF: {v3_ledger}")
        result = run_upper_bound_audit(
            task1_output_dir=task1_output,
            v3_ledger=v3_ledger,
            output_dir=output,
            models=_models(args.models),
            extra_ledgers=args.extra_ledger,
            extra_ledger_globs=args.extra_ledger_glob,
            bootstrap_samples=args.bootstrap_samples,
            ensemble_random_candidates=args.ensemble_random_candidates,
            random_seed=args.random_seed,
            clean_labels_csv=args.clean_labels_csv,
            strict=args.strict,
            repository=REPOSITORY_ROOT,
        )
    except Exception as exc:
        print(f"Task 1 upper-bound audit failed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        if "No legal channel OOF ledgers" in str(exc):
            task1_output = args.task1_output_dir or os.environ.get("DRE_TASK1_OUTPUT_DIR", "$DRE_TASK1_OUTPUT_DIR")
            v3_ledger = args.v3_ledger or os.environ.get("DRE_TASK1_V3_LEDGER_PATH", "$DRE_TASK1_V3_LEDGER_PATH")
            print("No training was started. Suggested feature-baseline OOF command:", file=sys.stderr)
            print(_training_command(task1_output, v3_ledger), file=sys.stderr)
        return 2

    payload = {
        "output_dir": str(result.output_dir.resolve()),
        "accepted_candidates": result.audit["accepted_candidates"],
        "best_strict_count_free_patient_macro_f1": float(result.by_model_seed["strict_count_free_patient_macro_f1"].max()),
        "best_true_k_patient_macro_f1": float(result.by_model_seed["true_k_patient_macro_f1"].max()),
        "best_single_model_oracle_threshold_f1": float(result.by_model_seed["patient_oracle_threshold_macro_f1"].max()),
        "best_oracle_prefix_f1": float(result.by_model_seed["patient_oracle_prefix_macro_f1"].max()),
        "patient_model_library_oracle_f1": result.library_summary["oracle_model_seed_selection_macro_f1"],
        "seed_ensemble_model_library_oracle_f1": result.library_summary["oracle_model_selection_after_seed_ensemble_macro_f1"],
        "best_constrained_ensemble_oracle_f1": result.ensemble_summary["best_simplex_ensemble_oracle_f1"],
        "optimistic_in_sample_global_threshold_ensemble_f1": result.ensemble_summary["optimistic_in_sample_global_threshold_f1"],
        "clean_label_status": result.audit["clean_label_status"],
    }
    progress("audit complete")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
