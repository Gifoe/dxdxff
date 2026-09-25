from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from outcome_hifos.baselines.suite import run_task2_baseline_suite


def _path(value: str | None, env: str) -> str:
    result = value or os.environ.get(env, "")
    if not result:
        raise ValueError(f"Path is required via CLI or {env}.")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run authoritative Task 2 baselines on the primary cohort.")
    parser.add_argument("--feature-cache")
    parser.add_argument("--raw-cache")
    parser.add_argument("--brainbert-embedding-cache")
    parser.add_argument("--cbramod-embedding-cache")
    parser.add_argument("--old90-manifest")
    parser.add_argument("--success-manifest", help="Audited success-patient manifest; overrides --old90-manifest.")
    parser.add_argument("--cohort-name", default="task2_primary")
    parser.add_argument("--positive-class", choices=["success", "failure"], default="success")
    parser.add_argument("--output-dir")
    parser.add_argument("--models", default="majority,elasticnet,rbf_svm,random_forest,lightgbm,h2_hier_pool,h3_attention_mil")
    parser.add_argument("--seeds", default="42,52,62")
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--fixed-baseline", action="store_true", help="Use fixed parameters and a fixed 0.5 threshold; no inner CV.")
    parser.add_argument("--run-loco", action="store_true", help="Additionally run leave-one-center-out analysis.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-outer-folds", type=int, default=0)
    parser.add_argument("--compact", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)
    return run_task2_baseline_suite(
        feature_cache_path=_path(args.feature_cache, "DRE_TASK2_FEATURE_CACHE_PATH"),
        raw_cache_path=args.raw_cache or os.environ.get("DRE_TASK2_RAW_CACHE_PATH"),
        brainbert_embedding_cache_path=args.brainbert_embedding_cache or os.environ.get("DRE_TASK2_BRAINBERT_EMBEDDING_CACHE"),
        cbramod_embedding_cache_path=args.cbramod_embedding_cache or os.environ.get("DRE_TASK2_CBRAMOD_EMBEDDING_CACHE"),
        old90_manifest_path=args.success_manifest or _path(args.old90_manifest, "DRE_TASK1_OLD90_MANIFEST"),
        cohort_name=args.cohort_name,
        positive_class=args.positive_class,
        output_dir=_path(args.output_dir, "DRE_TASK2_OUTPUT_DIR"),
        models=[value.strip() for value in args.models.split(",") if value.strip()],
        seeds=[int(value) for value in args.seeds.split(",") if value.strip()],
        inner_folds=args.inner_folds,
        fixed_baseline=args.fixed_baseline,
        run_loco=args.run_loco,
        compact=args.compact,
        strict=args.strict,
        device=args.device,
        epochs=args.epochs,
        max_outer_folds=args.max_outer_folds,
    )


if __name__ == "__main__":
    raise SystemExit(main())
