"""Run lightweight frozen-head baselines on frozen-FM window embeddings."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.fm_baselines.evaluation import evaluate_embeddings


def _split_methods(text: str) -> list[str]:
    return [item.strip() for item in str(text).split(",") if item.strip()]


REQUIRED_METADATA_COLUMNS = {
    "fold_idx",
    "split_role",
    "subject_id",
    "center",
    "channel_name",
    "run_id",
    "label_ez",
    "fm_model",
}


def _validate_and_align_metadata(embeddings: np.ndarray, metadata: pd.DataFrame, *, n_splits: int) -> pd.DataFrame:
    if embeddings.ndim != 2:
        raise ValueError("embedding array must be 2D [n_rows,n_features].")
    if len(metadata) != embeddings.shape[0]:
        raise ValueError(f"metadata row count {len(metadata)} does not match embedding rows {embeddings.shape[0]}.")
    missing = sorted(REQUIRED_METADATA_COLUMNS - set(metadata.columns))
    if missing:
        raise ValueError(f"metadata missing required columns: {missing}")
    if "embedding_row_idx" in metadata.columns:
        idx = metadata["embedding_row_idx"].astype(int).to_numpy()
        expected = np.arange(embeddings.shape[0])
        if not np.array_equal(idx, expected):
            raise ValueError(
                "metadata embedding_row_idx must match current embedding row order 0..N-1; "
                "refusing to reorder metadata without embeddings"
            )
    roles = set(metadata["split_role"].astype(str))
    if not roles.issubset({"train", "test"}):
        raise ValueError(f"metadata split_role must contain only train/test, got {sorted(roles)}")
    leaks = []
    for fold_idx, group in metadata.groupby("fold_idx", sort=True):
        train_subjects = set(group.loc[group["split_role"].eq("train"), "subject_id"].astype(str))
        test_subjects = set(group.loc[group["split_role"].eq("test"), "subject_id"].astype(str))
        overlap = sorted(train_subjects & test_subjects)
        if overlap:
            leaks.append({"fold_idx": int(fold_idx), "subjects": overlap})
    if leaks:
        raise ValueError(f"metadata has train/test subject leakage: {leaks}")
    expected_folds = list(range(1, int(n_splits) + 1))
    observed_folds = sorted(int(fold) for fold in metadata["fold_idx"].unique().tolist())
    missing_folds = [fold for fold in expected_folds if fold not in observed_folds]
    if missing_folds:
        raise RuntimeError(f"metadata missing expected folds: {missing_folds}")
    return metadata


def run_fm_frozen_head_baseline(args: argparse.Namespace) -> Dict:
    if str(args.positive_label).lower() != "ez":
        raise ValueError("FM frozen-head baseline requires positive_label='ez'.")
    embeddings = np.load(args.embedding_path)
    metadata = pd.read_csv(args.metadata_path)
    metadata = _validate_and_align_metadata(embeddings, metadata, n_splits=int(args.n_splits))
    fm_model = str(metadata["fm_model"].iloc[0]) if "fm_model" in metadata.columns and not metadata.empty else "unknown_fm"
    return evaluate_embeddings(
        embeddings,
        metadata,
        output_dir=args.output_dir,
        embedding_path=args.embedding_path,
        metadata_path=args.metadata_path,
        fm_model=fm_model,
        head_methods=_split_methods(args.head_methods),
        n_splits=int(args.n_splits),
        split_strategy=str(args.split_strategy),
        random_seed=int(args.random_seed),
        val_ratio=float(args.val_ratio),
        allow_incomplete_methods=bool(args.allow_incomplete_methods),
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding_path", required=True)
    parser.add_argument("--metadata_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split_strategy", default="5fold")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--positive_label", choices=["ez"], default="ez")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--head_methods", default="logistic_l2,linear_svm")
    parser.add_argument("--allow_incomplete_methods", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_fm_frozen_head_baseline(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
