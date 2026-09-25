"""Expose fixed 80-patient protocol discovery without duplicating protocol logic."""
from __future__ import annotations
import argparse
from .core import load_subjects_and_folds_by_seed

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--protocol-root", required=True); parser.add_argument("--seeds", type=int, nargs="+", default=[42, 52, 62]); args = parser.parse_args()
    subjects, fold_maps, path = load_subjects_and_folds_by_seed(args.protocol_root, args.seeds)
    print({"manifest": str(path), "n_subjects": len(subjects), "fold_sizes": {seed: {fold: len(value) for fold, value in folds.items()} for seed, folds in fold_maps.items()}})
