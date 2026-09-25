#!/usr/bin/env python3
"""Run strict post-hoc Task 1 analyses on frozen PRQ/BCR/CDEL OOF ledgers."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from task1_aaai.oof_analysis import analyze_oof_ledgers


def _read_specifications(values: list[str], *, kind: str) -> pd.DataFrame:
    frames = []
    for value in values:
        try:
            seed_text, path_text = value.split("=", 1)
            seed = int(seed_text)
        except ValueError as exc:
            raise ValueError(f"{kind} must use SEED=PATH, got {value!r}") from exc
        frame = pd.read_csv(path_text)
        frame["seed"] = seed if "seed" not in frame else pd.to_numeric(frame["seed"], errors="raise").astype(int)
        if not frame.seed.eq(seed).all():
            raise ValueError(f"{kind} seed mismatch in {path_text}")
        frames.append(frame)
    if not frames:
        raise ValueError(f"At least one --seed-{kind} is required")
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-ledger", action="append", default=[], metavar="SEED=PATH")
    parser.add_argument("--seed-thresholds", action="append", default=[], metavar="SEED=PATH")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260724)
    args = parser.parse_args()
    ledger = _read_specifications(args.seed_ledger, kind="ledger")
    thresholds = _read_specifications(args.seed_thresholds, kind="thresholds")
    print(analyze_oof_ledgers(ledger=ledger, thresholds=thresholds, output_dir=args.output_dir, bootstrap_samples=args.bootstrap_samples, bootstrap_seed=args.bootstrap_seed))


if __name__ == "__main__":
    main()
