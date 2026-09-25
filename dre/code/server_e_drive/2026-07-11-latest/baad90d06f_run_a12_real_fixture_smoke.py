"""Run all cache-dependent A12 variants on an anonymized real-structure fixture."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from a12_vcsn.config import A12Config
from a12_vcsn.io import load_window_feature_store
from a12_vcsn.schemas import build_canonical_ledger
from a12_vcsn.suite import run_a12_suite


def fixture_ledger(store) -> pd.DataFrame:
    rows = []
    for subject_index, subject_id in enumerate(store.subjects()):
        channels = []
        for run in store.runs(subject_id):
            for channel in run.channel_names:
                if channel not in channels:
                    channels.append(channel)
        for channel_index, channel in enumerate(channels):
            true_ez = int(channel_index < 2)
            predicted_ez = int(channel_index in {0, 2}) if len(channels) > 2 else true_ez
            rows.append({"fold_idx": subject_index % 2 + 1, "subject_id": subject_id,
                         "center": "real-cache-fixture", "channel_name": channel,
                         "true_ez": true_ez, "true_nez": 1 - true_ez,
                         "score_ez_probability": float(1. - (channel_index + 1) / (len(channels) + 1)),
                         "rank_ez_desc": channel_index + 1, "predicted_ez": predicted_ez})
    ledger, _ = build_canonical_ledger(pd.DataFrame(rows), strict=True)
    ledger["optional_hnc_score"] = 1. - ledger["old_v3_score_ez"]
    ledger["hnc_score_semantics"] = "p_nez"
    ledger["hnc_eject_priority"] = ledger["optional_hnc_score"]
    ledger["hnc_add_priority"] = -ledger["optional_hnc_score"]
    return ledger


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--variants", default="A12-V2,A12-V3,A12-V4,A12-V6,A12-V7,A12-V8,A12-V9,A12-V10")
    parser.add_argument("--strict", action="store_true", default=False)
    args = parser.parse_args(argv)
    features, trajectories, cache_audit = load_window_feature_store(args.cache_path, strict=False, invalid_record_policy="drop")
    ledger = fixture_ledger(trajectories)
    variants = tuple(item.strip() for item in args.variants.split(",") if item.strip())
    config = A12Config(variants=variants, seeds=(42,), inner_folds=2, max_swaps=1, strict=args.strict, device="cpu", smoke_mode=True, batch_size=16, max_epochs=1, patience=1, hidden_dim=4, catboost_max_iterations=2, max_eject_candidates=2, max_add_candidates=3, max_pairs_per_patient=6, anchor_min_channels=1)
    result = run_a12_suite(ledger, output_dir=args.output_dir, config=config, synthetic=True,
                           run_classification="real-cache filtered fixture", cache_audit=cache_audit,
                           hnc_available=True, channel_features=features, trajectories=trajectories)
    details = {name: {"status": value["status"], "reason": value.get("reason")} for name, value in result["variants"].items()}
    print(json.dumps({"output_dir": result["output_dir"], "n_subjects": int(ledger.subject_id.nunique()), "variants": details}, ensure_ascii=False))
    if any(value["status"] == "failed" for value in details.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
