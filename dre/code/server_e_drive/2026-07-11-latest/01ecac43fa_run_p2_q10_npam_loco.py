from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from neuroez_c.task2.data import load_cache
from neuroez_c.task2.outcomes import load_outcome_table


def main() -> int:
    parser = argparse.ArgumentParser(description="Leave-one-center-out P2-Q10-NPAM evaluation")
    parser.add_argument("--profile", required=True)
    parser.add_argument("--outcome_table", default=os.getenv("DRE_TASK2_OUTCOME_TABLE", "cache://patient_index"))
    parser.add_argument("--feature_cache", default=os.getenv("DRE_TASK1_FEATURE_CACHE_PATH"), required=os.getenv("DRE_TASK1_FEATURE_CACHE_PATH") is None)
    parser.add_argument("--raw_cache", default=os.getenv("DRE_TASK1_RAW_CACHE_PATH"))
    parser.add_argument("--graph_cache", default=os.getenv("DRE_TASK2_GRAPH_CACHE"))
    parser.add_argument("--p2_checkpoint_root", default=os.getenv("DRE_TASK1_P2_CHECKPOINT_ROOT"), required=os.getenv("DRE_TASK1_P2_CHECKPOINT_ROOT") is None)
    parser.add_argument("--p2_training_manifest", default=os.getenv("DRE_TASK1_P2_TRAINING_MANIFEST"))
    parser.add_argument("--p2_runtime_root", default=os.getenv("DRE_TASK1_P2_RUNTIME_ROOT"))
    parser.add_argument("--p2_config")
    parser.add_argument("--protocol", choices=("quick",), default="quick")
    parser.add_argument("--exclusion_manifest", default=os.getenv("DRE_TASK2_EXCLUSION_MANIFEST", str(PROJECT / "configs" / "data_exclusions.csv")))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    outcomes, _ = load_outcome_table(args.outcome_table, cache=load_cache(args.feature_cache))
    outcomes = outcomes[outcomes["outcome_group"].isin(["success", "failure"])]
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    summaries = []
    runner = Path(__file__).with_name("run_p2_q10_npam.py")
    for center in sorted(outcomes["center"].unique()):
        manifest = outcomes[["patient_key", "center"]].copy()
        manifest["outer_fold"] = (manifest["center"] != center).astype(int) + 1
        manifest_path = root / f"loco_{center}_fold_manifest.csv"
        manifest.to_csv(manifest_path, index=False)
        center_output = root / f"heldout_{center}"
        base_p2_root = Path(args.p2_checkpoint_root)
        center_p2_root = base_p2_root / f"heldout_{center}"
        center_manifest = center_p2_root / "outer_fold_ledger.csv"
        selected_p2_root = center_p2_root if center_p2_root.exists() else base_p2_root
        selected_manifest = center_manifest if center_manifest.exists() else (Path(args.p2_training_manifest) if args.p2_training_manifest else None)
        command = [sys.executable, str(runner), "--profile", args.profile, "--outcome_table", args.outcome_table, "--feature_cache", args.feature_cache, "--p2_checkpoint_root", str(selected_p2_root), "--fold_manifest", str(manifest_path), "--exclusion_manifest", args.exclusion_manifest, "--protocol", "quick", "--max_outer_folds", "1", "--seed", str(args.seed), "--device", args.device, "--output_dir", str(center_output)]
        if selected_manifest is not None:
            command.extend(("--p2_training_manifest", str(selected_manifest)))
        if args.raw_cache:
            command.extend(("--raw_cache", args.raw_cache))
        if args.graph_cache:
            command.extend(("--graph_cache", args.graph_cache))
        if args.p2_runtime_root:
            command.extend(("--p2_runtime_root", args.p2_runtime_root))
        if args.p2_config:
            command.extend(("--p2_config", args.p2_config))
        subprocess.run(command, check=True)
        metrics = pd.read_csv(center_output / "summary_metrics.csv")
        row = metrics[metrics["scope"] == "pooled_oof"].iloc[0].to_dict()
        summaries.append({"heldout_center": center, "p2_checkpoint_root": str(selected_p2_root.resolve()), "p2_training_manifest": str(selected_manifest.resolve()) if selected_manifest else "", **row})
    pd.DataFrame(summaries).to_csv(root / "loco_summary.csv", index=False)
    (root / "loco_protocol_audit.json").write_text(json.dumps({"center_as_input": False, "heldout_centers": sorted(outcomes["center"].unique()), "normalization_source": "train_centers_only", "decision_threshold": 0.5, "threshold_source": "fixed_predefined_0_5", "inner_cv_used": False, "paper_valid": False, "marker": "QUICK_SCREENING_NOT_PAPER_VALID"}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
