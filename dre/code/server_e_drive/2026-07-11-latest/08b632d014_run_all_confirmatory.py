#!/usr/bin/env python3
"""One strict entrypoint for Task 1 confirmatory execution."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from task1_confirmatory.audit import repository_entrypoint_audit, reproduce_seed42
from task1_confirmatory.config import load_config
from task1_confirmatory.efficiency import collect_efficiency
from task1_confirmatory.orchestrator import prepare_manifests, run_loco, run_pooled
from task1_confirmatory.statistics import summarize_statistics


def _seeds(value: str, configured: list[int]) -> list[int]:
    return configured if not value else [int(item) for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True); parser.add_argument("--seeds", default="")
    parser.add_argument("--run_pooled_cv", action="store_true"); parser.add_argument("--run_loco", action="store_true")
    parser.add_argument("--run_statistics", action="store_true"); parser.add_argument("--run_efficiency", action="store_true")
    parser.add_argument("--held_out_center", choices=["hup", "lzu", "multicenter", "pediatric"])
    parser.add_argument("--outer_fold", type=int, choices=[1, 2, 3, 4, 5]); parser.add_argument("--model", choices=["all", "p2", "v3", "fusion"], default="all")
    parser.add_argument("--resume", action="store_true"); parser.add_argument("--skip_completed", action="store_true")
    parser.add_argument("--strict", action="store_true"); parser.add_argument("--dry_run", action="store_true"); parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--audit_only", action="store_true")
    args = parser.parse_args(); config = load_config(args.config); output = Path(config["output_root"]); seeds = _seeds(args.seeds, list(config["seeds"]))
    if args.smoke and (len(seeds) != 1 or args.outer_fold is None and not args.held_out_center):
        raise ValueError("Smoke mode requires one seed and one outer fold or one held-out center")
    if args.strict and not bool(config.get("strict", False)):
        raise ValueError("Config strict must be true when --strict is requested")
    result: dict = {"entrypoint_audit": repository_entrypoint_audit(config, output_root=output)}
    result["seed42_reproduction"] = reproduce_seed42(config, output_root=output)
    result["manifests"] = prepare_manifests(config, output_root=output)
    if args.audit_only:
        print(json.dumps(result, indent=2, default=str)); return
    if args.run_pooled_cv:
        result["pooled"] = run_pooled(config, output_root=output, seeds=seeds, outer_fold=args.outer_fold, dry_run=args.dry_run, smoke=args.smoke, skip_completed=args.skip_completed or args.resume, model=args.model)
    if args.run_loco:
        result["loco"] = run_loco(config, output_root=output, seeds=seeds, held_out_center=args.held_out_center, dry_run=args.dry_run, smoke=args.smoke, skip_completed=args.skip_completed or args.resume, model=args.model)
    if args.run_statistics:
        if args.smoke or args.dry_run: raise ValueError("Statistics excludes smoke and dry-run outputs")
        result["statistics"] = summarize_statistics(output_root=output, seeds=seeds, bootstrap_repeats=int(config["bootstrap_repeats"]), permutation_repeats=int(config["permutation_repeats"]))
    if args.run_efficiency:
        result["efficiency"] = collect_efficiency(output_root=output)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
