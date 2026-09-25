from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from neuroez_c.task2.data import load_cache
from neuroez_c.task2.functional_graph import GraphConfig
from neuroez_c.task2.graph_cache import build_graph_cache


def main() -> int:
    parser = argparse.ArgumentParser(description="Build label-free phase raw AEC-Spearman graph cache")
    parser.add_argument("--raw_cache", default=os.getenv("DRE_TASK1_RAW_CACHE_PATH"), required=os.getenv("DRE_TASK1_RAW_CACHE_PATH") is None)
    parser.add_argument("--feature_cache", default=os.getenv("DRE_TASK1_FEATURE_CACHE_PATH"), required=os.getenv("DRE_TASK1_FEATURE_CACHE_PATH") is None)
    parser.add_argument("--graph_source", default="raw_aec_spearman", choices=("raw_aec_spearman",))
    parser.add_argument("--band_low", type=float, default=30.0)
    parser.add_argument("--band_high", type=float, default=80.0)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--exclusion_manifest", default=os.getenv("DRE_TASK2_EXCLUSION_MANIFEST", str(PROJECT / "configs" / "data_exclusions.csv")))
    parser.add_argument("--patients", help="Optional comma-separated bounded smoke patient list")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    patient_filter = set(filter(None, (args.patients or "").split(","))) or None
    config = GraphConfig(graph_source=args.graph_source, band_low=args.band_low, band_high=args.band_high, topk=args.topk)
    manifest = build_graph_cache(load_cache(args.raw_cache), load_cache(args.feature_cache), args.output_dir, raw_source_path=args.raw_cache, config=config, strict=args.strict, patient_filter=patient_filter, exclusion_manifest=args.exclusion_manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
