"""Summarize already completed A12 variant directories without re-tuning them."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from a12_vcsn.utils import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    root = Path(args.output_dir)
    rows = []
    for metrics_path in sorted((root / "variants").glob("*/metrics_summary.json")):
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        rows.append({"variant": metrics_path.parent.name, "patient_macro_f1": metrics.get("patient_macro_f1"), "patient_macro_ez_f1": metrics.get("patient_macro_ez_f1")})
    comparison = root / "comparison"
    comparison.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(comparison / "a12_variant_comparison.csv", index=False)
    write_json(comparison / "best_variant_posthoc.json", {"selection_is_posthoc": True, "do_not_reuse_outer_oof_for_further_tuning": True, "variants": rows})
    print(json.dumps({"n_variants": len(rows), "output": str(comparison.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
