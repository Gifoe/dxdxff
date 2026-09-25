from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import pandas as pd

from outcome_hifos.cache_audit import audit_caches


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the shared schema audit used only for adapter design.")
    parser.add_argument("--feature-cache")
    parser.add_argument("--raw-cache")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reuse-outcome-audit", help="Reuse an audit previously generated from the exact same cache files.")
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    source = Path(args.reuse_outcome_audit) if args.reuse_outcome_audit else output / "_outcome_audit"
    if not args.reuse_outcome_audit:
        if not args.feature_cache or not args.raw_cache:
            raise ValueError("--feature-cache and --raw-cache are required unless --reuse-outcome-audit is used.")
        audit_caches(args.feature_cache, args.raw_cache, source)
    required = {
        "outcome_cache_schema_feature.json": "feature_cache_schema.json",
        "outcome_cache_schema_raw.json": "raw_cache_schema.json",
        "outcome_run_manifest.csv": "_all_runs.csv",
    }
    for original, destination in required.items():
        path = source / original
        if not path.exists():
            raise FileNotFoundError(path)
        shutil.copyfile(path, output / destination)
    runs = pd.read_csv(output / "_all_runs.csv")
    runs[runs["cache_kind"] == "feature"].drop(columns="cache_kind").to_csv(output / "feature_run_schema.csv", index=False)
    runs[runs["cache_kind"] == "raw"].drop(columns="cache_kind").to_csv(output / "raw_run_schema.csv", index=False)
    (output / "_all_runs.csv").unlink()
    feature = json.loads((output / "feature_cache_schema.json").read_text(encoding="utf-8"))
    raw = json.loads((output / "raw_cache_schema.json").read_text(encoding="utf-8"))
    report = [
        "# Cache Key Mapping", "",
        "This audit describes storage only and never defines either task cohort.", "",
        f"- Feature records: {feature['n_run_records']}; patients: {feature['n_patient_index']}",
        f"- Raw records: {raw['n_run_records']}; patients: {raw['n_patient_index']}",
        "- Feature tensor: `sample.window_features` with `[window, channel, feature]`.",
        "- Raw tensor: `sample.raw_waveform` with `[channel, time]`; window centers are `sample.window_relative_centers_sec`.",
        "- Channel names: `record.channel_names_norm`.",
        "- Task 1 authoritative channel labels: `patient_index[subject].canonical_channels + labels`; record-aligned labels are fallback only when canonical labels are absent.",
        "- Task 2 outcomes: resolved only through `outcome_hifos.outcome_resolver`.",
    ]
    (output / "key_mapping_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
