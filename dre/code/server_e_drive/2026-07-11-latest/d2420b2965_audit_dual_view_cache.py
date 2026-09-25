"""Write the strict N6 feature/raw alignment audit without training."""

from __future__ import annotations

import argparse
import csv
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neuroez_c.dual_view_data import RawAlignmentStore


def _subjects(path: str | Path) -> set[str]:
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return {str(row["subject_id"]).strip() for row in csv.DictReader(handle) if str(row.get("subject_id", "")).strip()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-cache", required=True)
    parser.add_argument("--raw-cache", required=True)
    parser.add_argument("--subjects", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--raw-target-samples", type=int, default=500)
    parser.add_argument("--raw-target-sampling-rate", type=float, default=250.0)
    parser.add_argument("--min-channel-match-rate", type=float, default=0.95)
    parser.add_argument("--min-window-match-rate", type=float, default=0.90)
    args = parser.parse_args()
    with Path(args.feature_cache).open("rb") as handle:
        payload = pickle.load(handle)
    allowed = _subjects(args.subjects)
    records = [record for record in payload.get("run_records", []) if str(record.get("subject_id")) in allowed]
    store = RawAlignmentStore(
        records,
        feature_cache_path=args.feature_cache,
        raw_cache_path=args.raw_cache,
        raw_target_samples=args.raw_target_samples,
        raw_target_sampling_rate=args.raw_target_sampling_rate,
        output_audit_path=args.output,
    )
    store.assert_formal_coverage(
        min_channel_match_rate=args.min_channel_match_rate,
        min_window_match_rate=args.min_window_match_rate,
        expected_patients=len(allowed),
    )
    print(Path(args.output).read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
