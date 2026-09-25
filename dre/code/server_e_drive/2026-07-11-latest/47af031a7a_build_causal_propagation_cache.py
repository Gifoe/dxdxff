from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

# The offline VAR builder repeatedly solves dense channel systems.  Limiting
# BLAS before importing NumPy prevents Windows native access violations caused
# by excessive thread creation on large raw caches.
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neuroez_c.cane_path_cohort import EXPECTED_CENTER_COUNTS, read_exclusion_manifest, subject_center
from neuroez_c.causal_propagation_features import build_causal_propagation_cache


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build label-free ridge-VAR causal-propagation patient-channel features.")
    parser.add_argument("--raw-window-cache-path", required=True)
    parser.add_argument("--feature-window-cache-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--audit-output-dir", required=True)
    parser.add_argument("--exclude-subjects-file", required=True)
    parser.add_argument("--require-n-patients", type=int, default=80)
    parser.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--cp-max-preictal-windows", type=int, default=4)
    parser.add_argument("--cp-max-early-windows", type=int, default=6)
    parser.add_argument("--cp-early-ictal-seconds", type=float, default=30.0)
    parser.add_argument("--cp-ridge-alpha", type=float, default=0.10)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    with Path(args.feature_window_cache_path).open("rb") as handle:
        payload = pickle.load(handle)
    subjects = {str(value) for value in payload.get("patient_index", {})}
    exclusions = read_exclusion_manifest(args.exclude_subjects_file)
    canonical = {value.casefold(): value for value in subjects}
    excluded_keys = {row["subject_id"].casefold() for row in exclusions}
    if not excluded_keys.issubset(canonical):
        raise ValueError(f"Exclusion IDs missing from feature cache: {sorted(excluded_keys - set(canonical))}")
    retained = sorted(value for key, value in canonical.items() if key not in excluded_keys)
    if len(retained) != int(args.require_n_patients):
        raise ValueError(f"Expected {args.require_n_patients} retained patients, got {len(retained)}")
    counts = {center: sum(subject_center(value) == center for value in retained) for center in EXPECTED_CENTER_COUNTS}
    if counts != EXPECTED_CENTER_COUNTS:
        raise ValueError(f"Retained center counts mismatch: {counts}")
    summary = build_causal_propagation_cache(
        args.raw_window_cache_path, args.feature_window_cache_path, args.output_path,
        args.audit_output_dir, retained, ridge_alpha=args.cp_ridge_alpha,
        max_preictal_windows=args.cp_max_preictal_windows,
        max_early_windows=args.cp_max_early_windows,
        early_ictal_seconds=args.cp_early_ictal_seconds, strict=args.strict,
    )
    print(summary)


if __name__ == "__main__":
    main()
