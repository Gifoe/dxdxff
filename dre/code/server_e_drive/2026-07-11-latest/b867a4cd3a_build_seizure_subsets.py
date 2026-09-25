"""Build the immutable ID-only seizure sampling manifest."""
from __future__ import annotations

import argparse
from .core import SEEDS, build_subset_manifest, load_cache, load_subjects_and_folds_by_seed, patient_seizure_map, prepare_output_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-path", required=True)
    parser.add_argument("--protocol-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    paths = prepare_output_root(args.output_root)
    subjects, folds, _ = load_subjects_and_folds_by_seed(
        args.protocol_root,
        args.seeds,
    )
    records, _ = load_cache(args.cache_path)
    manifest = build_subset_manifest(patient_seizure_map(records, subjects), folds, args.seeds, args.repeats)
    manifest.to_csv(paths["manifests"] / "seizure_subset_manifest.csv", index=False)
    print({"status": "passed", "rows": len(manifest)})


if __name__ == "__main__":
    main()
