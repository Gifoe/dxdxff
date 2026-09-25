"""Audit checkpoints, cache hierarchy, and frozen 80-patient protocol."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd

from .core import SEEDS, discover_checkpoint, discover_config, git_commit, load_cache, load_fit_subjects_by_seed, load_model, load_subjects_and_folds_by_seed, patient_seizure_map, prepare_output_root, read_json, resolve_model_normalizer, sha256_file, write_json
from task1_confirmatory.provenance import (
    P2_METHOD_ID,
    P2_PROFILE_ID,
    V3_METHOD_ID,
    V3_PROFILE_ID,
    validate_branch_provenance,
)


def _require_config(config: dict, expected: dict, checkpoint: Path) -> None:
    errors = [
        f"{key}={config.get(key)!r}, expected {value!r}"
        for key, value in expected.items()
        if config.get(key) != value
    ]
    if errors:
        raise ValueError(
            f"Formal model configuration mismatch for {checkpoint}: {errors}"
        )


def _validate_branch(root: Path, model: str, seed: int) -> dict:
    if model == "PRQ-Net":
        return validate_branch_provenance(
            root,
            expected_method_id=P2_METHOD_ID,
            expected_profile_id=P2_PROFILE_ID,
            expected_seed=seed,
            expected_held_out_center="",
        )
    return validate_branch_provenance(
        root,
        expected_method_id=V3_METHOD_ID,
        expected_profile_id=V3_PROFILE_ID,
        expected_seed=seed,
        expected_held_out_center="",
    )


def audit(args: argparse.Namespace) -> dict:
    paths = prepare_output_root(args.output_root)
    subjects, fold_maps, manifest = load_subjects_and_folds_by_seed(
        args.protocol_root,
        args.seeds,
    )
    fit_maps = load_fit_subjects_by_seed(manifest, args.seeds)
    run_records, patient_index = load_cache(args.cache_path)
    seizures = patient_seizure_map(run_records, subjects)
    if set(patient_index) & subjects != subjects:
        raise ValueError("Cache patient_index does not cover the fixed 80-patient cohort")
    label_values = set()
    for subject in subjects:
        values = patient_index[subject].get("labels")
        if values is None:
            raise ValueError(f"Cache patient metadata lacks labels for {subject}")
        label_values.update(int(value) for value in pd.Series(values).dropna().astype(int).unique())
    if not label_values.issubset({0, 1}):
        raise ValueError(f"Cache labels must be binary EZ labels before NEZ conversion; got {sorted(label_values)}")
    inventory = []
    for model, root in (("PRQ-Net", args.prq_root), ("BCR-Net", args.bcr_root)):
        for seed in args.seeds:
            branch_name = "p2_q10" if model == "PRQ-Net" else "bcr_boundary_coverage"
            branch_root = Path(root) / f"seed_{seed}" / branch_name
            provenance = _validate_branch(branch_root, model, seed)
            provenance_by_fold = {
                int(row["fold"]): row for row in provenance["checkpoints"]
            }
            for fold in range(1, 6):
                checkpoint = discover_checkpoint(root, seed, fold, model)
                config_path = discover_config(checkpoint)
                config = read_json(config_path)
                if model == "PRQ-Net":
                    _require_config(config, {
                        "positive_label": "nez",
                        "use_p23_trn_nez": True,
                        "p23_profile": "P2_TEMPORAL_Q10",
                        "p23_direct_outer_only": True,
                        "p23_regression_protocol": True,
                        "p23_use_p2_loss": True,
                        "use_cane_path_cp_nez": False,
                        "use_patient_relative_z": True,
                    }, checkpoint)
                else:
                    _require_config(config, {
                        "v3_qbc_profile": "BCR_BOUNDARY_COVERAGE",
                        "positive_label": "ez",
                    }, checkpoint)
                checkpoint_hash = sha256_file(checkpoint)
                provenance_row = provenance_by_fold.get(fold)
                if provenance_row is None:
                    raise ValueError(
                        f"Formal provenance lacks fold {fold}: {branch_root}"
                    )
                if checkpoint_hash != provenance_row.get("sha256"):
                    raise ValueError(
                        f"Checkpoint does not match formal provenance for "
                        f"{model} seed={seed} fold={fold}: {checkpoint}"
                    )
                inventory.append({
                    "model": model,
                    "training_seed": seed,
                    "outer_fold": fold,
                    "checkpoint": str(checkpoint),
                    "sha256": checkpoint_hash,
                    "config": str(config_path),
                    "config_sha256": sha256_file(config_path),
                    "provenance": str(branch_root / "training_provenance.json"),
                    "provenance_sha256": sha256_file(branch_root / "training_provenance.json"),
                })
    frame = pd.DataFrame(inventory)
    if len(frame) != 30:
        raise RuntimeError("Expected exactly 30 frozen checkpoints")
    # Loading one checkpoint from each formal architecture is a strict smoke test;
    # parameters are frozen immediately by load_model and no forward/backward occurs.
    for model in ("PRQ-Net", "BCR-Net"):
        sample = frame[frame.model.eq(model)].iloc[0]
        fit_subjects = fit_maps[int(sample.training_seed)][int(sample.outer_fold)]
        normalizer = resolve_model_normalizer(
            Path(sample.checkpoint),
            model_kind=model,
            run_records=run_records,
            fit_subjects=fit_subjects,
        )
        loaded = load_model(
            Path(sample.checkpoint),
            device="cpu",
            model_kind=model,
            normalizer=normalizer,
        )
        del loaded
    frame.to_csv(paths["audit"] / "checkpoint_inventory.csv", index=False)
    frame.to_csv(paths["checkpoints_reference"] / "frozen_checkpoint_inventory.csv", index=False)
    seizure_frame = pd.DataFrame({"subject_id": list(seizures), "n_valid_seizures": [len(ids) for ids in seizures.values()]})
    seizure_frame.to_csv(paths["audit"] / "patient_seizure_counts.csv", index=False)
    counts = Counter(seizure_frame.n_valid_seizures)
    report = {"status": "passed", "n_patients": len(subjects), "fold_sizes": {
                  str(seed): {str(fold): len(values) for fold, values in folds.items()}
                  for seed, folds in fold_maps.items()
              },
              "fixed_manifest": str(manifest), "cache_path": str(args.cache_path), "cache_sha256": sha256_file(args.cache_path),
              "seizure_distribution": {str(key): int(value) for key, value in sorted(counts.items())},
              "exactly_one": int((seizure_frame.n_valid_seizures == 1).sum()), "at_least_two": int((seizure_frame.n_valid_seizures >= 2).sum()),
              "at_least_three": int((seizure_frame.n_valid_seizures >= 3).sum()), "git_commit": git_commit(Path(__file__).resolve().parents[2]),
              "label_semantics": "NEZ=1,EZ=0", "cache_ez_label_values": sorted(label_values), "no_training": True}
    pd.DataFrame([{"kind": "cache", "path": str(args.cache_path), "sha256": report["cache_sha256"]}, {"kind": "fixed_manifest", "path": str(manifest), "sha256": sha256_file(manifest)}]).to_csv(paths["audit"] / "input_inventory.csv", index=False)
    write_json(paths["audit"] / "input_audit.json", report)
    write_json(paths["audit"] / "input_hashes.json", {"cache": report["cache_sha256"], "manifest": sha256_file(manifest), "checkpoints": inventory})
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("prq_root", "bcr_root", "cache_path", "protocol_root", "output_root"):
        parser.add_argument("--" + name.replace("_", "-"), dest=name, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    args = parser.parse_args()
    print(audit(args))


if __name__ == "__main__":
    main()
