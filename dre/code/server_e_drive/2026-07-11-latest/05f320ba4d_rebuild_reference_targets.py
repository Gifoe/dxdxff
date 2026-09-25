from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REQUIRED = {
    "fold_idx", "subject_id", "center", "channel_name",
    "true_ez", "pseudo_core_q", "a9v3_oof_score",
}


def rebuild(reference_dir: Path, output_dir: Path) -> None:
    paths = sorted(reference_dir.glob("test_channel_predictions_neuroez_v2_fold_*.csv"))
    if len(paths) != 5:
        raise RuntimeError(f"Expected 5 reference test channel files, found {len(paths)} in {reference_dir}")
    all_oof = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    missing = sorted(REQUIRED - set(all_oof.columns))
    if missing:
        raise RuntimeError(f"Reference outputs are missing required columns: {missing}")
    all_oof = all_oof.rename(columns={"true_ez": "label_ez", "fold_idx": "fold_id"})
    all_oof["patient_id"] = all_oof["subject_id"].astype(str)
    columns = [
        "patient_id", "subject_id", "fold_id", "center", "channel_name",
        "label_ez", "pseudo_core_q", "a9v3_oof_score",
    ]
    all_oof = all_oof[columns].drop_duplicates(["subject_id", "channel_name"]).copy()
    output_dir.mkdir(parents=True, exist_ok=True)
    all_oof.to_csv(output_dir / "latent_core_targets_all_oof.csv", index=False)
    for fold_id in range(1, 6):
        all_oof[all_oof["fold_id"].astype(int) != fold_id].to_csv(
            output_dir / f"latent_core_targets_fold{fold_id}_train.csv", index=False
        )
    subjects = all_oof[["subject_id", "center"]].drop_duplicates().sort_values("subject_id")
    if len(subjects) != 90:
        raise RuntimeError(f"Expected 90 reference subjects, got {len(subjects)}")
    subjects[["subject_id"]].to_csv(output_dir.parent / "all90_subjects.csv", index=False)
    print(f"Rebuilt targets for {len(subjects)} subjects and {len(all_oof)} channels in {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    rebuild(args.reference_dir, args.output_dir)


if __name__ == "__main__":
    main()
