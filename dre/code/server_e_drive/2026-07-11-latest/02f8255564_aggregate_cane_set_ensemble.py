"""Aggregate fixed-seed CANE logits and predicted cardinalities without test tuning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from exp_ez_hybrid import _summarize_prediction_records


SEEDS = (42, 43, 44)
KEY_COLUMNS = ["outer_fold", "subject_id", "channel_name", "channel_id"]


def _read_seed_channels(root: Path, seed: int) -> pd.DataFrame:
    paths = sorted((root / f"seed_{seed}").glob("test_channel_predictions_neuroez_v2_fold_*.csv"))
    if not paths:
        raise FileNotFoundError(f"No CANE test channel predictions for seed {seed} under {root}")
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    frame = frame.rename(columns={"fold_idx": "outer_fold"})
    required = set(KEY_COLUMNS + [
        "true_nez", "true_ez", "center", "valid_channel", "final_nez_logit",
    ])
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Seed {seed} channel predictions are missing columns: {missing}")
    if frame.duplicated(KEY_COLUMNS).any():
        raise ValueError(f"Seed {seed} contains duplicate channel keys")
    return frame.sort_values(KEY_COLUMNS).reset_index(drop=True)


def _read_seed_patients(root: Path, seed: int) -> pd.DataFrame:
    paths = sorted((root / f"seed_{seed}").glob("test_patient_predictions_neuroez_v2_fold_*.csv"))
    if not paths:
        raise FileNotFoundError(f"No CANE test patient predictions for seed {seed} under {root}")
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    frame = frame.rename(columns={"fold_idx": "outer_fold"})
    required = {"outer_fold", "subject_id", "predicted_nez_fraction"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Seed {seed} patient predictions are missing columns: {missing}")
    if frame.duplicated(["outer_fold", "subject_id"]).any():
        raise ValueError(f"Seed {seed} contains duplicate patient keys")
    return frame.sort_values(["outer_fold", "subject_id"]).reset_index(drop=True)


def _assert_seed_alignment(channels: Sequence[pd.DataFrame], patients: Sequence[pd.DataFrame]) -> None:
    reference_channels = channels[0]
    reference_patients = patients[0]
    for index, frame in enumerate(channels[1:], start=1):
        if not reference_channels[KEY_COLUMNS].equals(frame[KEY_COLUMNS]):
            raise ValueError(f"CANE ensemble channel key mismatch for seed index {index}")
        for column in ("true_nez", "true_ez", "center", "valid_channel"):
            if not reference_channels[column].astype(str).equals(frame[column].astype(str)):
                raise ValueError(f"CANE ensemble {column} mismatch for seed index {index}")
    patient_keys = ["outer_fold", "subject_id"]
    for index, frame in enumerate(patients[1:], start=1):
        if not reference_patients[patient_keys].equals(frame[patient_keys]):
            raise ValueError(f"CANE ensemble patient key mismatch for seed index {index}")


def _stable_patient_topk(group: pd.DataFrame, fraction: float) -> tuple[np.ndarray, int]:
    valid = group["valid_channel"].astype(bool).to_numpy()
    valid_count = int(valid.sum())
    predicted_count = int(np.clip(np.rint(valid_count * float(fraction)), 0, valid_count))
    selected = np.zeros(len(group), dtype=bool)
    if predicted_count:
        candidates = group.loc[valid].sort_values(
            ["ensemble_score_nez", "channel_id"], ascending=[False, True], kind="mergesort"
        )
        selected[candidates.index.to_numpy()[:predicted_count] - group.index.min()] = True
    return selected, predicted_count


def _records_from_channels(channels: pd.DataFrame, patient_fractions: pd.DataFrame) -> list[dict]:
    fraction_lookup = {
        (int(row.outer_fold), str(row.subject_id)): float(row.ensemble_predicted_nez_fraction)
        for row in patient_fractions.itertuples(index=False)
    }
    records: list[dict] = []
    for (fold, subject), group_source in channels.groupby(["outer_fold", "subject_id"], sort=True):
        group = group_source.sort_values("channel_id", kind="mergesort").reset_index(drop=True)
        fraction = fraction_lookup[(int(fold), str(subject))]
        predicted_nez, predicted_count = _stable_patient_topk(group, fraction)
        valid = group["valid_channel"].astype(bool).to_numpy()
        predicted_ez = valid & ~predicted_nez
        records.append({
            "outer_fold": int(fold),
            "subject_id": str(subject),
            "center": str(group["center"].iloc[0]),
            "center_id": int(group["center_id"].iloc[0]) if "center_id" in group else 4,
            "canonical_channels": group["channel_name"].astype(str).tolist(),
            "labels_nez": group["true_nez"].to_numpy(dtype=np.float32),
            "labels_ez": group["true_ez"].to_numpy(dtype=np.float32),
            "labels": group["true_nez"].to_numpy(dtype=np.float32),
            "scores": group["ensemble_score_nez"].to_numpy(dtype=np.float32),
            "score_nez": group["ensemble_score_nez"].to_numpy(dtype=np.float32),
            "score_ez": group["ensemble_score_ez"].to_numpy(dtype=np.float32),
            "channel_mask": valid,
            "predicted_nez_mask": predicted_nez,
            "predicted_ez_mask": predicted_ez,
            "predicted_nez_fraction": fraction,
            "expected_nez_count": float(valid.sum() * fraction),
            "predicted_nez_count": float(predicted_count),
            "decision_rule": "predicted_nez_cardinality_topk",
            "classification_threshold": float("nan"),
            "threshold_source": "not_used",
            "true_count_used_for_prediction": False,
            "positive_label": "nez",
        })
    return records


def aggregate_cane_set_ensemble(
    output_root: str | Path,
    *,
    seeds: Iterable[int] = SEEDS,
) -> dict[str, Path]:
    root = Path(output_root)
    seed_values = tuple(int(seed) for seed in seeds)
    if seed_values != SEEDS:
        raise ValueError(f"Formal CANE ensemble requires fixed seeds {SEEDS}, got {seed_values}")
    seed_channels = [_read_seed_channels(root, seed) for seed in seed_values]
    seed_patients = [_read_seed_patients(root, seed) for seed in seed_values]
    _assert_seed_alignment(seed_channels, seed_patients)

    channels = seed_channels[0][KEY_COLUMNS + [
        "center", "center_id", "true_nez", "true_ez", "valid_channel",
    ]].copy()
    logit_columns: list[str] = []
    for seed, frame in zip(seed_values, seed_channels):
        column = f"final_nez_logit_seed_{seed}"
        channels[column] = frame["final_nez_logit"].to_numpy(dtype=np.float64)
        logit_columns.append(column)
    channels["ensemble_nez_logit"] = channels[logit_columns].mean(axis=1)
    channels["ensemble_score_nez"] = 1.0 / (1.0 + np.exp(-channels["ensemble_nez_logit"].clip(-80.0, 80.0)))
    channels["ensemble_score_ez"] = 1.0 - channels["ensemble_score_nez"]

    patient_keys = ["outer_fold", "subject_id"]
    fractions = seed_patients[0][patient_keys].copy()
    fraction_columns: list[str] = []
    for seed, frame in zip(seed_values, seed_patients):
        column = f"predicted_nez_fraction_seed_{seed}"
        fractions[column] = frame["predicted_nez_fraction"].to_numpy(dtype=np.float64)
        fraction_columns.append(column)
    fractions["ensemble_predicted_nez_fraction"] = fractions[fraction_columns].mean(axis=1)

    records = _records_from_channels(channels, fractions)
    summary, enriched = _summarize_prediction_records(records)
    summary.update({
        "method": "N7F_CANE_Set_NEZ_CountFree_3SeedEnsemble",
        "model_seeds": "42,43,44",
        "classification_threshold": float("nan"),
        "threshold_source": "not_used",
        "decision_rule": "predicted_nez_cardinality_topk",
        "true_count_used_for_prediction": False,
        "test_labels_used_for_ensemble": False,
    })

    ensemble_dir = root / "ensemble"
    ensemble_dir.mkdir(parents=True, exist_ok=True)
    patient_rows = pd.DataFrame([{key: value for key, value in record.items() if not isinstance(value, np.ndarray)} for record in enriched])
    channel_rows: list[dict] = []
    for record in enriched:
        fold = int(record["outer_fold"])
        subject = str(record["subject_id"])
        group = channels[(channels["outer_fold"] == fold) & (channels["subject_id"].astype(str) == subject)].sort_values("channel_id")
        pred_nez = np.asarray(record["predicted_nez_mask"], dtype=bool)
        pred_ez = np.asarray(record["predicted_ez_mask"], dtype=bool)
        for position, row in enumerate(group.itertuples(index=False)):
            values = row._asdict()
            values.update({"predicted_nez": int(pred_nez[position]), "predicted_ez": int(pred_ez[position])})
            channel_rows.append(values)
    channel_frame = pd.DataFrame(channel_rows)

    fold_rows: list[dict] = []
    for fold in sorted({int(record["outer_fold"]) for record in records}):
        fold_summary, _ = _summarize_prediction_records([record for record in records if int(record["outer_fold"]) == fold])
        fold_rows.append({"outer_fold": fold, **fold_summary})
        audit = {
            "method": "N7F_CANE_Set_NEZ_CountFree_3SeedEnsemble",
            "outer_fold": fold,
            "model_seeds": list(SEEDS),
            "channel_keys_identical": True,
            "labels_identical": True,
            "center_identical": True,
            "valid_mask_identical": True,
            "test_labels_used_for_ensemble": False,
            "decision_rule": "predicted_nez_cardinality_topk",
        }
        (ensemble_dir / f"cane_fold_{fold}_ensemble_audit.json").write_text(
            json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8"
        )

    summary_csv = ensemble_dir / "heldout_summary_cane_set.csv"
    summary_json = ensemble_dir / "heldout_summary_cane_set.json"
    pd.DataFrame([summary]).to_csv(summary_csv, index=False)
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=True), encoding="utf-8")
    pd.DataFrame(fold_rows).to_csv(ensemble_dir / "heldout_fold_summary_cane_set.csv", index=False)
    patient_rows.to_csv(ensemble_dir / "heldout_patient_predictions_cane_set.csv", index=False)
    channel_frame.to_csv(ensemble_dir / "heldout_channel_predictions_cane_set.csv", index=False)
    patient_rows[[
        "outer_fold", "subject_id", "center", "predicted_nez_fraction", "expected_nez_count",
        "predicted_nez_count", "true_nez_count", "nez_count_error", "nez_fraction_error",
    ]].to_csv(ensemble_dir / "cane_cardinality_audit.csv", index=False)
    patient_rows.groupby("center", as_index=False).agg(
        n_patients=("subject_id", "nunique"),
        patient_macro_f1=("patient_macro_f1", "mean"),
        patient_nez_f1=("patient_nez_f1", "mean"),
        patient_ez_f1=("patient_ez_f1", "mean"),
        nez_count_mae=("nez_count_error", "mean"),
    ).to_csv(ensemble_dir / "cane_center_summary.csv", index=False)
    return {"summary_csv": summary_csv, "summary_json": summary_json, "ensemble_dir": ensemble_dir}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    outputs = aggregate_cane_set_ensemble(args.output_root)
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()

