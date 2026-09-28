"""Rescore frozen v1 predictions under official labels on their observable overlap.

This cannot create scores for official channels absent from v1 predictions. The
coverage gap is reported explicitly; it is not an estimate of full-cohort D1.
Run only after the v2 model and threshold have been frozen.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import pandas as pd

from audit_official_labels import official_pathology_label
from evaluate_v2 import classification_metrics, patient_metrics, unique_patient_channels
from train_v2 import atomic_json, sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--v1-predictions", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.freeze.is_file():
        raise RuntimeError("V2 freeze absent: no official-test label rescore permitted")
    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    if not freeze.get("model_frozen_before_official_test") or not freeze.get(
            "threshold_frozen_before_official_test"):
        raise RuntimeError("V2 freeze incomplete")
    v1 = pd.read_csv(args.v1_predictions)
    if v1.duplicated(["patient", "channel"]).any() or not {
            "patient", "channel", "score_ez", "soz"} <= set(v1):
        raise RuntimeError("Unexpected frozen v1 prediction schema")
    v1_map = v1.set_index(["patient", "channel"])["score_ez"].to_dict()
    v1_label_map = v1.set_index(["patient", "channel"])["soz"].to_dict()
    official = pd.read_csv(args.cohort)
    official = official.loc[(official["official_split"] == "test") &
                            (official["official_labeled_channels"] > 0)]
    overlap = []
    totals = {"official_edf_channel_records": 0, "official_unique_patient_channels": set(),
              "v1_scored_official_edf_channel_records": 0}
    for row in official.itertuples(index=False):
        relative = Path(row.edf)
        if relative.is_absolute() or ".." in relative.parts or relative.suffix.lower() != ".edf":
            raise RuntimeError("Unsafe official EDF path")
        sidecar = args.source / Path(str(relative).replace("_ieeg.edf", "_channels.tsv"))
        h5_path = args.cache / relative.with_suffix(".edf.h5")
        with h5py.File(h5_path, "r") as h5:
            metadata = json.loads(h5["metadata_json"][()].decode("utf-8"))
            signal_names = {str(channel["label"]) for channel in metadata["signal_headers"]}
        channels = pd.read_csv(sidecar, sep="\t")
        channels = channels.loc[channels["name"].astype(str).isin(signal_names)]
        for channel in channels.itertuples(index=False):
            label = official_pathology_label(row.outcome, channel.resection,
                                             channel.soz, channel.good)
            if label is None:
                continue
            totals["official_edf_channel_records"] += 1
            key = (str(row.patient), str(channel.name))
            totals["official_unique_patient_channels"].add(key)
            if key not in v1_map:
                continue
            totals["v1_scored_official_edf_channel_records"] += 1
            overlap.append({"patient": key[0], "dataset": str(row.dataset),
                            "edf": str(row.edf), "channel": key[1],
                            "pathology": label, "v1_soz": int(v1_label_map[key]),
                            "score_pathology": float(v1_map[key])})
    overlap = pd.DataFrame(overlap)
    if overlap.empty or overlap.duplicated(["edf", "channel"]).any():
        raise RuntimeError("No valid v1/official overlap")
    unique = unique_patient_channels(overlap)
    patient = pd.DataFrame([patient_metrics(group) for _, group in unique.groupby("patient")])
    estimable = patient.loc[patient["estimable_for_ranking"]]
    original = overlap.drop(columns="pathology").rename(columns={"v1_soz": "pathology"})
    original_unique = unique.drop(columns="pathology").rename(columns={"v1_soz": "pathology"})
    original_patient = pd.DataFrame([patient_metrics(group)
                                     for _, group in original_unique.groupby("patient")])
    original_estimable = original_patient.loc[original_patient["estimable_for_ranking"]]
    original_metrics = classification_metrics(original, 0.5)
    official_metrics = classification_metrics(overlap, 0.5)
    summary = {
        "status": "INTERSECTION_ONLY_NOT_FULL_OFFICIAL_COHORT",
        "interpretation": "Frozen v1 predictions do not cover all official-labeled channels; this isolates official labels only on the score-observable intersection.",
        "v1_prediction_sha256": sha256(args.v1_predictions),
        "v2_freeze_sha256": sha256(args.freeze),
        "v1_original_test_patients": int(v1["patient"].nunique()),
        "v1_original_unique_patient_channels": len(v1),
        "official_test_edf_channel_records": totals["official_edf_channel_records"],
        "official_test_unique_patient_channels": len(totals["official_unique_patient_channels"]),
        "intersection_edf_channel_records": len(overlap),
        "intersection_unique_patient_channels": len(unique),
        "intersection_patients": int(overlap["patient"].nunique()),
        "intersection_coverage_of_official_edf_records": len(overlap) / totals["official_edf_channel_records"],
        "intersection_coverage_of_official_unique_channels": len(unique) / len(totals["official_unique_patient_channels"]),
        "intersection_estimable_ranking_patients": len(estimable),
        "intersection_patient_equal_ap": float(estimable["ap"].mean()),
        "intersection_v1_soz_estimable_ranking_patients": len(original_estimable),
        "intersection_v1_soz_patient_equal_ap": float(original_estimable["ap"].mean()),
        "intersection_v1_soz_edf_channel_pair_metrics_at_0_5": original_metrics,
        "unique_patient_channel_metrics_at_0_5": classification_metrics(unique, 0.5),
        "official_edf_channel_pair_metrics_at_0_5_on_overlap": official_metrics,
        "paired_label_only_delta_macro_f1_on_overlap": official_metrics["macro_f1"] - original_metrics["macro_f1"],
        "paired_label_only_delta_pooled_ap_on_overlap": official_metrics["pooled_ap"] - original_metrics["pooled_ap"],
        "paired_label_only_delta_auroc_on_overlap": official_metrics["pooled_auroc"] - original_metrics["pooled_auroc"],
        "full_cohort_D1_attribution_identifiable": False,
    }
    atomic_json(args.output, summary)
    print(json.dumps({"status": summary["status"],
                      "coverage": summary["intersection_coverage_of_official_edf_records"]}))


if __name__ == "__main__":
    main()
