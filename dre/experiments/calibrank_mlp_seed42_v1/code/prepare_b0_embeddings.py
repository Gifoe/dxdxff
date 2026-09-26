"""Rebuild frozen patient_z_mlp embeddings and verify exact OOF replay."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source-root", "feature-cache", "cohort-manifest", "fold-manifest", "split-manifest",
                 "feature-manifest", "b0-ledger", "checkpoint-root", "output-dir"):
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    sys.path.insert(0, str(args.source_root.resolve()))
    from outcome_hifos.cache_schema import load_cache_contract
    from task1_baselines.cache_io import task1_feature_records
    from task1_baselines.feature_aggregation import build_channel_feature_table
    from task1_baselines.fold_protocol import fold_manifest_hash, load_task1_sensitivity_protocol
    from task1_baselines.patient_controls import _ChannelMLP, _Preprocessor, feature_columns

    folds, splits = load_task1_sensitivity_protocol(args.cohort_manifest, args.fold_manifest,
                                                     args.split_manifest, expected_subjects=80)
    cache = load_cache_contract(args.feature_cache)
    table, manifest = build_channel_feature_table(task1_feature_records(cache, set(folds.subject_id)),
                                                  profile="p2_matched_simple")
    old_manifest = json.loads(args.feature_manifest.read_text(encoding="utf-8"))
    if manifest != old_manifest:
        raise RuntimeError("Feature manifest differs from B0 historical training")
    merged = table.merge(folds[["subject_id", "outer_fold"]], on="subject_id", how="inner", validate="many_to_one")
    columns = feature_columns(merged)
    if len(columns) != 88 or len(merged) != 7635:
        raise RuntimeError(f"Unexpected feature dimensionality or cohort: {len(columns)}, {len(merged)}")
    old = pd.read_csv(args.b0_ledger)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit = {"status": "PREPARING", "feature_dimension": len(columns), "fold_manifest_hash": fold_manifest_hash(folds),
             "feature_cache_sha256": sha256(args.feature_cache), "b0_ledger_sha256": sha256(args.b0_ledger),
             "source_code_sha256": sha256(Path(__file__)), "folds": []}
    for fold in range(1, 6):
        destination = args.output_dir / f"fold{fold}.npz"
        if destination.exists():
            raise RuntimeError(f"Preserve existing embeddings and inspect before rerun: {destination}")
        split = splits[splits.outer_fold == fold]
        groups = {name: merged[merged.subject_id.isin(set(split.loc[split.partition == name, "subject_id"]))].reset_index(drop=True)
                  for name in ("fit", "validation", "test")}
        pre = _Preprocessor(SimpleImputer(), StandardScaler(), True)
        prepared = {"fit": pre.fit_transform(groups["fit"], columns)}
        for name in ("validation", "test"):
            prepared[name] = pre.transform(groups[name], columns)
        model = _ChannelMLP(len(columns)).eval()
        checkpoint = args.checkpoint_root / f"fold_{fold}" / "best_model.pt"
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(saved["state_dict"], strict=True)
        arrays = {"b0_selected_threshold_nez": np.array(float(saved["selected_threshold"]), dtype=np.float64),
                  "b0_selected_epoch": np.array(int(saved["best_epoch"]), dtype=np.int32)}
        with torch.no_grad():
            for name in ("fit", "validation", "test"):
                x = torch.from_numpy(prepared[name])
                h = model.network[:-1](x)
                a_nez = model.network[-1](h).squeeze(-1)
                group = groups[name]
                arrays.update({f"{name}_hidden": h.numpy().astype(np.float32),
                               f"{name}_logit_ez": (-a_nez).numpy().astype(np.float32),
                               f"{name}_label_nez": group.label_nez.to_numpy(np.int8),
                               f"{name}_subject_id": group.subject_id.astype(str).to_numpy(),
                               f"{name}_channel_name": group.channel_name.astype(str).to_numpy(),
                               f"{name}_seizure_count": group.valid_seizure_count.to_numpy(np.float32)})
        test = groups["test"][["subject_id", "outer_fold", "channel_name", "label_nez"]].copy()
        test["replayed_score_nez"] = torch.sigmoid(torch.from_numpy(-arrays["test_logit_ez"])).numpy()
        reference = old.loc[old.outer_fold == fold, ["subject_id", "outer_fold", "channel_name", "label_nez",
                                                     "score_nez_probability", "selected_threshold"]]
        replay = test.merge(reference, on=["subject_id", "outer_fold", "channel_name"], how="inner", validate="one_to_one",
                            suffixes=("_replay", "_reference"))
        if len(replay) != len(test) or not np.array_equal(replay.label_nez_replay, replay.label_nez_reference):
            raise RuntimeError(f"Fold {fold} test membership or labels differ")
        difference = float(np.max(np.abs(replay.replayed_score_nez - replay.score_nez_probability)))
        if difference > 2e-6 or np.max(np.abs(replay.selected_threshold - float(saved["selected_threshold"]))) > 1e-12:
            raise RuntimeError(f"Fold {fold} B0 score/threshold replay failed: max diff {difference}")
        np.savez_compressed(destination, **arrays)
        audit["folds"].append({"fold": fold, "fit_patients": int(groups["fit"].subject_id.nunique()),
                               "validation_patients": int(groups["validation"].subject_id.nunique()),
                               "test_patients": int(groups["test"].subject_id.nunique()),
                               "test_channels": len(test), "max_score_difference": difference,
                               "checkpoint_sha256": sha256(checkpoint), "selected_epoch": int(saved["best_epoch"]),
                               "selected_threshold_nez": float(saved["selected_threshold"])})
        print(json.dumps(audit["folds"][-1]), flush=True)
    audit["status"] = "PASS"
    (args.output_dir / "REPLAY_AUDIT.json").write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    print("B0_REPLAY_PASS", flush=True)


if __name__ == "__main__":
    main()
