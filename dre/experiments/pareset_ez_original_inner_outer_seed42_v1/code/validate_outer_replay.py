"""Outcome-blind validation replay for the frozen outer evaluator."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--supplement-root", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.code_root.resolve()))
    sys.path.insert(0, str((args.code_root / "source").resolve()))
    sys.path.insert(0, str(args.supplement_root.resolve()))
    import epilens.models as models
    from epilens.data import load_records, load_partition_manifest, select_records, validate_protocol
    from train_repaired_control import repaired_mean_std
    from evaluate_outer_seed42 import load_model, predict
    models._masked_mean_std = repaired_mean_std
    records = load_records(args.data)
    manifest = load_partition_manifest(args.manifest)
    validate_protocol(records, manifest, expected_patients=80)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for fold in range(1, 6):
        validation = select_records(records, manifest, fold, "validation")
        for name in ("full", "base", "prq", "bcr"):
            cell = args.runtime_root / ("stage_a_seed42" if name in ("full", "base") else "matched_controls_seed42") / f"fold{fold}" / name
            selection = json.loads((cell / "selection.json").read_text(encoding="utf-8"))
            lock = {"thresholds": {f"fold{fold}": {name: float(selection["threshold"])}}}
            model, standardizer, threshold = load_model(name, cell / "best.pt", fold, lock, device)
            replay, _ = predict(model, standardizer, validation, name, device)
            saved = pd.read_csv(cell / "validation_predictions.csv")
            keys = ["patient_id", "center", "channel_name", "label_nez"]
            merged = saved[keys + ["probability_nez"]].merge(replay, on=keys, how="outer", validate="one_to_one", suffixes=("_saved", "_replay"), indicator=True)
            if len(merged) != len(saved) or len(merged) != len(replay) or not (merged["_merge"] == "both").all():
                raise RuntimeError(f"Validation replay keys disagree: fold={fold}, model={name}")
            maximum = float(np.max(np.abs(merged.probability_nez_saved - merged.probability_nez_replay)))
            mismatches = int(np.sum((merged.probability_nez_saved >= threshold) != (merged.probability_nez_replay >= threshold)))
            rows.append({"fold": fold, "method": name, "channels": len(merged), "max_probability_difference": maximum, "threshold_prediction_mismatches": mismatches})
            if maximum > 1e-6 or mismatches:
                raise RuntimeError(f"Validation replay failed: fold={fold}, method={name}, diff={maximum}, mismatches={mismatches}")
            print(f"VALIDATION_REPLAY_PASS fold={fold} method={name} max_diff={maximum:.3g}", flush=True)
    result = {"status": "PASS", "cells": len(rows), "max_probability_difference": max(row["max_probability_difference"] for row in rows), "threshold_prediction_mismatches": 0, "outer_test_outcome_accessed": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": result, "cells": rows}, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
