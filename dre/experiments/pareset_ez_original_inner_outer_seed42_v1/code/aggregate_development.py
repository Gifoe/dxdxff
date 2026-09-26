"""Validation-only summary of locked PaReSet, repaired controls, and CDEL."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--stage-a", type=Path, required=True)
    parser.add_argument("--controls", type=Path, required=True)
    parser.add_argument("--tabular", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.source_root.resolve()))
    from epilens.evaluation import patient_metrics, select_threshold

    variants = ("base", "full", "no_reference", "uniform_reference", "no_pool", "bce_only")
    rows = []
    for fold in (1, 2, 3, 4, 5):
        for variant in variants:
            selection_path = args.stage_a / f"fold{fold}" / variant / "selection.json"
            if not selection_path.exists():
                raise RuntimeError(f"Incomplete stage A: {selection_path}")
            selected = json.loads(selection_path.read_text(encoding="utf-8"))
            if selected.get("test_evaluated") is not False:
                raise RuntimeError(f"Test-tainted stage A cell: {selection_path}")
            rows.append({"fold": fold, "method": f"PaReSet_{variant}",
                         "patient_macro_f1": selected["patient_macro_f1"],
                         "patient_ez_f1": selected["patient_ez_f1"],
                         "patient_balanced_accuracy": selected["patient_balanced_accuracy"],
                         "selected_epoch": selected["epoch"]})
        control_ledgers = {}
        for branch in ("prq", "bcr"):
            cell = args.controls / f"fold{fold}" / branch
            selection_path = cell / "selection.json"
            if not selection_path.exists():
                raise RuntimeError(f"Incomplete matched control: {selection_path}")
            selected = json.loads(selection_path.read_text(encoding="utf-8"))
            if selected.get("test_evaluated") is not False:
                raise RuntimeError(f"Test-tainted control cell: {selection_path}")
            rows.append({"fold": fold, "method": f"repaired_{branch.upper()}",
                         "patient_macro_f1": selected["patient_macro_f1"],
                         "patient_ez_f1": selected["patient_ez_f1"],
                         "patient_balanced_accuracy": selected["patient_balanced_accuracy"],
                         "selected_epoch": selected["epoch"]})
            ledger = pd.read_csv(cell / "validation_predictions.csv")
            control_ledgers[branch] = ledger[["patient_id", "center", "channel_name", "label_nez", "probability_nez"]]
        keys = ["patient_id", "center", "channel_name", "label_nez"]
        joined = control_ledgers["prq"].merge(control_ledgers["bcr"], on=keys, how="outer", validate="one_to_one", suffixes=("_prq", "_bcr"), indicator=True)
        if not (joined["_merge"] == "both").all() or len(joined) != len(control_ledgers["prq"]):
            raise RuntimeError(f"PRQ/BCR validation channel mismatch at fold {fold}")
        fused = joined[keys].copy()
        fused["probability_nez"] = 0.8 * joined["probability_nez_prq"] + 0.2 * joined["probability_nez_bcr"]
        chosen = select_threshold(fused)
        rows.append({"fold": fold, "method": "repaired_CDEL_0p8_0p2",
                     "patient_macro_f1": chosen.patient_macro_f1,
                     "patient_ez_f1": chosen.patient_ez_f1,
                     "patient_balanced_accuracy": chosen.patient_balanced_accuracy,
                     "selected_epoch": None})
        fused["threshold"] = chosen.threshold
        fused["predicted_nez"] = (fused["probability_nez"] >= chosen.threshold).astype(int)
        patient_metrics(fused, chosen.threshold).to_csv(args.output / f"fold{fold}_repaired_cdel_validation_patient_metrics.csv", index=False)
        for name in ("logistic_regression", "lightgbm"):
            selection_path = args.tabular / f"fold{fold}" / name / "selection.json"
            if not selection_path.exists():
                raise RuntimeError(f"Incomplete tabular control: {selection_path}")
            selected = json.loads(selection_path.read_text(encoding="utf-8"))
            if selected.get("test_evaluated") is not False:
                raise RuntimeError(f"Test-tainted tabular cell: {selection_path}")
            rows.append({"fold": fold, "method": f"tabular_{name}_patient_z",
                         "patient_macro_f1": selected["patient_macro_f1"],
                         "patient_ez_f1": selected["patient_ez_f1"],
                         "patient_balanced_accuracy": selected["patient_balanced_accuracy"],
                         "selected_epoch": None})
    output = pd.DataFrame(rows)
    if len(output) != 55:
        raise RuntimeError(f"Expected 55 method-fold rows, got {len(output)}")
    output.to_csv(args.output / "DEVELOPMENT_FOLD_RESULTS.csv", index=False)
    aggregate = output.groupby("method", as_index=False).agg(
        folds=("fold", "nunique"),
        patient_macro_f1_mean=("patient_macro_f1", "mean"),
        patient_macro_f1_std=("patient_macro_f1", "std"),
        patient_ez_f1_mean=("patient_ez_f1", "mean"),
        patient_balanced_accuracy_mean=("patient_balanced_accuracy", "mean"),
    )
    aggregate.to_csv(args.output / "DEVELOPMENT_AGGREGATE_RESULTS.csv", index=False)
    (args.output / "DEVELOPMENT_STATUS.json").write_text(json.dumps({
        "status": "COMPLETE", "rows": 55, "five_folds": True, "seed": 42,
        "test_evaluated": False,
        "claim": "inner-validation development only; not an independent outer-test comparison",
    }, indent=2), encoding="utf-8")
    print(aggregate.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
