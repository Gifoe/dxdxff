"""Create an outcome-blind, hash-locked seed-42 outer-test protocol."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def status_complete(path: Path) -> None:
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("status") != "COMPLETE" or result.get("test_evaluated") is not False:
        raise RuntimeError(f"Incomplete or test-tainted development artifact: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--supplement-root", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".sha256").exists():
        raise RuntimeError("Outer lock already exists; refusing to overwrite")
    sys.path.insert(0, str(args.supplement_root.resolve()))
    from epilens.evaluation import select_threshold

    status_complete(args.runtime_root / "stage_a_seed42" / "DEVELOPMENT_STATUS.json")
    status_complete(args.runtime_root / "matched_controls_seed42" / "CONTROL_STATUS.json")
    status_complete(args.runtime_root / "tabular_seed42" / "TABULAR_STATUS.json")
    status_complete(args.runtime_root / "development_summary" / "AGGREGATION_STATUS.json")
    replay = json.loads((args.runtime_root / "VALIDATION_REPLAY.json").read_text(encoding="utf-8"))["summary"]
    if replay.get("status") != "PASS" or replay.get("cells") != 20 or replay.get("outer_test_outcome_accessed") is not False:
        raise RuntimeError("Validation replay did not pass before outer lock")
    files = {
        "data": args.data,
        "manifest": args.manifest,
        "pareset_model": args.code_root / "source" / "pareset_ez.py",
        "supplement_models": args.supplement_root / "epilens" / "models.py",
        "supplement_features": args.supplement_root / "epilens" / "features.py",
        "supplement_evaluation": args.supplement_root / "epilens" / "evaluation.py",
        "control_repair": args.code_root / "train_repaired_control.py",
        "outer_evaluator": args.code_root / "evaluate_outer_seed42.py",
        "lock_preparer": args.code_root / "prepare_outer_lock.py",
        "validation_replay": args.runtime_root / "VALIDATION_REPLAY.json",
    }
    thresholds = {}
    checkpoints = {}
    for fold in range(1, 6):
        key = f"fold{fold}"
        thresholds[key] = {}
        checkpoints[key] = {}
        for name in ("full", "base"):
            cell = args.runtime_root / "stage_a_seed42" / key / name
            selection = json.loads((cell / "selection.json").read_text(encoding="utf-8"))
            if selection.get("test_evaluated") is not False:
                raise RuntimeError(f"Test-tainted stage A cell {cell}")
            thresholds[key][name] = float(selection["threshold"])
            checkpoints[key][name] = {"path": str(cell / "best.pt"), "sha256": sha256(cell / "best.pt"), "selected_epoch": int(selection["epoch"])}
        validation = {}
        for name in ("prq", "bcr"):
            cell = args.runtime_root / "matched_controls_seed42" / key / name
            selection = json.loads((cell / "selection.json").read_text(encoding="utf-8"))
            if selection.get("test_evaluated") is not False:
                raise RuntimeError(f"Test-tainted control cell {cell}")
            thresholds[key][name] = float(selection["threshold"])
            checkpoints[key][name] = {"path": str(cell / "best.pt"), "sha256": sha256(cell / "best.pt"), "selected_epoch": int(selection["epoch"])}
            validation[name] = pd.read_csv(cell / "validation_predictions.csv")
        keys = ["patient_id", "center", "channel_name", "label_nez"]
        joined = validation["prq"].merge(validation["bcr"], on=keys, how="outer", validate="one_to_one", suffixes=("_prq", "_bcr"), indicator=True)
        if len(joined) != len(validation["prq"]) or len(joined) != len(validation["bcr"]) or not (joined["_merge"] == "both").all():
            raise RuntimeError(f"Control validation ledgers disagree in fold {fold}")
        fusion = joined[keys].copy()
        fusion["probability_nez"] = 0.8 * joined["probability_nez_prq"] + 0.2 * joined["probability_nez_bcr"]
        thresholds[key]["cdel"] = float(select_threshold(fusion).threshold)

    lock = {
        "lock_created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "FROZEN_BEFORE_OUTER_ACCESS",
        "primary_model": "PaReSet_full_default_18474_parameters",
        "comparators": ["PaReSet_base", "supplementary_repaired_PRQ", "supplementary_repaired_BCR", "supplementary_repaired_CDEL_0p8_0p2"],
        "scope": "80-patient five disjoint outer folds, seed 42 only",
        "folds": [1, 2, 3, 4, 5], "seed": 42,
        "model_selection": "each checkpoint and threshold selected on its frozen inner-validation partition only",
        "outer_test_used_for_tuning": False,
        "after_test": "report all outcomes and do not modify architecture, training, thresholds or method roster",
        "bootstrap": {"unit": "patient", "paired": True, "resamples": 10000, "seed": 4201, "interval": "percentile_95"},
        "paths": {name: str(path) for name, path in files.items()},
        "sha256": {name: sha256(path) for name, path in files.items()},
        "thresholds": thresholds,
        "checkpoints": checkpoints,
        "note": "The repaired 36-D controls are not literal historical 28-D paper checkpoints.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(lock, indent=2), encoding="utf-8")
    checksum = sha256(args.output)
    args.output.with_suffix(".sha256").write_text(checksum + "\n", encoding="ascii")
    print(json.dumps({"lock": str(args.output), "sha256": checksum, "folds": 5, "seed": 42, "test_outcome_accessed": False}), flush=True)


if __name__ == "__main__":
    main()
