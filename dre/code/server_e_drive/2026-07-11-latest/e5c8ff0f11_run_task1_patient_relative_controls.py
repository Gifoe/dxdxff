"""Run stronger patient-relative controls on the frozen 80-patient protocol."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from outcome_hifos.cache_schema import file_sha256, load_cache_contract
from task1_baselines.cache_io import task1_feature_records
from task1_baselines.feature_aggregation import build_channel_feature_table
from task1_baselines.fold_protocol import fold_manifest_hash, load_task1_sensitivity_protocol
from task1_baselines.patient_controls import CONTROL_MODELS, run_patient_relative_control_oof
from task1_baselines.reporting import summarize_task1


def _csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-cache", required=True)
    parser.add_argument("--cohort-manifest", required=True)
    parser.add_argument("--fold-manifest", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reference-ledger", required=True)
    parser.add_argument("--models", default="deepsets_bce,patient_z_mlp,patient_rank_mlp,patient_z_logistic,patient_z_rbf_svm")
    parser.add_argument("--seeds", default="42,52,62")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args(argv)
    models = _csv(args.models)
    unknown = set(models) - CONTROL_MODELS
    if unknown:
        raise ValueError(f"Unknown controls: {sorted(unknown)}")
    output = Path(args.output_dir)
    for directory in ("audit", "configs", "oof_ledgers", "checkpoints", "training", "metrics", "reports", "comparison"):
        (output / directory).mkdir(parents=True, exist_ok=True)
    folds, splits = load_task1_sensitivity_protocol(args.cohort_manifest, args.fold_manifest, args.split_manifest, expected_subjects=80)
    cache = load_cache_contract(args.feature_cache)
    table, feature_manifest = build_channel_feature_table(task1_feature_records(cache, set(folds["subject_id"])), profile="p2_matched_simple")
    provenance = {
        "protocol": "sensitivity80_p2_q10_fixed_validation",
        "label_semantics": "NEZ=1,EZ=0",
        "feature_cache": str(Path(args.feature_cache).resolve()),
        "feature_cache_sha256": file_sha256(args.feature_cache),
        "fold_manifest_hash": fold_manifest_hash(folds),
        "models": models,
        "seeds": [int(seed) for seed in _csv(args.seeds)],
        "epochs": args.epochs,
        "patience": args.patience,
        "test_labels_used_for_selection": False,
    }
    (output / "configs" / "feature_manifest.json").write_text(json.dumps(feature_manifest, indent=2, sort_keys=True), encoding="utf-8")
    (output / "configs" / "protocol.json").write_text(json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8")
    all_oof = []
    for model in models:
        for seed_text in _csv(args.seeds):
            seed = int(seed_text)
            path = output / "oof_ledgers" / model / f"seed_{seed}_channel_oof.csv"
            if args.skip_existing and path.exists():
                all_oof.append(__import__("pandas").read_csv(path))
                print(f"[PatientControls] skip {model} seed={seed}", flush=True)
                continue
            print(f"[PatientControls] start {model} seed={seed}", flush=True)
            result = run_patient_relative_control_oof(
                table, folds, splits, model_name=model, seed=seed, device=args.device,
                max_epochs=args.epochs, patience=args.patience,
                checkpoint_root=output / "checkpoints" / model / f"seed_{seed}",
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            result.oof.to_csv(path, index=False)
            result.training_audit.to_csv(output / "training" / f"{model}_seed_{seed}.csv", index=False)
            all_oof.append(result.oof)
            print(f"[PatientControls] complete {model} seed={seed} rows={len(result.oof)}", flush=True)
    import pandas as pd
    combined = pd.concat(all_oof, ignore_index=True)
    combined.to_csv(output / "oof_ledgers" / "all_patient_relative_controls.csv", index=False)
    summarize_task1(combined, output, v3_reference=pd.read_csv(args.reference_ledger))
    print(json.dumps({"status": "passed", "output": str(output), "config_hash": hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
