from __future__ import annotations

import csv
import pickle
import sys
from pathlib import Path
from typing import Any

from config import REPO_ROOT, VERSIONS
from utils import log, run_cmd, write_csv


def _resolve_n_splits(cache_path: Path, requested: int) -> int:
    with cache_path.open("rb") as fin:
        payload = pickle.load(fin)
    n_patients = len(payload.get("patient_index", {}))
    if n_patients < 2:
        raise RuntimeError(f"Need at least 2 HUP patients for patient-wise split, got {n_patients}.")
    return max(2, min(int(requested), int(n_patients)))


def write_manifest(args, caches: dict[str, Path]) -> None:
    notes = {
        "B0_HUP_Filtered": "Current NeuroEZ-B baseline restricted to HUP good/preview EDFs.",
        "M1_HUP_StrictInterNorm": "Strict ictal windows with patient-owned interictal baseline normalization.",
        "M2_HUP_InterPhysNode": "M1 plus lightweight physical node features appended in cache.",
        "M3_HUP_InterPhysGraph": "M2 plus precomputed interictal-delta adjacency; TFCCM causal adjacency skipped.",
    }
    rows = []
    for priority, version in enumerate(VERSIONS, start=1):
        rows.append(
            {
                "priority": priority,
                "experiment": f"{version}_seed{args.seed}",
                "family": version,
                "variant": version,
                "seed": args.seed,
                "threshold_tuning_metric": "patient_macro_weighted_f1",
                "early_stop_metric": "patient_macro_weighted_f1",
                "cache_path": str(caches[version]),
                "notes": notes[version],
            }
        )
    write_csv(Path(args.output_root) / "experiment_manifest.csv", rows)


def _base_train_args(args, *, cache_path: Path, output_dir: Path, n_splits: int) -> list[str]:
    return [
        sys.executable,
        str(REPO_ROOT / "run_neuroez_v2.py"),
        "--dataset_dir", str(args.dataset_dir),
        "--window_cache_path", str(cache_path),
        "--output_dir", str(output_dir),
        "--split_strategy", "5fold",
        "--n_splits", str(n_splits),
        "--random_seed", str(args.seed),
        "--positive_label", "nez",
        "--score_semantics", "nez_probability",
        "--rank_loss_weight", "0.0",
        "--count_loss_weight", "0.0",
        "--class_weight_mode", "none",
        "--threshold_tuning_metric", "patient_macro_weighted_f1",
        "--early_stop_metric", "patient_macro_weighted_f1",
        "--decision_rule", "threshold_nez",
        "--tune_decision_rule", "true",
        "--model_dim", str(args.model_dim),
        "--num_heads", str(args.num_heads),
        "--dropout", str(args.dropout),
        "--temporal_encoder", "mean",
        "--channel_layers", "1",
        "--seizure_pooling", "attention",
        "--use_adjacency_message_passing",
        "--use_channel_attention",
        "--no-use_raw_cnn",
        "--epochs", str(args.epochs),
        "--patience", str(args.patience),
        "--min_epochs_before_early_stop", str(args.min_epochs_before_early_stop),
        "--patient_batch_size", str(args.patient_batch_size),
        "--num_workers", str(args.num_workers),
        "--feature_num_workers", "1",
        "--device", str(args.device),
        "--log_interval", "1",
    ]


def _version_train_args(version: str) -> list[str]:
    if version == "B0_HUP_Filtered":
        return [
            "--feature_view", "self_comparison",
            "--self_compare_include_abs", "true",
            "--self_compare_include_delta", "true",
            "--self_compare_include_zdelta", "true",
            "--self_compare_include_ratio", "true",
            "--self_compare_include_channel_rank", "false",
            "--adjacency_view", "mixed_abs_delta",
            "--delta_adjacency_alpha", "0.5",
        ]
    return [
        "--feature_view", "interictal_comparison",
        "--include_inter_abs", "true",
        "--include_inter_delta", "true",
        "--include_inter_zdelta", "true",
        "--include_inter_ratio", "true",
        "--include_inter_percentile", "true",
        "--include_patient_relative", "true",
        "--include_patient_relative_z", "true",
        "--include_patient_relative_rank", "false",
        "--interictal_missing_indicators", "true",
        "--interictal_min_windows", "1",
        "--adjacency_view", "raw",
    ]


def run_training(args, caches: dict[str, Path]) -> None:
    write_manifest(args, caches)
    for version in VERSIONS:
        cache_path = caches[version]
        output_dir = Path(args.output_root) / f"{version}_seed{args.seed}"
        n_splits = _resolve_n_splits(cache_path, int(args.n_splits))
        log(f"Running {version}: n_splits={n_splits}, cache={cache_path}")
        cmd = [*_base_train_args(args, cache_path=cache_path, output_dir=output_dir, n_splits=n_splits), *_version_train_args(version)]
        run_cmd(cmd, cwd=REPO_ROOT, log_path=output_dir / "run.log")


def aggregate_outputs(args) -> None:
    output_root = Path(args.output_root)
    run_cmd(
        [sys.executable, str(REPO_ROOT / "aggregate_neuroez_results.py"), "--output_root", str(output_root)],
        cwd=REPO_ROOT,
        log_path=output_root / "aggregate.log",
    )
    metrics = [
        "patient_macro_accuracy",
        "patient_macro_balanced_accuracy",
        "patient_macro_macro_f1",
        "patient_macro_f1",
        "patient_macro_weighted_f1",
        "pooled_accuracy",
        "pooled_balanced_accuracy",
        "pooled_macro_f1",
        "pooled_weighted_f1",
    ]
    rows: list[dict[str, Any]] = []
    for summary_path in sorted(output_root.glob("*/heldout_summary_neuroez_v3.csv")):
        with summary_path.open("r", encoding="utf-8-sig", newline="") as fin:
            data = list(csv.DictReader(fin))
        if not data:
            continue
        row = {"experiment": summary_path.parent.name}
        for metric in metrics:
            row[metric] = data[0].get(metric, "")
        row["notes"] = ""
        rows.append(row)
    write_csv(output_root / "hup_primary_summary.csv", rows)
