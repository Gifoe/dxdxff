from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


RUN_ARGS_COLUMNS = [
    "config",
    "positive_label",
    "score_semantics",
    "drop_high_ez_fraction_lzu",
    "n_splits",
    "random_seed",
    "loss_mode",
    "use_negative_anchor_head",
    "negative_anchor_norm",
    "negative_anchor_distance_mode",
    "negative_anchor_center_gate",
    "use_hard_topk_loss",
    "hard_topk_loss_weight",
    "hard_topk_margin",
    "use_broad_ez_mil_loss",
    "broad_ez_mil_loss_weight",
    "broad_ez_core_frac",
    "broad_ez_min_fraction",
    "broad_ez_centers",
    "broad_ez_positive_bce_scale",
    "broad_ez_margin",
    "broad_ez_hard_neg_multiplier",
    "use_a9v8_lcbo",
    "latent_core_target_dir",
    "eval_score_fusion_gamma",
    "lambda_core_rank",
    "lambda_soft_mrr",
    "lambda_subset",
    "lambda_core_distill",
    "core_rank_margin",
    "soft_mrr_tau",
    "subset_eps",
    "lcbo_hard_neg_top_frac",
    "lcbo_min_core_mass",
    "use_two_expert_router",
    "two_expert_router_mode",
    "two_expert_gate_init_hup",
    "two_expert_gate_init_lzu",
    "two_expert_gate_init_multicenter",
    "two_expert_gate_init_pediatric",
    "two_expert_gate_l2",
    "two_expert_entropy_reg",
    "two_expert_static_preserve_pediatric",
    "two_expert_pediatric_preserve_weight",
    "two_expert_lambda_hup",
    "two_expert_lambda_lzu",
    "two_expert_lambda_multicenter",
    "two_expert_lambda_pediatric",
    "use_view_gated_fusion",
    "diffusion_center_mode",
    "diffusion_score_residual",
    "pretrain_masked_windows",
    "pretrain_epochs",
    "temporal_pooling",
    "temporal_pooling_top_p",
    "temporal_pooling_tau",
    "record_pooling",
    "record_pooling_top_p",
    "record_pooling_alpha",
    "use_physics_dynamics",
    "physics_state_features",
    "n_patient_rows",
    "n_unique_subjects",
    "centers_count_string",
]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"_read_error": str(exc)}


def _read_summary(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".json":
        return _read_json(path)
    df = pd.read_csv(path)
    return df.iloc[0].to_dict() if not df.empty else {}


def _patient_counts(run_dir: Path) -> tuple[int, int, str]:
    frames = [pd.read_csv(path) for path in sorted(run_dir.glob("test_patient_predictions_neuroez_v2_fold_*.csv"))]
    if not frames:
        return 0, 0, ""
    df = pd.concat(frames, ignore_index=True)
    center_counts = df["center"].value_counts().to_dict() if "center" in df.columns else {}
    centers = ",".join(f"{key}={center_counts[key]}" for key in sorted(center_counts))
    return int(len(df)), int(df["subject_id"].nunique()) if "subject_id" in df.columns else 0, centers


def summarize(root_dir: Path, baseline_pred_dir: Path | None, high_ez_threshold: float) -> dict[str, Path]:
    del baseline_pred_dir, high_ez_threshold
    root_dir = root_dir.resolve()
    summary_rows: list[dict[str, Any]] = []
    run_args_rows: list[dict[str, Any]] = []
    by_center_rows: list[pd.DataFrame] = []
    prevalence_rows: list[dict[str, Any]] = []
    failed_rows: list[dict[str, Any]] = []

    run_arg_paths = {path.parent: path for path in root_dir.rglob("run_args_b0_pruned.json")}
    summary_paths = list(root_dir.rglob("heldout_summary_neuroez_v3.json"))
    summary_dirs = {path.parent for path in summary_paths}

    for run_dir, args_path in sorted(run_arg_paths.items()):
        config = run_dir.name
        args = _read_json(args_path)
        patient_rows, unique_subjects, centers = _patient_counts(run_dir)
        if run_dir not in summary_dirs:
            failed_rows.append(
                {
                    "config": config,
                    "run_dir": str(run_dir),
                    "reason": "missing heldout_summary_neuroez_v3.json",
                    "positive_label": args.get("positive_label", ""),
                    "loss_mode": args.get("loss_mode", ""),
                }
            )
        row = {"config": config, "run_dir": str(run_dir)}
        for col in RUN_ARGS_COLUMNS:
            if col == "config":
                continue
            row[col] = args.get(col, "")
        row["n_patient_rows"] = patient_rows
        row["n_unique_subjects"] = unique_subjects
        row["centers_count_string"] = centers
        run_args_rows.append(row)

    for summary_path in sorted(summary_paths):
        run_dir = summary_path.parent
        config = run_dir.name
        row = _read_summary(summary_path)
        row["config"] = config
        row["run_dir"] = str(run_dir)
        summary_rows.append(row)
        patient_csvs = sorted(run_dir.glob("test_patient_predictions_neuroez_v2_fold_*.csv"))
        if patient_csvs:
            df = pd.concat([pd.read_csv(path) for path in patient_csvs], ignore_index=True)
            if "center" in df.columns:
                center_summary = df.groupby("center", dropna=False).mean(numeric_only=True).reset_index()
                center_summary.insert(0, "config", config)
                by_center_rows.append(center_summary)
        prevalence_path = run_dir / "audits" / "prevalence_lift_summary.json"
        if prevalence_path.exists():
            prev = _read_json(prevalence_path)
            prev["config"] = config
            prev["run_dir"] = str(run_dir)
            prevalence_rows.append(prev)

    outputs = {
        "grid": root_dir / "a8_a12_grid_summary.csv",
        "by_center": root_dir / "a8_a12_by_center_summary.csv",
        "prevalence": root_dir / "a8_a12_prevalence_lift_summary.csv",
        "run_args": root_dir / "a8_a12_run_args_audit.csv",
        "missing": root_dir / "a8_a12_missing_or_failed_runs.csv",
    }
    pd.DataFrame(summary_rows).to_csv(outputs["grid"], index=False)
    pd.concat(by_center_rows, ignore_index=True).to_csv(outputs["by_center"], index=False) if by_center_rows else pd.DataFrame().to_csv(outputs["by_center"], index=False)
    pd.DataFrame(prevalence_rows).to_csv(outputs["prevalence"], index=False)
    pd.DataFrame(run_args_rows, columns=["run_dir", *RUN_ARGS_COLUMNS]).to_csv(outputs["run_args"], index=False)
    pd.DataFrame(failed_rows).to_csv(outputs["missing"], index=False)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize A8-A12 NAR-EZ result directories.")
    parser.add_argument("--root_dir", required=True, type=Path)
    parser.add_argument("--baseline_pred_dir", default=None, type=Path)
    parser.add_argument("--high_ez_threshold", default=0.40, type=float)
    args = parser.parse_args()
    outputs = summarize(args.root_dir, args.baseline_pred_dir, args.high_ez_threshold)
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
