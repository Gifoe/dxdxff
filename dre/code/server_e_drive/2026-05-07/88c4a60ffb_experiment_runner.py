from __future__ import annotations

import csv
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .baseline_rankers import run_all_centers_baselines
from .config import BNPDGSConfig
from .data_audit import audit_patient_records, filter_trainable_patients, write_audit_csv, write_audit_summary
from .data_interface import load_patient_records
from .feature_cache import extract_features_for_center, load_cached_samples, write_all_centers_index
from .logging_utils import log
from .train import build_sample_splits, run_training_from_samples


CENTERS = ["lzu", "hup", "multicenter"]
RUN_PLAN = [
    {"name": "lzu_only", "center_filter": ["lzu"]},
    {"name": "hup_only", "center_filter": ["hup"]},
    {"name": "multicenter_only", "center_filter": ["multicenter"]},
    {"name": "all_centers", "center_filter": ["lzu", "hup", "multicenter"]},
]


def run_v1_experiment_suite(args: Any | None = None) -> dict[str, Any]:
    cfg = BNPDGSConfig.from_args(args)
    root = Path(cfg.output_dir)
    _make_layout(root)
    _write_fixed_config(cfg, root / "configs" / "v1_fixed_config.yaml")
    _write_json({"run_plan": _expanded_run_plan(root), "config": cfg.to_dict()}, root / "configs" / "run_manifest.json")

    center_records = {}
    audit_rows_by_center = {}
    manifests = {}
    for center_id in CENTERS:
        log(f"Step: loading center={center_id}")
        records = load_patient_records(_args_for_center(args, center_id))
        center_records[center_id] = records
        audit_rows = audit_patient_records(
            records,
            center_id=center_id,
            output_csv=root / "data_audit" / f"{center_id}_patient_label_audit.csv",
        )
        audit_rows_by_center[center_id] = audit_rows
        trainable_records = filter_trainable_patients(records, audit_rows)
        manifests[center_id] = extract_features_for_center(
            center_id=center_id,
            records=trainable_records,
            config=cfg,
            output_cache_dir=root / "feature_cache" / center_id,
            force_rebuild_cache=bool(cfg.force_rebuild_cache),
        )

    all_rows = [row for rows in audit_rows_by_center.values() for row in rows]
    write_audit_csv(all_rows, root / "data_audit" / "all_centers_patient_label_audit.csv")
    audit_summary = write_audit_summary(
        {**audit_rows_by_center, "all_centers": all_rows},
        root / "data_audit" / "data_audit_summary.json",
    )
    all_index = write_all_centers_index(root / "feature_cache", manifests)

    run_summaries = {}
    splits_by_run = {}
    for plan in RUN_PLAN:
        run_name = plan["name"]
        centers = plan["center_filter"]
        cache_sources = [root / "feature_cache" / center for center in centers]
        samples = load_cached_samples(cache_sources)
        splits = build_sample_splits(samples, seed=cfg.seed)
        splits_by_run[run_name] = splits
        summary = run_training_from_samples(
            run_name=run_name,
            samples=samples,
            cfg=cfg.replace(output_dir=root / "runs" / run_name),
            output_dir=root / "runs" / run_name,
            n_patients_loaded=sum(manifests.get(center, {}).get("n_patients_raw", 0) for center in centers),
            splits=splits,
        )
        run_summaries[run_name] = summary

    all_samples = load_cached_samples([root / "feature_cache" / center for center in CENTERS])
    baseline_summary = {}
    if splits_by_run.get("all_centers"):
        baseline_summary = run_all_centers_baselines(
            samples=all_samples,
            splits=splits_by_run["all_centers"],
            output_dir=root / "baselines" / "all_centers",
            seed=cfg.seed,
        )
    else:
        _write_json(
            {"status": "skipped", "reason": "all_centers_insufficient_patients"},
            root / "baselines" / "all_centers" / "baseline_summary.json",
        )
        _write_csv([], root / "baselines" / "all_centers" / "baseline_summary.csv")

    final = write_final_summary(root, run_summaries, baseline_summary, audit_summary, all_index)
    return final


def write_final_summary(
    root: Path,
    run_summaries: dict[str, dict[str, Any]],
    baseline_summary: dict[str, Any],
    audit_summary: dict[str, Any],
    all_index: dict[str, Any],
) -> dict[str, Any]:
    final_dir = root / "final_summary"
    final_dir.mkdir(parents=True, exist_ok=True)
    center_rows = []
    for run_name in ["lzu_only", "hup_only", "multicenter_only", "all_centers"]:
        summary = run_summaries.get(run_name, {})
        center_rows.append(
            {
                "run_name": run_name,
                "n_patients_loaded": summary.get("n_patients_loaded", 0),
                "n_dynamic_patients": summary.get("n_dynamic_patients", 0),
                "n_folds": summary.get("n_folds", 0),
                "mean_macro_AUC": summary.get("mean_macro_AUC"),
                "mean_macro_AUC_PR": summary.get("mean_macro_AUC_PR"),
                "mean_macro_F1": summary.get("mean_macro_F1"),
                "mean_macro_PREC": summary.get("mean_macro_PREC"),
                "mean_macro_REC": summary.get("mean_macro_REC"),
                "mean_macro_TOPK_RECALL": summary.get("mean_macro_TOPK_RECALL"),
                "mean_macro_TOP1_HIT": summary.get("mean_macro_TOP1_HIT"),
                "mean_macro_TOP3_HIT": summary.get("mean_macro_TOP3_HIT"),
                "mean_macro_COUNT_BIAS": summary.get("mean_macro_COUNT_BIAS"),
                "mean_macro_ABS_COUNT_BIAS_RATIO": summary.get("mean_macro_ABS_COUNT_BIAS_RATIO"),
                "mean_selection_score": summary.get("mean_selection_score"),
                "status": summary.get("status"),
            }
        )
    _write_csv(center_rows, final_dir / "center_comparison.csv")
    _write_json(center_rows, final_dir / "center_comparison.json")

    model_vs_baselines = []
    all_model = run_summaries.get("all_centers", {})
    model_vs_baselines.append(
        {
            "name": "bn_pdgs_v1_fixed",
            "selection_mode": "val_calibrated_pred_count_topk",
            "macro_AUC": all_model.get("mean_macro_AUC"),
            "macro_AUC_PR": all_model.get("mean_macro_AUC_PR"),
            "macro_F1": all_model.get("mean_macro_F1"),
            "macro_TOPK_RECALL": all_model.get("mean_macro_TOPK_RECALL"),
            "selection_score": all_model.get("mean_selection_score"),
        }
    )
    for row in baseline_summary.get("summary_rows", []):
        if row.get("selection_mode") == "validation_calibrated_topk":
            model_vs_baselines.append({"name": row.get("baseline_name"), **row})
    _write_csv(model_vs_baselines, final_dir / "all_centers_model_vs_baselines.csv")

    final_report = {
        "data_audit": audit_summary,
        "all_centers_cache_index": all_index,
        "run_summaries": run_summaries,
        "baseline_summary": baseline_summary,
        "best_model": _best_model(run_summaries),
        "failure_diagnostics": _failure_diagnostics(run_summaries, baseline_summary),
    }
    _write_json(final_report, final_dir / "final_report.json")
    return final_report


def _expanded_run_plan(root: Path) -> list[dict[str, Any]]:
    plans = []
    for plan in RUN_PLAN:
        centers = plan["center_filter"]
        plans.append(
            {
                **plan,
                "cache_sources": [str(root / "feature_cache" / center) for center in centers],
                "output_dir": str(root / "runs" / plan["name"]),
            }
        )
    return plans


def _args_for_center(args: Any, center_id: str) -> dict[str, Any]:
    payload = _object_to_dict(args)
    payload["datasets"] = (center_id,)
    return payload


def _object_to_dict(obj: Any) -> dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return dict(obj)
    if is_dataclass(obj):
        return asdict(obj)
    return dict(vars(obj))


def _make_layout(root: Path) -> None:
    dirs = [
        "configs",
        "data_audit",
        "feature_cache/lzu",
        "feature_cache/hup",
        "feature_cache/multicenter",
        "feature_cache/all_centers",
        "baselines/all_centers",
        "final_summary",
    ]
    for run in ["lzu_only", "hup_only", "multicenter_only", "all_centers"]:
        dirs.extend(
            [
                f"runs/{run}/logs",
                f"runs/{run}/checkpoints",
                f"runs/{run}/predictions",
                f"runs/{run}/reports",
                f"runs/{run}/folds",
            ]
        )
    for directory in dirs:
        (root / directory).mkdir(parents=True, exist_ok=True)


def _write_fixed_config(cfg: BNPDGSConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fout:
        for key, value in cfg.to_dict().items():
            fout.write(f"{key}: {json.dumps(value, ensure_ascii=False)}\n")


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with open(path, "w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fout:
        json.dump(payload, fout, ensure_ascii=False, indent=2, default=_json_default)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _best_model(run_summaries: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    candidates = []
    for run_name, summary in run_summaries.items():
        score = summary.get("mean_selection_score")
        if score is not None and np.isfinite(float(score)):
            candidates.append((float(score), run_name))
    if not candidates:
        return None
    score, run_name = max(candidates)
    return {"run_name": run_name, "mean_selection_score": score}


def _failure_diagnostics(
    run_summaries: dict[str, dict[str, Any]],
    baseline_summary: dict[str, Any],
) -> list[str]:
    issues = []
    for run_name, summary in run_summaries.items():
        if summary.get("status") == "insufficient_patients":
            issues.append(f"{run_name}: insufficient_patients")
        if float(summary.get("mean_macro_AUC_PR", 0.0) or 0.0) <= 0.05 and summary.get("status") == "completed":
            issues.append(f"{run_name}: very_low_auc_pr")
    if not baseline_summary:
        issues.append("all_centers baselines skipped or empty")
    return issues


__all__ = ["CENTERS", "RUN_PLAN", "run_v1_experiment_suite", "write_final_summary"]
