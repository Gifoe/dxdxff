from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


EXPECTED_METRICS: dict[str, float] = {
    "patient_macro_f1": 0.6427135781,
    "patient_macro_ez_f1": 0.4705912943,
    "patient_macro_auprc_ez": 0.5184277652,
    "patient_macro_ez_mrr": 0.7099510182,
}
EXPECTED_COUNTS: dict[str, int] = {
    "n_patient_rows": 90,
    "n_unique_subjects": 90,
}

TRUEBEST_REQUIRED_ARGS: dict[str, Any] = {
    "use_negative_anchor_head": True,
    "negative_anchor_loss_weight": 0.02,
    "use_ez_ranking_loss": True,
    "ez_ranking_loss_weight": 0.05,
    "ez_ranking_margin": 0.05,
    "use_physics_dynamics": True,
    "loss_mode": "patient_balanced_bce",
    "patient_loss_weighting": "uniform",
    "positive_label": "ez",
    "drop_high_ez_fraction_lzu": False,
    "split_strategy": "5fold",
    "n_splits": 5,
    "random_seed": 42,
    "epochs": 40,
    "patience": 8,
}

LEGACY_DISABLED_MODULE_ARGS: dict[str, Any] = {
    "use_patient_context_reranker": False,
    "use_multi_seizure_consistency": False,
    "use_shaft_local_residual": False,
    "use_clinical_mixture_head": False,
    "freeze_a9v3_backbone": False,
    "base_aux_loss_weight": 0,
}

REQUIRED_OUTPUTS = [
    "heldout_summary_neuroez_v3.csv",
    "heldout_summary_neuroez_v3.json",
    "run_args_b0_pruned.json",
]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fin:
        data = json.load(fin)
    if not isinstance(data, dict):
        raise ValueError(f"Expected object JSON in {path}")
    return data


def _load_summary(run_dir: Path) -> dict[str, Any]:
    summary_csv = run_dir / "heldout_summary_neuroez_v3.csv"
    if summary_csv.exists():
        frame = pd.read_csv(summary_csv)
        if frame.empty:
            raise ValueError(f"Empty summary CSV: {summary_csv}")
        return frame.iloc[0].to_dict()
    summary_json = run_dir / "heldout_summary_neuroez_v3.json"
    if summary_json.exists():
        return _load_json(summary_json)
    raise FileNotFoundError(f"Missing heldout_summary_neuroez_v3.csv/json in {run_dir}")


def _close_enough(actual: Any, expected: Any, tolerance: float) -> bool:
    if isinstance(expected, bool):
        return bool(actual) is expected
    if isinstance(expected, int) and not isinstance(expected, bool):
        try:
            return int(actual) == expected
        except (TypeError, ValueError):
            return False
    if isinstance(expected, float):
        try:
            return abs(float(actual) - expected) <= tolerance
        except (TypeError, ValueError):
            return False
    return str(actual) == str(expected)


def _validate_args(
    run_args: dict[str, Any],
    failures: list[str],
    *,
    tolerance: float,
    allow_missing_disabled_flags: bool,
) -> None:
    for key, expected in TRUEBEST_REQUIRED_ARGS.items():
        if key not in run_args:
            failures.append(f"missing_required_arg:{key}")
            continue
        if not _close_enough(run_args[key], expected, tolerance):
            failures.append(f"arg_mismatch:{key}:actual={run_args[key]!r}:expected={expected!r}")

    for key, expected in LEGACY_DISABLED_MODULE_ARGS.items():
        if key not in run_args:
            if not allow_missing_disabled_flags:
                failures.append(f"missing_required_arg:{key}")
            continue
        if not _close_enough(run_args[key], expected, tolerance):
            failures.append(f"arg_mismatch:{key}:actual={run_args[key]!r}:expected={expected!r}")


def _validate_outputs(run_dir: Path, failures: list[str]) -> None:
    for filename in REQUIRED_OUTPUTS:
        if not (run_dir / filename).exists():
            failures.append(f"missing_output:{filename}")
    for fold_idx in range(1, 6):
        for stem in ["test_patient_predictions_neuroez_v2", "test_channel_predictions_neuroez_v2"]:
            filename = f"{stem}_fold_{fold_idx}.csv"
            if not (run_dir / filename).exists():
                failures.append(f"missing_output:{filename}")


def _read_patient_predictions(run_dir: Path) -> pd.DataFrame:
    frames = []
    for fold_idx in range(1, 6):
        path = run_dir / f"test_patient_predictions_neuroez_v2_fold_{fold_idx}.csv"
        frames.append(pd.read_csv(path))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_center_summary(patient_rows: pd.DataFrame) -> pd.DataFrame:
    if patient_rows.empty or "center" not in patient_rows.columns:
        return pd.DataFrame()
    metric_cols = [
        col
        for col in [
            "patient_macro_f1",
            "patient_balanced_accuracy",
            "patient_weighted_f1",
            "patient_nez_f1",
            "patient_ez_f1",
            "ez_mrr",
            "ez_recall_at_true_count",
        ]
        if col in patient_rows.columns
    ]
    grouped = patient_rows.groupby("center", dropna=False)
    rows: list[dict[str, Any]] = []
    for center, group in grouped:
        row: dict[str, Any] = {
            "center": center,
            "n_patient_rows": int(len(group)),
        }
        if "subject_id" in group.columns:
            row["n_unique_subjects"] = int(group["subject_id"].nunique())
        for col in metric_cols:
            row[f"mean_{col}"] = float(pd.to_numeric(group[col], errors="coerce").mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("center").reset_index(drop=True)


def audit_run_dir(
    run_dir: str | Path,
    *,
    tolerance: float = 1e-6,
    allow_missing_disabled_flags: bool = False,
    write_outputs: bool = True,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    failures: list[str] = []
    if run_dir.name.lower() == "logs":
        failures.append("invalid_run_dir:logs_directory")
    if not run_dir.exists():
        failures.append(f"missing_run_dir:{run_dir}")
        return {"status": "fail", "run_dir": str(run_dir), "failures": failures, "metrics": {}}

    summary = _load_summary(run_dir)
    metrics: dict[str, Any] = {}
    for key, expected in EXPECTED_METRICS.items():
        actual = summary.get(key)
        metrics[key] = float(actual) if actual is not None else None
        if actual is None:
            failures.append(f"missing_metric:{key}")
        elif abs(float(actual) - expected) > tolerance:
            failures.append(f"metric_mismatch:{key}:actual={float(actual):.12g}:expected={expected:.12g}")
    for key, expected in EXPECTED_COUNTS.items():
        actual = summary.get(key)
        metrics[key] = int(float(actual)) if actual is not None else None
        if actual is None:
            failures.append(f"missing_metric:{key}")
        elif int(float(actual)) != expected:
            failures.append(f"count_mismatch:{key}:actual={actual}:expected={expected}")

    args_path = run_dir / "run_args_b0_pruned.json"
    if args_path.exists():
        run_args = _load_json(args_path)
        _validate_args(
            run_args,
            failures,
            tolerance=tolerance,
            allow_missing_disabled_flags=allow_missing_disabled_flags,
        )
    else:
        failures.append("missing_output:run_args_b0_pruned.json")
        run_args = {}

    _validate_outputs(run_dir, failures)

    center_summary_path = run_dir / "heldout_center_summary_neuroez_v3.csv"
    try:
        patient_rows = _read_patient_predictions(run_dir)
        center_summary = build_center_summary(patient_rows)
        if write_outputs and not center_summary.empty:
            center_summary.to_csv(center_summary_path, index=False)
            center_summary.to_csv(run_dir / "v3_truebest_clean_center_summary.csv", index=False)
    except Exception as exc:  # pragma: no cover - reported as audit failure.
        failures.append(f"center_summary_error:{type(exc).__name__}:{exc}")

    result = {
        "status": "pass" if not failures else "fail",
        "run_dir": str(run_dir),
        "metrics": metrics,
        "expected_metrics": EXPECTED_METRICS,
        "expected_counts": EXPECTED_COUNTS,
        "required_args": TRUEBEST_REQUIRED_ARGS,
        "legacy_disabled_module_args": LEGACY_DISABLED_MODULE_ARGS,
        "allow_missing_disabled_flags": allow_missing_disabled_flags,
        "failures": failures,
        "center_summary_csv": str(center_summary_path),
    }
    if write_outputs:
        with (run_dir / "v3_truebest_clean_audit.json").open("w", encoding="utf-8") as fout:
            json.dump(result, fout, indent=2, ensure_ascii=False, sort_keys=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit one explicit A9v3 TrueBest Clean run directory.")
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--allow_missing_disabled_flags", action="store_true")
    parser.add_argument("--no_write_outputs", action="store_true")
    args = parser.parse_args()

    result = audit_run_dir(
        args.run_dir,
        tolerance=args.tolerance,
        allow_missing_disabled_flags=args.allow_missing_disabled_flags,
        write_outputs=not args.no_write_outputs,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
