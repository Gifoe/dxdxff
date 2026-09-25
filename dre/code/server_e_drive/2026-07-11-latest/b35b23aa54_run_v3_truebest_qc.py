from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from run_neuroez_c import build_parser


VALUE_BOOL_KEYS = {"drop_high_ez_fraction_lzu", "use_patient_relative_z"}

V3_TRUEBEST_REQUIRED: dict[str, Any] = {
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
    "base_aux_loss_weight": 0.0,
}

QC_CONFIGS: dict[str, dict[str, Any]] = {
    "V3_TrueBest_Reproduce": {
        "use_edf_quality_weighting": False,
        "edf_quality_field": "",
        "review_weight": 1.0,
        "poor_weight": 1.0,
        "quality_weight_loss": False,
        "quality_weight_record_aggregation": "none",
        "quality_curriculum_epochs": 0,
    },
    "V3_QC_review_w070": {
        "use_edf_quality_weighting": True,
        "review_weight": 0.70,
        "poor_weight": 0.0,
        "quality_weight_loss": True,
        "quality_weight_record_aggregation": "weighted_mean_std",
        "quality_curriculum_epochs": 0,
    },
    "V3_QC_review_w050": {
        "use_edf_quality_weighting": True,
        "review_weight": 0.50,
        "poor_weight": 0.0,
        "quality_weight_loss": True,
        "quality_weight_record_aggregation": "weighted_mean_std",
        "quality_curriculum_epochs": 0,
    },
    "V3_QC_review_w030": {
        "use_edf_quality_weighting": True,
        "review_weight": 0.30,
        "poor_weight": 0.0,
        "quality_weight_loss": True,
        "quality_weight_record_aggregation": "weighted_mean_std",
        "quality_curriculum_epochs": 0,
    },
    "V3_QC_curriculum_good15_review_w050": {
        "use_edf_quality_weighting": True,
        "review_weight": 0.50,
        "poor_weight": 0.0,
        "quality_weight_loss": True,
        "quality_weight_record_aggregation": "weighted_mean_std",
        "quality_curriculum_epochs": 15,
    },
    "V3_QC_good_only_train_all_eval": {
        "use_edf_quality_weighting": True,
        "review_weight": 0.0,
        "poor_weight": 0.0,
        "quality_weight_loss": True,
        "quality_weight_record_aggregation": "train_good_only_all_eval",
        "quality_curriculum_epochs": 0,
    },
}

V3_TARGET_METRICS: dict[str, float] = {
    "patient_macro_f1": 0.6427135781,
    "patient_macro_ez_f1": 0.4705912943,
    "patient_macro_auprc_ez": 0.5184277652,
    "patient_macro_ez_mrr": 0.7099510182,
}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON file: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def _coerce_compare(value: Any, expected: Any) -> Any:
    if isinstance(expected, bool):
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y"}
        return bool(value)
    if isinstance(expected, int) and not isinstance(expected, bool):
        return int(value)
    if isinstance(expected, float):
        return float(value)
    if expected is None:
        return value
    return str(value)


def assert_truebest_base(base_args: dict[str, Any]) -> None:
    mismatches: dict[str, dict[str, Any]] = {}
    for key, expected in V3_TRUEBEST_REQUIRED.items():
        if key not in base_args:
            mismatches[key] = {"expected": expected, "actual": "<missing>"}
            continue
        actual = _coerce_compare(base_args[key], expected)
        ok = math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12) if isinstance(expected, float) else actual == expected
        if not ok:
            mismatches[key] = {"expected": expected, "actual": base_args[key]}
    if mismatches:
        raise ValueError(
            "Base run_args_b0_pruned.json is not the A9v3 TrueBest contract. "
            f"Refusing to override drifted base args: {json.dumps(mismatches, sort_keys=True)}"
        )


def known_cli_keys() -> set[str]:
    parser = build_parser()
    return {action.dest for action in parser._actions if action.dest and action.dest != "help"}


def stringify_value(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (list, tuple)):
        return ",".join(str(item) for item in value)
    return str(value)


def add_cli_arg(cli: list[str], name: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, str) and not value.strip():
        return
    flag = f"--{name}"
    if isinstance(value, bool):
        if name in VALUE_BOOL_KEYS:
            cli.extend([flag, stringify_value(value)])
        elif value:
            cli.append(flag)
        else:
            cli.append(f"--no-{name}")
        return
    cli.extend([flag, stringify_value(value)])


def build_effective_args(
    *,
    config_name: str,
    base_run_args_path: Path,
    output_dir: Path,
    cache_path: Path,
    edf_quality_field: str = "",
) -> tuple[dict[str, Any], dict[str, Any]]:
    if config_name not in QC_CONFIGS:
        raise KeyError(f"Unknown V3/QC config {config_name!r}. Available: {sorted(QC_CONFIGS)}")
    base_args = _read_json(base_run_args_path)
    assert_truebest_base(base_args)

    config = dict(QC_CONFIGS[config_name])
    if bool(config["use_edf_quality_weighting"]):
        if not edf_quality_field.strip():
            raise ValueError(f"{config_name} requires --edf_quality_field after auditing cache metadata.")
        config["edf_quality_field"] = edf_quality_field.strip()

    operational_overrides = {
        "config_name": config_name,
        "output_dir": str(output_dir),
        "window_cache_path": str(cache_path),
    }
    effective = dict(base_args)
    effective.update(operational_overrides)
    effective.update(LEGACY_DISABLED_MODULE_ARGS)
    effective.update(config)

    known = known_cli_keys()
    ignored_unknown = sorted(key for key in effective if key not in known)
    filtered = {key: value for key, value in effective.items() if key in known}
    diff = {
        "base_args_path": str(base_run_args_path),
        "base_args_inherited_from_a9v3": {
            key: value
            for key, value in base_args.items()
            if key not in operational_overrides and key not in LEGACY_DISABLED_MODULE_ARGS and key not in config
        },
        "truebest_contract_validated_not_overridden": V3_TRUEBEST_REQUIRED,
        "overridden_for_v3_qc_runtime": operational_overrides,
        "qc_config_args": config,
        "legacy_disabled_module_args": LEGACY_DISABLED_MODULE_ARGS,
        "ignored_unknown_base_or_effective_args": ignored_unknown,
    }
    return filtered, diff


def build_command(python_exe: str, run_args: dict[str, Any]) -> list[str]:
    cli = [python_exe, str(Path("run_neuroez_c.py"))]
    for key in sorted(run_args):
        add_cli_arg(cli, key, run_args[key])
    return cli


def read_summary(output_dir: Path) -> dict[str, Any]:
    csv_path = output_dir / "heldout_summary_neuroez_v3.csv"
    json_path = output_dir / "heldout_summary_neuroez_v3.json"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        if df.empty:
            raise ValueError(f"Empty summary CSV: {csv_path}")
        return df.iloc[0].to_dict()
    if json_path.exists():
        data = _read_json(json_path)
        return data
    raise FileNotFoundError(f"Missing heldout_summary_neuroez_v3.csv/json in {output_dir}")


def validate_outputs(config_name: str, output_dir: Path, *, metric_tolerance: float) -> dict[str, Any]:
    summary = read_summary(output_dir)
    checks: dict[str, Any] = {"config_name": config_name, "output_dir": str(output_dir)}
    expected = {
        "n_patient_rows": 90,
        "n_unique_subjects": 90,
        "positive_label": "ez",
        "score_semantics": "ez_probability",
        "drop_high_ez_fraction_lzu": False,
    }
    failures: dict[str, dict[str, Any]] = {}
    for key, wanted in expected.items():
        actual = summary.get(key)
        actual_cmp = _coerce_compare(actual, wanted)
        if actual_cmp != wanted:
            failures[key] = {"expected": wanted, "actual": actual}
    if config_name == "V3_TrueBest_Reproduce":
        metric_failures: dict[str, dict[str, Any]] = {}
        for key, wanted in V3_TARGET_METRICS.items():
            actual = float(summary.get(key, float("nan")))
            delta = abs(actual - wanted)
            if not math.isfinite(actual) or delta > metric_tolerance:
                metric_failures[key] = {"expected": wanted, "actual": actual, "abs_delta": delta}
        if metric_failures:
            failures["truebest_metrics"] = metric_failures
    if bool(QC_CONFIGS[config_name].get("use_edf_quality_weighting")):
        missing = [
            name
            for name in (
                "quality_by_center_summary.csv",
                "quality_by_patient_summary.csv",
                "quality_by_fold_split_summary.csv",
                "quality_audit_summary.json",
            )
            if not (output_dir / name).exists()
        ]
        if missing:
            failures["quality_audit_outputs"] = {"expected_present": missing, "actual": "missing"}
    checks["summary"] = summary
    checks["failures"] = failures
    checks["passed"] = not failures
    _write_json(output_dir / "v3_truebest_qc_validation.json", checks)
    if failures:
        raise AssertionError(f"{config_name} validation failed: {json.dumps(failures, sort_keys=True)}")
    return checks


def run_config(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.validate_only:
        validate_outputs(args.config_name, output_dir, metric_tolerance=float(args.metric_tolerance))
        return 0
    cache_path = Path(args.cache_path)
    if not cache_path.exists():
        raise FileNotFoundError(f"Missing A9v3 TrueBest/QC all90 window cache: {cache_path}")
    base_path = Path(args.base_run_args_path)
    run_args, diff = build_effective_args(
        config_name=args.config_name,
        base_run_args_path=base_path,
        output_dir=output_dir,
        cache_path=cache_path,
        edf_quality_field=args.edf_quality_field,
    )
    _write_json(output_dir / "v3_truebest_qc_base_args_diff.json", diff)
    _write_json(output_dir / "v3_truebest_qc_effective_args.json", run_args)
    command = build_command(args.python_exe, run_args)
    _write_json(output_dir / "v3_truebest_qc_launch_command.json", {"command": command})
    if args.dry_run:
        print(json.dumps({"dry_run": True, "command": command}, indent=2))
        return 0
    completed = subprocess.run(command, cwd=REPO_ROOT)
    if completed.returncode != 0:
        return int(completed.returncode)
    validate_outputs(args.config_name, output_dir, metric_tolerance=float(args.metric_tolerance))
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one A9v3 TrueBest reproduce or QC config without A9v14 modules.")
    parser.add_argument("--config_name", required=True, choices=sorted(QC_CONFIGS))
    parser.add_argument("--base_run_args_path", type=Path, required=True)
    parser.add_argument("--cache_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--edf_quality_field", type=str, default="")
    parser.add_argument("--python_exe", type=str, default=sys.executable)
    parser.add_argument("--metric_tolerance", type=float, default=0.003)
    parser.add_argument("--dry_run", action="store_true", default=False)
    parser.add_argument("--validate_only", action="store_true", default=False)
    raise SystemExit(run_config(parser.parse_args()))


if __name__ == "__main__":
    main()
