from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from run_neuroez_c import build_parser
from scripts.a9v14_candidate_configs import (
    DEFAULT_A9V14_ARGS,
    build_run_args,
    get_candidate_config,
    list_candidate_configs,
)
from scripts.source_propagation_diagnostic import _resolve_config_dir, run_source_propagation_diagnostic


VALUE_BOOL_KEYS = {"drop_high_ez_fraction_lzu", "use_patient_relative_z"}

DISABLED_INTENTIONALLY: dict[str, bool] = {
    "use_a9v8_lcbo": False,
    "use_teacher_anchor_eval": False,
    "use_two_expert_router": False,
    "use_feature_separated_two_expert": False,
    "use_broad_ez_mil_loss": False,
    "use_negative_anchor_head": False,
}


def _read_json(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing JSON file: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {p}")
    return data


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def _known_cli_keys() -> set[str]:
    parser = build_parser()
    return {
        action.dest
        for action in parser._actions
        if action.dest and action.dest != "help"
    }


def _stringify_value(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (list, tuple)):
        return ",".join(str(item) for item in value)
    return str(value)


def _add_cli_arg(cli: list[str], name: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, str) and not value.strip():
        return
    flag = f"--{name}"
    if isinstance(value, bool):
        if name in VALUE_BOOL_KEYS:
            cli.extend([flag, _stringify_value(value)])
        elif value:
            cli.append(flag)
        else:
            cli.append(f"--no-{name}")
        return
    cli.extend([flag, _stringify_value(value)])


def build_effective_args(
    *,
    config_name: str,
    output_dir: Path,
    cache_path: Path,
    base_run_args_path: Path | None = None,
    stage: str = "full",
    override_args_json: Path | None = None,
    output_config_name: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = get_candidate_config(config_name)
    base_args = _read_json(base_run_args_path) if base_run_args_path else {}
    override_args = _read_json(override_args_json) if override_args_json else {}

    candidate_overrides = build_run_args(config_name)
    candidate_overrides.update(
        {
            "config_name": output_config_name or config_name,
            "output_dir": str(output_dir),
            "window_cache_path": str(cache_path),
        }
    )
    stage_overrides: dict[str, Any] = {}
    if stage == "smoke":
        stage_overrides.update(
            {
                "epochs": 1,
                "patience": 0,
                "min_epochs_before_early_stop": 0,
                "max_outer_folds": 1,
                "enforce_fixed_all90_protocol": False,
                "log_interval": 1,
            }
        )

    effective = dict(base_args)
    effective.update(DISABLED_INTENTIONALLY)
    effective.update(candidate_overrides)
    effective.update(stage_overrides)
    effective.update(override_args)

    known_keys = _known_cli_keys()
    ignored_unknown = sorted(key for key in effective if key not in known_keys)
    filtered = {key: value for key, value in effective.items() if key in known_keys}

    inherited = {
        key: value
        for key, value in base_args.items()
        if key not in candidate_overrides and key not in stage_overrides and key not in override_args and key not in DISABLED_INTENTIONALLY
    }
    diff = {
        "base_args_path": str(base_run_args_path) if base_run_args_path else "",
        "base_config_name": config_name,
        "output_config_name": output_config_name or config_name,
        "module_tags": list(config.module_tags),
        "diagnostic_only": bool(config.diagnostic_only),
        "default_a9v14_args": DEFAULT_A9V14_ARGS,
        "inherited_from_base": inherited,
        "overridden_for_a9v14": candidate_overrides,
        "stage_overrides": stage_overrides,
        "override_args_json": str(override_args_json) if override_args_json else "",
        "override_args": override_args,
        "disabled_intentionally": DISABLED_INTENTIONALLY,
        "ignored_unknown_base_or_effective_args": ignored_unknown,
    }
    return filtered, diff


def build_command(python_exe: str, run_args: dict[str, Any]) -> list[str]:
    cli = [python_exe, str(Path("run_neuroez_c.py"))]
    for key in sorted(run_args):
        _add_cli_arg(cli, key, run_args[key])
    return cli


def _run_diagnostic(config_name: str, output_dir: Path) -> None:
    root = output_dir.parent
    source_config = "A9v3_Reproduce" if config_name.endswith("_A9v3") else "BEST"
    summary = run_source_propagation_diagnostic(root=root, config_name=source_config)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "source_propagation_diagnostic_summary.json", summary)
    target_dir = _resolve_config_dir(root, source_config)
    for name in (
        "source_propagation_diagnostic_by_patient.csv",
        "source_propagation_diagnostic_by_center.csv",
        "source_propagation_missing_features.json",
        "insufficient_feature_report.json",
    ):
        src = target_dir / name
        if src.exists():
            shutil.copy2(src, output_dir / name)


def run_candidate(args: argparse.Namespace) -> int:
    config = get_candidate_config(args.config_name)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if config.diagnostic_only:
        _run_diagnostic(args.config_name, output_dir)
        return 0

    cache_path = Path(args.cache_path)
    if not cache_path.exists():
        raise FileNotFoundError(f"Missing A9v14 all90 window cache: {cache_path}")
    base_path = Path(args.base_run_args_path) if args.base_run_args_path else None
    if base_path is not None and not base_path.exists():
        raise FileNotFoundError(f"Missing BaseRunArgsPath: {base_path}")

    run_args, diff = build_effective_args(
        config_name=args.config_name,
        output_dir=output_dir,
        cache_path=cache_path,
        base_run_args_path=base_path,
        stage=args.stage,
        override_args_json=Path(args.override_args_json) if args.override_args_json else None,
        output_config_name=args.output_config_name or None,
    )
    _write_json(output_dir / "a9v14_config.json", diff)
    _write_json(output_dir / "a9v14_effective_args.json", run_args)
    _write_json(output_dir / "a9v14_base_args_diff.json", diff)

    command = build_command(args.python_exe, run_args)
    _write_json(output_dir / "a9v14_launch_command.json", {"command": command})
    if args.dry_run:
        print(json.dumps({"dry_run": True, "command": command}, indent=2))
        return 0

    completed = subprocess.run(command, cwd=Path(__file__).resolve().parents[1])
    return int(completed.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one A9v14 candidate config through the existing A9v3 train/eval entrypoint.")
    parser.add_argument("--config_name", type=str, required=True, choices=list_candidate_configs())
    parser.add_argument("--cache_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--base_run_args_path", type=Path, default=None)
    parser.add_argument("--stage", type=str, default="full", choices=["smoke", "full"])
    parser.add_argument("--python_exe", type=str, default=sys.executable)
    parser.add_argument("--override_args_json", type=Path, default=None)
    parser.add_argument("--output_config_name", type=str, default="")
    parser.add_argument("--dry_run", action="store_true", default=False)
    args = parser.parse_args()
    raise SystemExit(run_candidate(args))


if __name__ == "__main__":
    main()
