#!/usr/bin/env python3
"""Launch the existing P2 entrypoint from a frozen JSON argument set."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from task1_confirmatory.provenance import (
    audit_argument_coverage,
    audit_effective_config_diff,
    effective_config,
    load_entrypoint_parser,
    write_json,
)


def _positive_flag(action: object) -> str:
    options = getattr(action, "option_strings", [])
    return next((value for value in options if value.startswith("--") and not value.startswith("--no-")), options[0])


def _append_action(command: list[str], action: argparse.Action, value: object) -> None:
    """Render one parsed value without inventing positional boolean strings."""
    if isinstance(action, argparse.BooleanOptionalAction):
        positive = _positive_flag(action)
        command.append(positive if bool(value) else "--no-" + positive[2:])
    elif isinstance(action, argparse._StoreTrueAction):
        if bool(value):
            command.append(_positive_flag(action))
    elif isinstance(action, argparse._StoreFalseAction):
        if not bool(value):
            command.append(_positive_flag(action))
    elif isinstance(value, bool):
        # Legacy optional-value booleans in this parser expect an explicit
        # true/false token (for example drop_high_ez_fraction_lzu).
        command.extend([_positive_flag(action), str(value).lower()])
    else:
        command.extend([_positive_flag(action), str(value)])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entrypoint", required=True); parser.add_argument("--base-args", required=True)
    parser.add_argument("--overrides-json", required=True); parser.add_argument("--command-output", required=True)
    parser.add_argument("--argument-coverage-output", required=True)
    parser.add_argument("--reference-effective-config-output", required=True)
    parser.add_argument("--effective-config-output", required=True)
    parser.add_argument("--effective-config-diff-output", required=True)
    parser.add_argument("--experiment-type", required=True)
    parser.add_argument("--allow-component-ablation", action="store_true")
    parser.add_argument("--strict-arg-coverage", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    entrypoint = Path(args.entrypoint).resolve(); base = json.loads(Path(args.base_args).read_text(encoding="utf-8")); overrides = json.loads(Path(args.overrides_json).read_text(encoding="utf-8"))
    target_parser = load_entrypoint_parser(entrypoint)
    coverage = audit_argument_coverage(base, target_parser, strict=False)
    reference = effective_config(base, {}, target_parser)
    candidate = effective_config(base, overrides, target_parser)
    diff = audit_effective_config_diff(reference, candidate, experiment_type=args.experiment_type)
    if args.allow_component_ablation:
        allowed = {
            "p23_profile",
            "use_patient_relative_z",
        }
        forbidden = dict(diff["forbidden_differences"])
        admitted = {key: forbidden.pop(key) for key in sorted(set(forbidden) & allowed)}
        diff["component_ablation_allowed_differences"] = admitted
        diff["forbidden_differences"] = forbidden
        diff["passed"] = not forbidden and not diff["forbidden_missing_reference_fields"] and not diff["forbidden_extra_confirmatory_fields"]
    write_json(args.argument_coverage_output, coverage)
    write_json(args.reference_effective_config_output, reference)
    write_json(args.effective_config_output, candidate)
    write_json(args.effective_config_diff_output, diff)
    if args.strict_arg_coverage and not coverage["passed"]:
        raise RuntimeError("P2 strict argument coverage failed; see " + args.argument_coverage_output)
    if not diff["passed"]:
        raise RuntimeError("P2 effective config diff failed; see " + args.effective_config_diff_output)
    values = {**base, **overrides}
    command = [sys.executable, str(entrypoint)]
    actions: dict[str, object] = {}
    for action in target_parser._actions:
        # A few legacy parsers declare an alias twice. Prefer the real
        # BooleanOptionalAction so false is rendered as --no-flag, not a
        # positional string that shifts the remaining command line.
        prior = actions.get(action.dest)
        if prior is None or action.__class__.__name__ == "BooleanOptionalAction":
            actions[action.dest] = action
    for action in actions.values():
        dest = action.dest
        if dest in {"help", ""} or dest not in values or values[dest] is None:
            continue
        value = values[dest]
        _append_action(command, action, value)
    output = Path(args.command_output); output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(command, indent=2), encoding="utf-8")
    print(json.dumps({"command": command, "dry_run": bool(args.dry_run), "argument_coverage": coverage, "effective_config_diff": diff}, indent=2))
    if not args.dry_run:
        try:
            subprocess.run(command, cwd=entrypoint.parent, check=True)
        except subprocess.CalledProcessError as error:
            # Preserve a native CUDA/driver exit status for the outer runner.
            # Otherwise Python turns 0xC0000005 into a generic exit code 1 and
            # the resumable PRQ runner cannot distinguish it from a code error.
            raise SystemExit(error.returncode) from error


if __name__ == "__main__":
    main()
