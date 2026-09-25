from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.summarize_a9v14_candidate_grid import summarize


SINGLE_MODULE_CONFIGS = {
    "M1": {"A9v14_M1_DeepsetReranker", "A9v14_M1_SetTransformerRank"},
    "M2": {"A9v14_M2_ConsistencyOnly"},
    "M3": {"A9v14_M3_LocalOnly"},
    "M4": {"A9v14_M4_MixtureOnly"},
}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def _read_summary(root: Path) -> pd.DataFrame:
    summary_path = root / "a9v14_candidate_grid_summary.csv"
    if not summary_path.exists():
        summarize(root)
    if not summary_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(summary_path)
    if "rank_robust_composite" in df.columns:
        df["rank_robust_composite"] = pd.to_numeric(df["rank_robust_composite"], errors="coerce").fillna(-1e9)
    return df


def _best_single_modules(summary_df: pd.DataFrame) -> tuple[list[dict[str, Any]], str]:
    if summary_df.empty:
        return [], "missing_summary"
    choices: list[dict[str, Any]] = []
    for module, names in SINGLE_MODULE_CONFIGS.items():
        subset = summary_df[summary_df["config_name"].astype(str).isin(names)].copy()
        if subset.empty:
            continue
        subset["rank_robust_composite"] = pd.to_numeric(subset["rank_robust_composite"], errors="coerce").fillna(-1e9)
        row = subset.sort_values("rank_robust_composite", ascending=False, kind="mergesort").iloc[0].to_dict()
        if str(row.get("failure_reasons", "")).startswith("summarizer_error"):
            continue
        row["module"] = module
        choices.append(row)
    choices = sorted(choices, key=lambda row: float(row.get("rank_robust_composite", -1e9)), reverse=True)
    if len(choices) < 2:
        return choices, "fewer_than_two_single_module_results"
    return choices[:2], ""


def _combo_overrides(selected: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    modules = sorted(str(row["module"]) for row in selected)
    overrides: dict[str, Any] = {
        "use_patient_context_reranker": False,
        "reranker_use_rank_features": False,
        "use_multi_seizure_consistency": False,
        "use_shaft_local_residual": False,
        "use_clinical_mixture_head": False,
    }
    name = "A9v14_AutoTop2_" + "".join(modules)
    for row in selected:
        module = str(row["module"])
        config_name = str(row.get("config_name", ""))
        if module == "M1":
            overrides["use_patient_context_reranker"] = True
            if "SetTransformerRank" in config_name:
                overrides["reranker_type"] = "set_transformer"
                overrides["reranker_use_rank_features"] = True
            else:
                overrides["reranker_type"] = "deepset"
                overrides["reranker_use_rank_features"] = False
        elif module == "M2":
            overrides["use_multi_seizure_consistency"] = True
        elif module == "M3":
            overrides["use_shaft_local_residual"] = True
            overrides["local_window"] = 1
        elif module == "M4":
            overrides["use_clinical_mixture_head"] = True
    if modules == ["M1", "M2"]:
        name = "A9v14_AutoTop2_M1M2"
    return name, overrides


def run_auto_top2(args: argparse.Namespace) -> int:
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    summary_df = _read_summary(root)
    selected, fallback_reason = _best_single_modules(summary_df)
    if fallback_reason:
        selected = [
            {"module": "M1", "config_name": "A9v14_M1_SetTransformerRank", "rank_robust_composite": None},
            {"module": "M2", "config_name": "A9v14_M2_ConsistencyOnly", "rank_robust_composite": None},
        ]
    output_config_name, overrides = _combo_overrides(selected)
    selection = {
        "selected_modules": selected,
        "output_config_name": output_config_name,
        "fallback_reason": fallback_reason,
        "overrides": overrides,
    }
    selection_path = root / "a9v14_auto_top2_selection.json"
    override_path = root / "a9v14_auto_top2_overrides.json"
    _write_json(selection_path, selection)
    _write_json(override_path, overrides)

    output_dir = root / output_config_name
    command = [
        args.python_exe,
        str(Path("scripts") / "run_a9v14_candidate.py"),
        "--config_name",
        "A9v14_AutoTop2_Combo",
        "--output_config_name",
        output_config_name,
        "--cache_path",
        str(args.cache_path),
        "--output_dir",
        str(output_dir),
        "--stage",
        args.stage,
        "--override_args_json",
        str(override_path),
        "--python_exe",
        args.python_exe,
    ]
    if args.base_run_args_path:
        command.extend(["--base_run_args_path", str(args.base_run_args_path)])
    if args.dry_run:
        _write_json(root / "a9v14_auto_top2_launch_command.json", {"command": command})
        print(json.dumps({"dry_run": True, "command": command, **selection}, indent=2))
        return 0
    completed = subprocess.run(command, cwd=Path(__file__).resolve().parents[1])
    if completed.returncode != 0:
        return int(completed.returncode)
    summarize(root)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Select top two A9v14 single modules and run their combination.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cache_path", type=Path, required=True)
    parser.add_argument("--base_run_args_path", type=Path, default=None)
    parser.add_argument("--stage", type=str, default="full", choices=["smoke", "full"])
    parser.add_argument("--python_exe", type=str, default=sys.executable)
    parser.add_argument("--dry_run", action="store_true", default=False)
    args = parser.parse_args()
    raise SystemExit(run_auto_top2(args))


if __name__ == "__main__":
    main()
