from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.run_v3_truebest_qc import QC_CONFIGS, V3_TARGET_METRICS


SKIP_DIR_NAMES = {"logs", "log", "debug", "tmp", "temp", "__pycache__"}


def _as_float(value: Any, default: float = float("nan")) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _read_summary(run_dir: Path) -> dict[str, Any]:
    csv_path = run_dir / "heldout_summary_neuroez_v3.csv"
    json_path = run_dir / "heldout_summary_neuroez_v3.json"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        if not df.empty:
            return df.iloc[0].to_dict()
    if json_path.exists():
        data = json.loads(json_path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    return {}


def iter_config_dirs(root: Path) -> list[Path]:
    dirs: list[Path] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if child.name.lower() in SKIP_DIR_NAMES:
            continue
        if (child / "heldout_summary_neuroez_v3.csv").exists() or (child / "heldout_summary_neuroez_v3.json").exists():
            dirs.append(child)
    return dirs


def summarize(root: Path, output_csv: Path | None = None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    baseline: dict[str, Any] | None = None
    for run_dir in iter_config_dirs(root):
        summary = _read_summary(run_dir)
        if not summary:
            continue
        config_name = str(summary.get("config_name") or run_dir.name)
        row = {"config_name": config_name, "output_dir": str(run_dir), **summary}
        row["qc_audit_present"] = all(
            (run_dir / name).exists()
            for name in (
                "quality_by_center_summary.csv",
                "quality_by_patient_summary.csv",
                "quality_by_fold_split_summary.csv",
                "quality_audit_summary.json",
            )
        )
        row["protocol_ok"] = (
            int(_as_float(row.get("n_patient_rows"), -1)) == 90
            and int(_as_float(row.get("n_unique_subjects"), -1)) == 90
            and str(row.get("positive_label", "")).lower() == "ez"
            and str(row.get("score_semantics", "")) == "ez_probability"
            and str(row.get("drop_high_ez_fraction_lzu", "")).lower() in {"false", "0"}
        )
        if config_name == "V3_TrueBest_Reproduce":
            row["reproduces_v3_truebest"] = all(
                math.isfinite(_as_float(row.get(key))) and abs(_as_float(row.get(key)) - target) <= 0.003
                for key, target in V3_TARGET_METRICS.items()
            )
            baseline = row
        else:
            row["reproduces_v3_truebest"] = False
        rows.append(row)

    if baseline is not None:
        for row in rows:
            row["patient_macro_f1_delta_vs_v3"] = _as_float(row.get("patient_macro_f1")) - _as_float(baseline.get("patient_macro_f1"))
            row["patient_macro_auprc_ez_delta_vs_v3"] = _as_float(row.get("patient_macro_auprc_ez")) - _as_float(
                baseline.get("patient_macro_auprc_ez")
            )
            row["patient_macro_ez_mrr_delta_vs_v3"] = _as_float(row.get("patient_macro_ez_mrr")) - _as_float(
                baseline.get("patient_macro_ez_mrr")
            )
            row["top1_is_ez_rate_delta_vs_v3"] = _as_float(row.get("top1_is_ez_rate")) - _as_float(baseline.get("top1_is_ez_rate"))

    df = pd.DataFrame(rows)
    if not df.empty:
        known_order = {name: idx for idx, name in enumerate(QC_CONFIGS)}
        df["_order"] = df["config_name"].map(lambda name: known_order.get(str(name), 999))
        df = df.sort_values(["_order", "config_name"]).drop(columns=["_order"])
    target = output_csv or (root / "quality_ablation_summary.csv")
    target.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target, index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize direct V3 TrueBest/QC config outputs without scanning logs/debug/tmp.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, default=None)
    args = parser.parse_args()
    df = summarize(args.root, args.output_csv)
    print(f"Wrote {len(df)} rows to {args.output_csv or (args.root / 'quality_ablation_summary.csv')}")


if __name__ == "__main__":
    main()
