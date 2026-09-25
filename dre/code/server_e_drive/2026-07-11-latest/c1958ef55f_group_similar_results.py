"""Group similar A9v8 results across configs and score variants.

Recursively collects heldout_summary_neuroez_v3.json, lcbo_rescore_summary.csv,
and lcbo_teacher_anchor_rescore_summary.csv, standardises fields, groups by
metric bins, and outputs best-per-group + main-gate candidates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

A9V3_GATE = {
    "patient_macro_f1": 0.642714,
    "patient_macro_ez_f1": 0.470591,
    "patient_macro_auprc_ez": 0.518635,
    "patient_macro_ez_mrr": 0.715136,
    "top1_is_ez_rate": 0.600000,
}

_METRICS = list(A9V3_GATE.keys())
_GROUP_COLS = ["f1_bin", "ez_f1_bin", "auprc_bin", "mrr_bin", "top1_bin"]


def _passes_main_gate(row: pd.Series) -> bool:
    return (
        float(row.get("patient_macro_f1", 0.0)) > A9V3_GATE["patient_macro_f1"]
        and float(row.get("patient_macro_ez_f1", 0.0)) > A9V3_GATE["patient_macro_ez_f1"]
        and float(row.get("patient_macro_auprc_ez", 0.0)) > A9V3_GATE["patient_macro_auprc_ez"]
        and float(row.get("patient_macro_ez_mrr", 0.0)) >= A9V3_GATE["patient_macro_ez_mrr"]
        and float(row.get("top1_is_ez_rate", 0.0)) >= A9V3_GATE["top1_is_ez_rate"]
    )


def _rescore_cols_map() -> dict[str, str]:
    return {
        "patient_macro_f1": "patient_macro_f1",
        "patient_macro_ez_f1": "patient_macro_ez_f1",
        "patient_macro_auprc_ez": "patient_macro_auprc_ez",
        "patient_macro_ez_mrr": "patient_macro_ez_mrr",
        "top1_is_ez_rate": "top1_is_ez_rate",
    }


def _collect_heldout(root_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(root_dir.rglob("heldout_summary_neuroez_v3.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        config = path.parent.name
        row = {
            "source_file": str(path.relative_to(root_dir)),
            "config": config,
        }
        for m in _METRICS:
            row[m] = float(data.get(m, np.nan))
        row["passes_main_gate"] = _passes_main_gate(pd.Series(row))
        row.update({col: "" for col in (
            "variant_family", "score_variant", "score_space", "alpha", "beta", "gamma",
        )})
        rows.append(row)
    return pd.DataFrame(rows)


def _collect_rescore(root_dir: Path, pattern: str, source_label: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(root_dir.rglob(pattern)):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if df.empty:
            continue
        rel = str(path.relative_to(root_dir))
        df["source_file"] = rel
        df["_source_cfg"] = path.parent.name if path.parent != root_dir else rel
        if "config" not in df.columns:
            df["config"] = df["_source_cfg"]
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    merged = pd.concat(frames, ignore_index=True)

    # Standardise column names
    for key in ("score_variant", "variant_family", "score_space", "score_name"):
        if key not in merged.columns:
            merged[key] = ""
    for key in ("alpha", "beta", "gamma"):
        if key not in merged.columns:
            merged[key] = np.nan
    for m in _METRICS:
        if m not in merged.columns:
            merged[m] = np.nan

    # Fill NaNs for non-numeric columns
    for col in ("variant_family", "score_variant", "score_space"):
        merged[col] = merged[col].fillna("")

    merged["passes_main_gate"] = merged.apply(_passes_main_gate, axis=1)

    # compute bins
    for m in _METRICS:
        bin_col = m.replace("patient_macro_", "").replace("_ez", "").replace("top1_is_ez_rate", "top1")
        merged[f"{bin_col}_bin"] = merged[m].astype(float).round(3)

    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Group similar A9v8 results across runs.")
    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()

    root_dir = Path(args.root_dir)
    out_dir = Path(args.out_dir) if args.out_dir else root_dir / "grouped_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    # collect
    heldout_df = _collect_heldout(root_dir)
    rescore_df = _collect_rescore(root_dir, "lcbo_rescore_summary.csv", "lcbo_rescore")
    teac_rescore_df = _collect_rescore(root_dir, "lcbo_teacher_anchor_rescore_summary.csv", "lcbo_teacher_anchor_rescore")

    all_frames = [df for df in (heldout_df, rescore_df, teac_rescore_df) if not df.empty]
    if not all_frames:
        raise RuntimeError(f"No result files found under {root_dir}")

    all_df = pd.concat(all_frames, ignore_index=True)

    # output 1: all variants
    all_df.to_csv(out_dir / "all_result_variants.csv", index=False)

    # group by bins
    bin_cols_present = [c for c in _GROUP_COLS if c in all_df.columns]
    if bin_cols_present:
        group_col_name = "similar_group_id"
        all_df[group_col_name] = all_df[bin_cols_present].astype(str).agg("_".join, axis=1)

        gdf_all = all_df.groupby(group_col_name, sort=True)
        grouped_list = []
        best_list = []
        for gid, grp in gdf_all:
            grp = grp.sort_values(
                ["passes_main_gate", "patient_macro_ez_mrr", "patient_macro_f1", "patient_macro_auprc_ez"],
                ascending=[False, False, False, False],
            )
            grp[group_col_name] = gid
            grouped_list.append(grp)
            best_list.append(grp.iloc[0])

        grouped_df = pd.concat(grouped_list, ignore_index=True)
        best_per_group_df = pd.DataFrame(best_list)
    else:
        grouped_df = all_df.copy()
        best_per_group_df = all_df.copy()

    grouped_df.to_csv(out_dir / "grouped_similar_results.csv", index=False)
    best_per_group_df.to_csv(out_dir / "best_per_group.csv", index=False)

    # main gate candidates
    candidates = all_df[all_df["passes_main_gate"] == True].sort_values(
        ["patient_macro_ez_mrr", "patient_macro_f1", "patient_macro_auprc_ez"],
        ascending=[False, False, False],
    )
    candidates.to_csv(out_dir / "main_gate_candidates.csv", index=False)

    print(f"Wrote {len(all_df)} rows to {out_dir}/all_result_variants.csv")
    print(f"Main gate candidates: {len(candidates)}")
    if len(bin_cols_present) > 0:
        print(f"Best per group: {len(best_per_group_df)} rows")


if __name__ == "__main__":
    main()
