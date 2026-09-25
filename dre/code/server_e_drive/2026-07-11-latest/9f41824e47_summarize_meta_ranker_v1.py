from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(output_dir: str | Path) -> Path:
    out = Path(output_dir)
    summary = _read_csv(out / "meta_ranker_summary.csv")
    by_fold = _read_csv(out / "meta_ranker_by_fold.csv")
    by_center = _read_csv(out / "meta_ranker_by_center.csv")
    selected = _read_csv(out / "meta_ranker_selected_params.csv")
    val = _read_csv(out / "meta_ranker_val_search_all.csv")
    inventory = _read_csv(out / "source_inventory.csv")
    score_audit = _read_json(out / "meta_ranker_score_bank_audit.json")

    compact_cols = [
        "patient_macro_f1",
        "patient_macro_ez_f1",
        "patient_macro_auprc_ez",
        "patient_macro_ez_mrr",
        "top1_is_ez_rate",
        "center_gap_f1",
        "passes_a9v3_gate",
        "passes_aaai_target_070",
    ]
    compact = summary[[col for col in compact_cols if col in summary.columns]].copy() if not summary.empty else pd.DataFrame(columns=compact_cols)
    compact.to_csv(out / "meta_ranker_compact_summary.csv", index=False)

    lines = ["# MetaRanker-v1 Report", ""]
    if not summary.empty:
        row = summary.iloc[0].to_dict()
        lines += [
            "## Overall Metrics",
            "",
            f"- patient_macro_f1: {float(row.get('patient_macro_f1', 0.0)):.6f}",
            f"- patient_macro_ez_f1: {float(row.get('patient_macro_ez_f1', 0.0)):.6f}",
            f"- patient_macro_auprc_ez: {float(row.get('patient_macro_auprc_ez', 0.0)):.6f}",
            f"- patient_macro_ez_mrr: {float(row.get('patient_macro_ez_mrr', 0.0)):.6f}",
            f"- top1_is_ez_rate: {float(row.get('top1_is_ez_rate', 0.0)):.6f}",
            f"- center_gap_f1: {float(row.get('center_gap_f1', 0.0)):.6f}",
            f"- passes_a9v3_gate: {bool(row.get('passes_a9v3_gate', False))}",
            f"- passes_aaai_target_070: {bool(row.get('passes_aaai_target_070', False))}",
            "",
        ]
    lines += ["## Selected Model Per Fold", ""]
    if not selected.empty:
        for _, row in selected.iterrows():
            lines.append(f"- fold {int(row.get('fold_idx', 0))}: {row.get('model_type', '')} / {row.get('feature_set', '')}")
    lines += ["", "## By-Center Performance", ""]
    if not by_center.empty:
        for _, row in by_center.iterrows():
            lines.append(f"- {row.get('center', 'unknown')}: patient_macro_f1={float(row.get('patient_macro_f1', 0.0)):.6f}")
    if not by_fold.empty and "patient_macro_f1" in by_fold.columns:
        worst = by_fold.sort_values("patient_macro_f1").iloc[0]
        best = by_fold.sort_values("patient_macro_f1", ascending=False).iloc[0]
        lines += [
            "",
            "## Fold Spread",
            "",
            f"- worst fold: {int(worst.get('fold_idx', 0))}, patient_macro_f1={float(worst.get('patient_macro_f1', 0.0)):.6f}",
            f"- best fold: {int(best.get('fold_idx', 0))}, patient_macro_f1={float(best.get('patient_macro_f1', 0.0)):.6f}",
        ]
    if not val.empty and "rank_robust_composite" in val.columns:
        top = val.sort_values("rank_robust_composite", ascending=False).head(10)
        lines += ["", "## Top Validation Configs", ""]
        for _, row in top.iterrows():
            lines.append(f"- fold {int(row.get('fold_idx', 0))}: {row.get('model_type', '')}/{row.get('feature_set', '')}, score={float(row.get('rank_robust_composite', 0.0)):.6f}")
    lines += [
        "",
        "## Sources",
        "",
        f"- inventory rows: {len(inventory)}",
        f"- score bank rows: {score_audit.get('n_rows', 'unknown')}",
        f"- score columns: {', '.join(score_audit.get('score_columns_found', [])) if score_audit else 'unknown'}",
    ]
    report_path = out / "meta_ranker_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize MetaRanker-v1 outputs.")
    parser.add_argument("--output_dir", required=True, type=Path)
    args = parser.parse_args(argv)
    print(summarize(args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
