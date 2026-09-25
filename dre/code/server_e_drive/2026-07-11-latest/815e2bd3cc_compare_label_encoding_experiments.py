from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neuroez_c.clean_nez_utils import json_safe


KEY_METHODS = [
    "V3 baseline predicted_ez",
    "V3 baseline oracle-K diagnostic",
    "CleanNEZ base score oracle-K diagnostic",
    "RawBrainBERT-NEZDistance oracle-K diagnostic",
    "CleanNEZ + RawDistance + SetTopo oracle-K diagnostic",
    "CleanNEZ + RawDistance + SetTopo + KCal no-leak",
]


def _read_csv(eval_dir: Path, name: str) -> pd.DataFrame:
    path = eval_dir / name
    if not path.exists():
        raise FileNotFoundError(f"Missing evaluation artifact: {path}")
    return pd.read_csv(path)


def _merge_metric_table(ez1: pd.DataFrame, ez0: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    merged = ez1.merge(ez0, on=keys, suffixes=("_ez1", "_ez0"), how="outer", indicator=True)
    if not merged["_merge"].eq("both").all():
        bad = merged.loc[~merged["_merge"].eq("both"), keys + ["_merge"]].head(20).to_dict("records")
        raise ValueError(f"Label-encoding comparison keys do not match for {keys}: {bad}")
    merged = merged.drop(columns=["_merge"])
    for col in [
        "patient_macro_f1",
        "patient_ez_f1",
        "patient_nez_f1",
        "patient_macro_auprc_ez",
        "patient_macro_ez_mrr",
        "top1_is_ez",
        "recall_at_true_count",
    ]:
        left = f"{col}_ez1"
        right = f"{col}_ez0"
        if left in merged.columns and right in merged.columns:
            merged[f"delta_{col}_ez0_minus_ez1"] = merged[right].astype(float) - merged[left].astype(float)
    return merged


def compare_label_encoding_experiments(
    ez1_eval_dir: str | Path,
    ez0_eval_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    ez1_eval_dir = Path(ez1_eval_dir)
    ez0_eval_dir = Path(ez0_eval_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = _merge_metric_table(
        _read_csv(ez1_eval_dir, "metrics_summary.csv"),
        _read_csv(ez0_eval_dir, "metrics_summary.csv"),
        ["method"],
    )
    by_fold = _merge_metric_table(
        _read_csv(ez1_eval_dir, "metrics_by_fold.csv"),
        _read_csv(ez0_eval_dir, "metrics_by_fold.csv"),
        ["method", "fold_idx"],
    )
    by_center = _merge_metric_table(
        _read_csv(ez1_eval_dir, "metrics_by_center.csv"),
        _read_csv(ez0_eval_dir, "metrics_by_center.csv"),
        ["method", "center"],
    )
    patient_ez1 = _read_csv(ez1_eval_dir, "patient_level_metrics.csv")
    patient_ez0 = _read_csv(ez0_eval_dir, "patient_level_metrics.csv")
    patient_keys = ["method", "subject_id", "fold_idx", "center"]
    patient_delta = _merge_metric_table(patient_ez1, patient_ez0, patient_keys)

    ez1_subjects = set(patient_ez1["subject_id"].astype(str).unique())
    ez0_subjects = set(patient_ez0["subject_id"].astype(str).unique())
    subject_sets_equal = ez1_subjects == ez0_subjects
    row_counts_equal = int(len(patient_ez1)) == int(len(patient_ez0))
    if not subject_sets_equal:
        raise ValueError(
            "Label-encoding comparison subject sets differ: "
            f"missing_in_ez0={sorted(ez1_subjects - ez0_subjects)[:20]}, "
            f"extra_in_ez0={sorted(ez0_subjects - ez1_subjects)[:20]}"
        )
    if not row_counts_equal:
        raise ValueError(f"Label-encoding comparison row counts differ: ez1={len(patient_ez1)}, ez0={len(patient_ez0)}")

    summary.to_csv(output_dir / "label_encoding_comparison_summary.csv", index=False)
    by_fold.to_csv(output_dir / "label_encoding_comparison_by_fold.csv", index=False)
    by_center.to_csv(output_dir / "label_encoding_comparison_by_center.csv", index=False)
    patient_delta.to_csv(output_dir / "label_encoding_comparison_patient_delta.csv", index=False)

    methods_present = sorted(set(summary["method"].astype(str)))
    audit = {
        "ez1_eval_dir": str(ez1_eval_dir),
        "ez0_eval_dir": str(ez0_eval_dir),
        "subject_sets_equal": bool(subject_sets_equal),
        "row_counts_equal": bool(row_counts_equal),
        "methods_present": methods_present,
        "expected_methods_missing": [method for method in KEY_METHODS if method not in methods_present],
        "ez1_label_semantics": "EZ=1, NEZ=0",
        "ez0_label_semantics": "EZ=0, NEZ=1",
        "clinical_metric_target": "clinical_true_ez",
    }
    with (output_dir / "label_encoding_comparison_audit.json").open("w", encoding="utf-8") as fout:
        json.dump(json_safe(audit), fout, indent=2, ensure_ascii=False, sort_keys=True)
    return audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare ez1 and ez0 CleanNEZ label-encoding experiments.")
    parser.add_argument("--ez1-eval-dir", required=True)
    parser.add_argument("--ez0-eval-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    audit = compare_label_encoding_experiments(args.ez1_eval_dir, args.ez0_eval_dir, args.output_dir)
    print(json.dumps(json_safe(audit), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
