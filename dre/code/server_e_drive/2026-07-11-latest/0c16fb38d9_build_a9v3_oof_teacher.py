"""Build leak-free A9v3 OOF channel-score teacher from patient-channel ledgers.

Each patient contributes its held-out/test fold prediction only.
Audit includes full metric recomputation from the teacher CSV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_recall_fscore_support


OUTPUT_COLUMNS = [
    "subject_id", "patient_id", "fold_id", "center", "center_id",
    "record_id", "channel_id", "channel_name",
    "label_ez", "patient_ez_count",
    "a9v3_oof_score", "a9v3_rank_eval", "a9v3_pred_topk", "a9v3_top1",
]


def normalize_center(value: object) -> str:
    return str(value).strip().lower().replace(" ", "_")


def build_center_mapping_audit(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["center"] = work["center"].map(normalize_center) if "center" in work.columns else "unknown"
    if "patient_id" not in work.columns and "subject_id" in work.columns:
        work["patient_id"] = work["subject_id"].astype(str)
    if "center_id" not in work.columns:
        work["center_id"] = "missing"
    global_informative = int(work["center_id"].dropna().astype(str).nunique()) > 1
    rows = []
    for center, group in work.groupby("center", sort=True, dropna=False):
        unique_ids = sorted(group["center_id"].dropna().astype(str).unique().tolist())
        rows.append(
            {
                "center": str(center),
                "n_patients": int(group["patient_id"].nunique()) if "patient_id" in group.columns else 0,
                "n_rows": int(len(group)),
                "unique_center_ids": ",".join(unique_ids),
                "n_unique_center_ids": int(len(unique_ids)),
                "center_id_is_informative": bool(global_informative),
                "center_id_warning": "" if global_informative else "center_id is non-informative; use normalized center string.",
            }
        )
    return pd.DataFrame(rows)


def _reciprocal_rank(y_ez: np.ndarray, score_ez: np.ndarray) -> float:
    positives = np.where(y_ez > 0.5)[0]
    if positives.size == 0:
        return 0.0
    order = np.argsort(score_ez)[::-1]
    ranks = {int(idx): rank for rank, idx in enumerate(order, start=1)}
    return float(max(1.0 / ranks[int(idx)] for idx in positives))


def build_oof_teacher_dataframe(ledgers: Iterable[pd.DataFrame]) -> pd.DataFrame:
    frames = []
    for ledger in ledgers:
        df = ledger.copy()
        if "split_role" in df.columns:
            df = df[df["split_role"].astype(str).str.lower() == "test"]
        elif "split" in df.columns:
            df = df[df["split"].astype(str).str.lower() == "test"]
        frames.append(df)
    if not frames:
        raise ValueError("No ledgers were provided.")
    df = pd.concat(frames, ignore_index=True)
    if df.empty:
        raise ValueError("No held-out test rows found in ledgers.")
    if "patient_id" not in df.columns:
        df["patient_id"] = df["subject_id"].astype(str)

    folds_per_patient = df.groupby("patient_id")["fold_id"].nunique()
    bad = folds_per_patient[folds_per_patient > 1]
    if not bad.empty:
        raise ValueError(f"Patients assigned to multiple held-out folds: {bad.index.tolist()[:5]}")

    duplicate_cols = ["patient_id", "channel_name"]
    if "record_id" in df.columns:
        duplicate_cols = ["patient_id", "record_id", "channel_name"]
    n_dupes = int(df.duplicated(duplicate_cols, keep=False).sum())
    if n_dupes > 0:
        raise ValueError(f"Duplicate OOF teacher rows ({n_dupes}) detected for keys {duplicate_cols}.")

    out = pd.DataFrame({
        "subject_id": df["subject_id"].astype(str),
        "patient_id": df["patient_id"].astype(str),
        "fold_id": df["fold_id"].astype(int),
        "center": df.get("center", "unknown").map(normalize_center) if "center" in df.columns else "unknown",
        "center_id": df.get("center_id", 4),
        "record_id": df.get("record_id", df["patient_id"].astype(str) + ":oof"),
        "channel_id": df.get("channel_id", df.groupby("patient_id").cumcount()),
        "channel_name": df["channel_name"].astype(str),
        "label_ez": df["label_ez"].astype(float),
        "patient_ez_count": df["patient_ez_count"].astype(float),
        "a9v3_oof_score": df["score_eval"].astype(float),
        "a9v3_rank_eval": df["rank_eval"].astype(int),
        "a9v3_pred_topk": df["pred_topk"].astype(int),
        "a9v3_top1": df["is_top1"].astype(int),
    })
    return out[OUTPUT_COLUMNS]


def recompute_metrics(teacher: pd.DataFrame) -> dict[str, float]:
    patient_f1s, patient_ez_f1s, patient_auprcs, patient_mrrs, top1_hits = [], [], [], [], []
    for _, group in teacher.groupby("patient_id", sort=False):
        y_ez = group["label_ez"].astype(float).to_numpy()
        y_nez = 1 - y_ez
        pred_ez = group["a9v3_pred_topk"].astype(int).to_numpy()
        pred_nez = 1 - pred_ez
        score_ez = group["a9v3_oof_score"].astype(float).to_numpy()
        _, _, f1_arr, _ = precision_recall_fscore_support(y_nez, pred_nez, labels=[1, 0], zero_division=0)
        patient_f1s.append(float(f1_score(y_nez, pred_nez, average="macro", zero_division=0)))
        patient_ez_f1s.append(float(f1_arr[1]))
        patient_auprcs.append(float(average_precision_score(y_ez, score_ez)) if np.unique(y_ez).size > 1 else 0.0)
        patient_mrrs.append(_reciprocal_rank(y_ez, score_ez))
        top1 = group[group["a9v3_top1"].astype(int) == 1]
        top1_hits.append(float(top1["label_ez"].max()) if not top1.empty else 0.0)
    def _m(a): return float(np.mean(a)) if a else 0.0
    return {
        "patient_macro_f1": _m(patient_f1s), "patient_macro_ez_f1": _m(patient_ez_f1s),
        "patient_macro_auprc_ez": _m(patient_auprcs), "patient_macro_ez_mrr": _m(patient_mrrs),
        "top1_is_ez_rate": _m(top1_hits),
    }


def _load_reference(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    if path.suffix == ".csv":
        df = pd.read_csv(path)
        return df.iloc[0].to_dict() if not df.empty else {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_audit(
    teacher: pd.DataFrame,
    ledger_paths: list[Path],
    *,
    expected_patients: int = 90,
    expected_patient_list_csv: str | None = None,
    reference_summary_json: str | None = None,
) -> dict[str, Any]:
    n_patients = int(teacher["patient_id"].nunique())
    missing_count = max(0, int(expected_patients) - n_patients)

    # missing patient list
    missing_list: list[str] = []
    if expected_patient_list_csv:
        elist = pd.read_csv(Path(expected_patient_list_csv))
        expected_ids = set(
            elist.get("patient_id", elist.get("subject_id", pd.Series(dtype=str))).astype(str)
        )
        observed_ids = set(teacher["patient_id"].astype(str))
        missing_list = sorted(expected_ids - observed_ids)
    else:
        missing_list = []

    # duplicate patient-fold entries: count patients assigned to multiple held-out folds
    folds_per_patient = teacher.groupby("patient_id")["fold_id"].nunique()
    patients_multi = folds_per_patient[folds_per_patient > 1]
    n_duplicate_pf = int(len(patients_multi))
    patients_multi_list = sorted(patients_multi.index.tolist())

    # channel-level duplicate check (separate field)
    dup_keys = ["patient_id", "record_id", "channel_name"] if "record_id" in teacher.columns else ["patient_id", "channel_name"]
    n_channel_dupes = int(teacher.duplicated(dup_keys, keep=False).sum())

    # teacher metric reproduction
    teacher_metrics = recompute_metrics(teacher)
    teacher_success = None
    teacher_diffs: dict[str, float] = {}
    if reference_summary_json:
        ref = _load_reference(Path(reference_summary_json))
        for key in ("patient_macro_f1", "patient_macro_ez_f1", "patient_macro_auprc_ez", "patient_macro_ez_mrr"):
            if key in ref:
                teacher_diffs[key] = float(teacher_metrics.get(key, 0.0)) - float(ref.get(key, 0.0))
        teacher_success = bool(teacher_diffs and all(abs(v) <= 1e-5 for v in teacher_diffs.values()))

    return {
        "n_teacher_rows": int(len(teacher)),
        "n_patients_expected": int(expected_patients),
        "n_patients_observed": n_patients,
        "missing_patient_count": missing_count,
        "missing_patients": missing_list,
        "n_folds": int(teacher["fold_id"].nunique()),
        "n_centers": int(teacher["center"].nunique()),
        "ledger_paths": [str(p) for p in ledger_paths],
        "one_heldout_fold_per_patient": bool(n_duplicate_pf == 0),
        "duplicate_patient_fold_entries": n_duplicate_pf,
        "patients_with_multiple_heldout_folds": patients_multi_list,
        "duplicate_channel_rows": n_channel_dupes,
        "teacher_metric_reproduction_success": teacher_success,
        "teacher_metric_diff_vs_reference": teacher_diffs,
        "metric_recomputed_from_teacher": teacher_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build leak-free A9v3 OOF channel-score teacher.")
    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument("--ledger_glob", type=str, default="**/patient_channel_ledger.csv")
    parser.add_argument("--output_csv", type=str, default="teacher/a9v3_oof_channel_scores.csv")
    parser.add_argument("--expected_patients", type=int, default=90)
    parser.add_argument("--fail_on_missing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--expected_patient_list_csv", type=str, default=None)
    parser.add_argument("--reference_summary_json", type=str, default=None)
    args = parser.parse_args()

    root_dir = Path(args.root_dir)
    paths = sorted(root_dir.glob(args.ledger_glob))
    if not paths:
        raise FileNotFoundError(f"No ledgers found under {root_dir} with glob {args.ledger_glob!r}")
    teacher = build_oof_teacher_dataframe(pd.read_csv(p) for p in paths)
    n_pts = int(teacher["patient_id"].nunique())
    if args.fail_on_missing and n_pts < int(args.expected_patients):
        raise ValueError(f"Teacher covers only {n_pts}/{args.expected_patients} patients.")

    output_csv = Path(args.output_csv)
    if not output_csv.is_absolute():
        output_csv = root_dir / output_csv
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    teacher.to_csv(output_csv, index=False)
    build_center_mapping_audit(teacher).to_csv(output_csv.parent / "center_mapping_audit.csv", index=False)

    audit = build_audit(teacher, paths, expected_patients=args.expected_patients,
                        expected_patient_list_csv=args.expected_patient_list_csv,
                        reference_summary_json=args.reference_summary_json)
    audit_path = output_csv.parent / "a9v3_oof_teacher_audit.json"
    with open(audit_path, "w", encoding="utf-8") as fout:
        json.dump(audit, fout, indent=2, ensure_ascii=False, sort_keys=True)
    print(f"Wrote {output_csv}")
    print(f"Teacher covers {n_pts}/{args.expected_patients} patients")


if __name__ == "__main__":
    main()
