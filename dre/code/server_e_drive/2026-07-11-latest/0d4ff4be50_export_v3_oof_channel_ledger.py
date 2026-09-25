from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neuroez_c.raw_brainbert_data import normalize_channel_name


REQUIRED_OUTPUT_COLUMNS = [
    "fold_idx",
    "subject_id",
    "center",
    "channel_id",
    "channel_name",
    "true_ez",
    "score_ez_probability",
    "rank_ez_desc",
    "predicted_ez",
]


def _first_present(df: pd.DataFrame, *names: str) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    return None


def _coerce_fold_prediction(df: pd.DataFrame, fold_idx: int) -> tuple[pd.DataFrame, list[str]]:
    out = pd.DataFrame()
    missing: list[str] = []

    fold_source = _first_present(df, "fold_idx", "fold_id")
    out["fold_idx"] = (
        pd.to_numeric(df[fold_source], errors="coerce").fillna(fold_idx).astype(int)
        if fold_source
        else pd.Series([int(fold_idx)] * len(df), index=df.index, dtype=int)
    )
    for target, aliases in {
        "subject_id": ("subject_id", "patient_id"),
        "center": ("center", "source_center", "source_dataset"),
        "channel_id": ("channel_id", "channel_index", "contact_idx"),
        "channel_name": ("channel_name", "channel_names_norm", "channel_name_norm", "contact_name"),
        "rank_ez_desc": ("rank_ez_desc", "rank_eval", "rank"),
        "predicted_ez": ("predicted_ez", "pred_topk", "pred_ez"),
    }.items():
        source = _first_present(df, *aliases)
        if source is None:
            if target == "center":
                out[target] = "unknown"
            elif target == "channel_id":
                out[target] = range(len(df))
            elif target == "channel_name":
                fallback = _first_present(df, "channel_id", "channel_index")
                out[target] = df[fallback].astype(str) if fallback else [f"ch{idx}" for idx in range(len(df))]
            elif target == "rank_ez_desc":
                out[target] = np.nan
            elif target == "predicted_ez":
                out[target] = 0
            else:
                missing.append(target)
        else:
            out[target] = df[source]

    true_ez_col = _first_present(df, "true_ez", "label_ez", "ez_label")
    if true_ez_col is not None:
        out["true_ez"] = pd.to_numeric(df[true_ez_col], errors="coerce").fillna(0).astype(int)
    elif "true_nez" in df.columns:
        out["true_ez"] = (1 - pd.to_numeric(df["true_nez"], errors="coerce").fillna(1)).astype(int)
    else:
        missing.append("true_ez")

    score_ez_col = _first_present(df, "score_ez_probability", "score_ez", "score_ez_final", "score_eval")
    if score_ez_col is not None:
        out["score_ez_probability"] = pd.to_numeric(df[score_ez_col], errors="coerce")
    elif "score_nez_probability" in df.columns:
        out["score_ez_probability"] = 1.0 - pd.to_numeric(df["score_nez_probability"], errors="coerce")
    else:
        missing.append("score_ez_probability")

    out["subject_id"] = out.get("subject_id", pd.Series(["unknown"] * len(df))).astype(str)
    out["center"] = out.get("center", pd.Series(["unknown"] * len(df))).fillna("unknown").astype(str)
    out["channel_name"] = out.get("channel_name", pd.Series([f"ch{idx}" for idx in range(len(df))])).astype(str)
    out["channel_id"] = pd.to_numeric(out.get("channel_id", pd.Series(range(len(df)))), errors="coerce").fillna(-1).astype(int)
    out["rank_ez_desc"] = pd.to_numeric(out["rank_ez_desc"], errors="coerce")
    if out["rank_ez_desc"].isna().any():
        out["rank_ez_desc"] = out["score_ez_probability"].rank(ascending=False, method="first")
    out["rank_ez_desc"] = out["rank_ez_desc"].astype(int)
    out["predicted_ez"] = pd.to_numeric(out["predicted_ez"], errors="coerce").fillna(0).astype(int)
    return out[REQUIRED_OUTPUT_COLUMNS], sorted(set(missing))


def _audit_for_ledger(ledger: pd.DataFrame, missing_required_columns: list[str]) -> dict[str, Any]:
    norm = ledger["channel_name"].map(normalize_channel_name) if "channel_name" in ledger.columns else pd.Series([])
    dup_cols = pd.DataFrame(
        {
            "fold_idx": ledger.get("fold_idx", pd.Series(dtype=int)),
            "subject_id": ledger.get("subject_id", pd.Series(dtype=str)),
            "channel_norm": norm,
        }
    )
    duplicated_mask = dup_cols.duplicated(["fold_idx", "subject_id", "channel_norm"], keep=False) if not dup_cols.empty else pd.Series([], dtype=bool)
    duplicated = int(duplicated_mask.sum()) if not dup_cols.empty else 0
    duplicate_preview: list[dict[str, Any]] = []
    if duplicated:
        preview_df = ledger.loc[duplicated_mask, ["fold_idx", "subject_id", "channel_name"]].copy()
        preview_df["normalized_channel_name"] = norm.loc[duplicated_mask].to_numpy()
        duplicate_preview = [
            {
                "fold_idx": int(row["fold_idx"]),
                "subject_id": str(row["subject_id"]),
                "channel_name": str(row["channel_name"]),
                "normalized_channel_name": str(row["normalized_channel_name"]),
            }
            for _, row in preview_df.head(20).iterrows()
        ]
    return {
        "num_rows": int(len(ledger)),
        "num_folds": int(ledger["fold_idx"].nunique()) if "fold_idx" in ledger.columns else 0,
        "folds_seen": sorted(int(v) for v in ledger["fold_idx"].dropna().unique().tolist()) if "fold_idx" in ledger.columns else [],
        "num_patients": int(ledger["subject_id"].nunique()) if "subject_id" in ledger.columns else 0,
        "rows_by_fold": {str(k): int(v) for k, v in ledger.groupby("fold_idx").size().to_dict().items()} if "fold_idx" in ledger.columns else {},
        "patients_by_fold": {
            str(k): int(v)
            for k, v in ledger.groupby("fold_idx")["subject_id"].nunique().to_dict().items()
        }
        if {"fold_idx", "subject_id"}.issubset(ledger.columns)
        else {},
        "rows_by_center": {str(k): int(v) for k, v in ledger.groupby("center").size().to_dict().items()} if "center" in ledger.columns else {},
        "true_ez_count": int(pd.to_numeric(ledger.get("true_ez", pd.Series(dtype=int)), errors="coerce").fillna(0).sum()),
        "missing_required_columns": list(missing_required_columns),
        "duplicated_subject_channel_rows": duplicated,
        "duplicated_subject_channel_preview": duplicate_preview,
    }


def export_v3_oof_channel_ledger(
    v3_output_dir: str | Path,
    output_path: str | Path,
    *,
    fold_start: int = 1,
    fold_end: int = 5,
    allow_duplicate_subject_channel: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    v3_output_dir = Path(v3_output_dir)
    output_path = Path(output_path)
    frames: list[pd.DataFrame] = []
    missing_by_fold: dict[str, list[str]] = {}
    for fold_idx in range(int(fold_start), int(fold_end) + 1):
        path = v3_output_dir / f"test_channel_predictions_neuroez_v2_fold_{fold_idx}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing V3 fold prediction CSV: {path}")
        frame, missing = _coerce_fold_prediction(pd.read_csv(path), fold_idx)
        frames.append(frame)
        if missing:
            missing_by_fold[str(fold_idx)] = missing
    ledger = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=REQUIRED_OUTPUT_COLUMNS)
    missing_required_columns = sorted(set(item for values in missing_by_fold.values() for item in values))
    audit = _audit_for_ledger(ledger, missing_required_columns)
    audit["missing_required_columns_by_fold"] = missing_by_fold
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ledger.to_csv(output_path, index=False)
    audit_path = output_path.parent / "v3_oof_ledger_audit.json"
    with audit_path.open("w", encoding="utf-8") as fout:
        json.dump(audit, fout, indent=2, ensure_ascii=False, sort_keys=True)
    if missing_required_columns:
        raise ValueError(f"Missing required V3 OOF ledger columns after alias mapping: {missing_by_fold}")
    if int(audit["duplicated_subject_channel_rows"]) > 0 and not bool(allow_duplicate_subject_channel):
        raise ValueError(
            "Duplicate V3 OOF subject-channel rows after channel normalization. "
            f"Preview: {audit['duplicated_subject_channel_preview']}"
        )
    return ledger, audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export V3 OOF channel prediction CSVs into one HNC ledger.")
    parser.add_argument("--v3-output-dir", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--fold-start", type=int, default=1)
    parser.add_argument("--fold-end", type=int, default=5)
    parser.add_argument("--allow-duplicate-subject-channel", action="store_true", default=False)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    ledger, audit = export_v3_oof_channel_ledger(
        args.v3_output_dir,
        args.output_path,
        fold_start=args.fold_start,
        fold_end=args.fold_end,
        allow_duplicate_subject_channel=args.allow_duplicate_subject_channel,
    )
    print(json.dumps({"output_path": args.output_path, **audit}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
