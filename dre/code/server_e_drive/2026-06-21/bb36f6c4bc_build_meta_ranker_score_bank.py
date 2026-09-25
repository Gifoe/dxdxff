from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_persistent_rank_features import get_forbidden_meta_feature_columns
from scripts.meta_ranker_common import forbidden_feature_columns, patientwise_rankpct, patientwise_zscore, read_json


SCORE_PRIORITY = ["score_ez_probability", "score_ez", "a9v3_oof_score", "score_eval", "probability_ez", "ez_score"]
EZ_LABEL_PRIORITY = ["label_ez", "true_ez", "ez_label", "y_ez", "target_ez"]
NEZ_LABEL_PRIORITY = ["label_nez", "true_nez", "nez_label", "y_nez", "target_nez"]
TRADITIONAL_METHOD_MAP = {
    "single_high_gamma": "score_single_high_gamma",
    "single_line_length": "score_single_line_length",
    "single_hfo_event_rate": "score_single_hfo_event_rate",
    "logistic_l2": "score_logistic_l2",
    "ridge": "score_ridge",
    "random_forest": "score_random_forest",
    "xgboost": "score_xgboost",
    "xgboost_optional": "score_xgboost",
}


def _score_column(df: pd.DataFrame) -> str | None:
    for col in SCORE_PRIORITY:
        if col in df.columns:
            return col
    for col in df.columns:
        lower = col.lower()
        if lower.startswith("score_") and ("ez" in lower or "prob" in lower):
            return col
    return None


def _label_columns(df: pd.DataFrame) -> tuple[str | None, str | None]:
    ez_col = next((col for col in EZ_LABEL_PRIORITY if col in df.columns), None)
    nez_col = next((col for col in NEZ_LABEL_PRIORITY if col in df.columns), None)
    return ez_col, nez_col


def _read_channel_paths(paths: list[str]) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in paths if path]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _standardize_source(df: pd.DataFrame, output_score_col: str) -> pd.DataFrame:
    if df.empty:
        return df
    score_col = _score_column(df)
    if score_col is None:
        raise ValueError(f"No EZ score column found for {output_score_col}")
    needed = ["subject_id", "channel_name"]
    for col in needed:
        if col not in df.columns:
            raise ValueError(f"Missing required column {col} for {output_score_col}")
    ez_col, nez_col = _label_columns(df)
    keep = ["subject_id", "channel_name", score_col, *[c for c in ("center", "fold_idx") if c in df.columns]]
    if ez_col:
        keep.append(ez_col)
    if nez_col and nez_col != ez_col:
        keep.append(nez_col)
    out = df[keep].copy()
    rename = {score_col: output_score_col}
    if ez_col:
        rename[ez_col] = "label_ez"
    if nez_col:
        rename[nez_col] = "label_nez"
    out = out.rename(columns=rename)
    if "label_ez" in out.columns and "label_nez" not in out.columns:
        labels = pd.to_numeric(out["label_ez"], errors="coerce")
        out["label_nez"] = 1 - labels
    if "label_nez" in out.columns and "label_ez" not in out.columns:
        labels = pd.to_numeric(out["label_nez"], errors="coerce")
        out["label_ez"] = 1 - labels
    out.attrs["label_source_columns_detected"] = {"label_ez": ez_col, "label_nez": nez_col}
    return out


def check_duplicate_keys(rows: pd.DataFrame, output_dir: Path) -> None:
    dup = rows[rows.duplicated(["subject_id", "channel_name"], keep=False)].copy()
    if not dup.empty:
        output_dir.mkdir(parents=True, exist_ok=True)
        dup.to_csv(output_dir / "duplicate_key_rows.csv", index=False)
        raise ValueError(f"Duplicate subject_id + channel_name rows: {len(dup)}")


def merge_source_frame(base: pd.DataFrame, source: pd.DataFrame, source_name: str, output_dir: Path) -> pd.DataFrame:
    if source.empty:
        return base
    check_duplicate_keys(source, output_dir)
    overlap_cols = [col for col in source.columns if col in base.columns and col not in {"subject_id", "channel_name"}]
    label_cols = [col for col in ("label_ez", "label_nez") if col in overlap_cols]
    fold_cols = [col for col in ("fold_idx",) if col in overlap_cols]
    merged = base.merge(source, on=["subject_id", "channel_name"], how="outer", suffixes=("", f"__{source_name}"))
    conflicts: list[dict[str, Any]] = []
    for label_col in label_cols:
        other = f"{label_col}__{source_name}"
        if other in merged.columns:
            left = pd.to_numeric(merged[label_col], errors="coerce")
            right = pd.to_numeric(merged[other], errors="coerce")
            mask = left.notna() & right.notna() & left.ne(right)
            if mask.any():
                for idx in merged.index[mask]:
                    conflicts.append(
                        {
                            "source_name": source_name,
                            "conflict_type": label_col,
                            "subject_id": str(merged.at[idx, "subject_id"]),
                            "channel_name": str(merged.at[idx, "channel_name"]),
                            "base_value": float(left.loc[idx]),
                            "source_value": float(right.loc[idx]),
                            "base_column": label_col,
                            "source_column": other,
                        }
                    )
            merged[label_col] = merged[label_col].combine_first(merged[other])
            merged = merged.drop(columns=[other])
    if conflicts:
        output_dir.mkdir(parents=True, exist_ok=True)
        conflict_df = pd.DataFrame(conflicts)
        conflict_df.to_csv(output_dir / "label_conflict_rows.csv", index=False)
        unique_pairs = conflict_df[["subject_id", "channel_name"]].drop_duplicates()
        binary = conflict_df[conflict_df["base_value"].isin([0.0, 1.0]) & conflict_df["source_value"].isin([0.0, 1.0])]
        inverse_rate = float(((binary["base_value"] + binary["source_value"]) == 1.0).mean()) if not binary.empty else 0.0
        (output_dir / "label_conflict_summary.json").write_text(
            json.dumps(
                {
                    "source_name": source_name,
                    "n_conflicts": int(len(conflict_df)),
                    "n_unique_subject_channel_conflicts": int(len(unique_pairs)),
                    "inverse_rate_if_binary": inverse_rate,
                    "subjects_with_conflicts": sorted(conflict_df["subject_id"].astype(str).unique().tolist()),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        raise ValueError("Label conflicts detected while merging score sources")
    fold_conflicts = []
    for fold_col in fold_cols:
        other = f"{fold_col}__{source_name}"
        if other in merged.columns:
            left = pd.to_numeric(merged[fold_col], errors="coerce")
            right = pd.to_numeric(merged[other], errors="coerce")
            mask = left.notna() & right.notna() & left.ne(right)
            if mask.any():
                fold_conflicts.append(merged.loc[mask, ["subject_id", "channel_name", fold_col, other]].copy())
            merged[fold_col] = merged[fold_col].combine_first(merged[other])
            merged = merged.drop(columns=[other])
    if fold_conflicts:
        output_dir.mkdir(parents=True, exist_ok=True)
        pd.concat(fold_conflicts, ignore_index=True).to_csv(output_dir / "fold_conflict_rows.csv", index=False)
        raise ValueError("fold_idx conflicts detected while merging score sources")
    for col in overlap_cols:
        other = f"{col}__{source_name}"
        if other in merged.columns:
            merged[col] = merged[col].combine_first(merged[other])
            merged = merged.drop(columns=[other])
    return merged


def validate_labels(bank: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    if "label_ez" not in bank.columns and "label_nez" in bank.columns:
        nez = pd.to_numeric(bank["label_nez"], errors="coerce")
        bank = bank.copy()
        bank["label_ez"] = 1 - nez
    if "label_ez" not in bank.columns:
        output_dir.mkdir(parents=True, exist_ok=True)
        bank.to_csv(output_dir / "missing_label_rows.csv", index=False)
        raise ValueError("score bank is missing label_ez")
    labels = pd.to_numeric(bank["label_ez"], errors="coerce")
    missing = bank[labels.isna()].copy()
    if not missing.empty:
        output_dir.mkdir(parents=True, exist_ok=True)
        missing.to_csv(output_dir / "missing_label_rows.csv", index=False)
        raise ValueError(f"score bank contains {len(missing)} rows with missing label_ez")
    invalid_mask = ~labels.isin([0, 1])
    if invalid_mask.any():
        output_dir.mkdir(parents=True, exist_ok=True)
        bank.loc[invalid_mask].to_csv(output_dir / "invalid_label_rows.csv", index=False)
        raise ValueError("score bank contains non-binary label_ez values")
    bank = bank.copy()
    bank["label_ez"] = labels.astype(int)
    bank["label_nez"] = 1 - bank["label_ez"]
    return bank


def validate_folds(bank: pd.DataFrame, output_dir: Path, *, allow_small_subject_count: bool = False) -> pd.DataFrame:
    if "fold_idx" not in bank.columns:
        output_dir.mkdir(parents=True, exist_ok=True)
        bank.to_csv(output_dir / "missing_fold_rows.csv", index=False)
        raise ValueError("score bank is missing fold_idx")
    folds = pd.to_numeric(bank["fold_idx"], errors="coerce")
    missing = bank[folds.isna()].copy()
    if not missing.empty:
        output_dir.mkdir(parents=True, exist_ok=True)
        missing.to_csv(output_dir / "missing_fold_rows.csv", index=False)
        raise ValueError(f"score bank contains {len(missing)} rows with missing fold_idx")
    bank = bank.copy()
    bank["fold_idx"] = folds.astype(int)
    subject_fold_counts = bank.groupby("subject_id")["fold_idx"].nunique()
    bad_subjects = subject_fold_counts[subject_fold_counts.ne(1)]
    if not bad_subjects.empty:
        output_dir.mkdir(parents=True, exist_ok=True)
        bank[bank["subject_id"].isin(bad_subjects.index)].to_csv(output_dir / "subject_fold_conflict_rows.csv", index=False)
        raise ValueError("subjects must have exactly one fold_idx")
    detected = sorted(int(v) for v in bank["fold_idx"].unique().tolist())
    if detected != [1, 2, 3, 4, 5] and not allow_small_subject_count:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "score_bank_subject_count_error.json").write_text(
            json.dumps({"error": "folds_not_fixed_all90", "folds_detected": detected}, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        raise ValueError(f"fixed all90 score bank requires folds [1, 2, 3, 4, 5], got {detected}")
    return bank


def _hard_score_bank_checks(bank: pd.DataFrame, feature_sets: dict[str, list[str]], output_dir: Path, *, allow_small_subject_count: bool = False) -> None:
    check_duplicate_keys(bank, output_dir)
    n_subjects = int(bank["subject_id"].nunique())
    if n_subjects != 90 and not allow_small_subject_count:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "score_bank_subject_count_error.json").write_text(
            json.dumps({"expected_n_subjects": 90, "actual_n_subjects": n_subjects}, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        raise ValueError(f"fixed all90 score bank requires 90 subjects, got {n_subjects}")
    no_ez = bank.groupby("subject_id")["label_ez"].sum()
    no_ez = no_ez[no_ez.le(0)]
    if not no_ez.empty:
        output_dir.mkdir(parents=True, exist_ok=True)
        bank[bank["subject_id"].isin(no_ez.index)].to_csv(output_dir / "no_ez_subject_rows.csv", index=False)
        raise ValueError("each subject must have at least one EZ channel")
    forbidden = forbidden_feature_columns(bank.columns)
    violations = {name: sorted(set(cols) & forbidden) for name, cols in feature_sets.items() if set(cols) & forbidden}
    if violations:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "forbidden_feature_columns_error.json").write_text(json.dumps(violations, indent=2, sort_keys=True), encoding="utf-8")
        raise ValueError("forbidden columns present in MetaRanker feature sets")
    if not any(feature_sets.values()):
        raise ValueError("at least one non-empty MetaRanker feature set is required")


def _load_a10_sources(rec: dict[str, Any], output_dir: Path) -> dict[str, pd.DataFrame]:
    out = {}
    mapping = {"A10_01_rank_w001_m003": "score_a10_01", "A10_04_rank_w003_m005": "score_a10_04"}
    for config, score_col in mapping.items():
        item = rec.get("a10_sources", {}).get(config, {})
        paths = item.get("channel_prediction_paths", [])
        if paths:
            out[score_col] = _standardize_source(_read_channel_paths(paths), score_col)
    return out


def _load_traditional(rec: dict[str, Any]) -> pd.DataFrame:
    path = rec.get("traditional_source", {}).get("channel_predictions", "")
    if not path:
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "method" in df.columns:
        frames = []
        for method, score_col in TRADITIONAL_METHOD_MAP.items():
            part = df[df["method"].astype(str).eq(method)].copy()
            if not part.empty:
                frames.append(_standardize_source(part, score_col))
        if frames:
            base = frames[0]
            for idx, frame in enumerate(frames[1:], start=1):
                base = merge_source_frame(base, frame, f"traditional_{idx}", Path(path).parent)
            return base
        return pd.DataFrame()
    score_cols = [col for col in df.columns if col.startswith("score_")]
    ez_col, nez_col = _label_columns(df)
    keep = ["subject_id", "channel_name", *[c for c in ("center", "fold_idx") if c in df.columns], *score_cols]
    if ez_col:
        keep.append(ez_col)
    if nez_col and nez_col != ez_col:
        keep.append(nez_col)
    out = df[keep].copy()
    rename = {}
    if ez_col:
        rename[ez_col] = "label_ez"
    if nez_col:
        rename[nez_col] = "label_nez"
    return out.rename(columns=rename)


def _allowed_persistent_features(rows: pd.DataFrame) -> list[str]:
    forbidden = get_forbidden_meta_feature_columns(list(rows.columns))
    return [
        col
        for col in rows.columns
        if col not in forbidden
        and pd.api.types.is_numeric_dtype(rows[col])
        and not col.startswith("score_")
        and not col.startswith("z_")
        and not col.startswith("rankpct_")
    ]


def _build_feature_sets(rows: pd.DataFrame, persistent_features: list[str]) -> dict[str, list[str]]:
    score_cols = [col for col in rows.columns if col.startswith("score_")]
    score_features = []
    for col in score_cols:
        for prefix in ("z_", "rankpct_"):
            feature = f"{prefix}{col}"
            if feature in rows.columns:
                score_features.append(feature)
    trad_names = ("single_high_gamma", "single_line_length", "single_hfo_event_rate", "logistic_l2", "ridge", "random_forest", "xgboost")
    trad_features = [f for f in score_features if any(name in f for name in trad_names)]
    a10_features = [f for f in score_features if "score_a10_01" in f or "score_a10_04" in f]
    a9_features = [f for f in score_features if "score_a9v3" in f]
    fm_features = [f for f in score_features if "score_fm_" in f]
    z_persistent = [f"z_persistent_{col}" for col in persistent_features if f"z_persistent_{col}" in rows.columns]
    marker_persistent = [col for col in persistent_features if any(token in col for token in ("high_gamma", "line_length", "hfo", "marker"))]
    return {
        "F0_scores_only": score_features,
        "F1_scores_plus_persistent": score_features + persistent_features + z_persistent,
        "F2_no_a10": a9_features + trad_features + persistent_features + z_persistent,
        "F3_a10_only": a10_features,
        "F4_marker_only": trad_features + marker_persistent,
        "F5_scores_plus_fm": score_features + fm_features,
    }


def build_score_bank(source_recommendations: str | Path, persistent_features: str | Path, output_dir: str | Path, *, allow_small_subject_count: bool = False) -> pd.DataFrame:
    source_recommendations = Path(source_recommendations)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rec = read_json(source_recommendations)
    persistent = pd.read_csv(persistent_features)
    check_duplicate_keys(persistent, output_dir)
    ez_col, nez_col = _label_columns(persistent)
    label_source_columns_detected = {"persistent": {"label_ez": ez_col, "label_nez": nez_col}}
    rename = {}
    if ez_col and ez_col != "label_ez":
        rename[ez_col] = "label_ez"
    if nez_col and nez_col != "label_nez":
        rename[nez_col] = "label_nez"
    if rename:
        persistent = persistent.rename(columns=rename)
    bank = persistent.copy()

    a9_paths = rec.get("a9v3_primary", {}).get("channel_prediction_paths", [])
    if a9_paths:
        a9_frame = _standardize_source(_read_channel_paths(a9_paths), "score_a9v3")
        label_source_columns_detected["a9v3"] = a9_frame.attrs.get("label_source_columns_detected", {})
        bank = merge_source_frame(bank, a9_frame, "a9v3", output_dir)
    for score_col, frame in _load_a10_sources(rec, output_dir).items():
        label_source_columns_detected[score_col] = frame.attrs.get("label_source_columns_detected", {})
        bank = merge_source_frame(bank, frame, score_col, output_dir)
    traditional = _load_traditional(rec)
    if not traditional.empty:
        bank = merge_source_frame(bank, traditional, "traditional", output_dir)
    for idx, item in enumerate(rec.get("fm_optional_sources", [])):
        paths = item.get("channel_prediction_paths", [])
        if paths:
            score_col = f"score_fm_{idx + 1}"
            fm_frame = _standardize_source(_read_channel_paths(paths), score_col)
            label_source_columns_detected[score_col] = fm_frame.attrs.get("label_source_columns_detected", {})
            bank = merge_source_frame(bank, fm_frame, score_col, output_dir)

    check_duplicate_keys(bank, output_dir)
    if "center" not in bank.columns:
        bank["center"] = "unknown"
    missing_label_count = int(pd.to_numeric(bank["label_ez"], errors="coerce").isna().sum()) if "label_ez" in bank.columns else int(len(bank))
    invalid_label_count = int((~pd.to_numeric(bank["label_ez"], errors="coerce").dropna().isin([0, 1])).sum()) if "label_ez" in bank.columns else 0
    bank = validate_labels(bank, output_dir)
    bank = validate_folds(bank, output_dir, allow_small_subject_count=allow_small_subject_count)

    score_cols = [col for col in bank.columns if col.startswith("score_")]
    for col in score_cols:
        bank[col] = pd.to_numeric(bank[col], errors="coerce").fillna(0.0)
        bank[f"z_{col}"] = patientwise_zscore(bank, col)
        bank[f"rankpct_{col}"] = patientwise_rankpct(bank, col)

    persistent_cols = _allowed_persistent_features(bank)
    for col in persistent_cols:
        if col.endswith("rank_pct") or col.endswith("fraction"):
            continue
        bank[f"z_persistent_{col}"] = patientwise_zscore(bank, col)

    feature_sets = _build_feature_sets(bank, persistent_cols)
    forbidden = forbidden_feature_columns(bank.columns)
    for key, features in list(feature_sets.items()):
        feature_sets[key] = [f for f in dict.fromkeys(features) if f in bank.columns and f not in forbidden and pd.api.types.is_numeric_dtype(bank[f])]
    _hard_score_bank_checks(bank, feature_sets, output_dir, allow_small_subject_count=allow_small_subject_count)
    (output_dir / "meta_ranker_feature_columns.json").write_text(json.dumps(feature_sets, indent=2, sort_keys=True), encoding="utf-8")
    bank.to_csv(output_dir / "meta_ranker_score_bank.csv", index=False)
    subject_fold = bank.drop_duplicates("subject_id")[["subject_id", "fold_idx"]]
    folds_detected = sorted(int(v) for v in bank["fold_idx"].unique().tolist())
    audit = {
        "n_rows": int(len(bank)),
        "n_subjects": int(bank["subject_id"].nunique()),
        "n_folds": int(len(folds_detected)),
        "folds_detected": folds_detected,
        "n_subjects_by_fold": {str(int(k)): int(v) for k, v in subject_fold["fold_idx"].value_counts().sort_index().to_dict().items()},
        "n_centers": int(bank["center"].nunique()),
        "center_counts": bank.drop_duplicates("subject_id")["center"].value_counts().sort_index().to_dict(),
        "score_columns_found": score_cols,
        "score_columns_missing": [col for col in ("score_a9v3", "score_a10_01", "score_a10_04") if col not in score_cols],
        "persistent_features_found": persistent_cols,
        "feature_sets": {key: len(value) for key, value in feature_sets.items()},
        "forbidden_columns_excluded": sorted(forbidden),
        "label_source_columns_detected": label_source_columns_detected,
        "missing_label_count": missing_label_count,
        "invalid_label_count": invalid_label_count,
        "label_check_result": "passed",
        "duplicate_check_result": "passed",
        "subject_fold_check_result": "passed",
        "hard_checks_result": "passed",
        "n_no_ez_subjects": 0,
        "forbidden_feature_check_result": "passed",
        "label_conflict_check_result": "passed",
        "fold_conflict_check_result": "passed",
        "missing_values_per_feature": {col: int(bank[col].isna().sum()) for features in feature_sets.values() for col in features if col in bank.columns},
    }
    (output_dir / "meta_ranker_score_bank_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    return bank


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build MetaRanker score bank.")
    parser.add_argument("--source_recommendations", required=True, type=Path)
    parser.add_argument("--persistent_features", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--allow_small_subject_count", action="store_true")
    args = parser.parse_args(argv)
    build_score_bank(args.source_recommendations, args.persistent_features, args.output_dir, allow_small_subject_count=bool(args.allow_small_subject_count))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
