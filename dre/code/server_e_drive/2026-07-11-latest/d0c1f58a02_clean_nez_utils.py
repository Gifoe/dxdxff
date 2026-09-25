from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from neuroez_c.raw_brainbert_data import load_embedding_table, normalize_channel_name


V3_LEDGER_COLUMNS = [
    "fold_idx",
    "subject_id",
    "center",
    "channel_id",
    "channel_name",
    "label_encoding_mode",
    "ez_label_value",
    "nez_label_value",
    "raw_binary_label",
    "clinical_true_ez",
    "clinical_true_nez",
    "true_ez",
    "true_nez",
    "score_ez_probability",
    "score_nez_probability",
    "rank_ez_desc",
    "rank_nez_desc",
    "p_clean_nez",
    "feature_non_nez_score",
    "feature_suspicious_z",
    "rank_feature_suspicious",
    "predicted_ez",
]

LABEL_ENCODING_COLUMNS = [
    "label_encoding_mode",
    "ez_label_value",
    "nez_label_value",
    "raw_binary_label",
    "clinical_true_ez",
    "clinical_true_nez",
]


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(child) for child in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _first_present(df: pd.DataFrame, names: Sequence[str]) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    return None


def _patient_zscore(values: pd.Series) -> pd.Series:
    arr = pd.to_numeric(values, errors="coerce").fillna(0.0).astype(float)
    std = float(arr.std(ddof=0))
    if not np.isfinite(std) or std < 1e-8:
        return pd.Series(np.zeros(len(arr), dtype=float), index=values.index)
    return (arr - float(arr.mean())) / std


def normalize_label_encoding_mode(label_encoding_mode: str | None) -> str:
    mode = str(label_encoding_mode or "ez1").strip().lower()
    if mode not in {"ez1", "ez0"}:
        raise ValueError("label_encoding_mode must be ez1 or ez0")
    return mode


def add_label_encoding_columns(df: pd.DataFrame, label_encoding_mode: str | None = None) -> pd.DataFrame:
    out = df.copy()
    if label_encoding_mode is None and "label_encoding_mode" in out.columns:
        modes = sorted(set(str(value).strip().lower() for value in out["label_encoding_mode"].dropna().astype(str)))
        mode = modes[0] if len(modes) == 1 else "ez1"
        if len(modes) > 1:
            raise ValueError(f"Ledger contains multiple label_encoding_mode values: {modes}")
    else:
        mode = normalize_label_encoding_mode(label_encoding_mode)
        if "label_encoding_mode" in out.columns:
            existing_modes = sorted(set(str(value).strip().lower() for value in out["label_encoding_mode"].dropna().astype(str)))
            if existing_modes and existing_modes != [mode]:
                raise ValueError(f"Ledger label_encoding_mode {existing_modes} does not match requested {mode}.")
    clinical_ez_source = "clinical_true_ez" if "clinical_true_ez" in out.columns else "true_ez"
    clinical_nez_source = "clinical_true_nez" if "clinical_true_nez" in out.columns else "true_nez"
    out["clinical_true_ez"] = pd.to_numeric(out[clinical_ez_source], errors="coerce").fillna(0).astype(int)
    if clinical_nez_source in out.columns:
        out["clinical_true_nez"] = pd.to_numeric(out[clinical_nez_source], errors="coerce").fillna(0).astype(int)
    else:
        out["clinical_true_nez"] = 1 - out["clinical_true_ez"]
    out["true_ez"] = out["clinical_true_ez"].astype(int)
    out["true_nez"] = out["clinical_true_nez"].astype(int)
    out["label_encoding_mode"] = mode
    if mode == "ez1":
        out["ez_label_value"] = 1
        out["nez_label_value"] = 0
        out["raw_binary_label"] = out["clinical_true_ez"].astype(int)
    else:
        out["ez_label_value"] = 0
        out["nez_label_value"] = 1
        out["raw_binary_label"] = out["clinical_true_nez"].astype(int)
    return out


def _read_allowed_subjects(
    allowed_subjects_ledger: str | Path | None = None,
    allowed_subjects_file: str | Path | None = None,
) -> set[str] | None:
    paths = [path for path in (allowed_subjects_ledger, allowed_subjects_file) if path]
    if not paths:
        return None
    allowed: set[str] = set()
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(f"Allowed-subjects file does not exist: {path}")
        if path.suffix.lower() in {".csv", ".tsv"}:
            sep = "\t" if path.suffix.lower() == ".tsv" else ","
            df = pd.read_csv(path, sep=sep)
            col = "subject_id" if "subject_id" in df.columns else df.columns[0]
            allowed.update(str(value) for value in df[col].dropna().astype(str))
        else:
            allowed.update(line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    return allowed


def apply_allowed_subject_filter(
    df: pd.DataFrame,
    *,
    allowed_subjects_ledger: str | Path | None = None,
    allowed_subjects_file: str | Path | None = None,
    require_n_patients: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    allowed = _read_allowed_subjects(allowed_subjects_ledger, allowed_subjects_file)
    out = df.copy()
    before_patients = int(out["subject_id"].astype(str).nunique()) if "subject_id" in out.columns else 0
    if allowed is not None:
        out = out[out["subject_id"].astype(str).isin(allowed)].copy()
    after_patients = int(out["subject_id"].astype(str).nunique()) if "subject_id" in out.columns else 0
    if require_n_patients is not None and int(require_n_patients) > 0 and after_patients != int(require_n_patients):
        raise ValueError(
            f"Expected {int(require_n_patients)} patients after allowed-subject filtering, got {after_patients}."
        )
    return out, {
        "allowed_subject_filter_used": allowed is not None,
        "n_allowed_subjects": None if allowed is None else len(allowed),
        "n_patients_before_allowed_filter": before_patients,
        "n_patients_after_allowed_filter": after_patients,
        "require_n_patients": None if require_n_patients is None else int(require_n_patients),
    }


def _coerce_v3_fold(df: pd.DataFrame, fold_idx: int, *, label_encoding_mode: str = "ez1") -> tuple[pd.DataFrame, list[str]]:
    missing: list[str] = []
    out = pd.DataFrame(index=df.index)
    fold_col = _first_present(df, ("fold_idx", "fold_id"))
    out["fold_idx"] = (
        pd.to_numeric(df[fold_col], errors="coerce").fillna(fold_idx).astype(int)
        if fold_col
        else int(fold_idx)
    )
    aliases = {
        "subject_id": ("subject_id", "patient_id"),
        "center": ("center", "source_center", "source_dataset"),
        "channel_id": ("channel_id", "channel_index", "contact_idx"),
        "channel_name": ("channel_name", "channel_names_norm", "channel_name_norm", "contact_name"),
    }
    for target, names in aliases.items():
        source = _first_present(df, names)
        if source is None:
            if target == "center":
                out[target] = "unknown"
            elif target == "channel_id":
                out[target] = np.arange(len(df), dtype=int)
            elif target == "predicted_ez":
                out[target] = 0
            elif target == "channel_name":
                out[target] = [f"ch{idx}" for idx in range(len(df))]
            else:
                missing.append(target)
        else:
            out[target] = df[source]

    predicted_ez_source = _first_present(df, ("predicted_ez", "pred_ez"))
    predicted_nez_source = _first_present(df, ("predicted_nez", "pred_nez"))
    pred_topk_source = _first_present(df, ("pred_topk",))
    mode = normalize_label_encoding_mode(label_encoding_mode)
    if predicted_ez_source:
        out["predicted_ez"] = df[predicted_ez_source]
    elif predicted_nez_source:
        out["predicted_ez"] = 1 - pd.to_numeric(df[predicted_nez_source], errors="coerce")
    elif pred_topk_source:
        raw_pred = pd.to_numeric(df[pred_topk_source], errors="coerce").fillna(0).astype(int)
        out["predicted_ez"] = raw_pred if mode == "ez1" else 1 - raw_pred
    else:
        out["predicted_ez"] = 0

    true_ez_col = _first_present(df, ("true_ez", "label_ez", "ez_label", "y_ez", "target_ez"))
    true_nez_col = _first_present(df, ("true_nez", "label_nez", "nez_label", "y_nez", "target_nez"))
    if true_ez_col:
        out["true_ez"] = pd.to_numeric(df[true_ez_col], errors="coerce")
    elif true_nez_col:
        out["true_ez"] = 1.0 - pd.to_numeric(df[true_nez_col], errors="coerce")
    else:
        missing.append("true_ez")
        out["true_ez"] = np.nan
    if true_nez_col:
        out["true_nez"] = pd.to_numeric(df[true_nez_col], errors="coerce")
    else:
        out["true_nez"] = 1.0 - pd.to_numeric(out["true_ez"], errors="coerce")

    score_ez_col = _first_present(df, ("score_ez_probability", "score_ez", "score_ez_final"))
    score_nez_col = _first_present(df, ("score_nez_probability", "score_nez"))
    score_eval_col = _first_present(df, ("score_eval",))
    if score_ez_col:
        out["score_ez_probability"] = pd.to_numeric(df[score_ez_col], errors="coerce")
    elif score_nez_col:
        out["score_ez_probability"] = 1.0 - pd.to_numeric(df[score_nez_col], errors="coerce")
    elif score_eval_col:
        score_eval = pd.to_numeric(df[score_eval_col], errors="coerce")
        out["score_ez_probability"] = score_eval if mode == "ez1" else 1.0 - score_eval
    else:
        missing.append("score_ez_probability")
        out["score_ez_probability"] = np.nan
    if score_nez_col:
        out["score_nez_probability"] = pd.to_numeric(df[score_nez_col], errors="coerce")
    elif score_ez_col:
        out["score_nez_probability"] = 1.0 - pd.to_numeric(out["score_ez_probability"], errors="coerce")
    elif score_eval_col:
        score_eval = pd.to_numeric(df[score_eval_col], errors="coerce")
        out["score_nez_probability"] = 1.0 - score_eval if mode == "ez1" else score_eval
    else:
        out["score_nez_probability"] = 1.0 - pd.to_numeric(out["score_ez_probability"], errors="coerce")

    out["subject_id"] = out["subject_id"].astype(str)
    out["center"] = out["center"].fillna("unknown").astype(str)
    out["channel_name"] = out["channel_name"].astype(str)
    out["channel_id"] = pd.to_numeric(out["channel_id"], errors="coerce").fillna(-1).astype(int)
    out["true_ez"] = pd.to_numeric(out["true_ez"], errors="coerce").fillna(0).astype(int)
    out["true_nez"] = pd.to_numeric(out["true_nez"], errors="coerce").fillna(1).astype(int)
    out = add_label_encoding_columns(out, label_encoding_mode=mode)
    out["score_ez_probability"] = pd.to_numeric(out["score_ez_probability"], errors="coerce")
    out["score_nez_probability"] = pd.to_numeric(out["score_nez_probability"], errors="coerce")
    out["p_clean_nez"] = out["score_nez_probability"].where(
        out["score_nez_probability"].notna(),
        1.0 - out["score_ez_probability"],
    )
    out["feature_non_nez_score"] = 1.0 - out["p_clean_nez"]
    out["feature_suspicious_z"] = out.groupby("subject_id", sort=False)["feature_non_nez_score"].transform(_patient_zscore)
    out["rank_ez_desc"] = out.groupby("subject_id")["score_ez_probability"].rank(ascending=False, method="first").astype(int)
    out["rank_nez_desc"] = out.groupby("subject_id")["p_clean_nez"].rank(ascending=False, method="first").astype(int)
    out["rank_feature_suspicious"] = (
        out.groupby("subject_id")["feature_non_nez_score"].rank(ascending=False, method="first").astype(int)
    )
    out["predicted_ez"] = pd.to_numeric(out["predicted_ez"], errors="coerce").fillna(0).astype(int)
    return out[V3_LEDGER_COLUMNS], sorted(set(missing))


def _ledger_audit(ledger: pd.DataFrame, missing_columns: Sequence[str]) -> dict[str, Any]:
    dup_frame = ledger[["fold_idx", "subject_id", "channel_name"]].copy()
    dup_frame["channel_norm"] = dup_frame["channel_name"].map(normalize_channel_name)
    duplicate_mask = dup_frame.duplicated(["fold_idx", "subject_id", "channel_norm"], keep=False)
    modes = sorted(set(ledger["label_encoding_mode"].astype(str))) if "label_encoding_mode" in ledger.columns and not ledger.empty else []
    return {
        "n_rows": int(len(ledger)),
        "n_patients": int(ledger["subject_id"].nunique()) if not ledger.empty else 0,
        "per_fold_patient_counts": {
            str(k): int(v) for k, v in ledger.groupby("fold_idx")["subject_id"].nunique().to_dict().items()
        },
        "per_center_counts": {str(k): int(v) for k, v in ledger.groupby("center").size().to_dict().items()},
        "true_ez_count": int(pd.to_numeric(ledger["true_ez"], errors="coerce").fillna(0).sum()),
        "true_nez_count": int(pd.to_numeric(ledger["true_nez"], errors="coerce").fillna(0).sum()),
        "missing_columns": list(missing_columns),
        "duplicate_subject_channel_rows": int(duplicate_mask.sum()),
        "label_encoding_mode": modes[0] if len(modes) == 1 else None,
        "ez_label_value": int(ledger["ez_label_value"].iloc[0]) if "ez_label_value" in ledger.columns and not ledger.empty else None,
        "nez_label_value": int(ledger["nez_label_value"].iloc[0]) if "nez_label_value" in ledger.columns and not ledger.empty else None,
        "raw_binary_label_used_as_clinical_label": False,
        "true_ez_equals_clinical_true_ez": bool((ledger["true_ez"].astype(int) == ledger["clinical_true_ez"].astype(int)).all()) if not ledger.empty else True,
        "true_nez_equals_clinical_true_nez": bool((ledger["true_nez"].astype(int) == ledger["clinical_true_nez"].astype(int)).all()) if not ledger.empty else True,
    }


def export_v3_clean_nez_ledger(
    v3_output_dir: str | Path,
    output_path: str | Path,
    *,
    fold_start: int = 1,
    fold_end: int = 5,
    allowed_subjects_ledger: str | Path | None = None,
    allowed_subjects_file: str | Path | None = None,
    require_n_patients: int | None = None,
    label_encoding_mode: str = "ez1",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    v3_output_dir = Path(v3_output_dir)
    output_path = Path(output_path)
    frames: list[pd.DataFrame] = []
    missing_by_fold: dict[str, list[str]] = {}
    for fold_idx in range(int(fold_start), int(fold_end) + 1):
        path = v3_output_dir / f"test_channel_predictions_neuroez_v2_fold_{fold_idx}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing V3 fold channel prediction CSV: {path}")
        frame, missing = _coerce_v3_fold(pd.read_csv(path), fold_idx, label_encoding_mode=label_encoding_mode)
        frames.append(frame)
        if missing:
            missing_by_fold[str(fold_idx)] = missing
    ledger = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=V3_LEDGER_COLUMNS)
    missing_columns = sorted(set(item for values in missing_by_fold.values() for item in values))
    ledger, filter_audit = apply_allowed_subject_filter(
        ledger,
        allowed_subjects_ledger=allowed_subjects_ledger,
        allowed_subjects_file=allowed_subjects_file,
        require_n_patients=require_n_patients,
    )
    audit = {**_ledger_audit(ledger, missing_columns), **filter_audit, "missing_columns_by_fold": missing_by_fold}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ledger.to_csv(output_path, index=False)
    with (output_path.parent / "v3_clean_nez_ledger_audit.json").open("w", encoding="utf-8") as fout:
        json.dump(json_safe(audit), fout, indent=2, ensure_ascii=False, sort_keys=True)
    if missing_columns:
        raise ValueError(f"Missing required V3 clean-NEZ ledger columns: {missing_by_fold}")
    if int(audit["duplicate_subject_channel_rows"]) > 0:
        raise ValueError("Duplicate V3 clean-NEZ subject/channel rows after normalization.")
    return ledger, audit


def _rawbb_cols(df: pd.DataFrame, prefix: str) -> list[str]:
    cols = [col for col in df.columns if str(col).startswith(prefix + "_")]
    return sorted(cols, key=lambda col: int(str(col).rsplit("_", 1)[1]))


def _cosine_distance(x: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    x_norm = np.linalg.norm(x, axis=1)
    c_norm = float(np.linalg.norm(centroid))
    denom = np.maximum(x_norm * max(c_norm, 1e-12), 1e-12)
    sim = np.matmul(x, centroid) / denom
    return 1.0 - sim


def _euclidean_distance(x: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    return np.linalg.norm(x - centroid.reshape(1, -1), axis=1)


def _mahalanobis_shrinkage_distance(x: np.ndarray, anchors: np.ndarray, centroid: np.ndarray) -> tuple[np.ndarray, bool]:
    if anchors.shape[0] < 3:
        return np.zeros((x.shape[0],), dtype=float), False
    var = np.var(anchors - centroid.reshape(1, -1), axis=0) + 1e-3
    diff = x - centroid.reshape(1, -1)
    return np.sqrt(np.sum((diff * diff) / var.reshape(1, -1), axis=1)), True


def _anchor_mask(
    group: pd.DataFrame,
    *,
    anchor_strategy: str,
    anchor_quantile: float,
    min_anchors: int,
    anchor_topk: int | None = None,
    anchor_threshold: float | None = None,
) -> pd.Series:
    scores = pd.to_numeric(group["p_clean_nez"], errors="coerce").fillna(0.0)
    strategy = str(anchor_strategy).strip().lower()
    if strategy == "quantile":
        threshold = float(scores.quantile(float(anchor_quantile)))
        mask = scores >= threshold
    elif strategy == "topk":
        k = max(1, int(anchor_topk or min_anchors))
        mask = pd.Series(False, index=group.index)
        mask.loc[scores.sort_values(ascending=False, kind="mergesort").head(k).index] = True
    elif strategy == "threshold":
        if anchor_threshold is None:
            raise ValueError("anchor_strategy='threshold' requires anchor_threshold.")
        mask = scores >= float(anchor_threshold)
    else:
        raise ValueError("anchor_strategy must be one of quantile, topk, threshold.")
    if int(mask.sum()) < int(min_anchors):
        top_idx = scores.sort_values(ascending=False, kind="mergesort").head(int(min_anchors)).index
        mask = pd.Series(False, index=group.index)
        mask.loc[top_idx] = True
    return mask.astype(bool)


def _load_fold_embeddings(embedding_dir: Path, fold_idx: int) -> pd.DataFrame:
    df = load_embedding_table(embedding_dir / f"rawbrainbert_patient_channel_embeddings_fold_{int(fold_idx)}")
    if not {"subject_id", "channel_name"}.issubset(df.columns):
        raise ValueError(f"Embedding table for fold {fold_idx} must contain subject_id and channel_name.")
    out = df.copy()
    out["subject_id"] = out["subject_id"].astype(str)
    out["channel_norm"] = out["channel_name"].map(normalize_channel_name)
    if out.duplicated(["subject_id", "channel_norm"], keep=False).any():
        raise ValueError(f"Duplicate embedding subject/channel keys for fold {fold_idx}.")
    return out


def build_clean_nez_distance_ledger(
    v3_ledger: str | Path,
    embedding_dir: str | Path,
    output_dir: str | Path,
    *,
    anchor_strategy: str = "quantile",
    anchor_quantile: float = 0.70,
    min_anchors: int = 5,
    anchor_topk: int | None = None,
    anchor_threshold: float | None = None,
    distance_modes: Sequence[str] = ("cosine", "euclidean"),
    allowed_subjects_ledger: str | Path | None = None,
    allowed_subjects_file: str | Path | None = None,
    require_n_patients: int | None = None,
    label_encoding_mode: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    ledger = pd.read_csv(v3_ledger)
    ledger["subject_id"] = ledger["subject_id"].astype(str)
    ledger = add_label_encoding_columns(ledger, label_encoding_mode=label_encoding_mode)
    ledger["channel_norm"] = ledger["channel_name"].map(normalize_channel_name)
    ledger, filter_audit = apply_allowed_subject_filter(
        ledger,
        allowed_subjects_ledger=allowed_subjects_ledger,
        allowed_subjects_file=allowed_subjects_file,
        require_n_patients=require_n_patients,
    )
    embedding_dir = Path(embedding_dir)
    merged_frames: list[pd.DataFrame] = []
    missing_embeddings: dict[str, int] = {}
    for fold_idx, fold_rows in ledger.groupby("fold_idx", sort=True):
        embeddings = _load_fold_embeddings(embedding_dir, int(fold_idx))
        before = len(fold_rows)
        merged = fold_rows.merge(
            embeddings.drop(columns=[col for col in ("fold_idx", "channel_name") if col in embeddings.columns]),
            on=["subject_id", "channel_norm"],
            how="left",
            suffixes=("", "_rawbb"),
        )
        if len(merged) != before:
            raise ValueError(f"Embedding merge changed row count for fold {fold_idx}: before={before}, after={len(merged)}")
        raw_cols = _rawbb_cols(merged, "rawbb_all")
        if not raw_cols:
            raise ValueError(f"No rawbb_all embedding columns found for fold {fold_idx}.")
        missing_embeddings[str(fold_idx)] = int(merged[raw_cols].isna().any(axis=1).sum())
        if missing_embeddings[str(fold_idx)]:
            raise ValueError(f"Missing RawBrainBERT embeddings for fold {fold_idx}: {missing_embeddings[str(fold_idx)]} rows.")
        merged_frames.append(merged)
    work = pd.concat(merged_frames, ignore_index=True) if merged_frames else ledger.copy()
    work["is_pseudo_clean_nez_anchor"] = False
    anchor_counts: dict[str, int] = {}
    low_anchor_patients: list[str] = []
    pseudo_anchor_precision_values: list[float] = []
    for subject_id, group in work.groupby("subject_id", sort=False):
        mask = _anchor_mask(
            group,
            anchor_strategy=anchor_strategy,
            anchor_quantile=anchor_quantile,
            min_anchors=min_anchors,
            anchor_topk=anchor_topk,
            anchor_threshold=anchor_threshold,
        )
        work.loc[mask.index, "is_pseudo_clean_nez_anchor"] = mask.to_numpy(dtype=bool)
        count = int(mask.sum())
        anchor_counts[str(subject_id)] = count
        if count < int(min_anchors):
            low_anchor_patients.append(str(subject_id))
        if count > 0 and "true_nez" in group.columns:
            pseudo_anchor_precision_values.append(
                float(pd.to_numeric(group.loc[mask, "true_nez"], errors="coerce").fillna(0).astype(int).mean())
            )

    mahalanobis_fallbacks: dict[str, int] = {}
    for emb_name, out_stem in (("rawbb_all", "all"), ("rawbb_onset", "onset"), ("rawbb_preictal", "preictal")):
        cols = _rawbb_cols(work, emb_name)
        if not cols:
            continue
        for mode in distance_modes:
            work[f"raw_dist_{out_stem}_{mode}"] = 0.0
        for subject_id, group in work.groupby("subject_id", sort=False):
            anchor_group = group[group["is_pseudo_clean_nez_anchor"].astype(bool)]
            if anchor_group.empty:
                anchor_group = group.sort_values("p_clean_nez", ascending=False, kind="mergesort").head(max(1, int(min_anchors)))
            centroid = anchor_group[cols].to_numpy(dtype=np.float64).mean(axis=0)
            x = group[cols].to_numpy(dtype=np.float64)
            if "cosine" in distance_modes:
                work.loc[group.index, f"raw_dist_{out_stem}_cosine"] = _cosine_distance(x, centroid)
            if "euclidean" in distance_modes:
                work.loc[group.index, f"raw_dist_{out_stem}_euclidean"] = _euclidean_distance(x, centroid)
            if "mahalanobis_shrinkage" in distance_modes:
                dist, ok = _mahalanobis_shrinkage_distance(x, anchor_group[cols].to_numpy(dtype=np.float64), centroid)
                work.loc[group.index, f"raw_dist_{out_stem}_mahalanobis_shrinkage"] = dist
                if not ok:
                    mahalanobis_fallbacks[out_stem] = mahalanobis_fallbacks.get(out_stem, 0) + 1
        source_col = f"raw_dist_{out_stem}_cosine" if f"raw_dist_{out_stem}_cosine" in work.columns else f"raw_dist_{out_stem}_euclidean"
        work[f"raw_dist_{out_stem}_z"] = work.groupby("subject_id", sort=False)[source_col].transform(_patient_zscore)

    for required in ("raw_dist_all_z", "raw_dist_onset_z", "raw_dist_preictal_z"):
        if required not in work.columns:
            work[required] = 0.0
    work["n_pseudo_anchors"] = work["subject_id"].map(anchor_counts).fillna(0).astype(int)
    keep_cols = [
        "fold_idx",
        "subject_id",
        "center",
        "channel_id",
        "channel_name",
        "label_encoding_mode",
        "ez_label_value",
        "nez_label_value",
        "raw_binary_label",
        "clinical_true_ez",
        "clinical_true_nez",
        "p_clean_nez",
        "feature_non_nez_score",
        "feature_suspicious_z",
        "is_pseudo_clean_nez_anchor",
        "raw_dist_all_cosine",
        "raw_dist_onset_cosine",
        "raw_dist_preictal_cosine",
        "raw_dist_all_euclidean",
        "raw_dist_onset_euclidean",
        "raw_dist_preictal_euclidean",
        "raw_dist_all_z",
        "raw_dist_onset_z",
        "raw_dist_preictal_z",
        "n_pseudo_anchors",
        "true_ez",
        "true_nez",
        "n_records",
    ]
    output = work[[col for col in keep_cols if col in work.columns]].copy()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_dir / "clean_nez_distance_ledger.csv", index=False)
    counts = list(anchor_counts.values())
    audit = {
        **filter_audit,
        "n_patients": int(output["subject_id"].nunique()) if not output.empty else 0,
        "anchor_count_distribution": {
            "min": int(min(counts)) if counts else 0,
            "max": int(max(counts)) if counts else 0,
            "mean": float(np.mean(counts)) if counts else 0.0,
        },
        "anchor_strategy": str(anchor_strategy),
        "anchor_quantile": float(anchor_quantile),
        "anchor_topk": None if anchor_topk is None else int(anchor_topk),
        "anchor_threshold": None if anchor_threshold is None else float(anchor_threshold),
        "min_anchors": int(min_anchors),
        "patients_with_low_anchor_count": low_anchor_patients,
        "whether_true_label_used_for_anchor_selection": False,
        "pseudo_anchor_precision_diagnostic": float(np.mean(pseudo_anchor_precision_values)) if pseudo_anchor_precision_values else None,
        "diagnostic_uses_true_labels": True,
        "mahalanobis_fallbacks_by_embedding": mahalanobis_fallbacks,
        "missing_embedding_rows_by_fold": missing_embeddings,
    }
    with (output_dir / "clean_nez_anchor_audit.json").open("w", encoding="utf-8") as fout:
        json.dump(json_safe(audit), fout, indent=2, ensure_ascii=False, sort_keys=True)
    return output, audit


__all__ = [
    "V3_LEDGER_COLUMNS",
    "LABEL_ENCODING_COLUMNS",
    "add_label_encoding_columns",
    "apply_allowed_subject_filter",
    "build_clean_nez_distance_ledger",
    "export_v3_clean_nez_ledger",
    "json_safe",
]
