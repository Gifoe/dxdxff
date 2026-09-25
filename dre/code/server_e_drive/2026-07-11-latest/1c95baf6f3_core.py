"""Traditional NeuroEZ-C baselines from engineered window_features only.

Foundation-model baselines are intentionally separate because they consume raw_waveform.
This module evaluates clinical-marker and classical-ML baselines on cache
window_features without modifying A9v3/A9v8 training, losses, labels, or split logic.
"""

from __future__ import annotations

import argparse
import itertools
import json
import pickle
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.utils.class_weight import compute_sample_weight

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data_factory import build_outer_splits, split_train_val_subjects


A9V3_GATE = {
    "patient_macro_f1": 0.642714,
    "patient_macro_ez_f1": 0.470591,
    "patient_macro_auprc_ez": 0.518635,
    "patient_macro_ez_mrr": 0.715136,
    "top1_is_ez_rate": 0.600000,
}

SUMMARY_COLUMNS = [
    "method",
    "method_group",
    "n_folds",
    "n_patients",
    "patient_macro_f1",
    "patient_macro_ez_f1",
    "patient_macro_auprc_ez",
    "patient_macro_ez_mrr",
    "top1_is_ez_rate",
    "delta_f1_vs_a9v3",
    "delta_ez_f1_vs_a9v3",
    "delta_auprc_vs_a9v3",
    "delta_mrr_vs_a9v3",
    "delta_top1_vs_a9v3",
    "passes_a9v3_gate",
]

PATIENT_ROW_COLUMNS = [
    "method",
    "fold_idx",
    "subject_id",
    "center",
    "valid_channel_count",
    "ez_channel_count",
    "ez_fraction",
    "patient_macro_f1",
    "patient_macro_ez_f1",
    "patient_macro_auprc_ez",
    "patient_macro_ez_mrr",
    "top1_is_ez",
    "selected_hyperparams_json",
]

CHANNEL_PREDICTION_COLUMNS = [
    "method",
    "fold_idx",
    "subject_id",
    "center",
    "channel_name",
    "label_ez",
    "label_nez",
    "score_ez",
    "score_nez",
    "pred_ez_topk",
    "rank_ez",
    "marker_column",
    "selected_hyperparams_json",
]

META_COLUMNS = {
    "subject_id",
    "center",
    "channel_name",
    "label_ez",
    "valid_channel",
    "n_records",
    "ez_fraction",
    "valid_channel_count",
    "ez_channel_count",
    "record_count_feature",
    "source_center",
    "source_dataset",
}


@dataclass(frozen=True)
class MethodResult:
    method: str
    method_group: str
    fold_idx: int
    selected_params: Dict[str, Any]
    channel_predictions: List[Dict[str, Any]]
    patient_rows: List[Dict[str, Any]]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window_cache_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split_strategy", default="5fold")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--positive_label", choices=["ez"], default="ez")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--methods", default="all", help="Comma-separated method names for smoke runs, or 'all'.")
    parser.add_argument(
        "--allow_incomplete_methods",
        action="store_true",
        help="Write diagnostics instead of failing when a requested completed method has fewer than n_splits folds.",
    )
    parser.add_argument(
        "--skip_mlp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip the optional sklearn MLP baseline by default to keep validation patient-wise.",
    )
    return parser.parse_args(argv)


def load_window_cache(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("rb") as handle:
        cache = pickle.load(handle)
    if not isinstance(cache, dict) or "run_records" not in cache or "patient_index" not in cache:
        raise ValueError("Window cache must be a dict with run_records and patient_index.")
    if not cache["run_records"]:
        raise ValueError("Window cache has no run_records.")
    if not cache["patient_index"]:
        raise ValueError("Window cache has no patient_index.")
    return cache


def normalize_center(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return "unknown"
    if "pediatric" in text or "ped" == text:
        return "pediatric"
    if "multicenter" in text or "multi" in text:
        return "multicenter"
    if "hup" in text:
        return "hup"
    if "lzu" in text:
        return "lzu"
    if text in {"all", "unknown"}:
        return text
    return text.split(":")[0].strip().lower() or "unknown"


def infer_center(subject_id: str, patient_meta: Mapping[str, Any], run_record: Mapping[str, Any], sample: Mapping[str, Any]) -> str:
    for source in (patient_meta, run_record, sample):
        for key in ("source_center", "center", "source_dataset"):
            if key in source and source[key] not in (None, ""):
                return normalize_center(source[key])
    if ":" in str(subject_id):
        return normalize_center(str(subject_id).split(":", 1)[0])
    return "unknown"


def _feature_names(sample: Mapping[str, Any], n_features: int) -> List[str]:
    names = list(sample.get("window_feature_names") or sample.get("feature_names") or [])
    if len(names) != n_features:
        names = [f"feature_{idx}" for idx in range(n_features)]
    return [str(name) for name in names]


def _channel_names(run_record: Mapping[str, Any], n_channels: int) -> List[str]:
    sample = run_record.get("sample") or {}
    names = list(run_record.get("channel_names_norm") or run_record.get("channel_names") or sample.get("channel_names_norm") or sample.get("channel_names") or [])
    if len(names) != n_channels:
        names = [f"ch{idx}" for idx in range(n_channels)]
    return [str(name) for name in names]


def _canonical_labels(patient_meta: Mapping[str, Any]) -> Tuple[List[str], np.ndarray, np.ndarray] | None:
    channels = list(patient_meta.get("canonical_channels") or [])
    labels = np.asarray(patient_meta.get("labels", []), dtype=np.float32).reshape(-1)
    if not channels or labels.size != len(channels):
        return None
    mask_raw = patient_meta.get("label_mask", None)
    if mask_raw is None:
        mask = np.ones(labels.shape[0], dtype=bool)
    else:
        mask = np.asarray(mask_raw, dtype=bool).reshape(-1)
        if mask.size != labels.size:
            mask = np.ones(labels.shape[0], dtype=bool)
    return [str(channel) for channel in channels], (labels > 0.5).astype(int), mask


def validate_traditional_rows_against_patient_index(rows: pd.DataFrame, patient_index: Mapping[str, Any], output_dir: str | Path) -> None:
    out = Path(output_dir)
    required = {"subject_id", "channel_name", "center", "label_ez"}
    missing_cols = sorted(required - set(rows.columns))
    if missing_cols:
        raise ValueError(f"Traditional rows missing required columns: {missing_cols}")
    labels = pd.to_numeric(rows["label_ez"], errors="coerce")
    missing_label = rows[labels.isna()].copy()
    if not missing_label.empty:
        out.mkdir(parents=True, exist_ok=True)
        missing_label.to_csv(out / "traditional_label_conflict_rows.csv", index=False)
        raise ValueError("Traditional rows contain missing label_ez")
    invalid_label = rows[~labels.isin([0, 1])].copy()
    if not invalid_label.empty:
        out.mkdir(parents=True, exist_ok=True)
        invalid_label.to_csv(out / "traditional_label_conflict_rows.csv", index=False)
        raise ValueError("Traditional rows contain non-binary label_ez")
    if "label_nez" in rows.columns:
        label_nez = pd.to_numeric(rows["label_nez"], errors="coerce")
        bad_nez = rows[label_nez.notna() & label_nez.ne(1 - labels)].copy()
        if not bad_nez.empty:
            out.mkdir(parents=True, exist_ok=True)
            bad_nez.to_csv(out / "traditional_label_conflict_rows.csv", index=False)
            raise ValueError("Traditional rows contain inconsistent label_nez")
    dup = rows[rows.duplicated(["subject_id", "channel_name"], keep=False)].copy()
    if not dup.empty:
        out.mkdir(parents=True, exist_ok=True)
        dup.to_csv(out / "traditional_duplicate_channel_rows.csv", index=False)
        raise ValueError("Traditional rows contain duplicate subject_id + channel_name rows")
    conflict_rows: List[Dict[str, Any]] = []
    noncanonical_rows: List[pd.DataFrame] = []
    for subject_id, group in rows.groupby("subject_id", sort=False):
        patient_meta = patient_index.get(str(subject_id), {}) or {}
        canonical = _canonical_labels(patient_meta)
        if canonical is None:
            continue
        channels, labels, mask = canonical
        label_map = {channels[idx]: int(labels[idx]) for idx in range(len(channels)) if bool(mask[idx])}
        bad = group[~group["channel_name"].astype(str).isin(label_map.keys())].copy()
        if not bad.empty:
            noncanonical_rows.append(bad)
        for _, row in group.iterrows():
            channel_name = str(row["channel_name"])
            if channel_name not in label_map:
                continue
            observed = int(pd.to_numeric(pd.Series([row["label_ez"]]), errors="coerce").fillna(-1).iloc[0])
            expected = int(label_map[channel_name])
            if observed != expected:
                conflict_rows.append(
                    {
                        "subject_id": str(subject_id),
                        "channel_name": channel_name,
                        "row_label_ez": observed,
                        "patient_index_label_ez": expected,
                    }
                )
    if noncanonical_rows:
        out.mkdir(parents=True, exist_ok=True)
        pd.concat(noncanonical_rows, ignore_index=True).to_csv(out / "traditional_noncanonical_channel_rows.csv", index=False)
        raise ValueError("Traditional rows contain noncanonical channels")
    if conflict_rows:
        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(conflict_rows).to_csv(out / "traditional_label_conflict_rows.csv", index=False)
        raise ValueError("Traditional rows conflict with patient_index canonical labels")
    if len(patient_index) >= 90:
        n_subjects = int(rows["subject_id"].nunique())
        if n_subjects != 90:
            raise ValueError(f"Traditional all90 rows must contain 90 subjects, got {n_subjects}")
        ez_by_subject = rows.groupby("subject_id")["label_ez"].sum()
        no_ez = ez_by_subject[ez_by_subject.le(0)]
        if not no_ez.empty:
            no_ez.reset_index().to_csv(out / "traditional_no_ez_subject_rows.csv", index=False)
            raise ValueError("Traditional rows contain subjects with no EZ channel")


def validate_traditional_output_against_patient_index(
    channel_df: pd.DataFrame,
    patient_index: Mapping[str, Any],
    output_dir: str | Path,
    *,
    expected_n_splits: int = 5,
) -> Dict[str, Any]:
    out = Path(output_dir)
    if channel_df.empty:
        return {
            "folds_detected": [],
            "n_noncanonical_output_rows": 0,
            "n_label_conflict_output_rows": 0,
            "n_duplicate_output_rows": 0,
        }
    dup = channel_df[channel_df.duplicated(["method", "subject_id", "channel_name"], keep=False)].copy()
    if not dup.empty:
        out.mkdir(parents=True, exist_ok=True)
        dup.to_csv(out / "traditional_output_duplicate_method_channel_rows.csv", index=False)
        raise ValueError("Traditional output contains duplicate method + subject_id + channel_name rows")
    noncanonical_rows: List[pd.DataFrame] = []
    conflict_rows: List[Dict[str, Any]] = []
    for subject_id, group in channel_df.groupby("subject_id", sort=False):
        patient_meta = patient_index.get(str(subject_id), {}) or {}
        canonical = _canonical_labels(patient_meta)
        if canonical is None:
            continue
        channels, labels, mask = canonical
        label_map = {channels[idx]: int(labels[idx]) for idx in range(len(channels)) if bool(mask[idx])}
        bad = group[~group["channel_name"].astype(str).isin(label_map.keys())].copy()
        if not bad.empty:
            noncanonical_rows.append(bad)
        for _, row in group.iterrows():
            channel_name = str(row["channel_name"])
            if channel_name not in label_map:
                continue
            label_ez = int(pd.to_numeric(pd.Series([row["label_ez"]]), errors="coerce").fillna(-1).iloc[0])
            expected = int(label_map[channel_name])
            if label_ez != expected:
                conflict_rows.append(
                    {
                        "method": str(row.get("method", "")),
                        "fold_idx": row.get("fold_idx", ""),
                        "subject_id": str(subject_id),
                        "channel_name": channel_name,
                        "output_label_ez": label_ez,
                        "patient_index_label_ez": expected,
                    }
                )
            if "label_nez" in channel_df.columns and not pd.isna(row.get("label_nez", np.nan)):
                label_nez = int(pd.to_numeric(pd.Series([row["label_nez"]]), errors="coerce").fillna(-1).iloc[0])
                if label_nez != 1 - label_ez:
                    conflict_rows.append(
                        {
                            "method": str(row.get("method", "")),
                            "fold_idx": row.get("fold_idx", ""),
                            "subject_id": str(subject_id),
                            "channel_name": channel_name,
                            "output_label_ez": label_ez,
                            "patient_index_label_ez": expected,
                            "conflict_reason": "label_nez_inconsistent",
                        }
                    )
    if noncanonical_rows:
        out.mkdir(parents=True, exist_ok=True)
        pd.concat(noncanonical_rows, ignore_index=True).to_csv(out / "traditional_output_noncanonical_channel_rows.csv", index=False)
        raise ValueError("Traditional output contains noncanonical channels")
    if conflict_rows:
        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(conflict_rows).to_csv(out / "traditional_output_label_conflict_rows.csv", index=False)
        raise ValueError("Traditional output labels conflict with patient_index canonical labels")
    folds_detected = sorted(int(v) for v in pd.to_numeric(channel_df["fold_idx"], errors="coerce").dropna().unique().tolist()) if "fold_idx" in channel_df else []
    expected_folds = list(range(1, int(expected_n_splits) + 1))
    if len(patient_index) >= 90 and expected_n_splits and folds_detected and folds_detected != expected_folds:
        raise ValueError(f"Traditional output folds must be {expected_folds}, got {folds_detected}")
    if len(patient_index) >= 90:
        for method, group in channel_df.groupby("method", sort=False):
            n_subjects = int(group["subject_id"].nunique())
            if n_subjects != 90:
                raise ValueError(f"Traditional output method {method} must contain 90 subjects, got {n_subjects}")
            no_ez = group.groupby("subject_id")["label_ez"].sum()
            no_ez = no_ez[no_ez.le(0)]
            if not no_ez.empty:
                no_ez.reset_index().to_csv(out / "traditional_output_no_ez_subject_rows.csv", index=False)
                raise ValueError(f"Traditional output method {method} contains subjects with no EZ channel")
    return {
        "folds_detected": folds_detected,
        "n_noncanonical_output_rows": 0,
        "n_label_conflict_output_rows": 0,
        "n_duplicate_output_rows": 0,
    }


def _record_channel_vector(windows_by_feature: np.ndarray) -> np.ndarray:
    arr = np.asarray(windows_by_feature, dtype=np.float32)
    mean = np.nanmean(arr, axis=0)
    max_v = np.nanmax(arr, axis=0)
    std = np.nanstd(arr, axis=0)
    k = max(1, int(np.ceil(arr.shape[0] * 0.20)))
    top_mean = np.nanmean(np.sort(arr, axis=0)[-k:, :], axis=0)
    return np.concatenate([mean, max_v, std, top_mean]).astype(np.float32)


def build_patient_channel_rows(cache: Mapping[str, Any]) -> pd.DataFrame:
    run_records = list(cache["run_records"])
    patient_index = {str(k): v for k, v in dict(cache["patient_index"]).items()}
    grouped: Dict[Tuple[str, str], Dict[str, Any]] = {}
    feature_names_out: List[str] | None = None
    warnings_list: List[str] = []
    n_channels_skipped_not_in_canonical = 0
    n_label_conflict_with_run_record_labels = 0

    for run_record in run_records:
        subject_id = str(run_record.get("subject_id", ""))
        if not subject_id:
            continue
        patient_meta = patient_index.get(subject_id, {}) or {}
        sample = run_record.get("sample") or {}
        window_features = np.asarray(sample.get("window_features"), dtype=np.float32)
        if window_features.ndim != 3:
            continue
        n_windows, n_channels, n_features = window_features.shape
        if n_windows == 0 or n_channels == 0 or n_features == 0:
            continue
        channel_names = _channel_names(run_record, n_channels)
        canonical = _canonical_labels(patient_meta)
        run_labels = np.asarray(run_record.get("labels", []), dtype=np.float32).reshape(-1)
        if canonical is None:
            if len(patient_index) >= 90:
                raise ValueError(f"Missing canonical_channels/labels in patient_index for all90 subject {subject_id}")
            if run_labels.size != n_channels:
                continue
            if "used_run_record_labels_fallback" not in warnings_list:
                warnings_list.append("used_run_record_labels_fallback")
            canonical_label_map = {str(channel_names[idx]): int(run_labels[idx] > 0.5) for idx in range(n_channels)}
        else:
            canonical_channels, canonical_labels, label_mask = canonical
            canonical_label_map = {canonical_channels[idx]: int(canonical_labels[idx]) for idx in range(len(canonical_channels)) if bool(label_mask[idx])}
            if run_labels.size == n_channels:
                for idx, channel_name in enumerate(channel_names):
                    channel_name = str(channel_name)
                    if channel_name in canonical_label_map and int(run_labels[idx] > 0.5) != canonical_label_map[channel_name]:
                        n_label_conflict_with_run_record_labels += 1
        base_names = _feature_names(sample, n_features)
        if feature_names_out is None:
            feature_names_out = [
                f"{agg}__{name}"
                for agg in ("mean", "max", "std", "top20pct_mean")
                for name in base_names
            ]
        center = infer_center(subject_id, patient_meta, run_record, sample)

        for channel_idx, channel_name in enumerate(channel_names):
            channel_name = str(channel_name)
            if channel_name not in canonical_label_map:
                n_channels_skipped_not_in_canonical += 1
                continue
            key = (subject_id, channel_name)
            entry = grouped.setdefault(
                key,
                {
                    "subject_id": subject_id,
                    "center": center,
                    "source_center": center,
                    "channel_name": channel_name,
                    "label_ez": int(canonical_label_map[channel_name]),
                    "vectors": [],
                },
            )
            entry["vectors"].append(_record_channel_vector(window_features[:, channel_idx, :]))

    if not grouped:
        raise ValueError("No valid patient-channel rows could be built from window_features.")
    feature_names_out = feature_names_out or []
    rows: List[Dict[str, Any]] = []
    for entry in grouped.values():
        vectors = np.vstack(entry.pop("vectors"))
        mean_vec = np.nanmean(vectors, axis=0)
        max_vec = np.nanmax(vectors, axis=0)
        final_vec = np.concatenate([mean_vec, max_vec, np.asarray([vectors.shape[0]], dtype=np.float32)])
        row = dict(entry)
        row["valid_channel"] = True
        row["n_records"] = int(vectors.shape[0])
        for idx, name in enumerate(feature_names_out):
            row[f"record_mean__{name}"] = float(mean_vec[idx])
            row[f"record_max__{name}"] = float(max_vec[idx])
        row["record_count_feature"] = float(final_vec[-1])
        rows.append(row)

    df = pd.DataFrame(rows)
    counts = df.groupby("subject_id")["channel_name"].count().rename("valid_channel_count")
    ez_counts = df.groupby("subject_id")["label_ez"].sum().rename("ez_channel_count")
    df = df.merge(counts, on="subject_id").merge(ez_counts, on="subject_id")
    df["ez_fraction"] = df["ez_channel_count"] / df["valid_channel_count"].clip(lower=1)
    df.attrs["label_source"] = "patient_index" if not any(w == "used_run_record_labels_fallback" for w in warnings_list) else "run_record_labels_fallback"
    df.attrs["n_channels_skipped_not_in_canonical"] = int(n_channels_skipped_not_in_canonical)
    df.attrs["n_label_conflict_with_run_record_labels"] = int(n_label_conflict_with_run_record_labels)
    df.attrs["warnings"] = sorted(set(warnings_list))
    return df


def feature_columns(rows: pd.DataFrame) -> List[str]:
    return [
        col
        for col in rows.columns
        if col not in META_COLUMNS
        and not col.endswith("_id")
        and pd.api.types.is_numeric_dtype(rows[col])
    ]


def _select_topk(scores: np.ndarray, k: int) -> np.ndarray:
    pred = np.zeros(scores.shape[0], dtype=bool)
    if scores.size == 0 or int(k) <= 0:
        return pred
    k = min(int(k), int(scores.size))
    order = np.argsort(scores)[::-1]
    pred[order[:k]] = True
    return pred


def _mrr(y_ez: np.ndarray, score_ez: np.ndarray) -> float:
    if y_ez.size == 0 or int(y_ez.sum()) <= 0:
        return 0.0
    order = np.argsort(score_ez)[::-1]
    hit = np.where(y_ez[order] == 1)[0]
    return float(1.0 / float(hit[0] + 1)) if hit.size else 0.0


def patient_topk_metrics(rows: pd.DataFrame, *, method: str, fold_idx: int, selected_params: Mapping[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    channel_predictions: List[Dict[str, Any]] = []
    patient_rows: List[Dict[str, Any]] = []
    selected_params_dict = dict(selected_params)
    params_json = json.dumps(selected_params_dict, sort_keys=True)
    marker_column = str(selected_params_dict.get("marker_column", ""))

    for subject_id, group in rows.groupby("subject_id", sort=False):
        group = group.reset_index(drop=True)
        y_ez = group["label_ez"].astype(int).to_numpy()
        score_ez = group["score_ez"].astype(float).to_numpy()
        pred = _select_topk(score_ez, int(y_ez.sum()))
        y_nez = 1 - y_ez
        pred_nez = (~pred).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(y_nez, pred_nez, labels=[1, 0], zero_division=0)
        order = np.argsort(score_ez)[::-1]
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, len(order) + 1)
        patient_rows.append(
            {
                "method": method,
                "fold_idx": int(fold_idx),
                "subject_id": str(subject_id),
                "center": str(group["center"].iloc[0]),
                "valid_channel_count": int(len(group)),
                "ez_channel_count": int(y_ez.sum()),
                "ez_fraction": float(y_ez.mean()) if y_ez.size else 0.0,
                "patient_macro_f1": float(f1_score(y_nez, pred_nez, average="macro", zero_division=0)),
                "patient_macro_ez_f1": float(f1[1]),
                "patient_macro_auprc_ez": float(average_precision_score(y_ez, score_ez)) if np.unique(y_ez).size > 1 else 0.0,
                "patient_macro_ez_mrr": _mrr(y_ez, score_ez),
                "top1_is_ez": float(y_ez[int(np.argmax(score_ez))] == 1) if score_ez.size else 0.0,
                "selected_hyperparams_json": params_json,
            }
        )
        for idx, row in group.iterrows():
            channel_predictions.append(
                {
                    "method": method,
                    "fold_idx": int(fold_idx),
                    "subject_id": str(subject_id),
                    "center": str(row["center"]),
                    "channel_name": str(row["channel_name"]),
                    "label_ez": int(row["label_ez"]),
                    "label_nez": int(1 - int(row["label_ez"])),
                    "score_ez": float(row["score_ez"]),
                    "score_nez": float(1.0 - row["score_ez"]),
                    "pred_ez_topk": int(pred[idx]),
                    "rank_ez": int(ranks[idx]),
                    "marker_column": marker_column,
                    "selected_hyperparams_json": params_json,
                }
            )
    return channel_predictions, patient_rows


def _summarize_patient_rows(patient_rows: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    if patient_rows.empty:
        return pd.DataFrame()
    summary = (
        patient_rows.groupby(group_cols, sort=True, dropna=False)[
            ["patient_macro_f1", "patient_macro_ez_f1", "patient_macro_auprc_ez", "patient_macro_ez_mrr", "top1_is_ez"]
        ]
        .mean()
        .reset_index()
        .rename(columns={"top1_is_ez": "top1_is_ez_rate"})
    )
    summary["n_patients"] = patient_rows.groupby(group_cols, sort=True, dropna=False)["subject_id"].nunique().to_numpy()
    return summary


def _selection_score(patient_rows: List[Dict[str, Any]]) -> Tuple[float, float, float, float]:
    if not patient_rows:
        return (-1.0, -1.0, -1.0, -1.0)
    df = pd.DataFrame(patient_rows)
    return (
        float(df["patient_macro_ez_mrr"].mean()),
        float(df["patient_macro_f1"].mean()),
        float(df["patient_macro_auprc_ez"].mean()),
        float(df["top1_is_ez"].mean()),
    )


def _score_orientation(value: pd.Series, sign: int) -> pd.Series:
    arr = value.astype(float)
    scored = arr if sign > 0 else -arr
    finite = scored[np.isfinite(scored)]
    if finite.empty:
        return pd.Series(np.zeros(len(scored), dtype=float), index=value.index)
    lo = float(finite.min())
    hi = float(finite.max())
    if hi <= lo:
        return pd.Series(np.full(len(scored), 0.5, dtype=float), index=value.index)
    return ((scored - lo) / (hi - lo)).fillna(0.0).clip(0.0, 1.0)


def _within_patient_zscore(rows: pd.DataFrame, cols: List[str]) -> pd.Series:
    values = pd.Series(np.zeros(len(rows), dtype=float), index=rows.index)
    used = 0
    for col in cols:
        z = rows.groupby("subject_id")[col].transform(lambda x: (x - x.mean()) / (x.std(ddof=0) or 1.0))
        values = values.add(z.fillna(0.0), fill_value=0.0)
        used += 1
    return values / max(used, 1)


MARKERS: Dict[str, Callable[[pd.DataFrame, List[str]], Tuple[str | None, str | None, pd.Series | None]]] = {}


def _marker_column_priority(col: str) -> Tuple[int, int, str]:
    lower = col.lower()
    if lower.startswith("record_mean__top20pct_mean__"):
        group = 0
    elif lower.startswith("record_mean__mean__"):
        group = 1
    elif lower.startswith("record_max__top20pct_mean__"):
        group = 2
    elif lower.startswith("record_max__mean__"):
        group = 3
    elif "top20pct_mean" in lower:
        group = 4
    elif "__mean__" in lower:
        group = 5
    else:
        group = 6
    return group, len(col), col


def _match_feature(cols: List[str], required: Sequence[str], excluded: Sequence[str] = ()) -> str | None:
    req = [item.lower() for item in required]
    exc = [item.lower() for item in excluded]
    candidates = []
    for col in cols:
        lower = col.lower()
        if all(item in lower for item in req) and not any(item in lower for item in exc):
            candidates.append(col)
    if not candidates:
        return None
    candidates.sort(key=_marker_column_priority)
    return candidates[0]


def _simple_marker(required: Sequence[str], excluded: Sequence[str] = ()) -> Callable[[pd.DataFrame, List[str]], Tuple[str | None, str | None, pd.Series | None]]:
    def build(rows: pd.DataFrame, cols: List[str]) -> Tuple[str | None, str | None, pd.Series | None]:
        col = _match_feature(cols, required, excluded)
        if col is None:
            return None, f"missing feature containing {required}", None
        return col, None, rows[col].astype(float)

    return build


def _high_gamma_marker(rows: pd.DataFrame, cols: List[str]) -> Tuple[str | None, str | None, pd.Series | None]:
    col = _match_feature(cols, ("high_gamma",), ("low_gamma", "theta", "delta", "alpha", "beta"))
    if col is None:
        col = _match_feature(cols, ("gamma",), ("low_gamma", "theta", "delta", "alpha", "beta"))
    if col is None:
        return None, "missing high_gamma or gamma feature", None
    return col, None, rows[col].astype(float)


MARKERS["single_high_gamma"] = _high_gamma_marker
MARKERS["single_line_length"] = _simple_marker(("line_length",))
MARKERS["single_hfo_event_rate"] = _simple_marker(("hfo", "event_rate"))
MARKERS["single_hfo_duration_fraction"] = _simple_marker(("hfo", "duration_fraction"))
MARKERS["single_onset_latency_high_gamma"] = _simple_marker(("onset_latency", "high_gamma"))
MARKERS["single_onset_latency_line_length"] = _simple_marker(("onset_latency", "line_length"))


def _average_ictal_evidence(rows: pd.DataFrame, cols: List[str]) -> Tuple[str | None, str | None, pd.Series | None]:
    matched = []
    for terms in (("high_gamma",), ("line_length",), ("hfo", "event_rate"), ("hfo", "duration_fraction")):
        col = _match_feature(cols, terms)
        if col is not None:
            matched.append(col)
    if not matched:
        return None, "missing high_gamma, line_length, or hfo evidence features", None
    return ",".join(matched), None, _within_patient_zscore(rows, matched)


MARKERS["average_ictal_evidence_score"] = _average_ictal_evidence


def run_single_marker_fold(
    rows: pd.DataFrame,
    method: str,
    fold_idx: int,
    fit_subjects: List[str],
    val_subjects: List[str],
    test_subjects: List[str],
    candidate_cols: List[str],
) -> MethodResult | str:
    marker_col, reason, raw = MARKERS[method](rows, candidate_cols)
    if raw is None:
        return reason or "marker unavailable"
    work = rows.copy()
    work["_raw_marker"] = raw
    best_sign = 1
    best_score = (-1.0, -1.0, -1.0, -1.0)
    for sign in (1, -1):
        val = work[work["subject_id"].isin(val_subjects)].copy()
        val["score_ez"] = _score_orientation(val["_raw_marker"], sign)
        _, val_patient_rows = patient_topk_metrics(
            val,
            method=method,
            fold_idx=fold_idx,
            selected_params={"orientation": sign, "marker_column": marker_col},
        )
        score = _selection_score(val_patient_rows)
        if score > best_score:
            best_score = score
            best_sign = sign
    test = work[work["subject_id"].isin(test_subjects)].copy()
    selected = {"orientation": best_sign, "marker_column": marker_col}
    test["score_ez"] = _score_orientation(test["_raw_marker"], best_sign)
    channel_rows, patient_rows = patient_topk_metrics(test, method=method, fold_idx=fold_idx, selected_params=selected)
    return MethodResult(method, "single_marker", int(fold_idx), selected, channel_rows, patient_rows)


def _constant_feature_filter(train_x: pd.DataFrame, all_x: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    keep = []
    for col in train_x.columns:
        series = train_x[col]
        if series.notna().sum() == 0:
            continue
        if series.nunique(dropna=True) <= 1:
            continue
        keep.append(col)
    return train_x[keep], all_x[keep], keep


def _classifier_score(model: Any, x: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)
        classes = list(getattr(model, "classes_", [0, 1]))
        if 1 in classes:
            return np.asarray(proba[:, classes.index(1)], dtype=float)
        return np.asarray(proba[:, -1], dtype=float)
    decision = np.asarray(model.decision_function(x), dtype=float)
    if decision.ndim > 1:
        decision = decision[:, -1]
    finite = decision[np.isfinite(decision)]
    if finite.size == 0:
        return np.zeros(decision.shape[0], dtype=float)
    lo, hi = float(finite.min()), float(finite.max())
    if hi <= lo:
        return np.full(decision.shape[0], 0.5, dtype=float)
    return np.clip((decision - lo) / (hi - lo), 0.0, 1.0)


def _pipeline(estimator: Any, *, scale: bool) -> Pipeline:
    steps: List[Tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", estimator))
    return Pipeline(steps)


def _fit_model(model: Pipeline, x: pd.DataFrame, y: np.ndarray, *, sample_weight: np.ndarray | None = None) -> Pipeline:
    if sample_weight is None:
        model.fit(x, y)
    else:
        try:
            model.fit(x, y, model__sample_weight=sample_weight)
        except TypeError:
            model.fit(x, y)
    return model


def classical_specs(random_seed: int, include_mlp: bool) -> Dict[str, Tuple[str, List[Dict[str, Any]], Callable[[Dict[str, Any]], Tuple[Any, bool, bool]]]]:
    specs: Dict[str, Tuple[str, List[Dict[str, Any]], Callable[[Dict[str, Any]], Tuple[Any, bool, bool]]]] = {
        "logistic_l2": (
            "classical_ml",
            [{"C": c} for c in (0.01, 0.1, 1.0, 10.0)],
            lambda p: (LogisticRegression(max_iter=5000, class_weight="balanced", C=p["C"], random_state=random_seed), True, False),
        ),
        "logistic_elasticnet": (
            "classical_ml",
            [{"C": c, "l1_ratio": r} for c, r in itertools.product((0.01, 0.1, 1.0), (0.1, 0.5, 0.9))],
            lambda p: (
                LogisticRegression(
                    penalty="elasticnet",
                    solver="saga",
                    max_iter=5000,
                    class_weight="balanced",
                    C=p["C"],
                    l1_ratio=p["l1_ratio"],
                    random_state=random_seed,
                ),
                True,
                False,
            ),
        ),
        "linear_svm": (
            "classical_ml",
            [{"C": c} for c in (0.01, 0.1, 1.0, 10.0)],
            lambda p: (LinearSVC(class_weight="balanced", C=p["C"], random_state=random_seed, max_iter=10000), True, False),
        ),
        "rbf_svm": (
            "classical_ml",
            [{"C": c, "gamma": g} for c, g in itertools.product((0.1, 1.0, 10.0), ("scale", 0.01, 0.1))],
            lambda p: (SVC(kernel="rbf", class_weight="balanced", probability=True, C=p["C"], gamma=p["gamma"], random_state=random_seed), True, False),
        ),
        "random_forest": (
            "classical_ml",
            [{"max_depth": d, "min_samples_leaf": leaf} for d, leaf in itertools.product((None, 4, 8), (1, 3, 5))],
            lambda p: (
                RandomForestClassifier(
                    class_weight="balanced_subsample",
                    n_estimators=300,
                    max_depth=p["max_depth"],
                    min_samples_leaf=p["min_samples_leaf"],
                    random_state=random_seed,
                    n_jobs=-1,
                ),
                False,
                False,
            ),
        ),
        "extra_trees": (
            "classical_ml",
            [{"max_features": mf} for mf in ("sqrt", None)],
            lambda p: (
                ExtraTreesClassifier(
                    class_weight="balanced",
                    n_estimators=300,
                    max_features=p["max_features"],
                    random_state=random_seed,
                    n_jobs=-1,
                ),
                False,
                False,
            ),
        ),
        "hist_gradient_boosting": (
            "classical_ml",
            [{"learning_rate": lr, "max_iter": mi} for lr, mi in itertools.product((0.05, 0.1), (100, 200))],
            lambda p: (
                HistGradientBoostingClassifier(learning_rate=p["learning_rate"], max_iter=p["max_iter"], random_state=random_seed),
                False,
                True,
            ),
        ),
    }
    try:
        from xgboost import XGBClassifier  # type: ignore

        specs["xgboost_optional"] = (
            "classical_ml",
            [{"max_depth": d, "learning_rate": lr} for d, lr in itertools.product((2, 3), (0.03, 0.1))],
            lambda p: (
                XGBClassifier(
                    n_estimators=200,
                    max_depth=p["max_depth"],
                    learning_rate=p["learning_rate"],
                    subsample=0.9,
                    colsample_bytree=0.9,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=random_seed,
                    n_jobs=-1,
                ),
                False,
                True,
            ),
        )
    except Exception:
        pass

    if include_mlp:
        from sklearn.neural_network import MLPClassifier

        specs["mlp_sklearn"] = (
            "shallow_neural",
            [{"hidden_layer_sizes": h, "alpha": a} for h, a in itertools.product(((64,), (128, 64)), (1e-4, 1e-3))],
            lambda p: (
                MLPClassifier(
                    hidden_layer_sizes=p["hidden_layer_sizes"],
                    alpha=p["alpha"],
                    early_stopping=True,
                    max_iter=500,
                    random_state=random_seed,
                ),
                True,
                False,
            ),
        )
    return specs


def resolve_requested_methods(
    methods: str,
    *,
    random_seed: int,
    include_mlp: bool,
) -> Tuple[List[str], Dict[str, Tuple[str, List[Dict[str, Any]], Callable[[Dict[str, Any]], Tuple[Any, bool, bool]]]]]:
    marker_methods = list(MARKERS.keys())
    ml_specs = classical_specs(random_seed, include_mlp=include_mlp)
    requested_text = str(methods or "all").strip()
    if requested_text.lower() == "all":
        return marker_methods, ml_specs

    requested = [item.strip() for item in requested_text.split(",") if item.strip()]
    known = set(marker_methods) | set(ml_specs) | {"xgboost_optional"}
    unknown = sorted(set(requested) - known)
    if unknown:
        raise ValueError(f"Unknown traditional baseline method(s): {', '.join(unknown)}")
    selected_markers = [method for method in marker_methods if method in requested]
    selected_ml = {method: ml_specs[method] for method in requested if method in ml_specs}
    return selected_markers, selected_ml


def run_classical_fold(
    rows: pd.DataFrame,
    method: str,
    method_group: str,
    params_grid: List[Dict[str, Any]],
    factory: Callable[[Dict[str, Any]], Tuple[Any, bool, bool]],
    fold_idx: int,
    fit_subjects: List[str],
    val_subjects: List[str],
    train_subjects: List[str],
    test_subjects: List[str],
    cols: List[str],
) -> MethodResult | str:
    train_rows = rows[rows["subject_id"].isin(fit_subjects)]
    val_rows = rows[rows["subject_id"].isin(val_subjects)]
    all_train_rows = rows[rows["subject_id"].isin(train_subjects)]
    test_rows = rows[rows["subject_id"].isin(test_subjects)]
    if train_rows["label_ez"].nunique() < 2:
        return "single-class fit data"

    train_x, _, kept = _constant_feature_filter(train_rows[cols], rows[cols])
    if not kept:
        return "no non-constant numeric features"

    best_params: Dict[str, Any] | None = None
    best_score = (-1.0, -1.0, -1.0, -1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for params in params_grid:
            estimator, scale, use_weights = factory(params)
            model = _pipeline(estimator, scale=scale)
            weights = compute_sample_weight("balanced", train_rows["label_ez"].to_numpy()) if use_weights else None
            try:
                _fit_model(model, train_x, train_rows["label_ez"].astype(int).to_numpy(), sample_weight=weights)
                val_scored = val_rows.copy()
                val_scored["score_ez"] = _classifier_score(model, val_rows[kept])
                _, val_patient_rows = patient_topk_metrics(val_scored, method=method, fold_idx=fold_idx, selected_params=params)
                score = _selection_score(val_patient_rows)
            except Exception:
                continue
            if score > best_score:
                best_score = score
                best_params = dict(params)

    if best_params is None:
        return "all hyperparameter candidates failed"
    if all_train_rows["label_ez"].nunique() < 2:
        return "single-class train data"

    all_train_x, all_x, kept = _constant_feature_filter(all_train_rows[cols], rows[cols])
    estimator, scale, use_weights = factory(best_params)
    model = _pipeline(estimator, scale=scale)
    weights = compute_sample_weight("balanced", all_train_rows["label_ez"].to_numpy()) if use_weights else None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _fit_model(model, all_train_x, all_train_rows["label_ez"].astype(int).to_numpy(), sample_weight=weights)
    test_scored = test_rows.copy()
    test_scored["score_ez"] = _classifier_score(model, all_x.loc[test_rows.index, kept])
    selected = dict(best_params)
    channel_rows, patient_rows = patient_topk_metrics(test_scored, method=method, fold_idx=fold_idx, selected_params=selected)
    return MethodResult(method, method_group, int(fold_idx), selected, channel_rows, patient_rows)


def _passes_gate(row: Mapping[str, Any]) -> bool:
    return (
        float(row.get("patient_macro_f1", 0.0)) > A9V3_GATE["patient_macro_f1"]
        and float(row.get("patient_macro_ez_f1", 0.0)) > A9V3_GATE["patient_macro_ez_f1"]
        and float(row.get("patient_macro_auprc_ez", 0.0)) > A9V3_GATE["patient_macro_auprc_ez"]
        and float(row.get("patient_macro_ez_mrr", 0.0)) >= A9V3_GATE["patient_macro_ez_mrr"]
        and float(row.get("top1_is_ez_rate", 0.0)) >= A9V3_GATE["top1_is_ez_rate"]
    )


def compute_completed_and_incomplete_methods(
    results: Sequence[MethodResult],
    *,
    requested_methods: Sequence[str],
    expected_n_folds: int,
    skipped: Sequence[Mapping[str, Any]],
) -> Tuple[Dict[str, List[int]], List[Dict[str, Any]]]:
    requested = list(dict.fromkeys(str(method) for method in requested_methods))
    completed: Dict[str, List[int]] = {}
    for result in results:
        if result.method not in requested:
            continue
        completed.setdefault(result.method, [])
        completed[result.method].append(int(result.fold_idx))
    completed = {method: sorted(set(folds)) for method, folds in completed.items()}

    skipped_by_method: Dict[str, List[Mapping[str, Any]]] = {}
    for item in skipped:
        skipped_by_method.setdefault(str(item.get("method", "")), []).append(item)

    incomplete: List[Dict[str, Any]] = []
    expected_folds = list(range(1, int(expected_n_folds) + 1))
    for method in requested:
        folds = completed.get(method, [])
        if not folds:
            skipped_items = skipped_by_method.get(method, [])
            if _is_optional_unavailable_method(method, skipped_items):
                continue
            incomplete.append(
                {
                    "method": method,
                    "completed_n_folds": 0,
                    "expected_n_folds": int(expected_n_folds),
                    "completed_folds": [],
                    "missing_folds": expected_folds,
                    "skipped_reasons": [str(item.get("reason", "")) for item in skipped_items],
                }
            )
            continue
        missing = [fold for fold in expected_folds if fold not in folds]
        if missing:
            incomplete.append(
                {
                    "method": method,
                    "completed_n_folds": int(len(folds)),
                    "expected_n_folds": int(expected_n_folds),
                    "completed_folds": folds,
                    "missing_folds": missing,
                    "skipped_reasons": [str(item.get("reason", "")) for item in skipped_by_method.get(method, [])],
                }
            )
    return completed, incomplete


def _is_optional_unavailable_method(method: str, skipped_items: Sequence[Mapping[str, Any]]) -> bool:
    method_name = str(method)
    if method_name != "xgboost_optional" and "xgboost_optional" not in method_name:
        return False
    if not skipped_items:
        return False
    reasons = " ".join(str(item.get("reason", "")).lower() for item in skipped_items)
    return "unavailable" in reasons or "not installed" in reasons or "missing dependency" in reasons


def _incomplete_error_message(incomplete_methods: Sequence[Mapping[str, Any]]) -> str:
    parts = []
    for item in incomplete_methods:
        parts.append(f"{item['method']} missing folds {item['missing_folds']}")
    return "Incomplete traditional baseline methods: " + "; ".join(parts)


def write_outputs(
    output_dir: str | Path,
    rows: pd.DataFrame,
    results: List[MethodResult],
    skipped: List[Dict[str, Any]],
    cache: Mapping[str, Any],
    args: argparse.Namespace,
    methods_attempted: List[str],
) -> Dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    validate_traditional_rows_against_patient_index(rows, {str(k): v for k, v in dict(cache["patient_index"]).items()}, out)
    channel_rows = list(itertools.chain.from_iterable(result.channel_predictions for result in results))
    patient_rows = list(itertools.chain.from_iterable(result.patient_rows for result in results))
    selected_rows = [
        {
            "method": result.method,
            "method_group": result.method_group,
            "fold_idx": result.fold_idx,
            "marker_column": str(result.selected_params.get("marker_column", "")),
            "selected_hyperparams_json": json.dumps(result.selected_params, sort_keys=True),
        }
        for result in results
    ]

    channel_df = pd.DataFrame(channel_rows, columns=CHANNEL_PREDICTION_COLUMNS)
    output_validation = validate_traditional_output_against_patient_index(
        channel_df,
        {str(k): v for k, v in dict(cache["patient_index"]).items()},
        out,
        expected_n_splits=int(getattr(args, "n_splits", 5)),
    )
    patient_df = pd.DataFrame(patient_rows, columns=PATIENT_ROW_COLUMNS)
    selected_df = pd.DataFrame(selected_rows)
    skipped_df = pd.DataFrame(skipped, columns=["method", "fold_idx", "reason"])
    expected_n_folds = int(getattr(args, "n_splits", 5))
    completed_folds_by_method, incomplete_methods = compute_completed_and_incomplete_methods(
        results,
        requested_methods=sorted(set(methods_attempted)),
        expected_n_folds=expected_n_folds,
        skipped=skipped,
    )
    incomplete_df = pd.DataFrame(incomplete_methods)
    by_fold = _summarize_patient_rows(patient_df, ["method", "fold_idx"]) if not patient_df.empty else pd.DataFrame()
    by_center = _summarize_patient_rows(patient_df, ["method", "center"]) if not patient_df.empty else pd.DataFrame()
    summary = _summarize_patient_rows(patient_df, ["method"]) if not patient_df.empty else pd.DataFrame(columns=["method"])
    if not summary.empty:
        group_map = {result.method: result.method_group for result in results}
        summary["method_group"] = summary["method"].map(group_map)
        summary["n_folds"] = patient_df.groupby("method")["fold_idx"].nunique().reindex(summary["method"]).to_numpy()
        summary["delta_f1_vs_a9v3"] = summary["patient_macro_f1"] - A9V3_GATE["patient_macro_f1"]
        summary["delta_ez_f1_vs_a9v3"] = summary["patient_macro_ez_f1"] - A9V3_GATE["patient_macro_ez_f1"]
        summary["delta_auprc_vs_a9v3"] = summary["patient_macro_auprc_ez"] - A9V3_GATE["patient_macro_auprc_ez"]
        summary["delta_mrr_vs_a9v3"] = summary["patient_macro_ez_mrr"] - A9V3_GATE["patient_macro_ez_mrr"]
        summary["delta_top1_vs_a9v3"] = summary["top1_is_ez_rate"] - A9V3_GATE["top1_is_ez_rate"]
        summary["passes_a9v3_gate"] = summary.apply(_passes_gate, axis=1)
        summary = summary.reindex(columns=SUMMARY_COLUMNS)

    summary.to_csv(out / "traditional_baseline_summary.csv", index=False)
    by_fold.to_csv(out / "traditional_baseline_by_fold.csv", index=False)
    by_center.to_csv(out / "traditional_baseline_by_center.csv", index=False)
    patient_df.to_csv(out / "traditional_baseline_patient_rows.csv", index=False)
    channel_df.to_csv(out / "traditional_baseline_channel_predictions.csv", index=False)
    selected_df.to_csv(out / "traditional_baseline_selected_params.csv", index=False)
    skipped_df.to_csv(out / "traditional_baseline_skipped_methods.csv", index=False)
    incomplete_df.to_csv(out / "traditional_baseline_incomplete_methods.csv", index=False)

    audit = {
        "input_cache_path": str(args.window_cache_path),
        "output_dir": str(out),
        "label_source": str(rows.attrs.get("label_source", "unknown")),
        "number_of_patients": int(rows["subject_id"].nunique()),
        "n_subjects": int(rows["subject_id"].nunique()),
        "number_of_run_records": int(len(cache["run_records"])),
        "center_counts": rows.drop_duplicates("subject_id")["center"].value_counts().sort_index().to_dict(),
        "number_of_final_patient_channel_rows": int(len(rows)),
        "n_base_channel_rows": int(len(rows)),
        "n_output_channel_rows": int(len(channel_df)),
        "n_channel_rows": int(len(rows)),
        "folds_detected": output_validation.get("folds_detected", []),
        "n_channels_skipped_not_in_canonical": int(rows.attrs.get("n_channels_skipped_not_in_canonical", 0)),
        "n_label_conflict_with_run_record_labels": int(rows.attrs.get("n_label_conflict_with_run_record_labels", 0)),
        "n_noncanonical_output_rows": int(output_validation.get("n_noncanonical_output_rows", 0)),
        "n_label_conflict_output_rows": int(output_validation.get("n_label_conflict_output_rows", 0)),
        "warnings": list(rows.attrs.get("warnings", [])),
        "feature_matrix_shape": [int(len(rows)), int(len(feature_columns(rows)))],
        "split_settings": {
            "split_strategy": args.split_strategy,
            "n_splits": int(args.n_splits),
            "random_seed": int(args.random_seed),
            "val_ratio": float(args.val_ratio),
        },
        "methods_attempted": methods_attempted,
        "methods_completed": sorted({result.method for result in results}),
        "methods_skipped": skipped,
        "expected_n_folds": expected_n_folds,
        "completed_folds_by_method": completed_folds_by_method,
        "incomplete_methods": incomplete_methods,
        "allow_incomplete_methods": bool(getattr(args, "allow_incomplete_methods", False)),
        "sklearn_version": __import__("sklearn").__version__,
        "random_seed": int(args.random_seed),
        "fixed_a9v3_gate_constants": A9V3_GATE,
        "warning": "All90 four-center traditional baseline only; does not change A9v3/A9v8 training.",
    }
    if incomplete_methods:
        audit["incomplete_methods_warning"] = _incomplete_error_message(incomplete_methods)
    (out / "traditional_baseline_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    if incomplete_methods and not bool(getattr(args, "allow_incomplete_methods", False)):
        raise RuntimeError(_incomplete_error_message(incomplete_methods))
    return audit


def run_traditional_baselines(args: argparse.Namespace) -> Dict[str, Any]:
    cache = load_window_cache(args.window_cache_path)
    patient_index_all = {str(k): v for k, v in dict(cache["patient_index"]).items()}
    rows = build_patient_channel_rows(cache)
    rows = rows[rows["valid_channel"].astype(bool)].copy()
    rows = rows[rows["center"].isin(["hup", "lzu", "multicenter", "pediatric"])].copy()
    if rows.empty:
        raise ValueError("No rows remain after retaining HUP, LZU, multicenter, and pediatric centers.")
    validate_traditional_rows_against_patient_index(
        rows,
        patient_index_all,
        args.output_dir,
    )
    patient_index = {sid: patient_index_all.get(str(sid), {}) for sid in sorted(rows["subject_id"].unique())}
    splits = build_outer_splits(patient_index, split_strategy=args.split_strategy, n_splits=args.n_splits, random_seed=args.random_seed)
    cols = feature_columns(rows)
    results: List[MethodResult] = []
    skipped: List[Dict[str, Any]] = []
    methods_attempted: List[str] = []

    requested_methods = str(getattr(args, "methods", "all"))
    marker_methods, ml_specs = resolve_requested_methods(
        requested_methods,
        random_seed=int(args.random_seed),
        include_mlp=not bool(args.skip_mlp),
    )
    for split in splits:
        fold_idx = int(split["fold_idx"])
        train_subjects = list(split["train_subjects"])
        test_subjects = list(split["test_subjects"])
        fit_subjects, val_subjects = split_train_val_subjects(
            train_subjects,
            val_ratio=float(args.val_ratio),
            random_seed=int(args.random_seed),
            fold_idx=fold_idx,
        )
        for method in marker_methods:
            methods_attempted.append(method)
            result = run_single_marker_fold(rows, method, fold_idx, fit_subjects, val_subjects, test_subjects, cols)
            if isinstance(result, str):
                skipped.append({"method": method, "fold_idx": fold_idx, "reason": result})
            else:
                results.append(result)
        for method, (method_group, grid, factory) in ml_specs.items():
            methods_attempted.append(method)
            result = run_classical_fold(
                rows,
                method,
                method_group,
                grid,
                factory,
                fold_idx,
                fit_subjects,
                val_subjects,
                train_subjects,
                test_subjects,
                cols,
            )
            if isinstance(result, str):
                skipped.append({"method": method, "fold_idx": fold_idx, "reason": result})
            else:
                results.append(result)

    requested_all = requested_methods.strip().lower() == "all"
    requested_xgb = "xgboost_optional" in {item.strip() for item in requested_methods.split(",")}
    if "xgboost_optional" not in ml_specs and (requested_all or requested_xgb):
        skipped.append({"method": "xgboost_optional", "fold_idx": "all", "reason": "xgboost unavailable"})
    return write_outputs(args.output_dir, rows, results, skipped, cache, args, sorted(set(methods_attempted)))


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_traditional_baselines(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
