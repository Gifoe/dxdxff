from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.meta_ranker_common import forbidden_feature_columns, normalize_center


MARKER_TERMS = {
    "high_gamma": (("log_bp_high_gamma",), ("high_gamma",), ("gamma",)),
    "line_length": (("line_length_per_sec",), ("line_length",)),
    "hfo": (("hfo_event_rate",), ("hfo_rate",), ("hfo",)),
    "rms": (("rms",),),
    "variance": (("variance",),),
}


def get_forbidden_meta_feature_columns(columns: list[str] | None = None) -> set[str]:
    base = forbidden_feature_columns(columns or [])
    base.update(
        {
            "subject_id",
            "channel_name",
            "center",
            "center_id",
            "fold_idx",
            "label_ez",
            "label_nez",
            "ez_fraction",
            "n_ez",
            "true_ez_count",
            "ez_channel_count",
            "valid_channel_count",
            "n_records",
            "n_valid_records",
            "n_valid_windows",
        }
    )
    return base


def _feature_names(sample: Mapping[str, Any], n_features: int) -> list[str]:
    names = list(sample.get("window_feature_names") or sample.get("feature_names") or [])
    if len(names) != n_features:
        names = [f"feature_{idx}" for idx in range(n_features)]
    return [str(name) for name in names]


def _match_marker(feature_names: list[str], marker: str) -> int | None:
    lowered = [name.lower() for name in feature_names]
    for terms in MARKER_TERMS[marker]:
        for idx, name in enumerate(lowered):
            if all(term in name for term in terms):
                return idx
    return None


def _channel_names(run_record: Mapping[str, Any], n_channels: int) -> list[str]:
    sample = run_record.get("sample") or {}
    names = list(run_record.get("channel_names_norm") or run_record.get("channel_names") or sample.get("channel_names_norm") or sample.get("channel_names") or [])
    if len(names) != n_channels:
        names = [f"ch{idx}" for idx in range(n_channels)]
    return [str(name) for name in names]


def _infer_center(subject_id: str, patient_meta: Mapping[str, Any], run_record: Mapping[str, Any], sample: Mapping[str, Any]) -> str:
    for source in (patient_meta, run_record, sample):
        for key in ("source_center", "center", "source_dataset"):
            if source.get(key):
                return normalize_center(source.get(key))
    return normalize_center(subject_id)


def _rank_values(values: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(np.asarray(values, dtype=float), nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
    order = np.argsort(values)[::-1]
    ranks = np.empty(values.shape[0], dtype=float)
    ranks[order] = np.arange(1, values.shape[0] + 1, dtype=float)
    return ranks


def _top20pct_mean(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    out = np.zeros(values.shape[1], dtype=float)
    top_k = max(1, int(np.ceil(values.shape[0] * 0.2)))
    for channel_idx in range(values.shape[1]):
        channel_values = values[:, channel_idx]
        channel_values = channel_values[np.isfinite(channel_values)]
        if channel_values.size == 0:
            out[channel_idx] = 0.0
            continue
        top = np.sort(channel_values)[-min(top_k, channel_values.size) :]
        out[channel_idx] = float(np.mean(top))
    return out


def _marker_pool_values(marker_tensor: np.ndarray) -> dict[str, np.ndarray]:
    marker_tensor = np.asarray(marker_tensor, dtype=float)
    return {
        "mean": np.nanmean(marker_tensor, axis=0),
        "max": np.nanmax(marker_tensor, axis=0),
        "top20pct_mean": _top20pct_mean(marker_tensor),
    }


def _rankpct(rank: float, n_channels: int) -> float:
    return float(1.0 - (rank - 1.0) / max(int(n_channels) - 1, 1))


def _patient_canonical_labels(patient_meta: Mapping[str, Any]) -> tuple[list[str], np.ndarray, np.ndarray] | None:
    channels = list(patient_meta.get("canonical_channels") or [])
    labels = np.asarray(patient_meta.get("labels", []), dtype=float).reshape(-1)
    if not channels or labels.size != len(channels):
        return None
    mask_raw = patient_meta.get("label_mask", None)
    if mask_raw is None:
        mask = np.ones(labels.shape[0], dtype=bool)
    else:
        mask = np.asarray(mask_raw, dtype=bool).reshape(-1)
        if mask.size != labels.size:
            mask = np.ones(labels.shape[0], dtype=bool)
    return [str(ch) for ch in channels], (labels > 0.5).astype(int), mask


def _empty_marker_features(row: dict[str, Any], marker: str) -> None:
    suffixes = ("rank_median", "rank_best", "rank_mean", "rank_iqr", "rank_std", "rank_pct_median", "rank_pct_best", "top1_record_fraction", "top3_record_fraction")
    for pooling in ("mean", "max", "top20pct_mean"):
        for suffix in suffixes:
            row[f"{marker}_{pooling}_{suffix}"] = 0.0
    for suffix in suffixes:
        row[f"{marker}_{suffix}"] = 0.0


def _fill_marker_features(row: dict[str, Any], marker_all: pd.DataFrame, marker: str) -> None:
    suffixes = ("rank_median", "rank_best", "rank_mean", "rank_iqr", "rank_std", "rank_pct_median", "rank_pct_best", "top1_record_fraction", "top3_record_fraction")
    if marker_all.empty:
        _empty_marker_features(row, marker)
        return
    for pooling in ("mean", "max", "top20pct_mean"):
        marker_group = marker_all[marker_all["pooling"] == pooling]
        if marker_group.empty:
            for suffix in suffixes:
                row[f"{marker}_{pooling}_{suffix}"] = 0.0
            continue
        ranks = marker_group["rank"].astype(float)
        pct = marker_group["rank_pct"].astype(float)
        row[f"{marker}_{pooling}_rank_median"] = float(ranks.median())
        row[f"{marker}_{pooling}_rank_best"] = float(ranks.min())
        row[f"{marker}_{pooling}_rank_mean"] = float(ranks.mean())
        row[f"{marker}_{pooling}_rank_iqr"] = float(ranks.quantile(0.75) - ranks.quantile(0.25))
        row[f"{marker}_{pooling}_rank_std"] = float(ranks.std(ddof=0))
        row[f"{marker}_{pooling}_rank_pct_median"] = float(pct.median())
        row[f"{marker}_{pooling}_rank_pct_best"] = float(pct.max())
        row[f"{marker}_{pooling}_top1_record_fraction"] = float(marker_group["is_top1"].mean())
        row[f"{marker}_{pooling}_top3_record_fraction"] = float(marker_group["is_top3"].mean())
    for suffix in suffixes:
        row[f"{marker}_{suffix}"] = float(row.get(f"{marker}_mean_{suffix}", 0.0))


def validate_persistent_rows_against_patient_index(rows: pd.DataFrame, patient_index: Mapping[str, Any], output_dir: Path) -> None:
    output_dir = Path(output_dir)
    dup = rows[rows.duplicated(["subject_id", "channel_name"], keep=False)].copy()
    if not dup.empty:
        output_dir.mkdir(parents=True, exist_ok=True)
        dup.to_csv(output_dir / "persistent_duplicate_key_rows.csv", index=False)
        raise ValueError("Duplicate persistent subject_id + channel_name rows")
    missing_subject = rows[~rows["subject_id"].astype(str).isin({str(k) for k in patient_index.keys()})].copy()
    if not missing_subject.empty:
        output_dir.mkdir(parents=True, exist_ok=True)
        missing_subject.to_csv(output_dir / "persistent_noncanonical_channel_rows.csv", index=False)
        raise ValueError("Persistent rows contain subjects not present in patient_index")
    noncanonical_rows: list[pd.DataFrame] = []
    conflict_rows: list[dict[str, Any]] = []
    for subject_id, group in rows.groupby("subject_id", sort=False):
        patient_meta = patient_index.get(str(subject_id), {}) or {}
        canonical = _patient_canonical_labels(patient_meta)
        if canonical is None:
            continue
        channels, labels, mask = canonical
        label_map = {channel: int(labels[idx]) for idx, channel in enumerate(channels) if bool(mask[idx])}
        bad = group[~group["channel_name"].astype(str).isin(label_map.keys())].copy()
        if not bad.empty:
            noncanonical_rows.append(bad)
        for _, row in group.iterrows():
            channel = str(row["channel_name"])
            if channel not in label_map:
                continue
            row_label = int(pd.to_numeric(pd.Series([row["label_ez"]]), errors="coerce").fillna(-1).iloc[0])
            expected = int(label_map[channel])
            if row_label != expected:
                conflict_rows.append({"subject_id": str(subject_id), "channel_name": channel, "row_label_ez": row_label, "patient_index_label_ez": expected})
    if noncanonical_rows:
        output_dir.mkdir(parents=True, exist_ok=True)
        pd.concat(noncanonical_rows, ignore_index=True).to_csv(output_dir / "persistent_noncanonical_channel_rows.csv", index=False)
        raise ValueError("Persistent rows contain non-canonical channels")
    if conflict_rows:
        output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(conflict_rows).to_csv(output_dir / "persistent_label_conflict_rows.csv", index=False)
        raise ValueError("Persistent labels conflict with patient_index canonical labels")
    no_ez_subjects = []
    for subject_id, patient_meta in patient_index.items():
        canonical = _patient_canonical_labels(patient_meta or {})
        if canonical is not None and int(canonical[1].sum()) <= 0:
            no_ez_subjects.append(str(subject_id))
    if no_ez_subjects:
        output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"subject_id": no_ez_subjects}).to_csv(output_dir / "persistent_no_ez_subject_rows.csv", index=False)
        raise ValueError("Every patient_index subject must have at least one EZ channel")


def build_persistent_rank_features(cache_path: str | Path, output_dir: str | Path) -> pd.DataFrame:
    cache_path = Path(cache_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with cache_path.open("rb") as fin:
        cache = pickle.load(fin)
    run_records = list(cache.get("run_records", []))
    patient_index = dict(cache.get("patient_index", {}))
    patient_index = {str(k): v for k, v in patient_index.items()}
    rank_rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    matched_features: dict[str, str | None] = {marker: None for marker in ("high_gamma", "line_length", "hfo")}
    base_rows: dict[tuple[str, str], dict[str, Any]] = {}
    base_order: list[tuple[str, str]] = []
    subject_channel_order: dict[str, list[str]] = {}
    subject_run_counts: dict[str, int] = {}
    n_channels_skipped_not_in_canonical = 0
    n_label_conflict_with_run_record_labels = 0

    for subject_id in sorted(patient_index.keys()):
        patient_meta = patient_index.get(subject_id, {}) or {}
        canonical = _patient_canonical_labels(patient_meta)
        if canonical is None:
            continue
        channels, labels, mask = canonical
        subject_channel_order[subject_id] = channels
        center = _infer_center(subject_id, patient_meta, {}, {})
        for idx, channel_name in enumerate(channels):
            if not bool(mask[idx]):
                continue
            key = (subject_id, channel_name)
            base_rows[key] = {
                "subject_id": subject_id,
                "channel_name": channel_name,
                "center": center,
                "label_ez": int(labels[idx]),
                "label_nez": int(1 - int(labels[idx])),
            }
            base_order.append(key)

    for run_idx, run_record in enumerate(run_records):
        subject_id = str(run_record.get("subject_id", ""))
        if not subject_id:
            continue
        subject_run_counts[subject_id] = subject_run_counts.get(subject_id, 0) + 1
        sample = run_record.get("sample") or {}
        features = np.asarray(sample.get("window_features"), dtype=np.float32)
        if features.ndim != 3:
            continue
        n_windows, n_channels, n_features = features.shape
        feature_names = _feature_names(sample, n_features)
        channel_names = _channel_names(run_record, n_channels)
        patient_meta = patient_index.get(subject_id, {}) or {}
        canonical = _patient_canonical_labels(patient_meta)
        if canonical is None:
            labels = np.asarray(run_record.get("labels"), dtype=np.float32).reshape(-1)
            if labels.size != n_channels:
                continue
            if "used_run_record_labels_fallback" not in warnings:
                warnings.append("used_run_record_labels_fallback")
            channels = channel_names
            subject_channel_order.setdefault(subject_id, channels)
            center = _infer_center(subject_id, patient_meta, run_record, sample)
            for idx, channel_name in enumerate(channels):
                key = (subject_id, str(channel_name))
                if key not in base_rows:
                    base_rows[key] = {
                        "subject_id": subject_id,
                        "channel_name": str(channel_name),
                        "center": center,
                        "label_ez": int(labels[idx] > 0.5),
                        "label_nez": int(labels[idx] <= 0.5),
                    }
                    base_order.append(key)
            canonical_set = set(channels)
            canonical_label_map = {str(ch): int(labels[idx] > 0.5) for idx, ch in enumerate(channels)}
        else:
            channels, labels, mask = canonical
            canonical_set = {str(channels[idx]) for idx in range(len(channels)) if bool(mask[idx])}
            canonical_label_map = {str(channels[idx]): int(labels[idx]) for idx in range(len(channels)) if bool(mask[idx])}
            run_labels = np.asarray(run_record.get("labels", []), dtype=float).reshape(-1)
            if run_labels.size == len(channel_names):
                for idx, channel_name in enumerate(channel_names):
                    channel = str(channel_name)
                    if channel in canonical_label_map and int(run_labels[idx] > 0.5) != canonical_label_map[channel]:
                        n_label_conflict_with_run_record_labels += 1
        for marker in ("high_gamma", "line_length", "hfo"):
            feat_idx = _match_marker(feature_names, marker)
            if feat_idx is None:
                continue
            matched_features[marker] = feature_names[feat_idx]
            for pooling, marker_values in _marker_pool_values(features[:, :, feat_idx]).items():
                ranks = _rank_values(marker_values)
                for channel_idx, channel_name in enumerate(channel_names):
                    channel_name = str(channel_name)
                    if channel_name not in canonical_set:
                        n_channels_skipped_not_in_canonical += 1
                        continue
                    rank_rows.append(
                        {
                            "subject_id": subject_id,
                            "channel_name": channel_name,
                            "run_idx": int(run_idx),
                            "marker": marker,
                            "pooling": pooling,
                            "rank": float(ranks[channel_idx]),
                            "rank_pct": _rankpct(float(ranks[channel_idx]), n_channels),
                            "is_top1": float(ranks[channel_idx] <= 1),
                            "is_top3": float(ranks[channel_idx] <= 3),
                            "n_windows": int(n_windows),
                            "n_channels": int(n_channels),
                        }
                    )

    if not base_rows:
        raise ValueError(f"No persistent base rows could be built from {cache_path}")
    rank_df = pd.DataFrame(rank_rows) if rank_rows else pd.DataFrame(columns=["subject_id", "channel_name", "run_idx", "marker", "pooling", "rank", "rank_pct", "is_top1", "is_top3", "n_windows", "n_channels"])
    out_rows = []
    for key in base_order:
        subject_id, channel_name = key
        group = rank_df[(rank_df["subject_id"].astype(str) == subject_id) & (rank_df["channel_name"].astype(str) == channel_name)] if not rank_df.empty else rank_df
        row = dict(base_rows[key])
        row["n_records"] = int(subject_run_counts.get(subject_id, 0))
        row["n_valid_records"] = int(group["run_idx"].nunique()) if not group.empty else 0
        row["n_valid_windows"] = int(group.drop_duplicates("run_idx")["n_windows"].sum()) if not group.empty else 0
        row["valid_channel_count"] = int(len(subject_channel_order.get(subject_id, [])))
        for marker in ("high_gamma", "line_length", "hfo"):
            marker_all = group[group["marker"] == marker]
            if marker_all.empty:
                warnings.append(f"missing marker {marker}")
                _empty_marker_features(row, marker)
                continue
            _fill_marker_features(row, marker_all, marker)
        rank_medians = [row.get(f"{marker}_rank_median", 0.0) for marker in ("high_gamma", "line_length", "hfo") if row.get(f"{marker}_rank_median", 0.0) > 0.0]
        rank_pcts = [row.get(f"{marker}_rank_pct_median", 0.0) for marker in ("high_gamma", "line_length", "hfo") if row.get(f"{marker}_rank_pct_median", 0.0) > 0.0]
        row["marker_rank_consensus_mean"] = float(np.mean(rank_medians)) if rank_medians else 0.0
        row["marker_rank_consensus_best"] = float(np.min(rank_medians)) if rank_medians else 0.0
        row["marker_rank_consistency_score"] = float(1.0 / (1.0 + np.std(rank_medians))) if len(rank_medians) > 1 else 0.0
        row["high_gamma_minus_line_length_rank_median"] = float(row.get("high_gamma_rank_median", 0.0) - row.get("line_length_rank_median", 0.0))
        row["best_marker_rank_pct"] = float(max(rank_pcts)) if rank_pcts else 0.0
        row["mean_marker_rank_pct"] = float(np.mean(rank_pcts)) if rank_pcts else 0.0
        out_rows.append(row)

    out = pd.DataFrame(out_rows)
    for subject_id, idx in out.groupby("subject_id").groups.items():
        count = len(idx)
        out.loc[idx, "valid_channel_count"] = count
        out.loc[idx, "n_ez"] = int(out.loc[idx, "label_ez"].sum())
        out.loc[idx, "true_ez_count"] = int(out.loc[idx, "label_ez"].sum())
        out.loc[idx, "ez_channel_count"] = int(out.loc[idx, "label_ez"].sum())
        out.loc[idx, "ez_fraction"] = float(out.loc[idx, "label_ez"].mean())

    validate_persistent_rows_against_patient_index(out, patient_index, output_dir)
    out.to_csv(output_dir / "persistent_rank_features.csv", index=False)
    audit = {
        "cache_path": str(cache_path),
        "label_source": "patient_index_canonical_labels" if any(_patient_canonical_labels(v or {}) is not None for v in patient_index.values()) else "run_record_labels_fallback",
        "n_rows": int(len(out)),
        "n_subjects": int(out["subject_id"].nunique()),
        "n_channels_skipped_not_in_canonical": int(n_channels_skipped_not_in_canonical),
        "n_label_conflict_with_run_record_labels": int(n_label_conflict_with_run_record_labels),
        "matched_features": matched_features,
        "warnings": sorted(set(warnings + [f"missing marker {m}" for m, v in matched_features.items() if v is None])),
        "forbidden_columns": sorted(get_forbidden_meta_feature_columns(list(out.columns))),
    }
    (output_dir / "persistent_rank_features_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build persistent patient-channel marker rank features.")
    parser.add_argument("--cache_path", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    args = parser.parse_args(argv)
    build_persistent_rank_features(args.cache_path, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
