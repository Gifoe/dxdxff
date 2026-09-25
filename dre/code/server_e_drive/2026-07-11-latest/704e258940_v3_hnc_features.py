from __future__ import annotations

from collections import Counter, defaultdict
import json
import math
import pickle
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data_factory import build_outer_splits
from ez_dataset import build_or_load_run_records
from neuroez_c.raw_brainbert_data import load_embedding_table, normalize_channel_name, parse_contact_topology


FORBIDDEN_CLASSIFIER_FEATURES = {
    "center_id",
    "outcome_group",
    "surgery_success",
    "true_ez_count",
    "patient_id",
    "subject_id",
    "fold_idx",
}
RAWBB_GROUPS = ("rawbb_all", "rawbb_preictal", "rawbb_onset", "rawbb_early", "rawbb_std", "rawbb_max")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _parse_float_list(text: str | Sequence[float]) -> list[float]:
    if isinstance(text, str):
        return [float(item.strip()) for item in text.split(",") if item.strip()]
    return [float(item) for item in text]


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-values))


def _logit(values: pd.Series | np.ndarray, eps: float) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    arr = np.clip(arr, float(eps), 1.0 - float(eps))
    return np.log(arr / (1.0 - arr))


def _load_success_patient_index(feature_cache_path: str | Path) -> dict[str, dict[str, Any]]:
    args = SimpleNamespace(
        window_cache_path=str(feature_cache_path),
        sample_cache_path=None,
        outcome_subset="success",
        drop_high_ez_fraction_lzu=False,
        output_dir=None,
    )
    _, patient_index = build_or_load_run_records(args)
    return patient_index


def _load_success_failure_patient_index(feature_cache_path: str | Path) -> dict[str, dict[str, Any]]:
    args = SimpleNamespace(
        window_cache_path=str(feature_cache_path),
        sample_cache_path=None,
        outcome_subset="success_failure",
        drop_high_ez_fraction_lzu=False,
        output_dir=None,
    )
    _, patient_index = build_or_load_run_records(args)
    return patient_index


def _subject_splits_from_feature_cache(
    feature_cache_path: str | Path,
    *,
    split_strategy: str,
    n_splits: int,
    random_seed: int,
) -> tuple[list[dict[str, Any]], list[str], dict[str, dict[str, Any]]]:
    success_index = _load_success_patient_index(feature_cache_path)
    splits = build_outer_splits(
        success_index,
        split_strategy=str(split_strategy),
        n_splits=int(n_splits),
        random_seed=int(random_seed),
    )
    success_failure_index = _load_success_failure_patient_index(feature_cache_path)
    failure_subjects = sorted(
        str(sid) for sid, meta in success_failure_index.items() if str(meta.get("outcome_group")) == "failure"
    )
    return splits, failure_subjects, success_index


def _coerce_ledger(df: pd.DataFrame) -> pd.DataFrame:
    required = ["fold_idx", "subject_id", "channel_name", "true_ez", "score_ez_probability"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"V3 OOF ledger missing required columns: {missing}")
    out = df.copy()
    out["fold_idx"] = pd.to_numeric(out["fold_idx"], errors="raise").astype(int)
    out["subject_id"] = out["subject_id"].astype(str)
    out["channel_name"] = out["channel_name"].astype(str)
    out["channel_name_norm_hnc"] = out["channel_name"].map(normalize_channel_name)
    out["true_ez"] = pd.to_numeric(out["true_ez"], errors="coerce").fillna(0).astype(int)
    out["score_ez_probability"] = pd.to_numeric(out["score_ez_probability"], errors="raise").astype(float)
    if "center" not in out.columns:
        out["center"] = "unknown"
    if "channel_id" not in out.columns:
        out["channel_id"] = out.groupby("subject_id").cumcount()
    if "rank_ez_desc" not in out.columns:
        out["rank_ez_desc"] = out.groupby("subject_id")["score_ez_probability"].rank(
            ascending=False, method="first"
        ).astype(int)
    if "predicted_ez" not in out.columns:
        out["predicted_ez"] = 0
    out["_ledger_row_id"] = np.arange(len(out), dtype=np.int64)
    return out


def add_v3_patient_features(df: pd.DataFrame, *, logit_eps: float) -> pd.DataFrame:
    out = df.copy()
    out["v3_logit"] = _logit(out["score_ez_probability"], logit_eps)
    out["num_channels"] = out.groupby("subject_id")["subject_id"].transform("size").astype(int)
    if "rank_ez_desc" not in out.columns or out["rank_ez_desc"].isna().any():
        out["rank_ez_desc"] = out.groupby("subject_id")["score_ez_probability"].rank(
            ascending=False, method="first"
        )
    out["rank_ez_desc"] = pd.to_numeric(out["rank_ez_desc"], errors="coerce").fillna(out["num_channels"]).astype(float)
    out["rank_percentile"] = out["rank_ez_desc"] / out["num_channels"].clip(lower=1)
    zscores = []
    top1_gap = []
    top3_gap = []
    top5_gap = []
    top1 = []
    top3 = []
    top5 = []
    for _, group in out.groupby("subject_id", sort=False):
        logits = group["v3_logit"].to_numpy(dtype=np.float64)
        mean = float(logits.mean()) if logits.size else 0.0
        std = float(logits.std()) if logits.size else 0.0
        z = np.zeros_like(logits) if std < 1e-6 else (logits - mean) / std
        sorted_logits = np.sort(logits)[::-1]
        top1_mean = float(sorted_logits[:1].mean()) if sorted_logits.size else 0.0
        top3_mean = float(sorted_logits[: min(3, sorted_logits.size)].mean()) if sorted_logits.size else 0.0
        top5_mean = float(sorted_logits[: min(5, sorted_logits.size)].mean()) if sorted_logits.size else 0.0
        ranks = group["rank_ez_desc"].to_numpy(dtype=np.float64)
        zscores.extend(z.tolist())
        top1_gap.extend((top1_mean - logits).tolist())
        top3_gap.extend((top3_mean - logits).tolist())
        top5_gap.extend((top5_mean - logits).tolist())
        top1.extend((ranks <= 1).astype(int).tolist())
        top3.extend((ranks <= 3).astype(int).tolist())
        top5.extend((ranks <= 5).astype(int).tolist())
    out["patient_zscore_logit"] = zscores
    out["gap_to_top1"] = top1_gap
    out["gap_to_top3_mean"] = top3_gap
    out["gap_to_top5_mean"] = top5_gap
    out["top1_indicator"] = top1
    out["top3_indicator"] = top3
    out["top5_indicator"] = top5
    return out


def _topology_from_patient_index(patient_index: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, tuple[str | None, int | None]]]:
    topology: dict[str, dict[str, tuple[str | None, int | None]]] = {}
    for subject_id, meta in patient_index.items():
        subject_map: dict[str, tuple[str | None, int | None]] = {}
        for item in meta.get("channel_meta", []) or []:
            if not isinstance(item, Mapping):
                continue
            name = (
                item.get("channel_name_norm")
                or item.get("channel_name")
                or item.get("name")
                or item.get("channel")
                or item.get("contact")
            )
            if name is None:
                continue
            group = item.get("contact_group")
            number = item.get("contact_number")
            if group is None or number is None:
                parsed_group, parsed_number = parse_contact_topology(name)
                group = group if group is not None else parsed_group
                number = number if number is not None else parsed_number
            try:
                number_int = None if number is None else int(number)
            except (TypeError, ValueError):
                number_int = None
            subject_map[normalize_channel_name(name)] = (str(group).upper() if group not in (None, "") else None, number_int)
        for name in meta.get("canonical_channels", []) or []:
            norm = normalize_channel_name(name)
            if norm not in subject_map:
                subject_map[norm] = parse_contact_topology(name)
        topology[str(subject_id)] = subject_map
    return topology


def add_candidate_and_topology_features(
    df: pd.DataFrame,
    *,
    candidate_rule: str,
    patient_topology: Mapping[str, Mapping[str, tuple[str | None, int | None]]],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = df.copy()
    is_candidate = np.zeros(len(out), dtype=bool)
    topology_missing_count = 0
    contact_group_codes = np.zeros(len(out), dtype=np.float32)
    contact_number_norm = np.zeros(len(out), dtype=np.float32)
    neighbor_pm1_mean = np.zeros(len(out), dtype=np.float32)
    neighbor_pm1_max = np.zeros(len(out), dtype=np.float32)
    neighbor_pm2_mean = np.zeros(len(out), dtype=np.float32)
    neighbor_pm2_max = np.zeros(len(out), dtype=np.float32)
    isolated = np.zeros(len(out), dtype=np.float32)

    for _, group in out.groupby("subject_id", sort=False):
        idxs = group.index.to_numpy()
        ordered = group.sort_values("score_ez_probability", ascending=False, kind="mergesort")
        n_channels = len(group)
        top20 = max(1, int(math.ceil(0.20 * n_channels)))
        if candidate_rule == "top12":
            candidate_names = set(ordered.head(min(12, n_channels))["channel_name_norm_hnc"])
        elif candidate_rule == "top8":
            candidate_names = set(ordered.head(min(8, n_channels))["channel_name_norm_hnc"])
        else:
            candidate_names = set(ordered.head(top20)["channel_name_norm_hnc"])
        subject_id = str(group["subject_id"].iloc[0])
        topology_map = patient_topology.get(subject_id, {})
        if candidate_rule == "top20pct_plus_neighbors":
            for _, top_row in ordered.head(min(5, n_channels)).iterrows():
                group_name, number = topology_map.get(str(top_row["channel_name_norm_hnc"]), parse_contact_topology(top_row["channel_name"]))
                if group_name is None or number is None:
                    continue
                for _, row in group.iterrows():
                    other_group, other_number = topology_map.get(
                        str(row["channel_name_norm_hnc"]), parse_contact_topology(row["channel_name"])
                    )
                    if other_group == group_name and other_number is not None and abs(int(other_number) - int(number)) <= 2:
                        candidate_names.add(str(row["channel_name_norm_hnc"]))
        is_candidate[idxs] = group["channel_name_norm_hnc"].astype(str).isin(candidate_names).to_numpy()

        parsed = []
        group_labels: dict[str, int] = {}
        for _, row in group.iterrows():
            group_name, number = topology_map.get(str(row["channel_name_norm_hnc"]), parse_contact_topology(row["channel_name"]))
            if group_name is None or number is None:
                topology_missing_count += 1
            if group_name is not None and group_name not in group_labels:
                group_labels[group_name] = len(group_labels) + 1
            parsed.append((group_name, number))
        numbers_by_group: dict[str, list[int]] = defaultdict(list)
        for group_name, number in parsed:
            if group_name is not None and number is not None:
                numbers_by_group[group_name].append(int(number))
        score_by_key = {
            (parsed[idx][0], parsed[idx][1]): float(group.iloc[idx]["v3_logit"])
            for idx in range(len(parsed))
            if parsed[idx][0] is not None and parsed[idx][1] is not None
        }
        top20_names = set(ordered.head(top20)["channel_name_norm_hnc"])
        for local_pos, row_idx in enumerate(idxs):
            group_name, number = parsed[local_pos]
            if group_name is None or number is None:
                continue
            contact_group_codes[row_idx] = float(group_labels.get(group_name, 0))
            nums = numbers_by_group.get(group_name, [])
            if nums:
                lo, hi = min(nums), max(nums)
                contact_number_norm[row_idx] = 0.0 if hi == lo else (float(number) - lo) / float(hi - lo)
            pm1 = [score for (grp, num), score in score_by_key.items() if grp == group_name and num != number and abs(num - number) <= 1]
            pm2 = [score for (grp, num), score in score_by_key.items() if grp == group_name and num != number and abs(num - number) <= 2]
            if pm1:
                neighbor_pm1_mean[row_idx] = float(np.mean(pm1))
                neighbor_pm1_max[row_idx] = float(np.max(pm1))
            if pm2:
                neighbor_pm2_mean[row_idx] = float(np.mean(pm2))
                neighbor_pm2_max[row_idx] = float(np.max(pm2))
            channel_norm = str(group.loc[row_idx, "channel_name_norm_hnc"])
            if channel_norm in top20_names and pm2 and float(group.loc[row_idx, "v3_logit"]) - float(np.mean(pm2)) > 1.0:
                isolated[row_idx] = 1.0

    out["is_candidate"] = is_candidate
    out["contact_group_code"] = contact_group_codes
    out["contact_number_norm"] = contact_number_norm
    out["neighbor_score_mean_pm1"] = neighbor_pm1_mean
    out["neighbor_score_max_pm1"] = neighbor_pm1_max
    out["neighbor_score_mean_pm2"] = neighbor_pm2_mean
    out["neighbor_score_max_pm2"] = neighbor_pm2_max
    out["is_isolated_high_score"] = isolated
    return out, {"topology_missing_count": int(topology_missing_count)}


def _rawbb_columns(df: pd.DataFrame) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for group in RAWBB_GROUPS:
        cols = [col for col in df.columns if re.match(rf"^{re.escape(group)}_[0-9]+$", str(col))]
        result[group] = sorted(cols, key=lambda col: int(str(col).rsplit("_", 1)[1]))
    return result


def _fill_missing_embeddings(
    df: pd.DataFrame,
    *,
    emb_cols: Sequence[str],
    train_mask: pd.Series,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = df.copy()
    if not emb_cols:
        return out, {"missing_embedding_count": int(len(out)), "missing_embedding_subjects": sorted(out["subject_id"].unique())}
    missing_mask = out[list(emb_cols)].isna().any(axis=1)
    missing_subjects = sorted(out.loc[missing_mask, "subject_id"].astype(str).unique().tolist())
    global_mean = out.loc[train_mask, list(emb_cols)].mean(axis=0, skipna=True)
    global_mean = global_mean.fillna(0.0)
    for row_idx in out.index[missing_mask]:
        subject = str(out.at[row_idx, "subject_id"])
        subject_mean = out.loc[out["subject_id"].astype(str).eq(subject), list(emb_cols)].mean(axis=0, skipna=True)
        fill_values = subject_mean.where(subject_mean.notna(), global_mean).fillna(0.0)
        out.loc[row_idx, list(emb_cols)] = fill_values.to_numpy(dtype=np.float64)
    out[list(emb_cols)] = out[list(emb_cols)].fillna(0.0)
    return out, {
        "missing_embedding_count": int(missing_mask.sum()),
        "missing_embedding_subjects": missing_subjects,
    }


def _build_rawbb_transforms(
    df: pd.DataFrame,
    *,
    train_candidate_mask: pd.Series,
    pca_dim: int,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    out = df.copy()
    feature_cols: list[str] = []
    audit: dict[str, Any] = {"rawbb_groups": {}, "pca_fit_scope": "per_fold_train_candidate_rows_only"}
    grouped_cols = _rawbb_columns(out)
    train_rows = out.loc[train_candidate_mask]
    for group_name, cols in grouped_cols.items():
        if not cols:
            continue
        x_all = out[cols].to_numpy(dtype=np.float64)
        x_train = train_rows[cols].to_numpy(dtype=np.float64)
        n_components = min(int(pca_dim), max(0, x_train.shape[0] - 1), x_train.shape[1])
        if x_train.shape[0] >= 2 and n_components >= 1:
            scaler = StandardScaler()
            x_train_scaled = scaler.fit_transform(x_train)
            pca = PCA(n_components=n_components, random_state=0)
            pca.fit(x_train_scaled)
            transformed = pca.transform(scaler.transform(x_all))
            cols_out = []
            for idx in range(transformed.shape[1]):
                col = f"{group_name}_pca_{idx}"
                out[col] = transformed[:, idx]
                cols_out.append(col)
            feature_cols.extend(cols_out)
            audit["rawbb_groups"][group_name] = {
                "mode": "pca",
                "n_components": int(n_components),
                "original_dim": int(x_train.shape[1]),
                "train_candidate_rows": int(x_train.shape[0]),
            }
        else:
            summary = np.column_stack(
                [
                    np.mean(x_all, axis=1),
                    np.std(x_all, axis=1),
                    np.max(x_all, axis=1),
                    np.min(x_all, axis=1),
                    np.linalg.norm(x_all, axis=1),
                ]
            )
            cols_out = [
                f"{group_name}_summary_mean",
                f"{group_name}_summary_std",
                f"{group_name}_summary_max",
                f"{group_name}_summary_min",
                f"{group_name}_summary_l2",
            ]
            for idx, col in enumerate(cols_out):
                out[col] = summary[:, idx]
            feature_cols.extend(cols_out)
            audit["rawbb_groups"][group_name] = {
                "mode": "summary",
                "original_dim": int(x_train.shape[1]) if x_train.ndim == 2 else 0,
                "train_candidate_rows": int(x_train.shape[0]) if x_train.ndim == 2 else 0,
            }
    return out, feature_cols, audit


def _load_fold_embeddings(embedding_dir: str | Path, fold_idx: int) -> pd.DataFrame:
    base = Path(embedding_dir) / f"rawbrainbert_patient_channel_embeddings_fold_{int(fold_idx)}"
    df = load_embedding_table(base)
    if "subject_id" not in df.columns or "channel_name" not in df.columns:
        raise ValueError(f"Embedding table for fold {fold_idx} must include subject_id and channel_name.")
    out = df.copy()
    out["subject_id"] = out["subject_id"].astype(str)
    out["channel_name_norm_hnc"] = out["channel_name"].map(normalize_channel_name)
    duplicated = out[out.duplicated(["subject_id", "channel_name_norm_hnc"], keep=False)].copy()
    if not duplicated.empty:
        preview = duplicated[["subject_id", "channel_name", "channel_name_norm_hnc"]].head(20).to_dict("records")
        raise ValueError(f"Duplicate Raw-BrainBERT embedding keys in fold {fold_idx}: {preview}")
    return out


def _classifier(classifier_name: str) -> LogisticRegression:
    if classifier_name == "elastic_logreg":
        return LogisticRegression(
            penalty="elasticnet",
            l1_ratio=0.5,
            C=1.0,
            class_weight="balanced",
            max_iter=2000,
            solver="saga",
        )
    return LogisticRegression(
        penalty="l2",
        C=1.0,
        class_weight="balanced",
        max_iter=2000,
        solver="lbfgs",
    )


def _auc_or_none(y_true: np.ndarray, score: np.ndarray) -> float | None:
    if np.unique(y_true).size < 2:
        return None
    try:
        return float(roc_auc_score(y_true, score))
    except Exception:
        return None


def _candidate_recall(rows: pd.DataFrame) -> float | None:
    ez = rows["true_ez"].astype(int).eq(1)
    if int(ez.sum()) <= 0:
        return None
    return float((rows.loc[ez, "is_candidate"].astype(bool)).mean())


def _rank_final(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["final_rank_ez_desc"] = out.groupby("subject_id")["final_score_ez_probability"].rank(
        ascending=False, method="first"
    ).astype(int)
    return out


def run_hnc_pipeline(args: Any) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    beta_list = _parse_float_list(getattr(args, "beta_list", "0.05,0.10,0.20,0.30"))
    ledger = _coerce_ledger(pd.read_csv(args.v3_oof_ledger))
    ledger = add_v3_patient_features(ledger, logit_eps=float(getattr(args, "logit_eps", 1e-5)))
    splits, failure_subjects, success_index = _subject_splits_from_feature_cache(
        args.feature_cache_path,
        split_strategy=str(getattr(args, "split_strategy", "5fold")),
        n_splits=int(getattr(args, "n_splits", 5)),
        random_seed=int(getattr(args, "random_seed", 42)),
    )
    topology = _topology_from_patient_index(success_index)
    ledger, topology_audit = add_candidate_and_topology_features(
        ledger,
        candidate_rule=str(args.candidate_rule),
        patient_topology=topology,
    )

    v3_feature_cols = [
        "score_ez_probability",
        "v3_logit",
        "patient_zscore_logit",
        "rank_ez_desc",
        "rank_percentile",
        "gap_to_top1",
        "gap_to_top3_mean",
        "gap_to_top5_mean",
        "top1_indicator",
        "top3_indicator",
        "top5_indicator",
        "num_channels",
        "contact_group_code",
        "contact_number_norm",
        "neighbor_score_mean_pm1",
        "neighbor_score_max_pm1",
        "neighbor_score_mean_pm2",
        "neighbor_score_max_pm2",
        "is_isolated_high_score",
        "n_records",
    ]
    corrected_by_beta: dict[float, list[pd.DataFrame]] = {beta: [] for beta in beta_list}
    missing_embedding_subjects: set[str] = set()
    missing_embedding_count = 0
    per_fold_train_subject_count: dict[str, int] = {}
    per_fold_test_subject_count: dict[str, int] = {}
    per_fold_train_candidate_count: dict[str, int] = {}
    per_fold_test_candidate_count: dict[str, int] = {}
    per_fold_hard_negative_rate: dict[str, float] = {}
    per_fold_classifier_auc_train: dict[str, float | None] = {}
    per_fold_classifier_auc_test: dict[str, float | None] = {}
    fold_classifier_fallback: dict[str, bool] = {}
    candidate_recall_by_fold: dict[str, float | None] = {}
    leakage_checks: dict[str, Any] = {}
    transform_audits: dict[str, Any] = {}
    missing_embedding_fraction_by_fold: dict[str, float] = {}
    row_count_before_embedding_merge_by_fold: dict[str, int] = {}
    row_count_after_embedding_merge_by_fold: dict[str, int] = {}
    embedding_columns_count_by_fold: dict[str, int] = {}
    duplicated_embedding_keys_by_fold: dict[str, list[dict[str, Any]]] = {}
    warnings: list[str] = []
    classifier_feature_columns: list[str] = []
    failure_subject_set = set(failure_subjects)

    for split in splits:
        fold_idx = int(split["fold_idx"])
        train_subjects = sorted(str(sid) for sid in split["train_subjects"])
        test_subjects = sorted(str(sid) for sid in split["test_subjects"])
        train_set = set(train_subjects)
        test_set = set(test_subjects)
        leakage_checks[str(fold_idx)] = {
            "train_test_subject_overlap": sorted(train_set.intersection(test_set)),
            "failure_subjects_in_hnc_train": sorted(train_set.intersection(failure_subject_set)),
        }
        if leakage_checks[str(fold_idx)]["train_test_subject_overlap"]:
            raise ValueError(f"Fold {fold_idx} train/test subject leakage detected.")
        fold_rows = ledger[ledger["subject_id"].isin(train_set.union(test_set))].copy()
        row_count_before_embedding_merge_by_fold[str(fold_idx)] = int(len(fold_rows))
        embeddings = _load_fold_embeddings(args.embedding_dir, fold_idx)
        duplicated_embedding_keys_by_fold[str(fold_idx)] = []
        fold_rows = fold_rows.merge(
            embeddings.drop(columns=[col for col in ("fold_idx",) if col in embeddings.columns]),
            on=["subject_id", "channel_name_norm_hnc"],
            how="left",
            suffixes=("", "_rawbb"),
        )
        row_count_after_embedding_merge_by_fold[str(fold_idx)] = int(len(fold_rows))
        if row_count_after_embedding_merge_by_fold[str(fold_idx)] != row_count_before_embedding_merge_by_fold[str(fold_idx)]:
            raise ValueError(
                f"Embedding merge changed row count for fold {fold_idx}: "
                f"before={row_count_before_embedding_merge_by_fold[str(fold_idx)]}, "
                f"after={row_count_after_embedding_merge_by_fold[str(fold_idx)]}."
            )
        if "channel_name_rawbb" in fold_rows.columns:
            fold_rows.drop(columns=["channel_name_rawbb"], inplace=True)
        if "n_records" not in fold_rows.columns:
            fold_rows["n_records"] = 0
        fold_rows["n_records"] = pd.to_numeric(fold_rows["n_records"], errors="coerce").fillna(0.0)
        emb_cols = [col for cols in _rawbb_columns(fold_rows).values() for col in cols]
        embedding_columns_count_by_fold[str(fold_idx)] = int(len(emb_cols))
        if not emb_cols:
            raise ValueError(f"No Raw-BrainBERT embedding columns found after merge for fold {fold_idx}.")
        train_mask = fold_rows["subject_id"].isin(train_set)
        train_candidate_mask = train_mask & fold_rows["is_candidate"].astype(bool)
        fold_rows, missing_audit = _fill_missing_embeddings(
            fold_rows,
            emb_cols=emb_cols,
            train_mask=train_candidate_mask if train_candidate_mask.any() else train_mask,
        )
        missing_fraction = float(missing_audit["missing_embedding_count"] / max(len(fold_rows), 1))
        missing_embedding_fraction_by_fold[str(fold_idx)] = missing_fraction
        if int(missing_audit["missing_embedding_count"]) == int(len(fold_rows)):
            raise ValueError(f"All HNC rows are missing Raw-BrainBERT embeddings for fold {fold_idx}.")
        if missing_fraction > 0.20:
            warnings.append(
                f"fold {fold_idx}: missing Raw-BrainBERT embeddings for {missing_fraction:.3f} of HNC rows."
            )
        missing_embedding_count += int(missing_audit["missing_embedding_count"])
        missing_embedding_subjects.update(missing_audit["missing_embedding_subjects"])
        fold_rows, rawbb_feature_cols, transform_audit = _build_rawbb_transforms(
            fold_rows,
            train_candidate_mask=train_candidate_mask,
            pca_dim=int(args.pca_dim),
        )
        transform_audits[str(fold_idx)] = transform_audit
        feature_cols = [col for col in v3_feature_cols + rawbb_feature_cols if col in fold_rows.columns]
        forbidden_used = sorted(FORBIDDEN_CLASSIFIER_FEATURES.intersection(feature_cols))
        if forbidden_used:
            raise ValueError(f"Forbidden HNC classifier feature(s) selected: {forbidden_used}")
        classifier_feature_columns = feature_cols

        y_train = (fold_rows.loc[train_candidate_mask, "true_ez"].astype(int).eq(0)).astype(int).to_numpy()
        x_train = fold_rows.loc[train_candidate_mask, feature_cols].to_numpy(dtype=np.float64)
        per_fold_train_subject_count[str(fold_idx)] = int(len(train_subjects))
        per_fold_test_subject_count[str(fold_idx)] = int(len(test_subjects))
        per_fold_train_candidate_count[str(fold_idx)] = int(train_candidate_mask.sum())
        test_mask = fold_rows["subject_id"].isin(test_set)
        test_candidate_mask = test_mask & fold_rows["is_candidate"].astype(bool)
        per_fold_test_candidate_count[str(fold_idx)] = int(test_candidate_mask.sum())
        hard_negative_rate = float(y_train.mean()) if y_train.size else 0.5
        per_fold_hard_negative_rate[str(fold_idx)] = hard_negative_rate

        fold_rows["p_hard_negative"] = 0.0
        if y_train.size == 0 or np.unique(y_train).size < 2:
            fallback_rate = hard_negative_rate if np.isfinite(hard_negative_rate) else 0.5
            fold_rows.loc[fold_rows["is_candidate"].astype(bool), "p_hard_negative"] = float(fallback_rate)
            fold_classifier_fallback[str(fold_idx)] = True
            per_fold_classifier_auc_train[str(fold_idx)] = None
            per_fold_classifier_auc_test[str(fold_idx)] = None
        else:
            model = Pipeline([("scaler", StandardScaler()), ("classifier", _classifier(str(args.classifier)))])
            model.fit(x_train, y_train)
            candidate_mask = fold_rows["is_candidate"].astype(bool)
            probs = model.predict_proba(fold_rows.loc[candidate_mask, feature_cols].to_numpy(dtype=np.float64))[:, 1]
            fold_rows.loc[candidate_mask, "p_hard_negative"] = probs
            train_scores = fold_rows.loc[train_candidate_mask, "p_hard_negative"].to_numpy(dtype=np.float64)
            per_fold_classifier_auc_train[str(fold_idx)] = _auc_or_none(y_train, train_scores)
            y_test = (fold_rows.loc[test_candidate_mask, "true_ez"].astype(int).eq(0)).astype(int).to_numpy()
            test_scores = fold_rows.loc[test_candidate_mask, "p_hard_negative"].to_numpy(dtype=np.float64)
            per_fold_classifier_auc_test[str(fold_idx)] = _auc_or_none(y_test, test_scores)
            fold_classifier_fallback[str(fold_idx)] = False

        test_rows = fold_rows[test_mask].copy()
        candidate_recall_by_fold[str(fold_idx)] = _candidate_recall(test_rows)
        for beta in beta_list:
            corrected = test_rows.copy()
            corrected["candidate_rule"] = str(args.candidate_rule)
            corrected["beta"] = float(beta)
            corrected["final_logit"] = corrected["patient_zscore_logit"].astype(float)
            candidate_mask = corrected["is_candidate"].astype(bool)
            corrected.loc[candidate_mask, "final_logit"] = (
                corrected.loc[candidate_mask, "patient_zscore_logit"].astype(float)
                - float(beta) * corrected.loc[candidate_mask, "p_hard_negative"].astype(float)
            )
            corrected["final_score_ez_probability"] = _sigmoid(corrected["final_logit"].to_numpy(dtype=np.float64))
            corrected_by_beta[beta].append(corrected)

    output_paths: dict[str, str] = {}
    num_rows_corrected = 0
    noncandidate_equal = True
    all_corrected_for_recall: list[pd.DataFrame] = []
    output_columns = [
        "fold_idx",
        "subject_id",
        "center",
        "channel_id",
        "channel_name",
        "true_ez",
        "score_ez_probability",
        "rank_ez_desc",
        "v3_logit",
        "patient_zscore_logit",
        "candidate_rule",
        "is_candidate",
        "p_hard_negative",
        "beta",
        "final_logit",
        "final_score_ez_probability",
        "final_rank_ez_desc",
    ]
    for beta, frames in corrected_by_beta.items():
        combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if not combined.empty:
            combined = combined.sort_values("_ledger_row_id", kind="mergesort").reset_index(drop=True)
            combined = _rank_final(combined)
            noncandidate = combined[~combined["is_candidate"].astype(bool)]
            if not noncandidate.empty:
                noncandidate_equal = noncandidate_equal and bool(
                    np.allclose(
                        noncandidate["final_logit"].to_numpy(dtype=np.float64),
                        noncandidate["patient_zscore_logit"].to_numpy(dtype=np.float64),
                        atol=0.0,
                        rtol=0.0,
                    )
                )
        file_name = f"corrected_oof_rawbrainbert_hnc_{args.candidate_rule}_beta{float(beta):.2f}.csv"
        output_path = output_dir / file_name
        cols = [col for col in output_columns if col in combined.columns]
        combined[cols].to_csv(output_path, index=False)
        output_paths[f"{float(beta):.2f}"] = str(output_path)
        num_rows_corrected = int(len(combined))
        all_corrected_for_recall.append(combined)

    candidate_recall_values = [value for value in candidate_recall_by_fold.values() if value is not None]
    audit = {
        "num_rows_v3_ledger": int(len(ledger)),
        "num_rows_corrected": int(num_rows_corrected),
        "row_count_equal": bool(num_rows_corrected == len(ledger)),
        "candidate_rule": str(args.candidate_rule),
        "beta_list": [float(beta) for beta in beta_list],
        "pca_dim": int(args.pca_dim),
        "classifier": str(args.classifier),
        "classifier_feature_columns": classifier_feature_columns,
        "forbidden_classifier_feature_intersection": sorted(
            FORBIDDEN_CLASSIFIER_FEATURES.intersection(classifier_feature_columns)
        ),
        "per_fold_train_subject_count": per_fold_train_subject_count,
        "per_fold_test_subject_count": per_fold_test_subject_count,
        "per_fold_train_candidate_count": per_fold_train_candidate_count,
        "per_fold_test_candidate_count": per_fold_test_candidate_count,
        "per_fold_hard_negative_rate": per_fold_hard_negative_rate,
        "per_fold_classifier_auc_train": per_fold_classifier_auc_train,
        "per_fold_classifier_auc_test_if_label_available": per_fold_classifier_auc_test,
        "fold_classifier_fallback": fold_classifier_fallback,
        "missing_embedding_count": int(missing_embedding_count),
        "missing_embedding_subjects": sorted(missing_embedding_subjects),
        "missing_embedding_fraction_by_fold": missing_embedding_fraction_by_fold,
        "row_count_before_embedding_merge_by_fold": row_count_before_embedding_merge_by_fold,
        "row_count_after_embedding_merge_by_fold": row_count_after_embedding_merge_by_fold,
        "embedding_columns_count_by_fold": embedding_columns_count_by_fold,
        "duplicated_embedding_keys_by_fold": duplicated_embedding_keys_by_fold,
        "warnings": warnings,
        "topology_missing_count": int(topology_audit["topology_missing_count"]),
        "candidate_recall_by_fold": candidate_recall_by_fold,
        "candidate_recall_overall": float(np.mean(candidate_recall_values)) if candidate_recall_values else None,
        "noncandidate_final_logit_equal_v3_check": bool(noncandidate_equal),
        "leakage_checks": leakage_checks,
        "pca_scaler_audit": transform_audits,
        "output_paths": output_paths,
    }
    with (output_dir / "rawbrainbert_hnc_audit.json").open("w", encoding="utf-8") as fout:
        json.dump(_json_safe(audit), fout, indent=2, ensure_ascii=False, sort_keys=True)
    return audit


__all__ = [
    "FORBIDDEN_CLASSIFIER_FEATURES",
    "RAWBB_GROUPS",
    "add_candidate_and_topology_features",
    "add_v3_patient_features",
    "normalize_channel_name",
    "run_hnc_pipeline",
]
