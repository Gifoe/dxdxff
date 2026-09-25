from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_factory import split_train_val_subjects
from scripts.meta_ranker_common import (
    REQUIRED_SUMMARY_COLUMNS,
    compute_rank_robust_composite,
    forbidden_feature_columns,
    patient_topk_evaluate,
    summarize_patient_rows,
    write_json,
)

PROGRESS_COLUMNS = [
    "timestamp",
    "fold_idx",
    "stage",
    "feature_set",
    "model_type",
    "candidate_idx",
    "candidate_total",
    "elapsed_s",
    "message",
]


def build_subject_splits(rows: pd.DataFrame, *, n_splits: int = 5, random_seed: int = 42) -> list[dict[str, Any]]:
    subjects = sorted(rows["subject_id"].dropna().astype(str).unique().tolist())
    n_splits = max(2, min(int(n_splits), len(subjects)))
    arr = np.asarray(subjects)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=int(random_seed))
    splits = []
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(arr), start=1):
        train_subjects = arr[train_idx].tolist()
        fit_subjects, val_subjects = split_train_val_subjects(train_subjects, random_seed=int(random_seed), fold_idx=fold_idx)
        splits.append(
            {
                "fold_idx": fold_idx,
                "split_source": "kfold_fallback",
                "fit_subjects": fit_subjects,
                "val_subjects": val_subjects,
                "train_subjects": train_subjects,
                "test_subjects": arr[test_idx].tolist(),
            }
        )
    return splits


def _assert_disjoint_subjects(split: dict[str, Any]) -> None:
    fit = set(split["fit_subjects"])
    val = set(split["val_subjects"])
    test = set(split["test_subjects"])
    if fit & val or fit & test or val & test:
        raise ValueError(f"Subject overlap detected in fold {split.get('fold_idx')}")


def build_subject_splits_from_fold_idx(
    rows: pd.DataFrame,
    *,
    n_splits: int = 5,
    random_seed: int = 42,
    allow_small_subject_count: bool = False,
) -> list[dict[str, Any]]:
    if "subject_id" not in rows.columns:
        raise ValueError("score bank must contain subject_id")
    if "fold_idx" not in rows.columns:
        if allow_small_subject_count:
            return build_subject_splits(rows, n_splits=n_splits, random_seed=random_seed)
        raise ValueError("score bank must contain fold_idx from source predictions")

    work = rows[["subject_id", "fold_idx"]].copy()
    work["subject_id"] = work["subject_id"].astype(str)
    work["fold_idx_numeric"] = pd.to_numeric(work["fold_idx"], errors="coerce")
    if work["fold_idx_numeric"].isna().any():
        raise ValueError("score bank contains missing or non-numeric fold_idx")
    work["fold_idx"] = work["fold_idx_numeric"].astype(int)

    subject_fold_counts = work.groupby("subject_id")["fold_idx"].nunique()
    bad_subjects = subject_fold_counts[subject_fold_counts.ne(1)]
    if not bad_subjects.empty:
        raise ValueError(f"Subjects assigned to multiple fold_idx values: {bad_subjects.index.tolist()[:10]}")

    subject_folds = work.drop_duplicates("subject_id")[["subject_id", "fold_idx"]]
    n_subjects = int(subject_folds["subject_id"].nunique())
    folds = sorted(int(v) for v in subject_folds["fold_idx"].unique().tolist())
    expected_folds = list(range(1, int(n_splits) + 1))
    if not allow_small_subject_count:
        if n_subjects != 90:
            raise ValueError(f"fixed all90 MetaRanker requires 90 subjects, got {n_subjects}")
        if int(n_splits) != 5:
            raise ValueError("fixed all90 MetaRanker requires n_splits=5")
        if folds != [1, 2, 3, 4, 5]:
            raise ValueError(f"fixed all90 MetaRanker requires fold_idx values [1, 2, 3, 4, 5], got {folds}")
    elif len(folds) < 2:
        return build_subject_splits(subject_folds, n_splits=n_splits, random_seed=random_seed)

    splits = []
    for fold_idx in folds:
        test_subjects = sorted(subject_folds.loc[subject_folds["fold_idx"].eq(fold_idx), "subject_id"].astype(str).tolist())
        train_subjects = sorted(subject_folds.loc[~subject_folds["fold_idx"].eq(fold_idx), "subject_id"].astype(str).tolist())
        fit_subjects, val_subjects = split_train_val_subjects(train_subjects, random_seed=int(random_seed), fold_idx=int(fold_idx))
        split = {
            "fold_idx": int(fold_idx),
            "split_source": "fold_idx",
            "fit_subjects": list(fit_subjects),
            "val_subjects": list(val_subjects),
            "train_subjects": list(train_subjects),
            "test_subjects": list(test_subjects),
            "n_subjects": n_subjects,
            "n_test_subjects": len(test_subjects),
        }
        _assert_disjoint_subjects(split)
        splits.append(split)
    if not allow_small_subject_count and len(splits) != 5:
        raise ValueError(f"fixed all90 MetaRanker requires 5 outer splits, got {len(splits)}")
    return splits


def _protocol_check(args: argparse.Namespace, rows: pd.DataFrame) -> None:
    if str(args.positive_label).lower() != "ez":
        raise ValueError("positive_label must be ez")
    if str(args.split_strategy).lower() != "5fold":
        raise ValueError("split_strategy must be 5fold")
    if int(args.n_splits) != 5 and not bool(getattr(args, "allow_small_subject_count", False)):
        raise ValueError("n_splits must be 5")
    if int(args.random_seed) != 42:
        raise ValueError("random_seed must be 42")
    if bool(args.drop_high_ez_fraction_lzu):
        raise ValueError("drop_high_ez_fraction_lzu must be false")
    n_subjects = int(rows["subject_id"].nunique())
    if n_subjects != 90 and not bool(getattr(args, "allow_small_subject_count", False)):
        raise ValueError(f"score bank must contain 90 unique subjects, got {n_subjects}")
    if "fold_idx" not in rows.columns:
        raise ValueError("score bank must contain fold_idx")
    fold_values = pd.to_numeric(rows["fold_idx"], errors="coerce")
    if fold_values.isna().any():
        raise ValueError("score bank contains missing fold_idx")
    folds = sorted(int(v) for v in fold_values.unique().tolist())
    if folds != [1, 2, 3, 4, 5] and not bool(getattr(args, "allow_small_subject_count", False)):
        raise ValueError(f"score bank must contain folds [1, 2, 3, 4, 5], got {folds}")
    subject_fold_counts = rows.assign(_fold=fold_values.astype(int)).groupby("subject_id")["_fold"].nunique()
    if subject_fold_counts.gt(1).any():
        raise ValueError("each subject must have exactly one fold_idx")


def validate_raw_feature_sets(feature_sets_raw: dict[str, list[str]], rows: pd.DataFrame, output_dir: Path) -> None:
    raw_cols = [str(col) for cols in feature_sets_raw.values() for col in cols]
    forbidden = forbidden_feature_columns(list(rows.columns) + raw_cols)
    violations: dict[str, list[str]] = {}
    for name, cols in feature_sets_raw.items():
        bad = sorted(str(col) for col in cols if str(col) in forbidden)
        if bad:
            violations[str(name)] = bad
    if violations:
        write_json(output_dir / "forbidden_feature_columns_error.json", {"forbidden_features_by_set": violations})
        raise ValueError(f"Forbidden model feature columns found in feature_columns_json: {violations}")


def _safe_feature_sets(feature_sets: dict[str, list[str]], rows: pd.DataFrame) -> dict[str, list[str]]:
    forbidden = forbidden_feature_columns(rows.columns)
    out = {}
    for name, cols in feature_sets.items():
        out[name] = [col for col in cols if col in rows.columns and col not in forbidden and pd.api.types.is_numeric_dtype(rows[col])]
    bad = {name: sorted(set(cols) & forbidden) for name, cols in out.items() if set(cols) & forbidden}
    if bad:
        raise ValueError(f"Forbidden model feature columns detected after filtering: {bad}")
    return out


def _score_from_linear_model(model: Pipeline, x: pd.DataFrame) -> np.ndarray:
    if hasattr(model[-1], "predict_proba"):
        proba = model.predict_proba(x)
        classes = list(model[-1].classes_)
        return np.asarray(proba[:, classes.index(1)] if 1 in classes else proba[:, -1], dtype=float)
    decision = np.asarray(model.decision_function(x), dtype=float)
    finite = decision[np.isfinite(decision)]
    if finite.size == 0:
        return np.zeros(decision.shape[0], dtype=float)
    lo, hi = float(finite.min()), float(finite.max())
    return np.zeros(decision.shape[0], dtype=float) if hi <= lo else (decision - lo) / (hi - lo)


def _candidate_weights(score_features: list[str]) -> list[dict[str, float]]:
    features = list(dict.fromkeys(score_features))
    candidates: list[dict[str, float]] = []
    for feature in features:
        candidates.append({feature: 1.0})
    for a, b in itertools.combinations(features, 2):
        for wa, wb in ((0.25, 0.75), (0.5, 0.5), (0.75, 0.25)):
            candidates.append({a: wa, b: wb})
    for combo in itertools.combinations(features, 3):
        for weights in set(itertools.permutations((0.2, 0.3, 0.5))):
            candidates.append(dict(zip(combo, weights)))
    if features:
        candidates.append({feature: 1.0 / len(features) for feature in features})
    return candidates


def _ensemble_score(rows: pd.DataFrame, weights: dict[str, float]) -> np.ndarray:
    score = np.zeros(len(rows), dtype=float)
    for col, weight in weights.items():
        score += float(weight) * rows[col].astype(float).fillna(0.0).to_numpy()
    return score


def _selection_tuple(summary: dict[str, Any], simplicity: int) -> tuple[float, float, float, float, int]:
    return (
        float(summary.get("rank_robust_composite", 0.0)),
        float(summary.get("patient_macro_f1", 0.0)),
        float(summary.get("patient_macro_ez_f1", 0.0)),
        float(summary.get("patient_macro_ez_mrr", 0.0)),
        -int(simplicity),
    )


def _fit_predict_linear(model_type: str, train_rows: pd.DataFrame, pred_rows: pd.DataFrame, features: list[str], params: dict[str, Any]) -> np.ndarray:
    if model_type == "logistic_l2":
        estimator = LogisticRegression(
            penalty="l2",
            solver="liblinear",
            C=float(params["C"]),
            class_weight=params.get("class_weight"),
            max_iter=5000,
        )
    elif model_type == "linear_svc":
        estimator = LinearSVC(C=float(params["C"]), max_iter=5000)
    else:
        raise ValueError(model_type)
    model = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", estimator)])
    model.fit(train_rows[features], train_rows["label_ez"].astype(int))
    return _score_from_linear_model(model, pred_rows[features])


def _evaluate_candidate(rows: pd.DataFrame, subjects: list[str], scores: np.ndarray, *, method: str, fold_idx: int, config: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    work = rows[rows["subject_id"].isin(subjects)].copy()
    work["score_ez_meta"] = scores
    channel_rows, patient_rows, summary = patient_topk_evaluate(work, score_col="score_ez_meta", method=method, fold_idx=fold_idx, selected_params=config)
    return summary, patient_rows, channel_rows


def _progress_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _emit_progress(
    progress_rows: list[dict[str, Any]],
    fold_dir: Path,
    *,
    fold_idx: int,
    stage: str,
    feature_set: str = "",
    model_type: str = "",
    candidate_idx: int = 0,
    candidate_total: int = 0,
    start_time: float | None = None,
    message: str = "",
    progress_enabled: bool = True,
) -> None:
    if not progress_enabled:
        return
    elapsed_s = 0.0 if start_time is None else max(0.0, time.perf_counter() - float(start_time))
    row = {
        "timestamp": _progress_timestamp(),
        "fold_idx": int(fold_idx),
        "stage": str(stage),
        "feature_set": str(feature_set or ""),
        "model_type": str(model_type or ""),
        "candidate_idx": int(candidate_idx),
        "candidate_total": int(candidate_total),
        "elapsed_s": float(elapsed_s),
        "message": str(message or ""),
    }
    progress_rows.append(row)
    fold_dir.mkdir(parents=True, exist_ok=True)
    with (fold_dir / "progress.jsonl").open("a", encoding="utf-8") as fout:
        fout.write(json.dumps(row, sort_keys=True) + "\n")
    print(
        "[MetaRanker] "
        f"fold={row['fold_idx']} stage={row['stage']} feature_set={row['feature_set']} "
        f"model={row['model_type']} candidate={row['candidate_idx']}/{row['candidate_total']} "
        f"elapsed={row['elapsed_s']:.1f}s {row['message']}".rstrip(),
        flush=True,
    )


def _write_fold_progress_csv(fold_dir: Path, progress_rows: list[dict[str, Any]], progress_enabled: bool) -> None:
    if not progress_enabled:
        return
    fold_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(progress_rows, columns=PROGRESS_COLUMNS).to_csv(fold_dir / "progress.csv", index=False)


def _numeric_passthrough_cols(rows: pd.DataFrame) -> list[str]:
    return [
        col
        for col in rows.columns
        if (col.startswith("score_") or col.startswith("z_score_") or col.startswith("rankpct_score_"))
        and pd.api.types.is_numeric_dtype(rows[col])
    ]


def _estimate_fold_candidate_total(rows: pd.DataFrame, feature_sets: dict[str, list[str]], split: dict[str, Any]) -> int:
    fit_rows = rows[rows["subject_id"].isin(split["fit_subjects"])].copy()
    total = len(_numeric_passthrough_cols(rows))
    can_fit_linear = fit_rows["label_ez"].nunique() >= 2
    for features in feature_sets.values():
        if not features:
            continue
        score_features = [f for f in features if f.startswith("z_score_") or f.startswith("rankpct_score_")]
        total += len(_candidate_weights(score_features))
        if can_fit_linear:
            total += 14
            total += 4
    return int(total)


def _should_emit_candidate(candidate_idx: int, candidate_total: int, progress_every: int) -> bool:
    progress_every = max(1, int(progress_every))
    return candidate_idx <= 0 or candidate_idx % progress_every == 0 or candidate_idx == candidate_total


def _search_fold(
    rows: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    split: dict[str, Any],
    output_dir: Path,
    *,
    progress_every: int = 50,
    progress_enabled: bool = True,
    progress_rows: list[dict[str, Any]] | None = None,
    progress_start_time: float | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    fold_idx = int(split["fold_idx"])
    fold_dir = output_dir / f"fold_{fold_idx}"
    fit_rows = rows[rows["subject_id"].isin(split["fit_subjects"])].copy()
    val_rows = rows[rows["subject_id"].isin(split["val_subjects"])].copy()
    results: list[dict[str, Any]] = []
    progress_rows = progress_rows if progress_rows is not None else []
    candidate_total = _estimate_fold_candidate_total(rows, feature_sets, split)
    candidate_idx = 0
    _emit_progress(
        progress_rows,
        fold_dir,
        fold_idx=fold_idx,
        stage="search_start",
        candidate_idx=0,
        candidate_total=candidate_total,
        start_time=progress_start_time,
        message="validation search started",
        progress_enabled=progress_enabled,
    )

    passthrough_cols = _numeric_passthrough_cols(rows)
    for col in passthrough_cols:
        candidate_idx += 1
        summary, _, _ = _evaluate_candidate(val_rows, split["val_subjects"], val_rows[col].astype(float).to_numpy(), method=f"passthrough:{col}", fold_idx=fold_idx, config={"score_col": col})
        results.append({"model_type": "passthrough", "feature_set": "single_score", "score_col": col, "selection_split": "val", "simplicity": 0, **summary})
        if _should_emit_candidate(candidate_idx, candidate_total, progress_every):
            _emit_progress(
                progress_rows,
                fold_dir,
                fold_idx=fold_idx,
                stage="passthrough",
                feature_set="single_score",
                model_type="passthrough",
                candidate_idx=candidate_idx,
                candidate_total=candidate_total,
                start_time=progress_start_time,
                message=f"score_col={col}",
                progress_enabled=progress_enabled,
            )

    for feature_set, features in feature_sets.items():
        if not features:
            continue
        score_features = [f for f in features if f.startswith("z_score_") or f.startswith("rankpct_score_")]
        for weights in _candidate_weights(score_features):
            candidate_idx += 1
            val_score = _ensemble_score(val_rows, weights)
            summary, _, _ = _evaluate_candidate(val_rows, split["val_subjects"], val_score, method="nonnegative_ensemble", fold_idx=fold_idx, config={"weights": weights})
            results.append({"model_type": "nonnegative_ensemble", "feature_set": feature_set, "weights_json": json.dumps(weights, sort_keys=True), "selection_split": "val", "simplicity": len(weights), **summary})
            if _should_emit_candidate(candidate_idx, candidate_total, progress_every):
                _emit_progress(
                    progress_rows,
                    fold_dir,
                    fold_idx=fold_idx,
                    stage="ensemble",
                    feature_set=feature_set,
                    model_type="nonnegative_ensemble",
                    candidate_idx=candidate_idx,
                    candidate_total=candidate_total,
                    start_time=progress_start_time,
                    message=f"weights={json.dumps(weights, sort_keys=True)}",
                    progress_enabled=progress_enabled,
                )
        if fit_rows["label_ez"].nunique() >= 2 and len(features) > 0:
            for class_weight in (None, "balanced"):
                for c in (0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0):
                    candidate_idx += 1
                    try:
                        val_score = _fit_predict_linear("logistic_l2", fit_rows, val_rows, features, {"C": c, "class_weight": class_weight})
                    except Exception:
                        if _should_emit_candidate(candidate_idx, candidate_total, progress_every):
                            _emit_progress(
                                progress_rows,
                                fold_dir,
                                fold_idx=fold_idx,
                                stage="logistic_l2",
                                feature_set=feature_set,
                                model_type="logistic_l2",
                                candidate_idx=candidate_idx,
                                candidate_total=candidate_total,
                                start_time=progress_start_time,
                                message=f"C={c} class_weight={class_weight or ''} skipped",
                                progress_enabled=progress_enabled,
                            )
                        continue
                    summary, _, _ = _evaluate_candidate(val_rows, split["val_subjects"], val_score, method="logistic_l2", fold_idx=fold_idx, config={"C": c, "class_weight": class_weight, "feature_set": feature_set})
                    results.append({"model_type": "logistic_l2", "feature_set": feature_set, "C": c, "class_weight": class_weight or "", "selection_split": "val", "simplicity": 10 + len(features), **summary})
                    if _should_emit_candidate(candidate_idx, candidate_total, progress_every):
                        _emit_progress(
                            progress_rows,
                            fold_dir,
                            fold_idx=fold_idx,
                            stage="logistic_l2",
                            feature_set=feature_set,
                            model_type="logistic_l2",
                            candidate_idx=candidate_idx,
                            candidate_total=candidate_total,
                            start_time=progress_start_time,
                            message=f"C={c} class_weight={class_weight or ''}",
                            progress_enabled=progress_enabled,
                        )
            for c in (0.01, 0.03, 0.1, 0.3):
                candidate_idx += 1
                try:
                    val_score = _fit_predict_linear("linear_svc", fit_rows, val_rows, features, {"C": c})
                except Exception:
                    if _should_emit_candidate(candidate_idx, candidate_total, progress_every):
                        _emit_progress(
                            progress_rows,
                            fold_dir,
                            fold_idx=fold_idx,
                            stage="linear_svc",
                            feature_set=feature_set,
                            model_type="linear_svc",
                            candidate_idx=candidate_idx,
                            candidate_total=candidate_total,
                            start_time=progress_start_time,
                            message=f"C={c} skipped",
                            progress_enabled=progress_enabled,
                        )
                    continue
                summary, _, _ = _evaluate_candidate(val_rows, split["val_subjects"], val_score, method="linear_svc", fold_idx=fold_idx, config={"C": c, "feature_set": feature_set})
                results.append({"model_type": "linear_svc", "feature_set": feature_set, "C": c, "selection_split": "val", "simplicity": 20 + len(features), **summary})
                if _should_emit_candidate(candidate_idx, candidate_total, progress_every):
                    _emit_progress(
                        progress_rows,
                        fold_dir,
                        fold_idx=fold_idx,
                        stage="linear_svc",
                        feature_set=feature_set,
                        model_type="linear_svc",
                        candidate_idx=candidate_idx,
                        candidate_total=candidate_total,
                        start_time=progress_start_time,
                        message=f"C={c}",
                        progress_enabled=progress_enabled,
                    )
    val_df = pd.DataFrame(results)
    if val_df.empty:
        raise RuntimeError(f"Fold {fold_idx}: no validation candidates succeeded")
    select_values = val_df.apply(lambda row: _selection_tuple(row, int(row.get("simplicity", 999))), axis=1)
    for idx in range(5):
        val_df[f"_select_{idx}"] = select_values.map(lambda item, idx=idx: item[idx])
    val_df["_tie_key"] = val_df.apply(
        lambda row: json.dumps(
            {
                "model_type": str(row.get("model_type", "")),
                "feature_set": str(row.get("feature_set", "")),
                "score_col": str(row.get("score_col", "")),
                "weights_json": str(row.get("weights_json", "")),
                "C": str(row.get("C", "")),
                "class_weight": str(row.get("class_weight", "")),
            },
            sort_keys=True,
        ),
        axis=1,
    )
    temp_cols = [f"_select_{idx}" for idx in range(5)] + ["_tie_key"]
    ranked = val_df.sort_values(
        temp_cols,
        ascending=[False, False, False, False, False, True],
        kind="mergesort",
    )
    selected = ranked.iloc[0].drop(labels=temp_cols).to_dict()
    fold_dir.mkdir(parents=True, exist_ok=True)
    val_df.drop(columns=temp_cols).to_csv(fold_dir / "val_search_results.csv", index=False)
    write_json(fold_dir / "selected_config.json", selected)
    _emit_progress(
        progress_rows,
        fold_dir,
        fold_idx=fold_idx,
        stage="search_done",
        feature_set=str(selected.get("feature_set", "")),
        model_type=str(selected.get("model_type", "")),
        candidate_idx=candidate_idx,
        candidate_total=candidate_total,
        start_time=progress_start_time,
        message=f"selected={selected.get('model_type')} score_col={selected.get('score_col', '')}",
        progress_enabled=progress_enabled,
    )
    return selected, val_df.drop(columns=temp_cols)


def _apply_selected(rows: pd.DataFrame, feature_sets: dict[str, list[str]], split: dict[str, Any], selected: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    fold_idx = int(split["fold_idx"])
    train_rows = rows[rows["subject_id"].isin(split["train_subjects"])].copy()
    test_rows = rows[rows["subject_id"].isin(split["test_subjects"])].copy()
    model_type = str(selected["model_type"])
    config = dict(selected)
    if model_type == "passthrough":
        col = str(selected["score_col"])
        scores = test_rows[col].astype(float).to_numpy()
        weights = {}
    elif model_type == "nonnegative_ensemble":
        weights = json.loads(str(selected.get("weights_json", "{}")))
        scores = _ensemble_score(test_rows, weights)
    else:
        features = feature_sets.get(str(selected["feature_set"]), [])
        params = {"C": selected.get("C", 1.0), "class_weight": selected.get("class_weight") or None}
        scores = _fit_predict_linear(model_type, train_rows, test_rows, features, params)
        weights = {}
    summary, patient_rows, channel_rows = _evaluate_candidate(test_rows, split["test_subjects"], scores, method=model_type, fold_idx=fold_idx, config=config)
    return summary, patient_rows, channel_rows, weights


def _split_audit_rows(rows: pd.DataFrame, split: dict[str, Any]) -> list[dict[str, Any]]:
    split_audit = []
    for split_name in ("fit_subjects", "val_subjects", "test_subjects"):
        for sid in split[split_name]:
            sub = rows[rows["subject_id"].eq(sid)]
            split_audit.append(
                {
                    "split_source": split.get("split_source", "fold_idx"),
                    "fold_idx": split["fold_idx"],
                    "split": split_name.replace("_subjects", ""),
                    "subject_id": sid,
                    "center": sub["center"].iloc[0],
                    "n_channels": len(sub),
                    "n_ez": int(sub["label_ez"].sum()),
                    "subject_fold_idx": int(pd.to_numeric(sub["fold_idx"], errors="coerce").dropna().iloc[0]),
                }
            )
    return split_audit


def _run_one_fold(
    rows: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    split: dict[str, Any],
    output_dir: Path | str,
    progress_every: int = 50,
    progress_enabled: bool = True,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    fold_idx = int(split["fold_idx"])
    fold_dir = output_dir / f"fold_{fold_idx}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.perf_counter()
    progress_rows: list[dict[str, Any]] = []
    if progress_enabled:
        print(f"MetaRanker fold {fold_idx} start", flush=True)
    try:
        candidate_total = _estimate_fold_candidate_total(rows, feature_sets, split)
        _emit_progress(
            progress_rows,
            fold_dir,
            fold_idx=fold_idx,
            stage="queued",
            candidate_idx=0,
            candidate_total=candidate_total,
            start_time=start_time,
            message="fold worker started",
            progress_enabled=progress_enabled,
        )
        split_audit_rows = _split_audit_rows(rows, split)
        selected, val_df = _search_fold(
            rows,
            feature_sets,
            split,
            output_dir,
            progress_every=progress_every,
            progress_enabled=progress_enabled,
            progress_rows=progress_rows,
            progress_start_time=start_time,
        )
        _emit_progress(
            progress_rows,
            fold_dir,
            fold_idx=fold_idx,
            stage="apply_start",
            feature_set=str(selected.get("feature_set", "")),
            model_type=str(selected.get("model_type", "")),
            candidate_idx=candidate_total,
            candidate_total=candidate_total,
            start_time=start_time,
            message="applying selected config to test fold",
            progress_enabled=progress_enabled,
        )
        summary, patient_rows, channel_rows, weights = _apply_selected(rows, feature_sets, split, selected)
        _emit_progress(
            progress_rows,
            fold_dir,
            fold_idx=fold_idx,
            stage="apply_done",
            feature_set=str(selected.get("feature_set", "")),
            model_type=str(selected.get("model_type", "")),
            candidate_idx=candidate_total,
            candidate_total=candidate_total,
            start_time=start_time,
            message="test fold evaluation written",
            progress_enabled=progress_enabled,
        )
        pd.DataFrame(patient_rows).to_csv(fold_dir / "test_patient_rows.csv", index=False)
        pd.DataFrame(channel_rows).to_csv(fold_dir / "test_channel_predictions.csv", index=False)
        selected_row = {"fold_idx": fold_idx, **selected}
        weights_rows = [{"fold_idx": fold_idx, "feature": key, "weight": value} for key, value in weights.items()]
        fold_summary = {"fold_idx": fold_idx, **summary}
        _emit_progress(
            progress_rows,
            fold_dir,
            fold_idx=fold_idx,
            stage="fold_done",
            feature_set=str(selected.get("feature_set", "")),
            model_type=str(selected.get("model_type", "")),
            candidate_idx=candidate_total,
            candidate_total=candidate_total,
            start_time=start_time,
            message="fold completed",
            progress_enabled=progress_enabled,
        )
        if progress_enabled:
            print(
                f"MetaRanker fold {fold_idx} end selected={selected.get('model_type')} feature_set={selected.get('feature_set', '')}",
                flush=True,
            )
        return {
            "fold_idx": fold_idx,
            "split_audit": split_audit_rows,
            "selected": selected_row,
            "weights_rows": weights_rows,
            "fold_summary": fold_summary,
            "patient_rows": patient_rows,
            "channel_rows": channel_rows,
            "val_df": val_df,
        }
    except Exception as exc:
        _emit_progress(
            progress_rows,
            fold_dir,
            fold_idx=fold_idx,
            stage="error",
            candidate_idx=max([int(row.get("candidate_idx", 0)) for row in progress_rows], default=0),
            candidate_total=max([int(row.get("candidate_total", 0)) for row in progress_rows], default=0),
            start_time=start_time,
            message=f"{type(exc).__name__}: {exc}",
            progress_enabled=progress_enabled,
        )
        raise
    finally:
        _write_fold_progress_csv(fold_dir, progress_rows, progress_enabled)


def _read_progress_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_global_progress_files(output_dir: Path, fold_results: list[dict[str, Any]], progress_enabled: bool) -> list[str]:
    progress_files = ["meta_ranker_progress.jsonl", "meta_ranker_progress_summary.csv"]
    if not progress_enabled:
        return progress_files

    global_path = output_dir / progress_files[0]
    summary_path = output_dir / progress_files[1]
    all_progress_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for result in sorted(fold_results, key=lambda item: int(item["fold_idx"])):
        fold_idx = int(result["fold_idx"])
        fold_progress_path = output_dir / f"fold_{fold_idx}" / "progress.jsonl"
        rows = _read_progress_jsonl(fold_progress_path)
        all_progress_rows.extend(rows)
        last_row = rows[-1] if rows else {}
        selected = dict(result.get("selected", {}))
        fold_summary = dict(result.get("fold_summary", {}))
        summary_rows.append(
            {
                "fold_idx": fold_idx,
                "last_stage": str(last_row.get("stage", "")),
                "candidate_total": int(max([int(row.get("candidate_total", 0)) for row in rows], default=0)),
                "candidates_evaluated": int(max([int(row.get("candidate_idx", 0)) for row in rows], default=0)),
                "elapsed_s": float(max([float(row.get("elapsed_s", 0.0)) for row in rows], default=0.0)),
                "selected_model_type": str(selected.get("model_type", "")),
                "selected_feature_set": str(selected.get("feature_set", "")),
                "selected_score_col": str(selected.get("score_col", "")),
                "selected_summary_patient_macro_f1": float(fold_summary.get("patient_macro_f1", 0.0)),
                "selected_summary_patient_macro_ez_f1": float(fold_summary.get("patient_macro_ez_f1", 0.0)),
                "selected_summary_patient_macro_auprc_ez": float(fold_summary.get("patient_macro_auprc_ez", 0.0)),
                "selected_summary_patient_macro_ez_mrr": float(fold_summary.get("patient_macro_ez_mrr", 0.0)),
                "selected_summary_top1_is_ez_rate": float(fold_summary.get("top1_is_ez_rate", 0.0)),
            }
        )

    with global_path.open("w", encoding="utf-8") as fout:
        for row in all_progress_rows:
            fout.write(json.dumps(row, sort_keys=True) + "\n")
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    return progress_files


def run_meta_ranker(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(args.score_bank)
    feature_sets_raw = json.loads(Path(args.feature_columns_json).read_text(encoding="utf-8"))
    _protocol_check(args, rows)
    validate_raw_feature_sets(feature_sets_raw, rows, output_dir)
    feature_sets = _safe_feature_sets(feature_sets_raw, rows)
    if not any(feature_sets.values()):
        raise ValueError("At least one non-empty MetaRanker feature set is required")
    splits = build_subject_splits_from_fold_idx(
        rows,
        n_splits=int(args.n_splits),
        random_seed=int(args.random_seed),
        allow_small_subject_count=bool(getattr(args, "allow_small_subject_count", False)),
    )

    n_jobs = int(getattr(args, "n_jobs", 1))
    progress_enabled = bool(getattr(args, "progress", True))
    progress_every = max(1, int(getattr(args, "progress_every", 50)))
    split_audit = []
    all_patient_rows: list[dict[str, Any]] = []
    all_channel_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    weights_rows: list[dict[str, Any]] = []
    all_val_search: list[pd.DataFrame] = []
    fold_summaries: list[dict[str, Any]] = []
    if n_jobs <= 1:
        fold_results = [
            _run_one_fold(
                rows,
                feature_sets,
                split,
                output_dir,
                progress_every=progress_every,
                progress_enabled=progress_enabled,
            )
            for split in splits
        ]
    else:
        fold_results = Parallel(n_jobs=n_jobs, backend="loky", verbose=10 if progress_enabled else 0)(
            delayed(_run_one_fold)(
                rows,
                feature_sets,
                split,
                output_dir,
                progress_every,
                progress_enabled,
            )
            for split in splits
        )

    for result in sorted(fold_results, key=lambda item: int(item["fold_idx"])):
        split_audit.extend(result["split_audit"])
        selected_rows.append(result["selected"])
        weights_rows.extend(result["weights_rows"])
        fold_summaries.append(result["fold_summary"])
        all_patient_rows.extend(result["patient_rows"])
        all_channel_rows.extend(result["channel_rows"])
        val_df = result["val_df"].copy()
        val_df.insert(0, "fold_idx", int(result["fold_idx"]))
        all_val_search.append(val_df)

    progress_files = _write_global_progress_files(output_dir, fold_results, progress_enabled)

    patient_df = pd.DataFrame(all_patient_rows)
    channel_df = pd.DataFrame(all_channel_rows)
    overall = summarize_patient_rows(patient_df)
    summary_df = pd.DataFrame([{key: overall.get(key, 0.0) for key in REQUIRED_SUMMARY_COLUMNS}])
    by_fold = pd.DataFrame(fold_summaries)
    by_center = patient_df.groupby("center", dropna=False).mean(numeric_only=True).reset_index() if not patient_df.empty else pd.DataFrame()
    selected_df = pd.DataFrame(selected_rows)
    weights_df = pd.DataFrame(weights_rows)
    val_all = pd.concat(all_val_search, ignore_index=True) if all_val_search else pd.DataFrame()
    pd.DataFrame(split_audit).to_csv(output_dir / "meta_ranker_split_audit.csv", index=False)
    summary_df.to_csv(output_dir / "meta_ranker_summary.csv", index=False)
    write_json(output_dir / "meta_ranker_summary.json", overall)
    by_fold.to_csv(output_dir / "meta_ranker_by_fold.csv", index=False)
    by_center.to_csv(output_dir / "meta_ranker_by_center.csv", index=False)
    patient_df.to_csv(output_dir / "meta_ranker_patient_rows.csv", index=False)
    channel_df.to_csv(output_dir / "meta_ranker_channel_predictions.csv", index=False)
    selected_df.to_csv(output_dir / "meta_ranker_selected_params.csv", index=False)
    weights_df.to_csv(output_dir / "meta_ranker_selected_weights.csv", index=False)
    val_all.to_csv(output_dir / "meta_ranker_val_search_all.csv", index=False)
    audit = {
        "n_rows": int(len(rows)),
        "n_subjects": int(rows["subject_id"].nunique()),
        "feature_sets": {key: len(value) for key, value in feature_sets.items()},
        "forbidden_columns": sorted(forbidden_feature_columns(rows.columns)),
        "split_source": "fold_idx",
        "n_jobs": n_jobs,
        "progress_enabled": progress_enabled,
        "progress_every": progress_every,
        "progress_files": progress_files,
        "folds_detected": sorted(int(v) for v in pd.to_numeric(rows["fold_idx"], errors="coerce").dropna().unique().tolist()),
        "n_test_subjects_by_fold": {int(split["fold_idx"]): len(split["test_subjects"]) for split in splits},
        "no_subject_overlap": True,
        "subject_has_single_fold_idx": bool(rows.groupby("subject_id")["fold_idx"].nunique().eq(1).all()),
        "validation_selection_only": True,
        "no_test_tuning": True,
        "no_center_features": not any("center" in col.lower() for features in feature_sets.values() for col in features),
        "selected_config_by_fold": selected_rows,
        "selection": "validation_only",
    }
    write_json(output_dir / "meta_ranker_audit.json", audit)
    return overall


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MetaRanker-v1.")
    parser.add_argument("--score_bank", required=True)
    parser.add_argument("--feature_columns_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split_strategy", required=True)
    parser.add_argument("--n_splits", required=True, type=int)
    parser.add_argument("--random_seed", required=True, type=int)
    parser.add_argument("--positive_label", required=True)
    parser.add_argument("--drop_high_ez_fraction_lzu", required=True, type=lambda x: str(x).lower() in {"true", "1", "yes"})
    parser.add_argument("--allow_small_subject_count", action="store_true")
    parser.add_argument("--n_jobs", type=int, default=1)
    parser.add_argument("--progress_every", type=int, default=50)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    run_meta_ranker(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
