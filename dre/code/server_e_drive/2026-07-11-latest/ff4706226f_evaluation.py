from __future__ import annotations

import itertools
import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from data_factory import split_train_val_subjects
from scripts.fm_baselines.aggregation import aggregate_window_embeddings_to_patient_channels
from scripts.traditional_baselines.core import (
    A9V3_GATE,
    MethodResult,
    _classifier_score,
    _fit_model,
    _selection_score,
    compute_completed_and_incomplete_methods,
    patient_topk_metrics,
)


def fm_feature_columns(rows: pd.DataFrame) -> List[str]:
    return [col for col in rows.columns if str(col).startswith("emb_") and pd.api.types.is_numeric_dtype(rows[col])]


def _head_specs(random_seed: int) -> Dict[str, Tuple[List[Dict[str, Any]], Any]]:
    return {
        "logistic_l2": (
            [{"C": c} for c in (0.01, 0.1, 1.0, 10.0)],
            lambda p: LogisticRegression(max_iter=5000, class_weight="balanced", C=p["C"], random_state=random_seed),
        ),
        "linear_svm": (
            [{"C": c} for c in (0.01, 0.1, 1.0, 10.0)],
            lambda p: LinearSVC(class_weight="balanced", C=p["C"], random_state=random_seed, max_iter=10000),
        ),
    }


def _pipeline(estimator: Any) -> Pipeline:
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", estimator)])


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


def _passes_gate(row: Mapping[str, Any]) -> bool:
    return (
        float(row.get("patient_macro_f1", 0.0)) > A9V3_GATE["patient_macro_f1"]
        and float(row.get("patient_macro_ez_f1", 0.0)) > A9V3_GATE["patient_macro_ez_f1"]
        and float(row.get("patient_macro_auprc_ez", 0.0)) > A9V3_GATE["patient_macro_auprc_ez"]
        and float(row.get("patient_macro_ez_mrr", 0.0)) >= A9V3_GATE["patient_macro_ez_mrr"]
        and float(row.get("top1_is_ez_rate", 0.0)) >= A9V3_GATE["top1_is_ez_rate"]
    )


def _fm_positioning_fields(fm_model: str) -> Dict[str, Any]:
    is_random = str(fm_model).lower() == "random_projection"
    return {
        "comparison_only_baseline": True,
        "true_pretrained_fm": not is_random,
        "debug_only": is_random,
        "paper_baseline": not is_random,
        "frozen_or_finetuned": "frozen",
        "frozen_encoder": True,
        "fine_tuned": False,
        "adapter_tuned": False,
        "test_time_selection": False,
        "backbone_trainable_params": 0,
        "trainable_component": "logistic_l2_or_linear_svm_head_only",
        "seeg_adaptation": "single-channel SEEG contact mode; no scalp montage remapping",
    }


def run_fm_frozen_head(
    patient_channel_rows: pd.DataFrame,
    *,
    fm_model: str,
    head_methods: Sequence[str],
    n_splits: int,
    random_seed: int,
    val_ratio: float,
) -> Tuple[List[MethodResult], List[Dict[str, Any]]]:
    specs = _head_specs(random_seed)
    requested = [item for item in head_methods if item]
    unknown = sorted(set(requested) - set(specs))
    if unknown:
        raise ValueError(f"Unknown head method(s): {', '.join(unknown)}")
    cols = fm_feature_columns(patient_channel_rows)
    if not cols:
        raise ValueError("No embedding feature columns found.")
    results: List[MethodResult] = []
    skipped: List[Dict[str, Any]] = []
    for fold_idx, fold_rows in patient_channel_rows.groupby("fold_idx", sort=True):
        train_subjects = sorted(fold_rows.loc[fold_rows["split_role"].eq("train"), "subject_id"].unique().tolist())
        test_subjects = sorted(fold_rows.loc[fold_rows["split_role"].eq("test"), "subject_id"].unique().tolist())
        fit_subjects, val_subjects = split_train_val_subjects(train_subjects, val_ratio=val_ratio, random_seed=random_seed, fold_idx=int(fold_idx))
        for head_method in requested:
            method = f"{fm_model}__{head_method}"
            grid, factory = specs[head_method]
            fit_rows = fold_rows[fold_rows["subject_id"].isin(fit_subjects)]
            val_rows = fold_rows[fold_rows["subject_id"].isin(val_subjects)]
            train_rows = fold_rows[fold_rows["subject_id"].isin(train_subjects)]
            test_rows = fold_rows[fold_rows["subject_id"].isin(test_subjects)]
            if fit_rows["label_ez"].nunique() < 2:
                skipped.append({"method": method, "fold_idx": int(fold_idx), "reason": "single-class fit data"})
                continue
            best_params = None
            best_score = (-1.0, -1.0, -1.0, -1.0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for params in grid:
                    model = _pipeline(factory(params))
                    try:
                        _fit_model(model, fit_rows[cols], fit_rows["label_ez"].astype(int).to_numpy())
                        scored = val_rows.copy()
                        scored["score_ez"] = _classifier_score(model, scored[cols])
                        _, val_patient_rows = patient_topk_metrics(scored, method=method, fold_idx=int(fold_idx), selected_params=params)
                        score = _selection_score(val_patient_rows)
                    except Exception:
                        continue
                    if score > best_score:
                        best_score = score
                        best_params = dict(params)
            if best_params is None or train_rows["label_ez"].nunique() < 2:
                skipped.append({"method": method, "fold_idx": int(fold_idx), "reason": "all candidates failed or single-class train data"})
                continue
            model = _pipeline(factory(best_params))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _fit_model(model, train_rows[cols], train_rows["label_ez"].astype(int).to_numpy())
            test_scored = test_rows.copy()
            test_scored["score_ez"] = _classifier_score(model, test_scored[cols])
            selected = dict(best_params)
            selected["head_method"] = head_method
            channel_rows, patient_rows = patient_topk_metrics(test_scored, method=method, fold_idx=int(fold_idx), selected_params=selected)
            results.append(MethodResult(method, "frozen_fm", int(fold_idx), selected, channel_rows, patient_rows))
    return results, skipped


def _incomplete_error(incomplete: Sequence[Mapping[str, Any]]) -> str:
    return "Incomplete FM frozen-head methods: " + "; ".join(f"{item['method']} missing folds {item['missing_folds']}" for item in incomplete)


def write_fm_outputs(
    *,
    output_dir: str | Path,
    fm_model: str,
    results: Sequence[MethodResult],
    skipped: Sequence[Mapping[str, Any]],
    requested_methods: Sequence[str],
    expected_n_folds: int,
    split_strategy: str,
    allow_incomplete_methods: bool,
    embedding_path: str,
    metadata_path: str,
    head_methods: Sequence[str],
    n_patient_channel_rows: int = 0,
    n_patients: int = 0,
) -> Dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    channel_df = pd.DataFrame(list(itertools.chain.from_iterable(result.channel_predictions for result in results)))
    patient_df = pd.DataFrame(list(itertools.chain.from_iterable(result.patient_rows for result in results)))
    selected_df = pd.DataFrame(
        [
            {
                "method": result.method,
                "fm_model": fm_model,
                "head_method": result.selected_params.get("head_method", ""),
                "fold_idx": result.fold_idx,
                "selected_hyperparams_json": json.dumps(result.selected_params, sort_keys=True),
            }
            for result in results
        ]
    )
    completed, incomplete = compute_completed_and_incomplete_methods(
        list(results),
        requested_methods=requested_methods,
        expected_n_folds=expected_n_folds,
        skipped=skipped,
    )
    summary = _summarize_patient_rows(patient_df, ["method"]) if not patient_df.empty else pd.DataFrame()
    if not summary.empty:
        positioning = _fm_positioning_fields(fm_model)
        summary["fm_model"] = fm_model
        summary["head_method"] = summary["method"].str.replace(f"{fm_model}__", "", regex=False)
        for key, value in positioning.items():
            summary[key] = value
        summary["n_folds"] = patient_df.groupby("method")["fold_idx"].nunique().reindex(summary["method"]).to_numpy()
        summary["delta_f1_vs_a9v3"] = summary["patient_macro_f1"] - A9V3_GATE["patient_macro_f1"]
        summary["delta_ez_f1_vs_a9v3"] = summary["patient_macro_ez_f1"] - A9V3_GATE["patient_macro_ez_f1"]
        summary["delta_auprc_vs_a9v3"] = summary["patient_macro_auprc_ez"] - A9V3_GATE["patient_macro_auprc_ez"]
        summary["delta_mrr_vs_a9v3"] = summary["patient_macro_ez_mrr"] - A9V3_GATE["patient_macro_ez_mrr"]
        summary["delta_top1_vs_a9v3"] = summary["top1_is_ez_rate"] - A9V3_GATE["top1_is_ez_rate"]
        summary["passes_a9v3_gate"] = summary.apply(_passes_gate, axis=1)
        summary = summary[
            [
                "method",
                "fm_model",
                "head_method",
                "frozen_or_finetuned",
                "comparison_only_baseline",
                "true_pretrained_fm",
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
        ]
    by_fold = _summarize_patient_rows(patient_df, ["method", "fold_idx"]) if not patient_df.empty else pd.DataFrame()
    by_center = _summarize_patient_rows(patient_df, ["method", "center"]) if not patient_df.empty else pd.DataFrame()
    pd.DataFrame(summary).to_csv(out / "fm_frozen_head_summary.csv", index=False)
    by_fold.to_csv(out / "fm_frozen_head_by_fold.csv", index=False)
    by_center.to_csv(out / "fm_frozen_head_by_center.csv", index=False)
    patient_df.to_csv(out / "fm_frozen_head_patient_rows.csv", index=False)
    channel_df.to_csv(out / "fm_frozen_head_channel_predictions.csv", index=False)
    selected_df.to_csv(out / "fm_frozen_head_selected_params.csv", index=False)
    pd.DataFrame(skipped).to_csv(out / "fm_frozen_head_skipped_methods.csv", index=False)
    pd.DataFrame(incomplete).to_csv(out / "fm_frozen_head_incomplete_methods.csv", index=False)
    audit = {
        "embedding_path": str(embedding_path),
        "metadata_path": str(metadata_path),
        "output_dir": str(out),
        "fm_model": fm_model,
        "frozen_or_finetuned": "frozen",
        "head_methods": list(head_methods),
        "n_patient_channel_rows": int(n_patient_channel_rows),
        "n_patients": int(n_patients),
        "n_folds": int(expected_n_folds),
        "expected_n_folds": int(expected_n_folds),
        "split_strategy": str(split_strategy),
        "completed_folds_by_method": completed,
        "incomplete_methods": incomplete,
        "allow_incomplete_methods": bool(allow_incomplete_methods),
        "A9V3_GATE": A9V3_GATE,
        **_fm_positioning_fields(fm_model),
        "warning": (
            "FM is used only as a frozen comparison baseline, not as the proposed method. "
            "Same all90 patient-level top-k evaluation. "
            "split roles are read from fm_window_metadata.csv generated by the manifest builder; "
            "split_strategy is retained for protocol logging."
        ),
    }
    if incomplete:
        audit["incomplete_methods_warning"] = _incomplete_error(incomplete)
    (out / "fm_frozen_head_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    if incomplete and not allow_incomplete_methods:
        raise RuntimeError(_incomplete_error(incomplete))
    return audit


def evaluate_embeddings(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    *,
    output_dir: str | Path,
    embedding_path: str,
    metadata_path: str,
    fm_model: str,
    head_methods: Sequence[str],
    n_splits: int,
    split_strategy: str,
    random_seed: int,
    val_ratio: float,
    allow_incomplete_methods: bool,
) -> Dict[str, Any]:
    rows = aggregate_window_embeddings_to_patient_channels(embeddings, metadata)
    results, skipped = run_fm_frozen_head(
        rows,
        fm_model=fm_model,
        head_methods=head_methods,
        n_splits=n_splits,
        random_seed=random_seed,
        val_ratio=val_ratio,
    )
    requested_methods = [f"{fm_model}__{method}" for method in head_methods]
    return write_fm_outputs(
        output_dir=output_dir,
        fm_model=fm_model,
        results=results,
        skipped=skipped,
        requested_methods=requested_methods,
        expected_n_folds=n_splits,
        split_strategy=split_strategy,
        allow_incomplete_methods=allow_incomplete_methods,
        embedding_path=embedding_path,
        metadata_path=metadata_path,
        head_methods=head_methods,
        n_patient_channel_rows=len(rows),
        n_patients=rows["subject_id"].nunique() if not rows.empty else 0,
    )
