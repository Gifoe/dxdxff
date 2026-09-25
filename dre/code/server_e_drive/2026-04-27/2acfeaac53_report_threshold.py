from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


METRIC_KEYS = [
    "n_true_ez",
    "n_pred_ez",
    "ACC",
    "PREC",
    "REC",
    "SPEC",
    "NPV",
    "F1",
    "AUC",
    "AUC_PR",
    "MCC",
    "TOPK_HIT",
    "TOPK_RECALL",
    "PRED_COUNT_BIAS",
    "ABS_COUNT_BIAS_RATIO",
]


def aggregate_window_predictions(
    sample_outputs: Iterable[Dict[str, object]],
    patient_index: Dict[str, Dict[str, object]],
    *,
    fold_idx: int,
) -> List[Dict[str, object]]:
    grouped: Dict[str, Dict[str, Dict[str, object]]] = {}
    for output in sample_outputs:
        subject_id = str(output["subject_id"])
        run_id = str(output["run_id"])
        grouped.setdefault(subject_id, {})[run_id] = output

    patient_outputs: List[Dict[str, object]] = []
    for subject_id, run_group in grouped.items():
        patient_meta = patient_index[subject_id]
        canonical_channels = list(patient_meta["canonical_channels"])
        canonical_index = {name: idx for idx, name in enumerate(canonical_channels)}
        labels = np.asarray(patient_meta["labels"], dtype=np.float32)

        patient_sum = np.zeros(len(canonical_channels), dtype=np.float64)
        patient_count = np.zeros(len(canonical_channels), dtype=np.float64)
        vote_count = np.zeros(len(canonical_channels), dtype=np.float64)
        rank_sum = np.zeros(len(canonical_channels), dtype=np.float64)
        rank_count = np.zeros(len(canonical_channels), dtype=np.float64)
        percentile_sum = np.zeros(len(canonical_channels), dtype=np.float64)
        percentile_count = np.zeros(len(canonical_channels), dtype=np.float64)
        sample_pred_count_values: List[float] = []
        sample_score_mass_values: List[float] = []
        sample_local_count_values: List[float] = []
        run_summaries: List[Dict[str, object]] = []

        for run_id, run_output in sorted(run_group.items(), key=lambda item: item[0]):
            aligned_scores = np.zeros(len(canonical_channels), dtype=np.float32)
            present_mask = np.zeros(len(canonical_channels), dtype=bool)

            channel_names = list(run_output["channel_names_norm"])
            scores = np.asarray(run_output["scores"], dtype=np.float32)
            for local_idx, channel_name in enumerate(channel_names):
                patient_idx = canonical_index[channel_name]
                aligned_scores[patient_idx] = float(scores[local_idx])
                present_mask[patient_idx] = True

            patient_sum[present_mask] += aligned_scores[present_mask]
            patient_count[present_mask] += 1.0

            present_indices = np.where(present_mask)[0]
            if present_indices.size > 0:
                ordered_present = present_indices[np.argsort(aligned_scores[present_indices])[::-1]]
                present_count = ordered_present.size
                if present_count > 1:
                    percentile_values = np.linspace(1.0, 0.0, present_count, dtype=np.float32)
                else:
                    percentile_values = np.asarray([1.0], dtype=np.float32)
                percentile_sum[ordered_present] += percentile_values
                percentile_count[ordered_present] += 1.0

                predicted_k = int(round(float(run_output["predicted_count"])))
                predicted_k = max(1, min(present_count, predicted_k))
                topk_idx = ordered_present[:predicted_k]
                vote_count[topk_idx] += 1.0

                for rank_position, patient_idx in enumerate(ordered_present, start=1):
                    rank_sum[patient_idx] += float(rank_position)
                    rank_count[patient_idx] += 1.0

            sample_pred_count_values.append(float(run_output["predicted_count"]))
            sample_score_mass_values.append(float(run_output["score_mass"]))
            sample_local_count_values.append(float(len(channel_names)))
            run_summaries.append(
                {
                    "run_id": run_id,
                    "sample_id": str(run_output["sample_id"]),
                    "analysis_phase": str(run_output["analysis_phase"]),
                    "segment_start_sec": float(run_output["start_sec"]),
                    "segment_end_sec": float(run_output["end_sec"]),
                    "seizure_onset_sec": float(run_output["seizure_onset_sec"]),
                    "seizure_offset_sec": float(run_output["seizure_offset_sec"]),
                    "ictal_duration_total_sec": float(run_output["ictal_duration_total_sec"]),
                    "ictal_duration_used_sec": float(run_output["ictal_duration_used_sec"]),
                    "predicted_count": float(run_output["predicted_count"]),
                    "score_mass": float(run_output["score_mass"]),
                    "channel_scores": aligned_scores.tolist(),
                    "channel_present_mask": present_mask.astype(int).tolist(),
                }
            )

        mean_scores = np.zeros(len(canonical_channels), dtype=np.float32)
        valid_mask = patient_count > 0.0
        mean_scores[valid_mask] = (patient_sum[valid_mask] / patient_count[valid_mask]).astype(np.float32)

        mean_percentile = np.zeros(len(canonical_channels), dtype=np.float32)
        percentile_mask = percentile_count > 0.0
        mean_percentile[percentile_mask] = (
            percentile_sum[percentile_mask] / percentile_count[percentile_mask]
        ).astype(np.float32)

        vote_rate = np.zeros(len(canonical_channels), dtype=np.float32)
        if run_summaries:
            vote_rate = (vote_count / float(len(run_summaries))).astype(np.float32)

        mean_rank = np.full(len(canonical_channels), np.nan, dtype=np.float32)
        rank_mask = rank_count > 0.0
        mean_rank[rank_mask] = (rank_sum[rank_mask] / rank_count[rank_mask]).astype(np.float32)
        inverse_mean_rank = np.zeros(len(canonical_channels), dtype=np.float32)
        if np.any(rank_mask):
            max_rank = np.nanmax(mean_rank[rank_mask])
            inverse_mean_rank[rank_mask] = 1.0 - ((mean_rank[rank_mask] - 1.0) / max(max_rank - 1.0, 1.0))

        mean_score_rank = np.zeros(len(canonical_channels), dtype=np.float32)
        if np.any(valid_mask):
            valid_indices = np.where(valid_mask)[0]
            ordered_valid = valid_indices[np.argsort(mean_scores[valid_indices])[::-1]]
            if ordered_valid.size > 1:
                rank_values = np.linspace(1.0, 0.0, ordered_valid.size, dtype=np.float32)
            else:
                rank_values = np.asarray([1.0], dtype=np.float32)
            mean_score_rank[ordered_valid] = rank_values

        combined_scores = (
            0.45 * mean_score_rank
            + 0.35 * mean_percentile
            + 0.20 * vote_rate
        ).astype(np.float32)

        patient_outputs.append(
            {
                "subject_id": subject_id,
                "fold_idx": int(fold_idx),
                "canonical_channels": canonical_channels,
                "labels": labels,
                "scores": combined_scores,
                "score_mean": mean_scores,
                "score_rank_mean": mean_score_rank,
                "percentile_mean": mean_percentile,
                "topk_vote_rate": vote_rate,
                "mean_seizure_rank": mean_rank,
                "inverse_mean_seizure_rank": inverse_mean_rank,
                "predicted_count_mean": float(np.mean(sample_pred_count_values)) if sample_pred_count_values else 0.0,
                "score_mass_mean": float(np.mean(sample_score_mass_values)) if sample_score_mass_values else 0.0,
                "local_channel_count_mean": float(np.mean(sample_local_count_values)) if sample_local_count_values else 0.0,
                "n_seizures": int(len(run_summaries)),
                "run_summaries": run_summaries,
            }
        )

    return patient_outputs


def predicted_count_topk(
    scores: np.ndarray,
    predicted_count: float,
    *,
    count_scale: float = 1.0,
    min_count: int = 1,
) -> np.ndarray:
    scores_arr = np.asarray(scores, dtype=np.float32)
    num_channels = int(scores_arr.size)
    if num_channels == 0:
        return np.zeros(0, dtype=bool)

    scaled_count = float(np.clip(float(predicted_count) * float(count_scale), 0.0, float(num_channels)))
    k = int(round(scaled_count))
    if min_count >= 0:
        k = max(int(min_count), k)
    k = min(num_channels, k)

    pred_mask = np.zeros(num_channels, dtype=bool)
    if k > 0:
        topk_idx = np.argsort(scores_arr)[::-1][:k]
        pred_mask[topk_idx] = True
    return pred_mask


def patient_specific_gap(
    scores: np.ndarray,
    predicted_count: float,
    *,
    prior_weight: float = 1.0,
    prior_scale: float = 4.0,
    min_count: int = 1,
) -> np.ndarray:
    scores_arr = np.asarray(scores, dtype=np.float32)
    num_channels = int(scores_arr.size)
    if num_channels == 0:
        return np.zeros(0, dtype=bool)
    if num_channels == 1:
        return np.ones(1, dtype=bool)

    order = np.argsort(scores_arr)[::-1]
    sorted_scores = scores_arr[order]
    gaps = sorted_scores[:-1] - sorted_scores[1:]
    if not np.any(np.isfinite(gaps)) or np.allclose(gaps, 0.0):
        return predicted_count_topk(scores_arr, predicted_count, count_scale=1.0, min_count=min_count)

    candidate_k = np.arange(1, num_channels, dtype=np.float32)
    prior_center = float(np.clip(round(float(predicted_count)), min_count, max(num_channels - 1, 1)))
    scale = max(float(prior_scale), 1.0)
    weights = 1.0 + float(prior_weight) * np.exp(-np.abs(candidate_k - prior_center) / scale)
    weighted_gaps = gaps * weights.astype(np.float32, copy=False)
    k = int(candidate_k[int(np.argmax(weighted_gaps))])
    k = max(int(min_count), min(num_channels, k))

    pred_mask = np.zeros(num_channels, dtype=bool)
    pred_mask[order[:k]] = True
    return pred_mask


def oracle_true_count_topk(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    min_count: int = 1,
) -> np.ndarray:
    scores_arr = np.asarray(scores, dtype=np.float32)
    labels_arr = np.asarray(labels, dtype=bool)
    num_channels = int(scores_arr.size)
    if num_channels == 0:
        return np.zeros(0, dtype=bool)

    k = int(labels_arr.sum())
    k = max(int(min_count), k)
    k = min(num_channels, k)

    pred_mask = np.zeros(num_channels, dtype=bool)
    if k > 0:
        topk_idx = np.argsort(scores_arr)[::-1][:k]
        pred_mask[topk_idx] = True
    return pred_mask


def apply_decision_rule(patient_output: Dict[str, object], decision_rule: Dict[str, object]) -> np.ndarray:
    strategy = str(decision_rule.get("strategy", "predicted_count_topk"))
    scores = np.asarray(patient_output["scores"], dtype=np.float32)
    predicted_count = float(patient_output.get("predicted_count_mean", float(scores.sum())))

    if strategy == "predicted_count_topk":
        return predicted_count_topk(
            scores,
            predicted_count,
            count_scale=float(decision_rule.get("count_scale", 1.0)),
            min_count=int(decision_rule.get("min_count", 1)),
        )
    if strategy == "patient_specific_gap":
        return patient_specific_gap(
            scores,
            predicted_count,
            prior_weight=float(decision_rule.get("prior_weight", 1.0)),
            prior_scale=float(decision_rule.get("prior_scale", 4.0)),
            min_count=int(decision_rule.get("min_count", 1)),
        )
    raise ValueError(f"Unsupported decision rule strategy: {strategy}")


def compute_patient_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    pred_mask: np.ndarray,
) -> Dict[str, Optional[float]]:
    labels_bool = np.asarray(labels, dtype=bool)
    pred_bool = np.asarray(pred_mask, dtype=bool)

    if len(np.unique(labels_bool.astype(int))) > 1:
        auc_val: Optional[float] = float(roc_auc_score(labels_bool, scores))
        auc_pr_val: Optional[float] = float(average_precision_score(labels_bool, scores))
    else:
        auc_val = None
        auc_pr_val = None

    acc = float(accuracy_score(labels_bool, pred_bool))
    prec = float(precision_score(labels_bool, pred_bool, zero_division=0))
    rec = float(recall_score(labels_bool, pred_bool, zero_division=0))
    f1 = float(f1_score(labels_bool, pred_bool, zero_division=0))
    mcc = float(matthews_corrcoef(labels_bool, pred_bool))

    cm = confusion_matrix(labels_bool, pred_bool, labels=[False, True])
    tn, fp, fn, tp = cm.ravel()
    spec = float(tn / (tn + fp + 1e-8))
    npv = float(tn / (tn + fn + 1e-8))

    n_true_ez = int(labels_bool.sum())
    n_pred_ez = int(pred_bool.sum())
    topk = max(1, n_true_ez)
    topk_idx = np.argsort(scores)[::-1][:topk]
    topk_hit = float(labels_bool[topk_idx].any()) if topk_idx.size > 0 else 0.0
    topk_recall = float(labels_bool[topk_idx].sum() / max(n_true_ez, 1))
    pred_count_bias = float(n_pred_ez - n_true_ez)
    abs_count_bias_ratio = float(abs(pred_count_bias) / max(n_true_ez, 1))

    return {
        "n_true_ez": n_true_ez,
        "n_pred_ez": n_pred_ez,
        "ACC": acc,
        "PREC": prec,
        "REC": rec,
        "SPEC": spec,
        "NPV": npv,
        "F1": f1,
        "AUC": auc_val,
        "AUC_PR": auc_pr_val,
        "MCC": mcc,
        "TOPK_HIT": topk_hit,
        "TOPK_RECALL": topk_recall,
        "PRED_COUNT_BIAS": pred_count_bias,
        "ABS_COUNT_BIAS_RATIO": abs_count_bias_ratio,
    }


def compute_oracle_metrics(labels: np.ndarray, scores: np.ndarray) -> Dict[str, Optional[float]]:
    oracle_mask = oracle_true_count_topk(scores, labels, min_count=1)
    return compute_patient_metrics(labels, scores, oracle_mask)


def _score_metric_list(metric_list: List[Dict[str, float]]) -> Tuple[float, Dict[str, float]]:
    auc_pr_values = [float(m["AUC_PR"]) for m in metric_list if m["AUC_PR"] is not None]
    macro_f1 = float(np.mean([m["F1"] for m in metric_list]))
    macro_recall = float(np.mean([m["REC"] for m in metric_list]))
    macro_precision = float(np.mean([m["PREC"] for m in metric_list]))
    macro_auc_pr = float(np.mean(auc_pr_values)) if auc_pr_values else 0.0
    macro_topk_recall = float(np.mean([m["TOPK_RECALL"] for m in metric_list]))
    macro_bias_ratio = float(np.mean([m["ABS_COUNT_BIAS_RATIO"] for m in metric_list]))
    selection_score = (
        1.20 * macro_f1
        + 0.10 * macro_recall
        + 0.40 * macro_precision
        + 0.15 * macro_auc_pr
        + 0.10 * macro_topk_recall
        - 0.35 * macro_bias_ratio
    )
    return selection_score, {
        "macro_f1": macro_f1,
        "macro_recall": macro_recall,
        "macro_precision": macro_precision,
        "macro_auc_pr": macro_auc_pr,
        "macro_topk_recall": macro_topk_recall,
        "macro_abs_count_bias_ratio": macro_bias_ratio,
        "selection_score": selection_score,
    }


def _candidate_decision_rules(candidate_scales: np.ndarray) -> List[Dict[str, object]]:
    rules: List[Dict[str, object]] = []
    for count_scale in candidate_scales.tolist():
        rules.append(
            {
                "strategy": "predicted_count_topk",
                "count_scale": float(count_scale),
                "min_count": 1,
            }
        )
    rules.extend(
        [
            {
                "strategy": "patient_specific_gap",
                "prior_weight": 0.75,
                "prior_scale": 3.0,
                "min_count": 1,
            },
            {
                "strategy": "patient_specific_gap",
                "prior_weight": 1.25,
                "prior_scale": 5.0,
                "min_count": 1,
            },
        ]
    )
    return rules


def _decision_rule_primary_value(decision_rule: Dict[str, object]) -> float:
    strategy = str(decision_rule.get("strategy", "predicted_count_topk"))
    if strategy == "predicted_count_topk":
        return float(decision_rule.get("count_scale", np.nan))
    if strategy == "patient_specific_gap":
        return float(decision_rule.get("prior_weight", np.nan))
    return float("nan")


def _decision_rule_secondary_value(decision_rule: Dict[str, object]) -> float:
    strategy = str(decision_rule.get("strategy", "predicted_count_topk"))
    if strategy == "patient_specific_gap":
        return float(decision_rule.get("prior_scale", np.nan))
    return float("nan")


def select_best_decision_rule(
    patient_outputs: Iterable[Dict[str, np.ndarray]],
    candidate_scales: Optional[np.ndarray] = None,
) -> Tuple[Dict[str, object], Dict[str, float]]:
    patient_outputs = list(patient_outputs)
    if not patient_outputs:
        raise ValueError("select_best_decision_rule received no patient outputs.")

    if candidate_scales is None:
        candidate_scales = np.asarray([0.25, 0.35, 0.50, 0.65, 0.80, 0.95, 1.10, 1.25, 1.50, 1.75], dtype=np.float32)

    best_rule: Dict[str, object] = {
        "strategy": "predicted_count_topk",
        "count_scale": float(candidate_scales[0]),
        "min_count": 1,
    }
    best_summary: Dict[str, float] = {}
    best_score = -1e9

    for decision_rule in _candidate_decision_rules(candidate_scales):
        metric_list = []
        for item in patient_outputs:
            pred_mask = apply_decision_rule(item, decision_rule)
            metric_list.append(
                compute_patient_metrics(
                    np.asarray(item["labels"], dtype=np.float32),
                    np.asarray(item["scores"], dtype=np.float32),
                    pred_mask,
                )
            )

        selection_score, summary = _score_metric_list(metric_list)
        if selection_score > best_score:
            best_score = selection_score
            best_rule = dict(decision_rule)
            best_summary = {
                "strategy": str(decision_rule.get("strategy", "predicted_count_topk")),
                "primary_value": _decision_rule_primary_value(decision_rule),
                "secondary_value": _decision_rule_secondary_value(decision_rule),
                **summary,
            }

    return best_rule, best_summary


def build_patient_prediction(
    patient_output: Dict[str, np.ndarray],
    decision_rule: Dict[str, object],
) -> Dict[str, object]:
    scores = np.asarray(patient_output["scores"], dtype=np.float32)
    labels = np.asarray(patient_output["labels"], dtype=np.float32)
    pred_mask = apply_decision_rule(patient_output, decision_rule)
    metrics = compute_patient_metrics(labels, scores, pred_mask)
    oracle_mask = oracle_true_count_topk(scores, labels, min_count=1)
    oracle_metrics = compute_patient_metrics(labels, scores, oracle_mask)

    channels = list(patient_output["canonical_channels"])
    true_ez = [channels[idx] for idx, value in enumerate(labels) if value == 1.0]
    pred_ez = [channels[idx] for idx, value in enumerate(pred_mask) if value]
    tp = [channels[idx] for idx, value in enumerate(pred_mask & (labels == 1.0)) if value]
    fp = [channels[idx] for idx, value in enumerate(pred_mask & (labels == 0.0)) if value]
    fn = [channels[idx] for idx, value in enumerate((~pred_mask) & (labels == 1.0)) if value]
    oracle_pred_ez = [channels[idx] for idx, value in enumerate(oracle_mask) if value]
    oracle_tp = [channels[idx] for idx, value in enumerate(oracle_mask & (labels == 1.0)) if value]
    oracle_fp = [channels[idx] for idx, value in enumerate(oracle_mask & (labels == 0.0)) if value]
    oracle_fn = [channels[idx] for idx, value in enumerate((~oracle_mask) & (labels == 1.0)) if value]

    channel_rows = pd.DataFrame(
        {
            "channel_name_norm": channels,
            "patient_score_combined": scores,
            "patient_score_mean": np.asarray(patient_output.get("score_mean", scores), dtype=np.float32),
            "patient_score_rank_mean": np.asarray(
                patient_output.get("score_rank_mean", np.zeros_like(scores)),
                dtype=np.float32,
            ),
            "percentile_mean": np.asarray(
                patient_output.get("percentile_mean", np.zeros_like(scores)),
                dtype=np.float32,
            ),
            "topk_vote_rate": np.asarray(patient_output.get("topk_vote_rate", np.zeros_like(scores)), dtype=np.float32),
            "mean_seizure_rank": np.asarray(
                patient_output.get("mean_seizure_rank", np.full_like(scores, np.nan)),
                dtype=np.float32,
            ),
            "inverse_mean_seizure_rank": np.asarray(
                patient_output.get("inverse_mean_seizure_rank", np.zeros_like(scores)),
                dtype=np.float32,
            ),
            "is_true_ez": labels.astype(int),
            "predicted_ez": pred_mask.astype(int),
            "oracle_predicted_ez": oracle_mask.astype(int),
        }
    ).sort_values("patient_score_combined", ascending=False)
    channel_rows["rank"] = np.arange(1, len(channel_rows) + 1)

    selected_k = int(pred_mask.sum())
    patient_aggregation = {
        "n_seizures": int(patient_output.get("n_seizures", len(patient_output.get("run_summaries", [])))),
        "primary_score": "patient_wise_ranker_score",
        "secondary_stats": ["patient_score_mean", "topk_vote_rate", "mean_seizure_rank"],
        "predicted_count_mean": float(patient_output.get("predicted_count_mean", float(scores.sum()))),
        "score_mass_mean": float(patient_output.get("score_mass_mean", float(scores.sum()))),
        "local_channel_count_mean": float(patient_output.get("local_channel_count_mean", float(scores.size))),
        "decision_rule": {
            **{key: value for key, value in decision_rule.items()},
            "selected_k": selected_k,
        },
    }

    return {
        "subject_id": patient_output["subject_id"],
        "fold_idx": int(patient_output["fold_idx"]),
        "decision_rule_strategy": str(decision_rule.get("strategy", "predicted_count_topk")),
        "decision_rule_count_scale": float(decision_rule.get("count_scale", np.nan)),
        "decision_rule_primary_value": _decision_rule_primary_value(decision_rule),
        "decision_rule_secondary_value": _decision_rule_secondary_value(decision_rule),
        "metrics": metrics,
        "oracle_metrics": oracle_metrics,
        "true_ez": true_ez,
        "predicted_ez": pred_ez,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "oracle_predicted_ez": oracle_pred_ez,
        "oracle_true_positive": oracle_tp,
        "oracle_false_positive": oracle_fp,
        "oracle_false_negative": oracle_fn,
        "channel_table": channel_rows,
        "patient_aggregation": patient_aggregation,
        "run_summaries": patient_output.get("run_summaries", []),
    }


def save_patient_report(prediction: Dict[str, object], output_dir: str) -> None:
    output_root = Path(output_dir)
    patient_dir = output_root / "per_patient" / str(prediction["subject_id"])
    patient_dir.mkdir(parents=True, exist_ok=True)

    report_json = {
        "subject_id": prediction["subject_id"],
        "fold_idx": int(prediction["fold_idx"]),
        "decision_rule_strategy": prediction["decision_rule_strategy"],
        "decision_rule_count_scale": float(prediction["decision_rule_count_scale"]),
        "decision_rule_primary_value": float(prediction["decision_rule_primary_value"]),
        "decision_rule_secondary_value": float(prediction["decision_rule_secondary_value"]),
        "metrics": prediction["metrics"],
        "oracle_metrics": prediction["oracle_metrics"],
        "patient_aggregation": prediction["patient_aggregation"],
        "results": {
            "true_ez": prediction["true_ez"],
            "predicted_ez": prediction["predicted_ez"],
            "true_positive": prediction["true_positive"],
            "false_positive": prediction["false_positive"],
            "false_negative": prediction["false_negative"],
            "oracle_predicted_ez": prediction["oracle_predicted_ez"],
            "oracle_true_positive": prediction["oracle_true_positive"],
            "oracle_false_positive": prediction["oracle_false_positive"],
            "oracle_false_negative": prediction["oracle_false_negative"],
        },
        "run_summaries": prediction["run_summaries"],
    }
    with open(patient_dir / "patient_report_threshold.json", "w", encoding="utf-8") as fout:
        json.dump(report_json, fout, indent=2)

    prediction["channel_table"].to_csv(patient_dir / "channel_scores_threshold.csv", index=False)


def summarize_cv_results(predictions: List[Dict[str, object]], output_dir: str) -> Dict[str, object]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for prediction in predictions:
        save_patient_report(prediction, output_dir)
        row = {
            "subject_id": prediction["subject_id"],
            "fold_idx": prediction["fold_idx"],
            "decision_rule_strategy": prediction["decision_rule_strategy"],
            "decision_rule_count_scale": prediction["decision_rule_count_scale"],
            "decision_rule_primary_value": prediction["decision_rule_primary_value"],
            "decision_rule_secondary_value": prediction["decision_rule_secondary_value"],
        }
        row.update(prediction["metrics"])
        row.update({f"ORACLE_{key}": value for key, value in prediction["oracle_metrics"].items()})
        if prediction["oracle_metrics"]["F1"] is not None:
            row["F1_GAP_TO_ORACLE"] = float(prediction["oracle_metrics"]["F1"] - prediction["metrics"]["F1"])
            row["F1_TO_ORACLE_RATIO"] = float(
                prediction["metrics"]["F1"] / max(float(prediction["oracle_metrics"]["F1"]), 1e-8)
            )
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_root / "channel_results_threshold_all_folds.csv", index=False)

    summary = {}
    oracle_summary = {}
    gap_summary = {}
    if not df.empty:
        summary_columns = [
            "fold_idx",
            "decision_rule_count_scale",
            "decision_rule_primary_value",
            "decision_rule_secondary_value",
            *METRIC_KEYS,
        ]
        oracle_columns = [f"ORACLE_{column}" for column in METRIC_KEYS]
        gap_columns = ["F1_GAP_TO_ORACLE", "F1_TO_ORACLE_RATIO"]

        means = df[summary_columns].mean(numeric_only=True)
        stds = df[summary_columns].std(numeric_only=True)
        summary = {f"{column}_mean": float(value) for column, value in means.to_dict().items()}
        summary.update({f"{column}_std": float(value) for column, value in stds.to_dict().items()})

        oracle_means = df[oracle_columns].mean(numeric_only=True)
        oracle_stds = df[oracle_columns].std(numeric_only=True)
        oracle_summary = {f"{column}_mean": float(value) for column, value in oracle_means.to_dict().items()}
        oracle_summary.update({f"{column}_std": float(value) for column, value in oracle_stds.to_dict().items()})

        available_gap_columns = [column for column in gap_columns if column in df.columns]
        if available_gap_columns:
            gap_means = df[available_gap_columns].mean(numeric_only=True)
            gap_stds = df[available_gap_columns].std(numeric_only=True)
            gap_summary = {f"{column}_mean": float(value) for column, value in gap_means.to_dict().items()}
            gap_summary.update({f"{column}_std": float(value) for column, value in gap_stds.to_dict().items()})

    unique_strategies = sorted({str(prediction["decision_rule_strategy"]) for prediction in predictions})
    combined = {
        "n_patients": int(len(predictions)),
        "selection_strategy": "patient_wise_ranker_with_oracle_diagnostics",
        "selected_decision_strategies": unique_strategies,
        "summary_metrics": summary,
        "oracle_upper_bound_metrics": oracle_summary,
        "gap_to_oracle_metrics": gap_summary,
    }
    with open(output_root / "summary_metrics_threshold.json", "w", encoding="utf-8") as fout:
        json.dump(combined, fout, indent=2)
    return combined


__all__ = [
    "aggregate_window_predictions",
    "apply_decision_rule",
    "build_patient_prediction",
    "compute_patient_metrics",
    "compute_oracle_metrics",
    "oracle_true_count_topk",
    "patient_specific_gap",
    "predicted_count_topk",
    "select_best_decision_rule",
    "summarize_cv_results",
]
