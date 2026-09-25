from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import torch


DEFAULT_SELECTION_MODE = "val_calibrated_pred_count_topk"
SELECTION_MODES = [
    "oracle_true_count_topk",
    "pred_count_topk",
    "val_calibrated_pred_count_topk",
    "val_calibrated_threshold",
]


@torch.no_grad()
def predict_batches(
    model: torch.nn.Module,
    dataloader: Iterable[dict[str, Any]],
    device: str | torch.device = "cpu",
) -> list[dict[str, Any]]:
    model.eval()
    records: list[dict[str, Any]] = []
    for batch in dataloader:
        batch = move_batch_to_device(batch, device)
        outputs = model(batch)
        records.extend(batch_prediction_records(batch, outputs))
    return records


def batch_prediction_records(batch: dict[str, Any], outputs: dict[str, torch.Tensor]) -> list[dict[str, Any]]:
    scores = outputs["scores"].detach().cpu().numpy()
    predicted_count = outputs["predicted_count"].detach().cpu().numpy()
    predicted_count_ratio = outputs.get("predicted_count_ratio")
    predicted_count_ratio_np = (
        predicted_count_ratio.detach().cpu().numpy() if predicted_count_ratio is not None else None
    )
    labels = batch["labels"].detach().cpu().numpy()
    channel_mask = batch["channel_mask"].detach().cpu().numpy().astype(bool)
    temporal_attention = outputs.get("temporal_attention")
    seizure_attention = outputs.get("seizure_attention")
    temporal_np = temporal_attention.detach().cpu().numpy() if temporal_attention is not None else None
    seizure_np = seizure_attention.detach().cpu().numpy() if seizure_attention is not None else None

    records = []
    for idx, subject_id in enumerate(batch["subject_ids"]):
        record = raw_patient_prediction_record(
            subject_id=str(subject_id),
            center_id=_optional_str(batch.get("center_ids", [None])[idx]),
            patient_id=_optional_str(batch.get("patient_ids", [None])[idx]),
            channel_names=list(batch["channel_names"][idx]),
            scores=scores[idx],
            labels=labels[idx],
            channel_mask=channel_mask[idx],
            predicted_count=float(predicted_count[idx]),
            predicted_count_ratio=float(predicted_count_ratio_np[idx]) if predicted_count_ratio_np is not None else None,
        )
        if temporal_np is not None:
            record["temporal_attention"] = temporal_np[idx]
        if seizure_np is not None:
            record["seizure_attention"] = seizure_np[idx]
        records.append(record)
    return records


def raw_patient_prediction_record(
    subject_id: str,
    center_id: str | None,
    patient_id: str | None,
    channel_names: list[str],
    scores: np.ndarray,
    labels: np.ndarray,
    channel_mask: np.ndarray,
    predicted_count: float,
    predicted_count_ratio: float | None = None,
) -> dict[str, Any]:
    mask = np.asarray(channel_mask, dtype=bool)
    return {
        "subject_id": subject_id,
        "global_patient_id": subject_id,
        "center_id": center_id,
        "patient_id": patient_id,
        "channel_names": list(channel_names),
        "scores": np.asarray(scores, dtype=np.float32),
        "labels": np.asarray(labels, dtype=np.float32),
        "channel_mask": mask,
        "predicted_count_float": float(predicted_count),
        "predicted_count_ratio": predicted_count_ratio,
        "n_channels": int(mask.sum()),
        "n_true_ez": int(np.asarray(labels, dtype=np.float32)[mask].sum()) if mask.any() else 0,
    }


def calibrate_selection(records: list[dict[str, Any]]) -> dict[str, float]:
    count_errors = []
    true_ratios = []
    all_scores = []
    all_labels = []
    for record in records:
        valid_scores, valid_labels, _ = _valid_arrays(record)
        if valid_scores.size == 0:
            continue
        true_count = float((valid_labels > 0.5).sum())
        count_errors.append(float(record.get("predicted_count_float", 0.0)) - true_count)
        true_ratios.append(true_count / max(valid_scores.size, 1))
        all_scores.extend(valid_scores.tolist())
        all_labels.extend((valid_labels > 0.5).astype(np.float32).tolist())
    count_bias = float(np.mean(count_errors)) if count_errors else 0.0
    ez_ratio = float(np.median(true_ratios)) if true_ratios else 0.05
    threshold = _search_threshold(np.asarray(all_scores), np.asarray(all_labels))
    return {
        "count_bias_val": count_bias,
        "count_correction": -count_bias,
        "val_median_ez_ratio": ez_ratio,
        "threshold": threshold,
    }


def apply_selection_mode(
    records: list[dict[str, Any]],
    mode: str = DEFAULT_SELECTION_MODE,
    calibration: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    return [patient_metrics_from_raw(record, mode=mode, calibration=calibration) for record in records]


def patient_metrics_from_raw(
    record: dict[str, Any],
    mode: str = DEFAULT_SELECTION_MODE,
    calibration: dict[str, float] | None = None,
) -> dict[str, Any]:
    valid_scores, valid_labels, valid_channels = _valid_arrays(record)
    n_channels = len(valid_channels)
    true_indices = set(np.flatnonzero(valid_labels > 0.5).tolist())
    true_count = len(true_indices)
    order = np.argsort(-valid_scores) if n_channels else np.asarray([], dtype=int)
    selected_indices = _select_indices(valid_scores, valid_labels, order, record, mode, calibration)
    pred_count = len(selected_indices)

    tp = len(selected_indices & true_indices)
    fp = len(selected_indices - true_indices)
    fn = len(true_indices - selected_indices)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)

    topk = max(true_count, 1)
    topk_indices = set(order[:topk].tolist())
    topk_recall = len(topk_indices & true_indices) / max(true_count, 1) if true_count > 0 else float("nan")

    rows = []
    for rank, channel_idx in enumerate(order.tolist(), start=1):
        is_pred = channel_idx in selected_indices
        is_true = channel_idx in true_indices
        rows.append(
            {
                "subject_id": record["subject_id"],
                "center_id": record.get("center_id"),
                "channel_name": valid_channels[channel_idx],
                "rank": rank,
                "ez_score": float(valid_scores[channel_idx]),
                "predicted_ez": int(is_pred),
                "true_ez": int(is_true),
                "predicted_count": pred_count,
                "selection_mode": mode,
                "is_tp": int(is_pred and is_true),
                "is_fp": int(is_pred and not is_true),
                "is_fn": int((not is_pred) and is_true),
            }
        )

    return {
        **{key: value for key, value in record.items() if key not in {"scores", "labels", "channel_mask"}},
        "n_channels": n_channels,
        "n_true_ez": true_count,
        "n_pred_ez": pred_count,
        "selection_mode": mode,
        "true_ez_channels": [valid_channels[idx] for idx in sorted(true_indices)],
        "pred_ez_channels": [valid_channels[idx] for idx in sorted(selected_indices)],
        "top_channels": rows[: min(10, len(rows))],
        "channel_scores": rows,
        "metrics": {
            "AUC": _safe_metric("roc_auc_score", valid_labels, valid_scores),
            "AUC_PR": _safe_metric("average_precision_score", valid_labels, valid_scores),
            "F1": float(f1),
            "PREC": float(precision),
            "REC": float(recall),
            "TOPK_RECALL": float(topk_recall),
            "TOP1_HIT": float(order[0] in true_indices) if n_channels and true_count else float("nan"),
            "TOP3_HIT": float(bool(set(order[:3].tolist()) & true_indices)) if n_channels and true_count else float("nan"),
            "COUNT_BIAS": float(pred_count - true_count),
            "ABS_COUNT_BIAS_RATIO": float(abs(pred_count - true_count) / max(n_channels, 1)),
        },
    }


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    metric_names = [
        "AUC",
        "AUC_PR",
        "F1",
        "PREC",
        "REC",
        "TOPK_RECALL",
        "TOP1_HIT",
        "TOP3_HIT",
        "COUNT_BIAS",
        "ABS_COUNT_BIAS_RATIO",
    ]
    summary = {"n_patients": len(records)}
    warnings = []
    for name in metric_names:
        values = np.asarray([record["metrics"].get(name, np.nan) for record in records], dtype=np.float32)
        if np.any(np.isfinite(values)):
            summary[f"macro_{name}"] = float(np.nanmean(values))
            if np.any(~np.isfinite(values)):
                warnings.append(f"{name} has {int((~np.isfinite(values)).sum())} NaN patient values")
        else:
            summary[f"macro_{name}"] = float("nan")
            warnings.append(f"{name} is all NaN")
    summary["selection_score"] = early_stop_score(summary)
    if records:
        summary["selection_mode"] = records[0].get("selection_mode")
    if warnings:
        summary["warnings"] = warnings
    return summary


def site_stratified_summaries(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for center_id in sorted({str(record.get("center_id")) for record in records if record.get("center_id")}):
        subset = [record for record in records if str(record.get("center_id")) == center_id]
        result[f"site_{center_id}"] = summarize_records(subset)
    return result


def early_stop_score(summary: dict[str, Any]) -> float:
    return float(
        0.35 * _nan0(summary.get("macro_AUC_PR"))
        + 0.35 * _nan0(summary.get("macro_TOPK_RECALL"))
        + 0.20 * _nan0(summary.get("macro_F1"))
        - 0.10 * _nan0(summary.get("macro_ABS_COUNT_BIAS_RATIO"))
    )


def move_batch_to_device(batch: dict[str, Any], device: str | torch.device) -> dict[str, Any]:
    result = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            result[key] = value.to(device)
        else:
            result[key] = value
    return result


def _select_indices(
    valid_scores: np.ndarray,
    valid_labels: np.ndarray,
    order: np.ndarray,
    record: dict[str, Any],
    mode: str,
    calibration: dict[str, float] | None,
) -> set[int]:
    n_channels = int(valid_scores.size)
    if n_channels == 0:
        return set()
    calibration = calibration or {}
    if mode == "oracle_true_count_topk":
        k = int((valid_labels > 0.5).sum())
    elif mode == "pred_count_topk":
        k = int(round(float(record.get("predicted_count_float", 1.0))))
    elif mode == "val_calibrated_pred_count_topk":
        k = int(round(float(record.get("predicted_count_float", 1.0)) + float(calibration.get("count_correction", 0.0))))
    elif mode == "val_calibrated_threshold":
        threshold = float(calibration.get("threshold", 0.5))
        return set(np.flatnonzero(valid_scores >= threshold).tolist())
    elif mode == "validation_calibrated_topk":
        ratio = float(calibration.get("val_median_ez_ratio", 0.05))
        k = int(round(ratio * n_channels))
    else:
        raise ValueError(f"Unsupported selection mode: {mode}")
    k = int(np.clip(k, 1, n_channels))
    return set(order[:k].tolist())


def _valid_arrays(record: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    mask = np.asarray(record["channel_mask"], dtype=bool)
    scores = np.asarray(record["scores"], dtype=np.float32)[mask]
    labels = np.asarray(record["labels"], dtype=np.float32)[mask]
    channels = [name for name, keep in zip(record["channel_names"], mask) if keep]
    return scores, labels, channels


def _search_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    if scores.size == 0:
        return 0.5
    labels = (labels > 0.5).astype(np.float32)
    if labels.sum() <= 0:
        return float(np.quantile(scores, 0.95))
    candidates = np.unique(np.quantile(scores, np.linspace(0.05, 0.95, 37)))
    best_threshold = float(candidates[0])
    best_score = -float("inf")
    for threshold in candidates:
        pred = scores >= threshold
        tp = float(((pred == 1) & (labels == 1)).sum())
        fp = float(((pred == 1) & (labels == 0)).sum())
        fn = float(((pred == 0) & (labels == 1)).sum())
        precision = tp / max(tp + fp, 1.0)
        recall = tp / max(tp + fn, 1.0)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)
        if f1 > best_score:
            best_score = f1
            best_threshold = float(threshold)
    return best_threshold


def _safe_metric(metric_name: str, labels: np.ndarray, scores: np.ndarray) -> float:
    if labels.size == 0 or np.unique(labels).size < 2:
        return float("nan")
    try:
        from sklearn import metrics

        fn = getattr(metrics, metric_name)
        return float(fn(labels, scores))
    except Exception:
        return float("nan")


def _nan0(value: Any) -> float:
    try:
        return 0.0 if value is None or not np.isfinite(float(value)) else float(value)
    except Exception:
        return 0.0


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)


__all__ = [
    "DEFAULT_SELECTION_MODE",
    "SELECTION_MODES",
    "apply_selection_mode",
    "batch_prediction_records",
    "calibrate_selection",
    "early_stop_score",
    "move_batch_to_device",
    "patient_metrics_from_raw",
    "predict_batches",
    "raw_patient_prediction_record",
    "site_stratified_summaries",
    "summarize_records",
]
