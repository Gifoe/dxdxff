from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import torch


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
    labels = batch["labels"].detach().cpu().numpy()
    channel_mask = batch["channel_mask"].detach().cpu().numpy().astype(bool)
    temporal_attention = outputs.get("temporal_attention")
    seizure_attention = outputs.get("seizure_attention")
    temporal_np = temporal_attention.detach().cpu().numpy() if temporal_attention is not None else None
    seizure_np = seizure_attention.detach().cpu().numpy() if seizure_attention is not None else None

    records = []
    for idx, subject_id in enumerate(batch["subject_ids"]):
        valid = channel_mask[idx]
        channel_names = list(batch["channel_names"][idx])
        record = patient_metrics(
            subject_id=str(subject_id),
            channel_names=channel_names,
            scores=scores[idx],
            labels=labels[idx],
            channel_mask=valid,
            predicted_count=float(predicted_count[idx]),
        )
        if temporal_np is not None:
            record["temporal_attention"] = temporal_np[idx]
        if seizure_np is not None:
            record["seizure_attention"] = seizure_np[idx]
        records.append(record)
    return records


def patient_metrics(
    subject_id: str,
    channel_names: list[str],
    scores: np.ndarray,
    labels: np.ndarray,
    channel_mask: np.ndarray,
    predicted_count: float,
) -> dict[str, Any]:
    valid_scores = np.asarray(scores, dtype=np.float32)[channel_mask]
    valid_labels = np.asarray(labels, dtype=np.float32)[channel_mask]
    valid_channels = [name for name, keep in zip(channel_names, channel_mask) if keep]
    n_channels = len(valid_channels)
    true_count = int(valid_labels.sum())
    pred_count = int(np.clip(round(float(predicted_count)), 1, max(n_channels, 1)))

    order = np.argsort(-valid_scores)
    pred_indices = set(order[:pred_count].tolist())
    true_indices = set(np.flatnonzero(valid_labels > 0.5).tolist())
    tp = len(pred_indices & true_indices)
    fp = len(pred_indices - true_indices)
    fn = len(true_indices - pred_indices)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)
    topk = max(true_count, 1)
    topk_indices = set(order[:topk].tolist())
    topk_recall = len(topk_indices & true_indices) / max(true_count, 1) if true_count > 0 else float("nan")

    rows = []
    for rank, channel_idx in enumerate(order.tolist(), start=1):
        is_pred = channel_idx in pred_indices
        is_true = channel_idx in true_indices
        rows.append(
            {
                "subject_id": subject_id,
                "channel_name": valid_channels[channel_idx],
                "rank": rank,
                "ez_score": float(valid_scores[channel_idx]),
                "predicted_ez": int(is_pred),
                "true_ez": int(is_true),
                "predicted_count": pred_count,
                "is_tp": int(is_pred and is_true),
                "is_fp": int(is_pred and not is_true),
                "is_fn": int((not is_pred) and is_true),
            }
        )

    return {
        "subject_id": subject_id,
        "n_channels": n_channels,
        "n_true_ez": true_count,
        "n_pred_ez": pred_count,
        "true_ez_channels": [valid_channels[idx] for idx in sorted(true_indices)],
        "pred_ez_channels": [valid_channels[idx] for idx in order[:pred_count].tolist()],
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
    for name in metric_names:
        values = np.asarray([record["metrics"].get(name, np.nan) for record in records], dtype=np.float32)
        summary[f"macro_{name}"] = float(np.nanmean(values)) if np.any(np.isfinite(values)) else float("nan")
    summary["selection_score"] = float(
        0.40 * _nan0(summary["macro_AUC_PR"])
        + 0.30 * _nan0(summary["macro_TOPK_RECALL"])
        + 0.20 * _nan0(summary["macro_F1"])
        - 0.10 * _nan0(summary["macro_ABS_COUNT_BIAS_RATIO"])
    )
    return summary


def move_batch_to_device(batch: dict[str, Any], device: str | torch.device) -> dict[str, Any]:
    result = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            result[key] = value.to(device)
        else:
            result[key] = value
    return result


def _safe_metric(metric_name: str, labels: np.ndarray, scores: np.ndarray) -> float:
    if labels.size == 0 or np.unique(labels).size < 2:
        return float("nan")
    try:
        from sklearn import metrics

        fn = getattr(metrics, metric_name)
        return float(fn(labels, scores))
    except Exception:
        return float("nan")


def _nan0(value: float) -> float:
    return 0.0 if value is None or not np.isfinite(value) else float(value)


__all__ = [
    "batch_prediction_records",
    "move_batch_to_device",
    "patient_metrics",
    "predict_batches",
    "summarize_records",
]

