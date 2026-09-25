from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from data_factory import data_provider, split_train_val_subjects
from ez_dataset import flatten_window_samples
from ez_patient_dataset import (
    PatientEZDataset,
    build_patient_examples,
    collate_patient_ez_batch,
    fit_window_tensor_normalizer,
)
from neuroez_hybrid_model import NeuroEZHybridModel


def _set_random_seed(seed: int) -> None:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _acquire_device(args: Any) -> torch.device:
    preferred = str(getattr(args, "device", "auto")).lower()
    if preferred == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if preferred.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(preferred)


def _move_tensors_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    moved: Dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


def _masked_bce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    labels_ez: torch.Tensor,
    mask: torch.Tensor,
    *,
    class_weight_mode: str,
    ez_negative_weight: torch.Tensor,
) -> torch.Tensor:
    valid = mask & (labels >= 0.0)
    if not torch.any(valid):
        return logits.sum() * 0.0
    losses = F.binary_cross_entropy_with_logits(logits[valid], labels[valid], reduction="none")
    if str(class_weight_mode).lower() == "ez_negative":
        weights = torch.ones_like(losses)
        weights = torch.where(labels_ez[valid] > 0.5, ez_negative_weight.to(logits.device).expand_as(weights), weights)
        return (losses * weights).sum() / weights.sum().clamp_min(1e-6)
    return losses.mean()


def _pairwise_rank_loss(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
    losses: List[torch.Tensor] = []
    for patient_idx in range(logits.shape[0]):
        valid = mask[patient_idx] & (labels[patient_idx] >= 0.0)
        pos = logits[patient_idx][valid & (labels[patient_idx] > 0.5)]
        neg = logits[patient_idx][valid & (labels[patient_idx] <= 0.5)]
        if pos.numel() == 0 or neg.numel() == 0:
            continue
        losses.append(F.relu(float(margin) - pos[:, None] + neg[None, :]).mean())
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


def _count_loss(outputs: Dict[str, torch.Tensor], batch: Dict[str, Any], args: Any) -> torch.Tensor:
    mask = batch["channel_mask"]
    labels_nez = batch["labels_nez"]
    valid_count = mask.float().sum(dim=1).clamp_min(1.0)
    true_nez_count = (labels_nez.clamp_min(0.0) * mask.float()).sum(dim=1)
    true_ez_count = valid_count - true_nez_count

    pred_nez_count = outputs["predicted_count"].clamp(min=0.0, max=float(mask.shape[1]))
    mass_nez = outputs["score_mass"].clamp(min=0.0, max=float(mask.shape[1]))
    pred_ez_count = (valid_count - pred_nez_count).clamp_min(0.0)
    mass_ez = (valid_count - mass_nez).clamp_min(0.0)

    nez_loss = 0.5 * (
        F.smooth_l1_loss(pred_nez_count, true_nez_count)
        + F.smooth_l1_loss(mass_nez, true_nez_count)
    )
    ez_loss = 0.5 * (
        F.smooth_l1_loss(pred_ez_count, true_ez_count)
        + F.smooth_l1_loss(mass_ez, true_ez_count)
    )
    count_target = str(getattr(args, "count_target", "both")).lower()
    if count_target == "ez":
        return ez_loss
    if count_target == "nez":
        return nez_loss
    return float(getattr(args, "nez_count_loss_weight", 0.5)) * nez_loss + float(getattr(args, "ez_count_loss_weight", 1.0)) * ez_loss


def _patient_targets(batch: Dict[str, Any], patient_id_to_index: Dict[str, int], device: torch.device) -> torch.Tensor:
    targets = [int(patient_id_to_index.get(str(subject_id), -1)) for subject_id in batch["subject_id"]]
    return torch.tensor(targets, dtype=torch.long, device=device)


def _supervised_contrastive_loss(outputs: Dict[str, torch.Tensor], batch: Dict[str, Any], args: Any) -> torch.Tensor:
    embeddings = outputs.get("task_embedding", outputs["patient_channel_embedding"])
    labels = batch["labels_nez"]
    mask = batch["channel_mask"] & (labels >= 0.0)
    if embeddings is None or int(mask.sum().item()) < 2:
        return outputs["logits"].sum() * 0.0

    b, c, d = embeddings.shape
    z = F.normalize(embeddings.reshape(b * c, d), dim=-1)
    y = labels.reshape(b * c)
    valid = mask.reshape(b * c)
    patient_ids = torch.arange(b, device=embeddings.device).unsqueeze(1).expand(b, c).reshape(b * c)

    z = z[valid]
    y = y[valid].long()
    patient_ids = patient_ids[valid]
    if z.shape[0] < 2:
        return outputs["logits"].sum() * 0.0

    temperature = max(float(getattr(args, "supcon_temperature", 0.20)), 1e-3)
    logits = torch.matmul(z, z.t()) / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    eye = torch.eye(logits.shape[0], device=logits.device, dtype=torch.bool)
    same_label = y[:, None] == y[None, :]
    different_patient = patient_ids[:, None] != patient_ids[None, :]
    positive_mask = same_label & different_patient & ~eye
    denominator_mask = ~eye
    valid_anchor = positive_mask.any(dim=1)
    if not torch.any(valid_anchor):
        return outputs["logits"].sum() * 0.0

    exp_logits = torch.exp(logits) * denominator_mask.float()
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-8))
    pos_log_prob = (log_prob * positive_mask.float()).sum(dim=1) / positive_mask.float().sum(dim=1).clamp_min(1.0)
    class_weights = torch.where(
        y == 0,
        torch.full_like(pos_log_prob, float(getattr(args, "supcon_ez_class_weight", 2.0))),
        torch.full_like(pos_log_prob, float(getattr(args, "supcon_nez_class_weight", 1.0))),
    )
    losses = -pos_log_prob[valid_anchor] * class_weights[valid_anchor]
    return losses.sum() / class_weights[valid_anchor].sum().clamp_min(1e-6)


def _orthogonality_loss(outputs: Dict[str, torch.Tensor], batch: Dict[str, Any]) -> torch.Tensor:
    task = outputs.get("task_embedding")
    patient = outputs.get("patient_specific_embedding")
    if task is None or patient is None:
        return outputs["logits"].sum() * 0.0
    if task.shape[-1] != patient.shape[-1]:
        patient = outputs.get("patient_specific_for_orthogonality", patient[..., : task.shape[-1]])
        if patient.shape[-1] != task.shape[-1]:
            min_dim = min(task.shape[-1], patient.shape[-1])
            task = task[..., :min_dim]
            patient = patient[..., :min_dim]
    mask = batch["channel_mask"] & (batch["labels"] >= 0.0)
    if not torch.any(mask):
        return outputs["logits"].sum() * 0.0
    cos = F.cosine_similarity(task[mask], patient[mask], dim=-1).abs()
    return cos.mean()


def _patient_classification_loss(logits: torch.Tensor | None, batch: Dict[str, Any], patient_id_to_index: Dict[str, int]) -> torch.Tensor:
    if logits is None:
        return batch["labels"].sum() * 0.0
    targets = _patient_targets(batch, patient_id_to_index, logits.device)
    valid = targets >= 0
    if not torch.any(valid) or logits.shape[-1] <= 1:
        return logits.sum() * 0.0
    return F.cross_entropy(logits[valid], targets[valid])


def _coral_loss(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if source.shape[0] < 2 or target.shape[0] < 2:
        return source.sum() * 0.0 + target.sum() * 0.0
    source_centered = source - source.mean(dim=0, keepdim=True)
    target_centered = target - target.mean(dim=0, keepdim=True)
    source_cov = source_centered.t().matmul(source_centered) / float(source.shape[0] - 1)
    target_cov = target_centered.t().matmul(target_centered) / float(target.shape[0] - 1)
    return (source_cov - target_cov).pow(2).mean() + (source.mean(dim=0) - target.mean(dim=0)).pow(2).mean()


def _conditional_alignment_loss(outputs: Dict[str, torch.Tensor], batch: Dict[str, Any], args: Any) -> torch.Tensor:
    embeddings = outputs.get("task_embedding", outputs["patient_channel_embedding"])
    labels = batch["labels_nez"]
    mask = batch["channel_mask"] & (labels >= 0.0)
    min_channels = int(getattr(args, "alignment_min_channels_per_class", 2))
    losses: List[torch.Tensor] = []
    for class_value in (0, 1):
        patient_groups: List[torch.Tensor] = []
        for patient_idx in range(embeddings.shape[0]):
            class_mask = mask[patient_idx] & (labels[patient_idx].long() == class_value)
            if int(class_mask.sum().item()) >= min_channels:
                patient_groups.append(embeddings[patient_idx][class_mask])
        for src_idx in range(len(patient_groups)):
            for dst_idx in range(src_idx + 1, len(patient_groups)):
                losses.append(_coral_loss(patient_groups[src_idx], patient_groups[dst_idx]))
    if not losses:
        return outputs["logits"].sum() * 0.0
    return torch.stack(losses).mean()


def _estimate_ez_negative_weight(dataset: PatientEZDataset, cap: float = 20.0) -> float:
    ez = 0.0
    nez = 0.0
    for item in dataset.patient_examples:
        labels_ez = np.asarray(item["labels_ez"], dtype=np.float32)
        mask = np.asarray(item["channel_mask"], dtype=bool)
        ez += float(((labels_ez == 1.0) & mask).sum())
        nez += float(((labels_ez == 0.0) & mask).sum())
    if ez <= 0.0:
        return 1.0
    return float(np.clip(nez / max(ez, 1.0), 1.0, float(cap)))


def _select_topk(scores: np.ndarray, k: int, valid_mask: np.ndarray, *, descending: bool = True) -> np.ndarray:
    pred = np.zeros(scores.shape[0], dtype=bool)
    valid_idx = np.where(valid_mask)[0]
    if valid_idx.size == 0:
        return pred
    k = max(1, min(int(k), int(valid_idx.size)))
    order = valid_idx[np.argsort(scores[valid_idx])]
    if descending:
        order = order[::-1]
    pred[order[:k]] = True
    return pred


def _selection_scores(record: Dict[str, Any], rule: Dict[str, Any]) -> np.ndarray:
    selection_score = str(rule.get("selection_score", "ez_from_nez")).lower()
    if selection_score in {"ez_from_nez", "ez", "score_ez"}:
        return np.asarray(record.get("score_ez", 1.0 - np.asarray(record["scores"], dtype=np.float32)), dtype=np.float32)
    if selection_score in {"nez", "score_nez", "p_nez"}:
        return np.asarray(record.get("score_nez", record["scores"]), dtype=np.float32)
    raise ValueError(f"Unsupported selection_score={selection_score!r}.")


def _selection_descending(rule: Dict[str, Any]) -> bool:
    selection_target = str(rule.get("selection_target", "ez")).lower()
    selection_score = str(rule.get("selection_score", "ez_from_nez")).lower()
    return not (selection_target == "ez" and selection_score in {"nez", "score_nez", "p_nez"})


def _apply_decision_rule(record: Dict[str, Any], rule: Dict[str, Any]) -> np.ndarray:
    scores = _selection_scores(record, rule)
    valid_mask = np.asarray(record["channel_mask"], dtype=bool)
    strategy = str(rule.get("strategy", "threshold_nez"))
    descending = _selection_descending(rule)
    if strategy == "existing_prediction" and "predicted_ez_mask" in record:
        return np.asarray(record["predicted_ez_mask"], dtype=bool) & valid_mask
    if strategy == "threshold_nez":
        score_nez = np.asarray(record.get("score_nez", record["scores"]), dtype=np.float32)
        threshold = float(rule.get("threshold", 0.5))
        # Return an EZ-positive mask for downstream compatibility.
        return (score_nez < threshold) & valid_mask
    if strategy == "threshold":
        threshold = float(rule.get("threshold", 0.5))
        pred = ((scores >= threshold) if descending else (scores <= threshold)) & valid_mask
        if not np.any(pred):
            return _select_topk(scores, 1, valid_mask, descending=descending)
        return pred
    if strategy == "score_mass_topk":
        k = int(round(float(record.get("score_mass_ez", 1.0)) * float(rule.get("count_scale", 1.0))))
        return _select_topk(scores, max(int(rule.get("min_count", 1)), k), valid_mask, descending=descending)
    if strategy in {"fixed_true_ez_count_topk", "true_ez_count_topk"}:
        k = int(round(float(record.get("true_ez_count", 1.0))))
        return _select_topk(scores, max(int(rule.get("min_count", 1)), k), valid_mask, descending=descending)
    k = int(round(float(record.get("predicted_ez_count", 1.0)) * float(rule.get("count_scale", 1.0))))
    return _select_topk(scores, max(int(rule.get("min_count", 1)), k), valid_mask, descending=descending)


def _reciprocal_rank(y_true: np.ndarray, scores: np.ndarray) -> float:
    if y_true.size == 0 or int((y_true == 1).sum()) == 0:
        return 0.0
    order = np.argsort(scores)[::-1]
    positive_ranks = np.where(y_true[order] == 1)[0]
    return float(1.0 / float(positive_ranks[0] + 1)) if positive_ranks.size else 0.0


def _recall_at_true_count(y_true: np.ndarray, scores: np.ndarray) -> float:
    true_count = int((y_true == 1).sum())
    if y_true.size == 0 or true_count <= 0:
        return 0.0
    pred = _select_topk(scores, true_count, np.ones_like(y_true, dtype=bool))
    return float(((y_true == 1) & pred).sum() / max(true_count, 1))


def _summarize_prediction_records(records: Sequence[Dict[str, Any]], rule: Dict[str, Any]) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    patient_metrics: Dict[str, List[float]] = {
        "accuracy": [],
        "balanced_accuracy": [],
        "macro_f1": [],
        "weighted_f1": [],
        "nez_precision": [],
        "nez_recall": [],
        "nez_f1": [],
        "ez_precision": [],
        "ez_recall": [],
        "ez_f1": [],
        "auroc_nez": [],
        "auprc_nez": [],
        "auroc_ez": [],
        "auprc_ez": [],
        "ez_recall_at_true_count": [],
        "ez_mrr": [],
    }
    pooled_y_nez: List[np.ndarray] = []
    pooled_pred_nez: List[np.ndarray] = []
    pooled_score_nez: List[np.ndarray] = []
    pooled_score_ez: List[np.ndarray] = []
    enriched: List[Dict[str, Any]] = []

    for record in records:
        labels_nez = np.asarray(record.get("labels_nez", record["labels"]), dtype=np.float32)
        labels_ez = np.asarray(record.get("labels_ez", 1.0 - labels_nez), dtype=np.float32)
        score_nez = np.asarray(record.get("score_nez", record["scores"]), dtype=np.float32)
        score_ez = np.asarray(record.get("score_ez", 1.0 - score_nez), dtype=np.float32)
        valid_mask = np.asarray(record["channel_mask"], dtype=bool)
        pred_ez_mask = _apply_decision_rule(record, rule)
        pred_nez_mask = (~pred_ez_mask) & valid_mask

        y_nez = labels_nez[valid_mask].astype(int)
        y_ez = labels_ez[valid_mask].astype(int)
        pred_nez = pred_nez_mask[valid_mask].astype(int)
        pred_ez = pred_ez_mask[valid_mask].astype(int)
        score_nez_valid = score_nez[valid_mask]
        score_ez_valid = score_ez[valid_mask]
        if y_nez.size == 0:
            continue

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_nez,
            pred_nez,
            labels=[1, 0],
            zero_division=0,
        )
        patient_accuracy = float(accuracy_score(y_nez, pred_nez))
        patient_balanced_accuracy = float(balanced_accuracy_score(y_nez, pred_nez))
        patient_macro_f1 = float(f1_score(y_nez, pred_nez, average="macro", zero_division=0))
        patient_weighted_f1 = float(f1_score(y_nez, pred_nez, average="weighted", zero_division=0))
        patient_metrics["accuracy"].append(patient_accuracy)
        patient_metrics["balanced_accuracy"].append(patient_balanced_accuracy)
        patient_metrics["macro_f1"].append(patient_macro_f1)
        patient_metrics["weighted_f1"].append(patient_weighted_f1)
        patient_metrics["nez_precision"].append(float(precision[0]))
        patient_metrics["nez_recall"].append(float(recall[0]))
        patient_metrics["nez_f1"].append(float(f1[0]))
        patient_metrics["ez_precision"].append(float(precision[1]))
        patient_metrics["ez_recall"].append(float(recall[1]))
        patient_metrics["ez_f1"].append(float(f1[1]))
        patient_metrics["ez_recall_at_true_count"].append(_recall_at_true_count(y_ez, score_ez_valid))
        patient_metrics["ez_mrr"].append(_reciprocal_rank(y_ez, score_ez_valid))

        record_nez_auc = 0.0
        record_nez_auprc = 0.0
        record_ez_auc = 0.0
        record_ez_auprc = 0.0
        if np.unique(y_nez).size > 1:
            record_nez_auc = float(roc_auc_score(y_nez, score_nez_valid))
            record_nez_auprc = float(average_precision_score(y_nez, score_nez_valid))
            record_ez_auc = float(roc_auc_score(y_ez, score_ez_valid))
            record_ez_auprc = float(average_precision_score(y_ez, score_ez_valid))
            patient_metrics["auroc_nez"].append(record_nez_auc)
            patient_metrics["auprc_nez"].append(record_nez_auprc)
            patient_metrics["auroc_ez"].append(record_ez_auc)
            patient_metrics["auprc_ez"].append(record_ez_auprc)

        pooled_y_nez.append(y_nez)
        pooled_pred_nez.append(pred_nez)
        pooled_score_nez.append(score_nez_valid)
        pooled_score_ez.append(score_ez_valid)

        channel_names = list(record["canonical_channels"])
        enriched_record = dict(record)
        enriched_record["predicted_mask"] = pred_ez_mask.astype(int).tolist()
        enriched_record["predicted_ez_mask"] = pred_ez_mask.astype(int).tolist()
        enriched_record["predicted_nez_mask"] = pred_nez_mask.astype(int).tolist()
        enriched_record["predicted_ez_channels"] = [channel_names[idx] for idx, flag in enumerate(pred_ez_mask) if flag]
        enriched_record["predicted_nez_channels"] = [channel_names[idx] for idx, flag in enumerate(pred_nez_mask) if flag]
        enriched_record["predicted_channels"] = enriched_record["predicted_ez_channels"]
        enriched_record["true_ez_channels"] = [channel_names[idx] for idx, flag in enumerate(labels_ez == 1.0) if flag]
        enriched_record["true_nez_channels"] = [channel_names[idx] for idx, flag in enumerate(labels_nez == 1.0) if flag]
        enriched_record["patient_accuracy"] = patient_accuracy
        enriched_record["patient_balanced_accuracy"] = patient_balanced_accuracy
        enriched_record["patient_macro_f1"] = patient_macro_f1
        enriched_record["patient_weighted_f1"] = patient_weighted_f1
        enriched_record["patient_nez_precision"] = float(precision[0])
        enriched_record["patient_nez_recall"] = float(recall[0])
        enriched_record["patient_nez_f1"] = float(f1[0])
        enriched_record["patient_ez_precision"] = float(precision[1])
        enriched_record["patient_ez_recall"] = float(recall[1])
        enriched_record["patient_ez_f1"] = float(f1[1])
        enriched_record["patient_precision_ez"] = float(precision[1])
        enriched_record["patient_macro_f1_ez"] = patient_macro_f1
        enriched_record["patient_macro_f1_nez"] = patient_macro_f1
        enriched_record["ez_recall_at_true_count"] = patient_metrics["ez_recall_at_true_count"][-1]
        enriched_record["patient_topk_recall"] = patient_metrics["ez_recall_at_true_count"][-1]
        enriched_record["ez_mrr"] = patient_metrics["ez_mrr"][-1]
        enriched_record["nez_auc_roc"] = record_nez_auc
        enriched_record["nez_auprc"] = record_nez_auprc
        enriched_record["ez_auc_roc"] = record_ez_auc
        enriched_record["ez_auprc"] = record_ez_auprc
        enriched_record["true_nez_count"] = float((labels_nez[valid_mask] == 1.0).sum())
        enriched_record["true_ez_count"] = float((labels_ez[valid_mask] == 1.0).sum())
        enriched_record["predicted_nez_count"] = float(pred_nez.sum())
        enriched_record["predicted_ez_count"] = float(pred_ez.sum())
        enriched.append(enriched_record)

    summary: Dict[str, float] = {}
    for name, values in patient_metrics.items():
        summary[f"patient_macro_{name}"] = float(np.mean(values)) if values else 0.0

    if pooled_y_nez:
        y_nez_all = np.concatenate(pooled_y_nez).astype(int)
        pred_nez_all = np.concatenate(pooled_pred_nez).astype(int)
        score_nez_all = np.concatenate(pooled_score_nez).astype(np.float32)
        score_ez_all = np.concatenate(pooled_score_ez).astype(np.float32)
        y_ez_all = 1 - y_nez_all
        tn, fp, fn, tp = confusion_matrix(y_nez_all, pred_nez_all, labels=[0, 1]).ravel()
        summary.update(
            {
                "pooled_accuracy": float(accuracy_score(y_nez_all, pred_nez_all)),
                "pooled_balanced_accuracy": float(balanced_accuracy_score(y_nez_all, pred_nez_all)),
                "pooled_macro_f1": float(f1_score(y_nez_all, pred_nez_all, average="macro", zero_division=0)),
                "pooled_weighted_f1": float(f1_score(y_nez_all, pred_nez_all, average="weighted", zero_division=0)),
                "confusion_ez_true_ez": float(tn),
                "confusion_ez_pred_nez": float(fp),
                "confusion_nez_pred_ez": float(fn),
                "confusion_nez_true_nez": float(tp),
            }
        )
        if np.unique(y_nez_all).size > 1:
            summary.update(
                {
                    "pooled_auroc_nez": float(roc_auc_score(y_nez_all, score_nez_all)),
                    "pooled_auprc_nez": float(average_precision_score(y_nez_all, score_nez_all)),
                    "pooled_auroc_ez": float(roc_auc_score(y_ez_all, score_ez_all)),
                    "pooled_auprc_ez": float(average_precision_score(y_ez_all, score_ez_all)),
                }
            )
        else:
            summary.update({"pooled_auroc_nez": 0.0, "pooled_auprc_nez": 0.0, "pooled_auroc_ez": 0.0, "pooled_auprc_ez": 0.0})

    summary["patient_macro_f1"] = summary.get("patient_macro_macro_f1", 0.0)
    summary["patient_balanced_accuracy"] = summary.get("patient_macro_balanced_accuracy", 0.0)
    summary["patient_weighted_f1"] = summary.get("patient_macro_weighted_f1", 0.0)
    summary["patient_accuracy"] = summary.get("patient_macro_accuracy", 0.0)
    summary["macro_f1"] = summary["patient_macro_f1"]
    summary["macro_topk_recall"] = summary.get("patient_macro_ez_recall_at_true_count", 0.0)
    summary["macro_auc_pr"] = summary.get("patient_macro_auprc_ez", 0.0)
    summary["macro_auc_roc"] = summary.get("patient_macro_auroc_ez", 0.0)
    summary["ez_auprc"] = summary.get("patient_macro_auprc_ez", 0.0)
    summary["ez_auc_roc"] = summary.get("patient_macro_auroc_ez", 0.0)
    summary["nez_auprc"] = summary.get("patient_macro_auprc_nez", 0.0)
    summary["nez_auc_roc"] = summary.get("patient_macro_auroc_nez", 0.0)
    summary["ez_mrr"] = summary.get("patient_macro_ez_mrr", 0.0)
    summary["ez_recall_at_true_count"] = summary.get("patient_macro_ez_recall_at_true_count", 0.0)
    return summary, enriched


def _default_decision_rule(args: Any | None = None) -> Dict[str, Any]:
    return {
        "strategy": str(getattr(args, "decision_rule", "threshold_nez") if args is not None else "threshold_nez"),
        "selection_target": str(getattr(args, "selection_target", "ez") if args is not None else "ez"),
        "selection_score": str(getattr(args, "selection_score", "ez_from_nez") if args is not None else "ez_from_nez"),
        "threshold": float(getattr(args, "decision_threshold", 0.5) if args is not None else 0.5),
        "count_scale": 1.0,
        "min_count": 1,
    }


def _parse_threshold_grid(args: Any | None = None) -> List[float]:
    raw_grid = getattr(args, "threshold_grid", None) if args is not None else None
    if raw_grid:
        thresholds = []
        for raw_value in str(raw_grid).split(","):
            value = raw_value.strip()
            if value:
                thresholds.append(float(value))
        if thresholds:
            return [float(np.clip(value, 0.0, 1.0)) for value in thresholds]
    return [float(value) for value in np.linspace(0.05, 0.95, 19)]


def _summary_score(summary: Dict[str, float], args: Any | None = None, *, val_loss: float | None = None) -> float:
    metric = str(getattr(args, "early_stop_metric", "patient_macro_f1") if args is not None else "patient_macro_f1")
    metric_aliases = {
        "patient_macro_f1_ez": "patient_macro_ez_f1",
        "patient_macro_f1_nez": "patient_macro_nez_f1",
        "macro_auc_pr": "ez_auprc",
        "macro_auc_roc": "ez_auc_roc",
        "macro_topk_recall": "ez_recall_at_true_count",
    }
    metric = metric_aliases.get(metric, metric)
    if metric == "val_loss":
        return -float(val_loss if val_loss is not None else 0.0)
    return float(summary.get(metric, summary.get("patient_macro_f1", summary.get("ez_auprc", 0.0))))


def select_best_decision_rule(records: Sequence[Dict[str, Any]], args: Any | None = None) -> Tuple[Dict[str, Any], Dict[str, float], List[Dict[str, Any]]]:
    if not bool(getattr(args, "tune_decision_rule", False) if args is not None else False):
        rule = _default_decision_rule(args)
        summary, enriched = _summarize_prediction_records(records, rule)
        summary = dict(summary)
        summary["selection_score"] = _summary_score(summary, args)
        summary["strategy"] = str(rule.get("strategy", "unknown"))
        return rule, summary, enriched

    candidates: List[Dict[str, Any]] = []
    base_rule = _default_decision_rule(args)
    if str(base_rule.get("strategy", "threshold_nez")) == "threshold_nez":
        for threshold in _parse_threshold_grid(args):
            candidates.append({**base_rule, "strategy": "threshold_nez", "threshold": float(threshold)})
    else:
        for count_scale in (0.5, 0.75, 1.0, 1.25, 1.5):
            candidates.append({**base_rule, "strategy": "fixed_predicted_ez_count_topk", "count_scale": count_scale, "min_count": 1})
            candidates.append({**base_rule, "strategy": "score_mass_topk", "count_scale": count_scale, "min_count": 1})
        for threshold in np.linspace(0.10, 0.90, 17):
            candidates.append({**base_rule, "strategy": "threshold", "threshold": float(threshold)})

    best_rule = candidates[0]
    best_summary, best_enriched = _summarize_prediction_records(records, best_rule)
    tune_metric_args = copy.copy(args) if args is not None else None
    if tune_metric_args is not None:
        setattr(tune_metric_args, "early_stop_metric", str(getattr(args, "threshold_tuning_metric", getattr(args, "early_stop_metric", "patient_macro_f1"))))
    best_score = _summary_score(best_summary, tune_metric_args)
    for rule in candidates[1:]:
        summary, enriched = _summarize_prediction_records(records, rule)
        score = _summary_score(summary, tune_metric_args)
        if score > best_score + 1e-8:
            best_rule = rule
            best_summary = summary
            best_enriched = enriched
            best_score = score
    best_summary = dict(best_summary)
    best_summary["selection_score"] = float(best_score)
    best_summary["strategy"] = str(best_rule.get("strategy", "unknown"))
    return dict(best_rule), best_summary, best_enriched


class Exp_EZHybridLocalization:
    """Nested patient-level CNN-GNN-MIL pipeline for EZ localization.

    V3-small treats EZ localization as NEZ-positive classification plus inverse
    EZ ranking: labels_nez=1, labels_ez=0, scores=p_NEZ, and EZ selection uses
    the lowest p_NEZ / highest 1-p_NEZ channels.
    """

    @staticmethod
    def _log(message: str) -> None:
        print(f"[NeuroEZ-V2][Train] {message}", flush=True)

    def __init__(self, args: Any) -> None:
        self.args = args
        setattr(self.args, "extract_window_tensors", bool(getattr(self.args, "extract_window_tensors", True)))
        self.device = _acquire_device(args)
        self.run_records, self.patient_index, self.outer_splits = data_provider(args)
        self.patient_id_to_index = {subject_id: idx for idx, subject_id in enumerate(sorted(self.patient_index.keys()))}
        setattr(self.args, "num_patient_domains", len(self.patient_id_to_index))
        self.current_epoch = 0
        self._log(
            f"Experiment ready with {len(self.run_records)} ictal records, "
            f"{len(self.patient_index)} patients, {len(self.outer_splits)} outer split(s), device={self.device}."
        )

    def _make_loader(self, dataset: PatientEZDataset, *, shuffle: bool, batch_size: int) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=int(getattr(self.args, "num_workers", 0)),
            collate_fn=collate_patient_ez_batch,
            pin_memory=self.device.type == "cuda",
        )

    def _build_datasets(
        self,
        fit_subjects: Sequence[str],
        val_subjects: Sequence[str],
        test_subjects: Sequence[str],
    ) -> Tuple[PatientEZDataset, PatientEZDataset, PatientEZDataset, Any]:
        fit_samples = flatten_window_samples(self.run_records, subject_ids=fit_subjects)
        val_samples = flatten_window_samples(self.run_records, subject_ids=val_subjects)
        test_samples = flatten_window_samples(self.run_records, subject_ids=test_subjects)
        normalizer = fit_window_tensor_normalizer(fit_samples, args=self.args)
        fit_examples = build_patient_examples(fit_samples, self.patient_index, normalizer=normalizer, subject_ids=fit_subjects, args=self.args)
        val_examples = build_patient_examples(val_samples, self.patient_index, normalizer=normalizer, subject_ids=val_subjects, args=self.args)
        test_examples = build_patient_examples(test_samples, self.patient_index, normalizer=normalizer, subject_ids=test_subjects, args=self.args)
        if not val_examples:
            val_examples = fit_examples
        return PatientEZDataset(fit_examples), PatientEZDataset(val_examples), PatientEZDataset(test_examples), normalizer

    def _compute_loss(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, Any], ez_negative_weight: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        logits = outputs["logits"]
        labels = batch["labels"]
        mask = batch["channel_mask"]
        bce = _masked_bce_loss(
            logits,
            labels,
            batch["labels_ez"],
            mask,
            class_weight_mode=str(getattr(self.args, "class_weight_mode", "ez_negative")),
            ez_negative_weight=ez_negative_weight,
        )
        rank_target = str(getattr(self.args, "rank_target", "nez_higher")).lower()
        if rank_target == "ez_higher":
            rank_logits = -logits
            rank_labels = batch["labels_ez"]
        else:
            rank_logits = logits
            rank_labels = batch["labels_nez"]
        rank = _pairwise_rank_loss(rank_logits, rank_labels, mask, margin=float(getattr(self.args, "rank_margin", 1.0)))
        count = _count_loss(outputs, batch, self.args)
        supcon = _supervised_contrastive_loss(outputs, batch, self.args) if bool(getattr(self.args, "use_supervised_contrastive", False)) else logits.sum() * 0.0
        orth = _orthogonality_loss(outputs, batch) if bool(getattr(self.args, "use_disentanglement", False)) else logits.sum() * 0.0
        patient_specific = _patient_classification_loss(outputs.get("patient_specific_logits"), batch, self.patient_id_to_index) if bool(getattr(self.args, "use_disentanglement", False)) else logits.sum() * 0.0
        patient_adv = _patient_classification_loss(outputs.get("task_patient_logits"), batch, self.patient_id_to_index) if bool(getattr(self.args, "use_patient_adversarial", False)) else logits.sum() * 0.0
        alignment = _conditional_alignment_loss(outputs, batch, self.args) if bool(getattr(self.args, "use_conditional_alignment", False)) else logits.sum() * 0.0
        graph_sparsity = outputs.get("graph_sparsity_loss", logits.sum() * 0.0) if bool(getattr(self.args, "use_graph_sparsity_loss", False)) else logits.sum() * 0.0

        adv_max_weight = float(getattr(self.args, "adv_max_weight", getattr(self.args, "adv_weight", 0.0)))
        adv_warmup_epochs = max(1, int(getattr(self.args, "adv_warmup_epochs", 5)))
        adv_weight = adv_max_weight * min(1.0, float(max(self.current_epoch, 1)) / float(adv_warmup_epochs))
        total = (
            bce
            + float(getattr(self.args, "rank_loss_weight", 0.05)) * rank
            + float(getattr(self.args, "count_loss_weight", 0.0)) * count
            + float(getattr(self.args, "supcon_weight", 0.0)) * supcon
            + float(getattr(self.args, "orthogonality_weight", 0.0)) * orth
            + float(getattr(self.args, "disentangle_weight", 0.0)) * patient_specific
            + adv_weight * patient_adv
            + float(getattr(self.args, "alignment_weight", 0.0)) * alignment
            + float(getattr(self.args, "graph_sparsity_weight", 0.0)) * graph_sparsity
        )
        parts = {
            "bce": float(bce.detach().cpu()),
            "rank": float(rank.detach().cpu()),
            "count": float(count.detach().cpu()),
            "supcon": float(supcon.detach().cpu()),
            "orth": float(orth.detach().cpu()),
            "patient_specific": float(patient_specific.detach().cpu()),
            "patient_adv": float(patient_adv.detach().cpu()),
            "alignment": float(alignment.detach().cpu()),
            "graph_sparsity": float(graph_sparsity.detach().cpu()),
            "adv_weight": float(adv_weight),
        }
        return total, parts

    def _dry_initialize_lazy_layers(self, model: NeuroEZHybridModel, loader: DataLoader) -> None:
        model.eval()
        with torch.no_grad():
            for batch in loader:
                _ = model(_move_tensors_to_device(batch, self.device))
                return

    def _train_one_epoch(
        self,
        model: NeuroEZHybridModel,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        ez_negative_weight: torch.Tensor,
    ) -> Dict[str, float]:
        model.train()
        losses: List[float] = []
        bce_losses: List[float] = []
        rank_losses: List[float] = []
        count_losses: List[float] = []
        supcon_losses: List[float] = []
        orth_losses: List[float] = []
        patient_adv_losses: List[float] = []
        alignment_losses: List[float] = []
        graph_sparsity_losses: List[float] = []
        for batch in loader:
            batch = _move_tensors_to_device(batch, self.device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch)
            loss, parts = self._compute_loss(outputs, batch, ez_negative_weight)
            loss.backward()
            grad_clip = float(getattr(self.args, "grad_clip", 1.0))
            if grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            bce_losses.append(parts["bce"])
            rank_losses.append(parts["rank"])
            count_losses.append(parts["count"])
            supcon_losses.append(parts["supcon"])
            orth_losses.append(parts["orth"])
            patient_adv_losses.append(parts["patient_adv"])
            alignment_losses.append(parts["alignment"])
            graph_sparsity_losses.append(parts["graph_sparsity"])
        return {
            "loss": float(np.mean(losses)) if losses else 0.0,
            "bce": float(np.mean(bce_losses)) if bce_losses else 0.0,
            "rank": float(np.mean(rank_losses)) if rank_losses else 0.0,
            "count": float(np.mean(count_losses)) if count_losses else 0.0,
            "supcon": float(np.mean(supcon_losses)) if supcon_losses else 0.0,
            "orth": float(np.mean(orth_losses)) if orth_losses else 0.0,
            "patient_adv": float(np.mean(patient_adv_losses)) if patient_adv_losses else 0.0,
            "alignment": float(np.mean(alignment_losses)) if alignment_losses else 0.0,
            "graph_sparsity": float(np.mean(graph_sparsity_losses)) if graph_sparsity_losses else 0.0,
        }

    def _evaluate(
        self,
        model: NeuroEZHybridModel,
        loader: DataLoader,
        ez_negative_weight: torch.Tensor,
        *,
        decision_rule: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float, Dict[str, float], Dict[str, Any], List[Dict[str, Any]]]:
        model.eval()
        losses: List[float] = []
        records: List[Dict[str, Any]] = []
        with torch.no_grad():
            for batch in loader:
                batch_device = _move_tensors_to_device(batch, self.device)
                outputs = model(batch_device)
                loss, _ = self._compute_loss(outputs, batch_device, ez_negative_weight)
                losses.append(float(loss.detach().cpu()))
                score_nez = outputs["scores"].detach().cpu().numpy()
                score_ez = (1.0 - outputs["scores"]).detach().cpu().numpy()
                predicted_nez_count = outputs["predicted_count"].detach().cpu().numpy()
                score_mass_nez = outputs["score_mass"].detach().cpu().numpy()
                labels = batch["labels"].cpu().numpy()
                labels_nez = batch["labels_nez"].cpu().numpy()
                labels_ez = batch["labels_ez"].cpu().numpy()
                channel_mask = batch["channel_mask"].cpu().numpy().astype(bool)
                for idx, subject_id in enumerate(batch["subject_id"]):
                    c = len(batch["canonical_channels"][idx])
                    valid_count = float(channel_mask[idx, :c].sum())
                    pred_nez_count = float(np.clip(predicted_nez_count[idx], 0.0, valid_count))
                    mass_nez = float(np.clip(score_mass_nez[idx], 0.0, valid_count))
                    records.append(
                        {
                            "subject_id": str(subject_id),
                            "canonical_channels": list(batch["canonical_channels"][idx]),
                            "channel_meta": list(batch.get("channel_meta", [[]])[idx]),
                            "labels": labels[idx, :c].astype(np.float32),
                            "labels_nez": labels_nez[idx, :c].astype(np.float32),
                            "labels_ez": labels_ez[idx, :c].astype(np.float32),
                            "scores": score_nez[idx, :c].astype(np.float32),
                            "score_nez": score_nez[idx, :c].astype(np.float32),
                            "score_ez": score_ez[idx, :c].astype(np.float32),
                            "channel_mask": channel_mask[idx, :c],
                            "predicted_count": pred_nez_count,
                            "predicted_nez_count": pred_nez_count,
                            "predicted_ez_count": float(max(0.0, valid_count - pred_nez_count)),
                            "score_mass": mass_nez,
                            "score_mass_nez": mass_nez,
                            "score_mass_ez": float(max(0.0, valid_count - mass_nez)),
                            "true_nez_count": float((labels_nez[idx, :c] * channel_mask[idx, :c]).sum()),
                            "true_ez_count": float((labels_ez[idx, :c] * channel_mask[idx, :c]).sum()),
                            "run_ids": list(batch.get("run_ids", [[]])[idx]),
                            "sample_ids": list(batch.get("sample_ids", [[]])[idx]),
                        }
                    )
        if decision_rule is None:
            rule, summary, enriched = select_best_decision_rule(records, self.args)
        else:
            summary, enriched = _summarize_prediction_records(records, decision_rule)
            rule = dict(decision_rule)
            summary = dict(summary)
            summary["strategy"] = str(rule.get("strategy", "unknown"))
            summary["selection_score"] = _summary_score(summary, self.args, val_loss=float(np.mean(losses)) if losses else 0.0)
        return float(np.mean(losses)) if losses else 0.0, summary, rule, enriched

    def _save_outputs(self, records: Sequence[Dict[str, Any]], *, fold_idx: int, split_name: str) -> None:
        output_dir = Path(getattr(self.args, "output_dir", "outputs"))
        output_dir.mkdir(parents=True, exist_ok=True)
        patient_rows = []
        channel_rows = []
        for record in records:
            patient_rows.append(
                {
                    "fold_idx": int(fold_idx),
                    "split": split_name,
                    "subject_id": record["subject_id"],
                    "true_nez_count": float(record.get("true_nez_count", 0.0)),
                    "true_ez_count": float(record.get("true_ez_count", 0.0)),
                    "predicted_nez_count": float(record.get("predicted_nez_count", record.get("predicted_count", 0.0))),
                    "predicted_ez_count": float(record.get("predicted_ez_count", 0.0)),
                    "score_mass_nez": float(record.get("score_mass_nez", record.get("score_mass", 0.0))),
                    "score_mass_ez": float(record.get("score_mass_ez", 0.0)),
                    "patient_macro_f1_nez": float(record.get("patient_macro_f1_nez", 0.0)),
                    "patient_macro_f1_ez": float(record.get("patient_macro_f1_ez", record.get("patient_macro_f1", 0.0))),
                    "patient_macro_f1": float(record.get("patient_macro_f1", 0.0)),
                    "patient_balanced_accuracy": float(record.get("patient_balanced_accuracy", 0.0)),
                    "patient_weighted_f1": float(record.get("patient_weighted_f1", 0.0)),
                    "patient_nez_precision": float(record.get("patient_nez_precision", 0.0)),
                    "patient_nez_recall": float(record.get("patient_nez_recall", 0.0)),
                    "patient_nez_f1": float(record.get("patient_nez_f1", 0.0)),
                    "patient_ez_precision": float(record.get("patient_ez_precision", 0.0)),
                    "patient_ez_recall": float(record.get("patient_ez_recall", 0.0)),
                    "patient_ez_f1": float(record.get("patient_ez_f1", 0.0)),
                    "patient_precision_ez": float(record.get("patient_precision_ez", 0.0)),
                    "patient_accuracy": float(record.get("patient_accuracy", 0.0)),
                    "nez_auc_roc": float(record.get("nez_auc_roc", 0.0)),
                    "nez_auprc": float(record.get("nez_auprc", 0.0)),
                    "ez_auc_roc": float(record.get("ez_auc_roc", 0.0)),
                    "ez_auprc": float(record.get("ez_auprc", 0.0)),
                    "ez_mrr": float(record.get("ez_mrr", 0.0)),
                    "ez_recall_at_true_count": float(record.get("ez_recall_at_true_count", record.get("patient_topk_recall", 0.0))),
                    "predicted_ez_channels": ";".join(record.get("predicted_ez_channels", record.get("predicted_channels", []))),
                    "predicted_nez_channels": ";".join(record.get("predicted_nez_channels", [])),
                    "true_ez_channels": ";".join(record.get("true_ez_channels", [])),
                    "true_nez_channels": ";".join(record.get("true_nez_channels", [])),
                    "n_seizures": len(record.get("run_ids", [])),
                }
            )
            labels_nez = np.asarray(record.get("labels_nez", record["labels"]), dtype=np.float32)
            labels_ez = np.asarray(record.get("labels_ez", 1.0 - labels_nez), dtype=np.float32)
            score_nez = np.asarray(record.get("score_nez", record["scores"]), dtype=np.float32)
            score_ez = np.asarray(record.get("score_ez", 1.0 - score_nez), dtype=np.float32)
            pred_ez = np.asarray(record.get("predicted_ez_mask", record.get("predicted_mask", np.zeros_like(labels_nez))), dtype=int)
            pred_nez = np.asarray(record.get("predicted_nez_mask", 1 - pred_ez), dtype=int)
            valid_mask = np.asarray(record["channel_mask"], dtype=bool)
            rank_nez_desc = np.full(score_nez.shape[0], -1, dtype=int)
            rank_ez_asc_nez = np.full(score_nez.shape[0], -1, dtype=int)
            valid_idx = np.where(valid_mask)[0]
            if valid_idx.size > 0:
                for rank, idx in enumerate(valid_idx[np.argsort(score_nez[valid_idx])[::-1]], start=1):
                    rank_nez_desc[idx] = rank
                for rank, idx in enumerate(valid_idx[np.argsort(score_nez[valid_idx])], start=1):
                    rank_ez_asc_nez[idx] = rank
            for channel_idx, channel_name in enumerate(record["canonical_channels"]):
                channel_rows.append(
                    {
                        "fold_idx": int(fold_idx),
                        "split": split_name,
                        "subject_id": record["subject_id"],
                        "channel_name": channel_name,
                        "true_nez": float(labels_nez[channel_idx]),
                        "true_ez": float(labels_ez[channel_idx]),
                        "score_nez_probability": float(score_nez[channel_idx]),
                        "score_ez_probability": float(score_ez[channel_idx]),
                        "rank_nez_desc": int(rank_nez_desc[channel_idx]),
                        "rank_ez_asc_nez": int(rank_ez_asc_nez[channel_idx]),
                        "predicted_nez": int(pred_nez[channel_idx]),
                        "predicted_ez": int(pred_ez[channel_idx]),
                    }
                )
        pd.DataFrame(patient_rows).to_csv(output_dir / f"{split_name}_patient_predictions_neuroez_v2_fold_{fold_idx}.csv", index=False)
        pd.DataFrame(channel_rows).to_csv(output_dir / f"{split_name}_channel_predictions_neuroez_v2_fold_{fold_idx}.csv", index=False)

    def run(self) -> List[Dict[str, Any]]:
        all_test_records: List[Dict[str, Any]] = []
        base_seed = int(getattr(self.args, "random_seed", 42))
        batch_size = int(getattr(self.args, "patient_batch_size", getattr(self.args, "batch_size", 2)))
        epochs = int(getattr(self.args, "epochs", 50))
        patience = int(getattr(self.args, "patience", 15))
        min_epochs_before_early_stop = max(0, int(getattr(self.args, "min_epochs_before_early_stop", 50)))
        log_interval = int(getattr(self.args, "log_interval", 1))

        self._log(
            "Starting patient-level NeuroEZ-V2 cross-validation | "
            f"folds={len(self.outer_splits)} | epochs={epochs} | batch_size={batch_size} | "
            f"label_semantics=1:NEZ,0:EZ | scores=p_NEZ | decision=threshold_nez"
        )

        for split in self.outer_splits:
            fold_idx = int(split["fold_idx"])
            fold_seed = base_seed + fold_idx
            _set_random_seed(fold_seed)
            fit_subjects, val_subjects = split_train_val_subjects(
                list(split["train_subjects"]),
                val_ratio=float(getattr(self.args, "val_ratio", 0.2)),
                random_seed=base_seed,
                fold_idx=fold_idx,
            )
            test_subjects = list(split["test_subjects"])
            train_dataset, val_dataset, test_dataset, normalizer = self._build_datasets(fit_subjects, val_subjects, test_subjects)
            if len(train_dataset) == 0 or len(test_dataset) == 0:
                self._log(f"Fold {fold_idx}: skipped because train/test dataset is empty.")
                continue

            train_loader = self._make_loader(train_dataset, shuffle=True, batch_size=batch_size)
            val_loader = self._make_loader(val_dataset, shuffle=False, batch_size=batch_size)
            test_loader = self._make_loader(test_dataset, shuffle=False, batch_size=batch_size)

            model = NeuroEZHybridModel(self.args).to(self.device)
            self._dry_initialize_lazy_layers(model, train_loader)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=float(getattr(self.args, "learning_rate", 3e-4)),
                weight_decay=float(getattr(self.args, "weight_decay", 1e-4)),
            )
            ez_weight_arg = str(getattr(self.args, "ez_negative_weight", "2")).lower()
            if ez_weight_arg == "auto":
                ez_negative_weight_value = _estimate_ez_negative_weight(
                    train_dataset,
                    cap=float(getattr(self.args, "ez_negative_weight_cap", getattr(self.args, "pos_weight_cap", 20.0))),
                )
            else:
                ez_negative_weight_value = float(ez_weight_arg)
            ez_negative_weight = torch.tensor(ez_negative_weight_value, dtype=torch.float32, device=self.device)

            self._log(
                f"Fold {fold_idx}/{len(self.outer_splits)} ready | "
                f"fit_patients={len(train_dataset)} | val_patients={len(val_dataset)} | test_patients={len(test_dataset)} | "
                f"window_feature_dim={normalizer.feature_dim} | ez_negative_weight={ez_negative_weight_value:.3f}"
            )

            best_state = copy.deepcopy(model.state_dict())
            best_rule: Dict[str, Any] = _default_decision_rule(self.args)
            best_summary: Dict[str, float] = {"patient_macro_f1": 0.0, "patient_balanced_accuracy": 0.0, "pooled_macro_f1": 0.0, "selection_score": -1.0}
            best_score = -1.0
            epochs_without_improvement = 0
            fold_dir = Path(getattr(self.args, "output_dir", "outputs")) / f"fold_{fold_idx}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            best_model_path = fold_dir / "best_neuroez_v2_model.pth"

            for epoch in range(1, epochs + 1):
                self.current_epoch = epoch
                train_metrics = self._train_one_epoch(model, train_loader, optimizer, ez_negative_weight)
                val_loss, val_summary, val_rule, val_records = self._evaluate(model, val_loader, ez_negative_weight)
                val_score = _summary_score(val_summary, self.args, val_loss=val_loss)
                val_summary["selection_score"] = float(val_score)
                improved = val_score > best_score + 1e-6
                if improved:
                    best_score = val_score
                    best_state = copy.deepcopy(model.state_dict())
                    best_rule = dict(val_rule)
                    best_summary = dict(val_summary)
                    epochs_without_improvement = 0
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": best_state,
                            "optimizer_state_dict": optimizer.state_dict(),
                            "decision_rule": best_rule,
                            "val_summary": best_summary,
                            "normalizer_mean": normalizer.mean,
                            "normalizer_std": normalizer.std,
                            "label_semantics": "1=NEZ,0=EZ",
                            "score_semantics": "scores=p_NEZ, score_ez=1-p_NEZ",
                            "positive_label": str(getattr(self.args, "positive_label", "nez")),
                            "ez_negative_weight": float(ez_negative_weight_value),
                        },
                        best_model_path,
                    )
                    self._save_outputs(val_records, fold_idx=fold_idx, split_name="val")
                else:
                    epochs_without_improvement += 1

                if epoch == 1 or epoch % log_interval == 0 or improved:
                    self._log(
                        f"Fold {fold_idx} epoch {epoch}/{epochs} | "
                        f"train_loss={train_metrics['loss']:.4f} | val_loss={val_loss:.4f} | "
                        f"val_patient_macro_f1={float(val_summary.get('patient_macro_f1', 0.0)):.4f} | "
                        f"val_patient_bal_acc={float(val_summary.get('patient_balanced_accuracy', 0.0)):.4f} | "
                        f"val_pooled_macro_f1={float(val_summary.get('pooled_macro_f1', 0.0)):.4f} | "
                        f"rule={val_rule.get('strategy', 'unknown')}@{float(val_rule.get('threshold', 0.5)):.2f} | "
                        f"{'improved' if improved else 'no_improve'}"
                    )
                if patience > 0 and epoch > min_epochs_before_early_stop and epochs_without_improvement >= patience:
                    self._log(
                        f"Fold {fold_idx}: early stop at epoch {epoch} "
                        f"after minimum {min_epochs_before_early_stop} epochs."
                    )
                    break

            model.load_state_dict(best_state)
            test_loss, test_summary, _, test_records = self._evaluate(model, test_loader, ez_negative_weight, decision_rule=best_rule)
            self._save_outputs(test_records, fold_idx=fold_idx, split_name="test")
            all_test_records.extend(test_records)
            self._log(
                f"Fold {fold_idx}/{len(self.outer_splits)} complete | "
                f"test_loss={test_loss:.4f} | test_patient_macro_f1={float(test_summary.get('patient_macro_f1', 0.0)):.4f} | "
                f"test_patient_bal_acc={float(test_summary.get('patient_balanced_accuracy', 0.0)):.4f} | "
                f"test_pooled_macro_f1={float(test_summary.get('pooled_macro_f1', 0.0)):.4f} | "
                f"test_pooled_bal_acc={float(test_summary.get('pooled_balanced_accuracy', 0.0)):.4f} | "
                f"best_val_patient_macro_f1={float(best_summary.get('patient_macro_f1', 0.0)):.4f}"
            )

        if all_test_records:
            overall_rule = {**_default_decision_rule(self.args), "strategy": "existing_prediction"}
            overall_summary, _ = _summarize_prediction_records(all_test_records, overall_rule)
            output_dir = Path(getattr(self.args, "output_dir", "outputs"))
            output_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([overall_summary]).to_csv(output_dir / "heldout_summary_neuroez_v3.csv", index=False)
            pd.Series(overall_summary).to_json(output_dir / "heldout_summary_neuroez_v3.json", indent=2)
            self._log(
                "Cross-validation held-out mean | "
                f"patients={len(all_test_records)} | "
                f"patient_macro_f1={float(overall_summary.get('patient_macro_f1', 0.0)):.4f} | "
                f"patient_bal_acc={float(overall_summary.get('patient_balanced_accuracy', 0.0)):.4f} | "
                f"pooled_macro_f1={float(overall_summary.get('pooled_macro_f1', 0.0)):.4f} | "
                f"pooled_bal_acc={float(overall_summary.get('pooled_balanced_accuracy', 0.0)):.4f} | "
                f"pooled_auroc_nez={float(overall_summary.get('pooled_auroc_nez', 0.0)):.4f} | "
                f"pooled_auprc_ez={float(overall_summary.get('pooled_auprc_ez', 0.0)):.4f}"
            )
        self._log(f"Cross-validation finished. Total held-out patient records: {len(all_test_records)}")
        return all_test_records


__all__ = ["Exp_EZHybridLocalization", "select_best_decision_rule"]
