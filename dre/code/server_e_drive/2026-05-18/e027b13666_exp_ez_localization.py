from __future__ import annotations

import copy
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler, random_split

from data_factory import data_provider
from ez_dataset import flatten_window_samples
from exp_basic import Exp_Basic

try:
    from report_threshold import build_patient_prediction, select_best_decision_rule
except Exception:  # pragma: no cover - fallback keeps this module importable in the v2 bundle.
    from exp_ez_hybrid import select_best_decision_rule as _select_best_decision_rule_v2

    def select_best_decision_rule(patient_outputs: Sequence[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, float]]:
        records = []
        for item in patient_outputs:
            labels = np.asarray(item["labels"], dtype=np.float32)
            records.append(
                {
                    "subject_id": item["subject_id"],
                    "canonical_channels": list(item["canonical_channels"]),
                    "labels": labels,
                    "scores": np.asarray(item["scores"], dtype=np.float32),
                    "channel_mask": np.ones(labels.shape[0], dtype=bool),
                    "predicted_count": float(item.get("predicted_count_mean", labels.sum())),
                    "score_mass": float(item.get("score_mass_mean", np.asarray(item["scores"]).sum())),
                }
            )
        rule, summary, _ = _select_best_decision_rule_v2(records)
        return rule, summary

    def build_patient_prediction(patient_output: Dict[str, Any], decision_rule: Dict[str, Any]) -> Dict[str, Any]:
        scores = np.asarray(patient_output["scores"], dtype=np.float32)
        labels = np.asarray(patient_output["labels"], dtype=np.float32)
        channels = list(patient_output["canonical_channels"])
        valid_idx = np.arange(len(channels))
        k = max(1, min(len(valid_idx), int(round(float(patient_output.get("predicted_count_mean", scores.sum()))))))
        ordered = valid_idx[np.argsort(scores[valid_idx])[::-1]]
        pred_mask = np.zeros(len(channels), dtype=int)
        pred_mask[ordered[:k]] = 1
        output = dict(patient_output)
        output["predicted_mask"] = pred_mask.tolist()
        output["predicted_channels"] = [channels[idx] for idx, flag in enumerate(pred_mask) if flag]
        output["true_ez_channels"] = [channels[idx] for idx, value in enumerate(labels) if value == 1.0]
        return output

try:
    from TeChEZ import NeuralCNN, NeuralCNNPreProcessing
except Exception:  # pragma: no cover - TeChEZ can be provided by the original project environment.
    NeuralCNN = None
    NeuralCNNPreProcessing = None


def _set_random_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ChannelSegmentDataset(Dataset):
    """One item is one channel-level waveform segment.

    Local labels are stored as EZ/pathology labels. This dataset keeps the same
    publication-facing semantics: 1 means EZ/pathological and 0 means non-EZ.
    """

    def __init__(self, window_samples: Sequence[Dict[str, Any]], *, flip: bool = False) -> None:
        self.items: List[Dict[str, Any]] = []
        self.flip = bool(flip)
        for sample in window_samples:
            raw_waveform = np.asarray(sample["raw_waveform"], dtype=np.float32)
            labels_ez = np.asarray(sample["labels"], dtype=np.float32)
            channel_names = list(sample["channel_names_norm"])
            if raw_waveform.ndim != 2:
                raise ValueError("raw_waveform must have shape [channels, time].")
            if raw_waveform.shape[0] != len(channel_names) or raw_waveform.shape[0] != labels_ez.shape[0]:
                raise ValueError("raw_waveform, channel_names, and labels must have matching channel counts.")

            for channel_idx, channel_name in enumerate(channel_names):
                ez_label = float(labels_ez[channel_idx])
                pathology_label = ez_label if ez_label in {0.0, 1.0} else -1.0
                self.items.append(
                    {
                        "data": raw_waveform[channel_idx].astype(np.float32, copy=False),
                        "labels": float(pathology_label),
                        "ez_label": float(ez_label),
                        "name": str(channel_name),
                        "channel_idx": int(channel_idx),
                        "subject_id": str(sample["subject_id"]),
                        "patient": str(sample["subject_id"]),
                        "run_id": str(sample["run_id"]),
                        "edf_name": str(sample["run_id"]),
                        "sample_id": str(sample["sample_id"]),
                        "analysis_phase": str(sample["analysis_phase"]),
                        "start_indices": float(sample["start_sec"]),
                        "end_indices": float(sample["end_sec"]),
                        "start_sec": float(sample["start_sec"]),
                        "end_sec": float(sample["end_sec"]),
                        "seizure_onset_sec": float(sample["seizure_onset_sec"]),
                        "seizure_offset_sec": float(sample["seizure_offset_sec"]),
                        "ictal_duration_total_sec": float(sample["ictal_duration_total_sec"]),
                        "ictal_duration_used_sec": float(sample["ictal_duration_used_sec"]),
                    }
                )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item = dict(self.items[index])
        data = torch.as_tensor(item["data"], dtype=torch.float32)
        if self.flip and torch.rand(()) < 0.5:
            data = torch.flip(data, dims=[0])
        item["data"] = data
        return item


def _collate_channel_segments(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not batch:
        raise ValueError("_collate_channel_segments received an empty batch.")
    data = torch.stack([item["data"] for item in batch]).float()
    labels = torch.as_tensor([item["labels"] for item in batch], dtype=torch.float32)
    metadata = {
        key: [item[key] for item in batch]
        for key in batch[0].keys()
        if key not in {"data", "labels"}
    }
    return {"data": data, "labels": labels, "metadata": metadata}


class Exp_EZLocalization(Exp_Basic):
    @staticmethod
    def _log(message: str) -> None:
        print(f"[TeChEZ-OmniCNN][Train] {message}", flush=True)

    def __init__(self, args: Any) -> None:
        super().__init__(args)
        if NeuralCNN is None or NeuralCNNPreProcessing is None:
            raise ImportError(
                "Exp_EZLocalization requires the original TeChEZ.NeuralCNN components. "
                "Use Exp_EZHybridLocalization for the self-contained CNN-GNN-MIL v2 pipeline."
            )
        self.run_records, self.patient_index, self.outer_splits = data_provider(args)
        self.pre_processing = NeuralCNNPreProcessing.from_args(args)
        self._log(
            f"Experiment ready with {len(self.run_records)} run records, "
            f"{len(self.patient_index)} patients, and {len(self.outer_splits)} outer split(s)."
        )

    def _build_model(self) -> NeuralCNN:
        if NeuralCNN is None:
            raise ImportError("NeuralCNN is unavailable in this environment.")
        return NeuralCNN.from_args(self.args).to(self.device)

    @staticmethod
    def _compute_class_distribution(dataset: ChannelSegmentDataset, num_classes: int = 2) -> Dict[int, int]:
        label_counts = {i: 0 for i in range(num_classes)}
        label_counts[-1] = 0
        for item in dataset.items:
            label = int(item["labels"])
            if label in label_counts:
                label_counts[label] += 1
            else:
                label_counts[-1] += 1
        return label_counts

    def _calculate_class_weights(self, class_counts: Dict[int, int], num_classes: int = 2) -> torch.Tensor:
        total_samples = sum(int(class_counts.get(i, 0)) for i in range(num_classes))
        class_weights = torch.zeros(num_classes, dtype=torch.double)
        for class_idx in range(num_classes):
            count = int(class_counts.get(class_idx, 0))
            if count > 0 and total_samples > 0:
                class_weights[class_idx] = total_samples / (num_classes * count)
        if class_weights.sum() > 0:
            class_weights = class_weights / class_weights.sum()
        return class_weights

    @staticmethod
    def _subset_labels(subset: Subset) -> torch.Tensor:
        dataset = subset.dataset
        if not isinstance(dataset, ChannelSegmentDataset):
            raise TypeError("Expected a Subset of ChannelSegmentDataset.")
        labels = [float(dataset.items[int(idx)]["labels"]) for idx in subset.indices]
        return torch.as_tensor(labels, dtype=torch.float32)

    def _make_train_loader(
        self,
        train_subset: Subset,
        class_counts: Dict[int, int],
        *,
        batch_size: int,
    ) -> DataLoader:
        num_workers = int(getattr(self.args, "num_workers", 0))
        pin_memory = self.device.type == "cuda"
        use_weighted_sampling = bool(getattr(self.args, "use_weighted_sampling", True))
        drop_last = len(train_subset) > batch_size and (len(train_subset) % batch_size == 1)

        if use_weighted_sampling:
            class_weights = self._calculate_class_weights(class_counts, num_classes=2)
            all_labels = self._subset_labels(train_subset)
            sample_weights = torch.zeros(len(all_labels), dtype=torch.double)
            for idx, label_value in enumerate(all_labels):
                label = int(label_value.item())
                if 0 <= label < 2:
                    sample_weights[idx] = class_weights[label]
            if sample_weights.sum() > 0:
                sampler = WeightedRandomSampler(
                    weights=sample_weights,
                    num_samples=len(sample_weights),
                    replacement=True,
                )
                return DataLoader(
                    train_subset,
                    batch_size=batch_size,
                    sampler=sampler,
                    num_workers=num_workers,
                    collate_fn=_collate_channel_segments,
                    pin_memory=pin_memory,
                    drop_last=drop_last,
                )
            self._log("Weighted sampling requested, but no valid weighted samples were found; using shuffle.")

        return DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=_collate_channel_segments,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )

    def _make_eval_loader(self, dataset_or_subset: Dataset, *, batch_size: int) -> DataLoader:
        return DataLoader(
            dataset_or_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=int(getattr(self.args, "num_workers", 0)),
            collate_fn=_collate_channel_segments,
            pin_memory=self.device.type == "cuda",
        )

    def _random_segment_split(self, dataset: ChannelSegmentDataset, *, fold_seed: int) -> Tuple[Subset, Subset]:
        total_size = len(dataset)
        if total_size == 0:
            raise ValueError("Training dataset is empty.")
        if total_size == 1:
            return Subset(dataset, [0]), Subset(dataset, [0])

        val_ratio = float(getattr(self.args, "val_ratio", 0.2))
        val_size = int(total_size * val_ratio)
        val_size = max(1, min(total_size - 1, val_size))
        train_size = total_size - val_size
        generator = torch.Generator().manual_seed(int(fold_seed))
        train_subset, val_subset = random_split(dataset, [train_size, val_size], generator=generator)
        return train_subset, val_subset

    @staticmethod
    def _segment_metrics(pathology_probs: np.ndarray, pathology_labels: np.ndarray) -> Dict[str, float]:
        valid_mask = pathology_labels != -1
        if not np.any(valid_mask):
            return {"f1_pathology_macro": 0.0, "balanced_accuracy_pathology": 0.0}
        y_true_pathology = pathology_labels[valid_mask].astype(int)
        y_pred_pathology = (pathology_probs[valid_mask] >= 0.5).astype(int)
        return {
            "f1_pathology_macro": float(
                f1_score(y_true_pathology, y_pred_pathology, average="macro", zero_division=0)
            ),
            "balanced_accuracy_pathology": float(
                balanced_accuracy_score(y_true_pathology, y_pred_pathology)
            ),
        }

    def _evaluate_loader(
        self,
        model: NeuralCNN,
        dataloader: DataLoader,
        criterion: nn.Module,
    ) -> Tuple[float, Dict[str, float], Dict[str, Any]]:
        model.eval()
        self.pre_processing.disable_random_shift()

        total_loss = 0.0
        total_batches = 0
        pathology_probs: List[np.ndarray] = []
        pathology_logits: List[np.ndarray] = []
        pathology_labels: List[np.ndarray] = []
        combined_metadata: Dict[str, List[Any]] = defaultdict(list)

        with torch.no_grad():
            for batch in dataloader:
                waveforms = batch["data"].to(self.device)
                labels = batch["labels"].to(self.device)
                processed_waveforms = self.pre_processing(waveforms)
                logits = model(processed_waveforms).squeeze(-1)
                valid_mask = labels != -1
                if valid_mask.any():
                    loss = criterion(logits[valid_mask], labels[valid_mask]).mean()
                    total_loss += float(loss.detach().cpu())
                    total_batches += 1

                probs = torch.sigmoid(logits).detach().cpu().numpy()
                pathology_probs.append(probs)
                pathology_logits.append(logits.detach().cpu().numpy())
                pathology_labels.append(labels.detach().cpu().numpy())
                for key, values in batch["metadata"].items():
                    combined_metadata[key].extend(values)

        probs_np = np.concatenate(pathology_probs, axis=0) if pathology_probs else np.zeros(0, dtype=np.float32)
        logits_np = np.concatenate(pathology_logits, axis=0) if pathology_logits else np.zeros(0, dtype=np.float32)
        labels_np = np.concatenate(pathology_labels, axis=0) if pathology_labels else np.zeros(0, dtype=np.float32)
        metrics = self._segment_metrics(probs_np, labels_np)

        metadata = dict(combined_metadata)
        metadata["channel_pred"] = logits_np
        metadata["normal_probability"] = 1.0 - probs_np
        metadata["pathology_probability"] = probs_np
        metadata["channel_true"] = labels_np
        metadata["channel_true_ez"] = labels_np
        avg_loss = total_loss / max(total_batches, 1)
        return avg_loss, metrics, metadata

    def _train_one_epoch(
        self,
        model: NeuralCNN,
        train_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        model.train()
        self.pre_processing.enable_random_shift()

        losses: List[float] = []
        for batch in train_loader:
            waveforms = batch["data"].to(self.device)
            labels = batch["labels"].to(self.device)
            valid_mask = labels != -1
            if waveforms.shape[0] < 2 or not valid_mask.any():
                continue

            processed_waveforms = self.pre_processing(waveforms)
            optimizer.zero_grad(set_to_none=True)
            logits = model(processed_waveforms).squeeze(-1)
            loss = criterion(logits[valid_mask], labels[valid_mask]).mean()
            loss.backward()
            grad_clip = float(getattr(self.args, "grad_clip", 0.0))
            if grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        return float(np.mean(losses)) if losses else 0.0

    def _aggregate_segment_predictions(
        self,
        metadata: Dict[str, Any],
        *,
        fold_idx: int,
    ) -> List[Dict[str, Any]]:
        subject_channel_sum: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        subject_channel_count: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        run_channel_scores: Dict[Tuple[str, str, str], Dict[str, float]] = defaultdict(dict)
        run_meta: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

        n_rows = len(metadata.get("subject_id", []))
        pathology_probs = np.asarray(metadata.get("pathology_probability", np.zeros(n_rows)), dtype=np.float32)
        for row_idx in range(n_rows):
            subject_id = str(metadata["subject_id"][row_idx])
            run_id = str(metadata["run_id"][row_idx])
            sample_id = str(metadata["sample_id"][row_idx])
            channel_name = str(metadata["name"][row_idx])
            pathology_prob = float(pathology_probs[row_idx])

            subject_channel_sum[subject_id][channel_name] += pathology_prob
            subject_channel_count[subject_id][channel_name] += 1
            run_key = (subject_id, run_id, sample_id)
            run_channel_scores[run_key][channel_name] = pathology_prob
            if run_key not in run_meta:
                run_meta[run_key] = {
                    "run_id": run_id,
                    "sample_id": sample_id,
                    "analysis_phase": str(metadata["analysis_phase"][row_idx]),
                    "segment_start_sec": float(metadata["start_sec"][row_idx]),
                    "segment_end_sec": float(metadata["end_sec"][row_idx]),
                    "seizure_onset_sec": float(metadata["seizure_onset_sec"][row_idx]),
                    "seizure_offset_sec": float(metadata["seizure_offset_sec"][row_idx]),
                    "ictal_duration_total_sec": float(metadata["ictal_duration_total_sec"][row_idx]),
                    "ictal_duration_used_sec": float(metadata["ictal_duration_used_sec"][row_idx]),
                }

        patient_outputs: List[Dict[str, Any]] = []
        for subject_id in sorted(subject_channel_sum.keys()):
            patient_meta = self.patient_index[subject_id]
            canonical_channels = list(patient_meta["canonical_channels"])
            labels = np.asarray(patient_meta["labels"], dtype=np.float32)
            channel_meta = list(patient_meta.get("channel_meta", []))

            pathology_scores = np.zeros(len(canonical_channels), dtype=np.float32)
            counts = np.zeros(len(canonical_channels), dtype=np.float32)
            channel_to_idx = {name: idx for idx, name in enumerate(canonical_channels)}
            for channel_name, score_sum in subject_channel_sum[subject_id].items():
                if channel_name not in channel_to_idx:
                    continue
                idx = channel_to_idx[channel_name]
                count = max(1, int(subject_channel_count[subject_id][channel_name]))
                pathology_scores[idx] = float(score_sum / count)
                counts[idx] = float(count)

            run_summaries = []
            vote_count = np.zeros(len(canonical_channels), dtype=np.float32)
            rank_sum = np.zeros(len(canonical_channels), dtype=np.float32)
            rank_count = np.zeros(len(canonical_channels), dtype=np.float32)
            percentile_sum = np.zeros(len(canonical_channels), dtype=np.float32)
            percentile_count = np.zeros(len(canonical_channels), dtype=np.float32)

            for run_key, channel_scores in sorted(run_channel_scores.items(), key=lambda item: item[0]):
                run_subject, _, _ = run_key
                if run_subject != subject_id:
                    continue
                aligned_scores = np.zeros(len(canonical_channels), dtype=np.float32)
                present_mask = np.zeros(len(canonical_channels), dtype=bool)
                for channel_name, score in channel_scores.items():
                    if channel_name not in channel_to_idx:
                        continue
                    idx = channel_to_idx[channel_name]
                    aligned_scores[idx] = float(score)
                    present_mask[idx] = True

                present_indices = np.where(present_mask)[0]
                predicted_count = 0.0
                if present_indices.size > 0:
                    ordered = present_indices[np.argsort(aligned_scores[present_indices])[::-1]]
                    predicted_count = float(aligned_scores[present_indices].sum())
                    k = max(1, min(int(round(predicted_count)), int(present_indices.size)))
                    vote_count[ordered[:k]] += 1.0
                    if ordered.size > 1:
                        percentiles = np.linspace(1.0, 0.0, ordered.size, dtype=np.float32)
                    else:
                        percentiles = np.asarray([1.0], dtype=np.float32)
                    percentile_sum[ordered] += percentiles
                    percentile_count[ordered] += 1.0
                    for rank_position, channel_idx in enumerate(ordered, start=1):
                        rank_sum[channel_idx] += float(rank_position)
                        rank_count[channel_idx] += 1.0

                run_summary = dict(run_meta[run_key])
                run_summary.update(
                    {
                        "predicted_count": float(predicted_count),
                        "score_mass": float(aligned_scores[present_mask].sum()),
                        "channel_scores": aligned_scores.tolist(),
                        "channel_present_mask": present_mask.astype(int).tolist(),
                    }
                )
                run_summaries.append(run_summary)

            topk_vote_rate = np.zeros(len(canonical_channels), dtype=np.float32)
            if run_summaries:
                topk_vote_rate = vote_count / float(len(run_summaries))

            mean_rank = np.full(len(canonical_channels), np.nan, dtype=np.float32)
            rank_mask = rank_count > 0.0
            mean_rank[rank_mask] = rank_sum[rank_mask] / rank_count[rank_mask]
            inverse_mean_rank = np.zeros(len(canonical_channels), dtype=np.float32)
            if np.any(rank_mask):
                max_rank = float(np.nanmax(mean_rank[rank_mask]))
                inverse_mean_rank[rank_mask] = 1.0 - ((mean_rank[rank_mask] - 1.0) / max(max_rank - 1.0, 1.0))

            score_rank_mean = np.zeros(len(canonical_channels), dtype=np.float32)
            valid_score_mask = counts > 0
            if np.any(valid_score_mask):
                valid_indices = np.where(valid_score_mask)[0]
                ordered_valid = valid_indices[np.argsort(pathology_scores[valid_indices])[::-1]]
                if ordered_valid.size > 1:
                    rank_values = np.linspace(1.0, 0.0, ordered_valid.size, dtype=np.float32)
                else:
                    rank_values = np.asarray([1.0], dtype=np.float32)
                score_rank_mean[ordered_valid] = rank_values

            percentile_mean = np.zeros(len(canonical_channels), dtype=np.float32)
            percentile_mask = percentile_count > 0.0
            percentile_mean[percentile_mask] = percentile_sum[percentile_mask] / percentile_count[percentile_mask]

            patient_outputs.append(
                {
                    "subject_id": subject_id,
                    "fold_idx": int(fold_idx),
                    "canonical_channels": canonical_channels,
                    "labels": labels,
                    "scores": pathology_scores,
                    "score_mean": pathology_scores,
                    "score_rank_mean": score_rank_mean,
                    "percentile_mean": percentile_mean,
                    "topk_vote_rate": topk_vote_rate,
                    "run_vote_score": topk_vote_rate,
                    "mean_seizure_rank": mean_rank,
                    "inverse_mean_seizure_rank": inverse_mean_rank,
                    "predicted_count_mean": float(pathology_scores.sum()),
                    "score_mass_mean": float(pathology_scores.sum()),
                    "local_channel_count_mean": float(len(canonical_channels)),
                    "n_seizures": int(len(run_summaries)),
                    "channel_meta": channel_meta,
                    "run_summaries": run_summaries,
                    "primary_score": "omni_cnn_pathology_probability",
                }
            )

        return patient_outputs

    def _save_segment_metadata(self, metadata: Dict[str, Any], *, fold_idx: int, split_name: str) -> None:
        output_dir = Path(getattr(self.args, "output_dir", "outputs"))
        output_dir.mkdir(parents=True, exist_ok=True)
        if not metadata:
            return
        pd.DataFrame(metadata).to_csv(output_dir / f"{split_name}_segment_metadata_fold_{fold_idx}.csv", index=False)

    def _fit_cnn(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        class_counts: Dict[int, int],
        *,
        fold_idx: int,
        fold_seed: int,
    ) -> Tuple[NeuralCNN, Dict[str, object], Dict[str, float]]:
        _set_random_seed(fold_seed)
        model = self._build_model()
        criterion = nn.BCEWithLogitsLoss(reduction="none").to(self.device)
        optimizer = torch.optim.Adam(
            filter(lambda param: param.requires_grad, model.parameters()),
            lr=float(getattr(self.args, "learning_rate", 3e-4)),
        )

        class_weights = self._calculate_class_weights(class_counts, num_classes=2)
        self._log(
            "Training channel sampling | "
            f"weighted={bool(getattr(self.args, 'use_weighted_sampling', True))} | "
            f"class0_non_ez={class_counts.get(0, 0)} | "
            f"class1_ez={class_counts.get(1, 0)} | "
            f"sampling_weights={[round(float(value), 4) for value in class_weights.tolist()]} | "
            f"weighted_loss={bool(getattr(self.args, 'use_weighted_loss', False))} ignored_for_bce=True"
        )

        best_state = copy.deepcopy(model.state_dict())
        best_rule: Dict[str, object] = {"strategy": "predicted_count_topk", "count_scale": 0.50, "min_count": 1}
        best_summary: Dict[str, float] = {
            "strategy": "predicted_count_topk",
            "macro_f1": 0.0,
            "macro_auc_pr": 0.0,
            "macro_topk_recall": 0.0,
            "macro_abs_count_bias_ratio": 0.0,
        }
        best_score = -1.0
        epochs = int(getattr(self.args, "epochs", 10))
        log_interval = int(getattr(self.args, "log_interval", 1))
        fold_dir = Path(getattr(self.args, "output_dir", "outputs")) / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        best_model_path = fold_dir / "best_model.pth"

        for epoch in range(1, epochs + 1):
            train_loss = self._train_one_epoch(model, train_loader, criterion, optimizer)
            val_loss, val_segment_metrics, val_metadata = self._evaluate_loader(model, val_loader, criterion)
            val_patient_outputs = self._aggregate_segment_predictions(val_metadata, fold_idx=fold_idx)
            if val_patient_outputs:
                decision_rule, summary = select_best_decision_rule(val_patient_outputs)
                val_score = float(summary["macro_f1"])
            else:
                decision_rule, summary, val_score = best_rule, best_summary, 0.0

            improved = val_score > best_score + 1e-6
            if improved:
                best_score = val_score
                best_state = copy.deepcopy(model.state_dict())
                best_rule = dict(decision_rule)
                best_summary = dict(summary)
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": best_state,
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_macro_f1": best_score,
                        "decision_rule": best_rule,
                    },
                    best_model_path,
                )

            if epoch == 1 or epoch % log_interval == 0 or improved:
                self._log(
                    f"Epoch {epoch}/{epochs} | "
                    f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                    f"val_segment_f1={val_segment_metrics['f1_pathology_macro']:.4f} | "
                    f"val_patient_f1={float(summary.get('macro_f1', 0.0)):.4f} | "
                    f"val_auc_pr={float(summary.get('macro_auc_pr', 0.0)):.4f} | "
                    f"{'improved' if improved else 'no_improve'}"
                )

        model.load_state_dict(best_state)
        if not best_model_path.exists():
            torch.save(
                {
                    "epoch": epochs,
                    "model_state_dict": best_state,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_macro_f1": best_score,
                    "decision_rule": best_rule,
                },
                best_model_path,
            )
        self._log(f"Fold {fold_idx} best_model saved at {best_model_path}")
        return model, best_rule, best_summary

    def run(self) -> List[Dict[str, Any]]:
        all_predictions: List[Dict[str, Any]] = []
        base_seed = int(getattr(self.args, "random_seed", 42))
        total_folds = len(self.outer_splits)
        batch_size = int(getattr(self.args, "batch_size", 32))

        self._log(
            f"Starting Omni-style channel CNN cross-validation: {total_folds} fold(s) | "
            f"event_length_ms={getattr(self.args, 'cnn_event_length_ms', int(getattr(self.args, 'prepost_context_sec', 30.0) * 2000))} | "
            f"sfreq={getattr(self.args, 'raw_temporal_sfreq', 1000)} | "
            f"freq_range={getattr(self.args, 'cnn_freq_min_hz', 10.0)}-{getattr(self.args, 'cnn_freq_max_hz', 300.0)} Hz | "
            f"epochs={getattr(self.args, 'epochs', 10)}"
        )

        for split in self.outer_splits:
            fold_idx = int(split["fold_idx"])
            fold_seed = base_seed + fold_idx
            train_subjects = list(split["train_subjects"])
            test_subjects = list(split["test_subjects"])

            train_samples = flatten_window_samples(self.run_records, subject_ids=train_subjects)
            test_samples = flatten_window_samples(self.run_records, subject_ids=test_subjects)
            train_dataset = ChannelSegmentDataset(train_samples, flip=bool(getattr(self.args, "flip", True)))
            test_dataset = ChannelSegmentDataset(test_samples, flip=False)
            train_subset, val_subset = self._random_segment_split(train_dataset, fold_seed=fold_seed)
            class_counts = self._compute_class_distribution(train_dataset)

            self._log(
                f"Fold {fold_idx}/{total_folds} - segment split ready | "
                f"train_subjects={len(train_subjects)} | test_subjects={len(test_subjects)} | "
                f"train_segments={len(train_subset)} | val_segments={len(val_subset)} | "
                f"test_segments={len(test_dataset)}"
            )

            train_loader = self._make_train_loader(train_subset, class_counts, batch_size=batch_size)
            val_loader = self._make_eval_loader(val_subset, batch_size=batch_size)
            test_loader = self._make_eval_loader(test_dataset, batch_size=batch_size)

            model, decision_rule, summary = self._fit_cnn(
                train_loader,
                val_loader,
                class_counts,
                fold_idx=fold_idx,
                fold_seed=fold_seed,
            )
            self._log(
                f"Fold {fold_idx}/{total_folds} - best validation summary | "
                f"strategy={summary.get('strategy', decision_rule.get('strategy', 'unknown'))} | "
                f"f1={float(summary.get('macro_f1', 0.0)):.4f} | "
                f"auc_pr={float(summary.get('macro_auc_pr', 0.0)):.4f} | "
                f"topk_recall={float(summary.get('macro_topk_recall', 0.0)):.4f}"
            )

            criterion = nn.BCEWithLogitsLoss(reduction="none").to(self.device)
            _, test_segment_metrics, test_metadata = self._evaluate_loader(model, test_loader, criterion)
            self._save_segment_metadata(test_metadata, fold_idx=fold_idx, split_name="test")
            test_outputs = self._aggregate_segment_predictions(test_metadata, fold_idx=fold_idx)
            for patient_output in test_outputs:
                all_predictions.append(build_patient_prediction(patient_output, decision_rule))
            self._log(
                f"Fold {fold_idx}/{total_folds} complete | "
                f"test_segment_f1={test_segment_metrics['f1_pathology_macro']:.4f} | "
                f"patients={len(test_outputs)}"
            )

        self._log(f"Cross-validation finished. Total prediction records: {len(all_predictions)}")
        return all_predictions


__all__ = ["Exp_EZLocalization"]
