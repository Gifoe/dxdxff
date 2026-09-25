from __future__ import annotations

import copy
from collections import defaultdict
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from data_factory import data_provider, split_train_val_subjects
from ez_dataset import fit_feature_normalizer, flatten_window_samples
from exp_basic import Exp_Basic
from report_threshold import build_patient_prediction, select_best_decision_rule
from TeChEZ import PatientChannelRanker


def _rank_percentiles(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)[::-1]
    if order.size <= 1:
        percentiles = np.ones(order.size, dtype=np.float32)
    else:
        percentiles = np.linspace(1.0, 0.0, order.size, dtype=np.float32)
    result = np.zeros(order.size, dtype=np.float32)
    result[order] = percentiles
    return result


def _compute_hjorth_parameters(channel_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    activity = np.var(channel_data, axis=-1)
    diff1 = np.diff(channel_data, axis=-1)
    diff2 = np.diff(diff1, axis=-1)

    diff1_var = np.var(diff1, axis=-1) + 1e-8
    diff2_var = np.var(diff2, axis=-1) + 1e-8
    activity_safe = activity + 1e-8
    mobility = np.sqrt(diff1_var / activity_safe)
    complexity = np.sqrt(diff2_var / diff1_var) / (mobility + 1e-8)
    return mobility.astype(np.float32, copy=False), complexity.astype(np.float32, copy=False)


def _set_random_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class PatientRankDataset(Dataset):
    def __init__(self, patient_payloads: Sequence[Dict[str, Any]]) -> None:
        self.patient_payloads = list(patient_payloads)

    def __len__(self) -> int:
        return len(self.patient_payloads)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        payload = self.patient_payloads[index]
        return {
            "features": np.asarray(payload["feature_matrix"], dtype=np.float32),
            "labels": np.asarray(payload["labels"], dtype=np.float32),
            "subject_id": payload["subject_id"],
        }


def _collate_patient_rank_batch(items: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    batch_size = len(items)
    max_channels = max(int(item["features"].shape[0]) for item in items)
    feature_dim = int(items[0]["features"].shape[1])

    features = torch.zeros(batch_size, max_channels, feature_dim, dtype=torch.float32)
    labels = torch.zeros(batch_size, max_channels, dtype=torch.float32)
    channel_mask = torch.zeros(batch_size, max_channels, dtype=torch.bool)

    for batch_idx, item in enumerate(items):
        x = torch.as_tensor(item["features"], dtype=torch.float32)
        y = torch.as_tensor(item["labels"], dtype=torch.float32)
        num_channels = int(x.shape[0])
        features[batch_idx, :num_channels] = x
        labels[batch_idx, :num_channels] = y
        channel_mask[batch_idx, :num_channels] = True

    return {
        "x": features,
        "labels": labels,
        "channel_mask": channel_mask,
    }


class Exp_EZLocalization(Exp_Basic):
    @staticmethod
    def _log(message: str) -> None:
        print(f"[TeChEZ-Ranker][Train] {message}", flush=True)

    def __init__(self, args: Any) -> None:
        super().__init__(args)
        self.run_records, self.patient_index, self.outer_splits = data_provider(args)
        self._log(
            f"Experiment ready with {len(self.run_records)} run records, "
            f"{len(self.patient_index)} patients, and {len(self.outer_splits)} outer split(s)."
        )

    def _build_model(self):
        return PatientChannelRanker(self.args).to(self.device)

    def _compute_raw_channel_features(self, sample: Dict[str, Any]) -> np.ndarray:
        raw = np.asarray(sample["raw_waveform"], dtype=np.float32)
        if raw.ndim != 2:
            raise ValueError("raw_waveform must have shape [channels, time].")

        valid_samples = max(2, min(int(sample.get("raw_valid_samples", raw.shape[-1])), int(raw.shape[-1])))
        raw_seg = raw[:, :valid_samples]
        raw_sfreq = float(sample.get("raw_temporal_sfreq", 256.0))
        duration_sec = max(float(valid_samples) / max(raw_sfreq, 1e-8), 1e-6)

        abs_mean = np.mean(np.abs(raw_seg), axis=-1)
        rms = np.sqrt(np.mean(np.square(raw_seg), axis=-1) + 1e-8)
        variance = np.var(raw_seg, axis=-1)
        peak_to_peak = np.ptp(raw_seg, axis=-1)
        max_abs = np.max(np.abs(raw_seg), axis=-1)
        line_length_per_sec = np.sum(np.abs(np.diff(raw_seg, axis=-1)), axis=-1) / duration_sec
        zero_cross_rate = np.mean((raw_seg[:, 1:] * raw_seg[:, :-1]) < 0.0, axis=-1)
        mobility, complexity = _compute_hjorth_parameters(raw_seg)

        raw_features = np.stack(
            [
                abs_mean,
                rms,
                variance,
                peak_to_peak,
                max_abs,
                line_length_per_sec,
                zero_cross_rate.astype(np.float32, copy=False),
                mobility,
                complexity,
            ],
            axis=-1,
        ).astype(np.float32, copy=False)
        mean = raw_features.mean(axis=0, keepdims=True)
        std = np.clip(raw_features.std(axis=0, keepdims=True), 1e-5, None)
        return ((raw_features - mean) / std).astype(np.float32, copy=False)

    def _normalize_sample_features(self, sample: Dict[str, Any], normalizer) -> np.ndarray:
        feature_matrix = normalizer.transform(sample["spectral_features"], sample["graph_features"]).T
        raw_weight = float(getattr(self.args, "raw_feature_weight", 1.0))
        if raw_weight <= 0.0:
            return feature_matrix.astype(np.float32, copy=False)

        raw_features = self._compute_raw_channel_features(sample) * raw_weight
        combined = np.concatenate([feature_matrix, raw_features], axis=-1)
        return combined.astype(np.float32, copy=False)

    def _build_patient_payloads(
        self,
        subject_ids: Sequence[str],
        normalizer,
    ) -> List[Dict[str, Any]]:
        seizure_samples = flatten_window_samples(self.run_records, subject_ids=subject_ids)
        samples_by_subject: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for sample in seizure_samples:
            samples_by_subject[str(sample["subject_id"])].append(sample)

        patient_payloads: List[Dict[str, Any]] = []
        for subject_id in subject_ids:
            patient_meta = self.patient_index[subject_id]
            canonical_channels = list(patient_meta["canonical_channels"])
            labels = np.asarray(patient_meta["labels"], dtype=np.float32)
            channel_meta = list(patient_meta.get("channel_meta", []))
            subject_samples = samples_by_subject.get(subject_id, [])
            n_seizures = len(subject_samples)
            if n_seizures == 0:
                continue

            feature_bags: Dict[str, List[np.ndarray]] = defaultdict(list)
            anomaly_percentiles: Dict[str, List[float]] = defaultdict(list)
            anomaly_top25_votes: Dict[str, List[float]] = defaultdict(list)
            seizure_presence: Dict[str, int] = defaultdict(int)
            feature_dim: int | None = None

            for sample in subject_samples:
                normalized_features = self._normalize_sample_features(sample, normalizer)
                feature_dim = int(normalized_features.shape[1])
                channel_names = list(sample["channel_names_norm"])
                anomaly_energy = np.sqrt(np.mean(np.square(normalized_features), axis=1))
                anomaly_percentile = _rank_percentiles(anomaly_energy)
                top25_cutoff = max(1, int(np.ceil(0.25 * len(channel_names))))
                top25_indices = set(np.argsort(anomaly_energy)[::-1][:top25_cutoff].tolist())

                for local_idx, channel_name in enumerate(channel_names):
                    feature_bags[channel_name].append(normalized_features[local_idx])
                    anomaly_percentiles[channel_name].append(float(anomaly_percentile[local_idx]))
                    anomaly_top25_votes[channel_name].append(1.0 if local_idx in top25_indices else 0.0)
                    seizure_presence[channel_name] += 1

            if feature_dim is None:
                continue

            rows = []
            feature_rows = []
            consistency_scores = []
            presence_scores = []
            for channel_idx, channel_name in enumerate(canonical_channels):
                vectors = feature_bags.get(channel_name)
                if vectors:
                    stacked = np.stack(vectors, axis=0).astype(np.float32, copy=False)
                    mean_feat = stacked.mean(axis=0)
                    max_feat = stacked.max(axis=0)
                    std_feat = stacked.std(axis=0)
                    q75_feat = np.quantile(stacked, 0.75, axis=0).astype(np.float32, copy=False)
                    presence_ratio = float(seizure_presence[channel_name] / max(n_seizures, 1))
                    percentile_mean = float(np.mean(anomaly_percentiles[channel_name]))
                    top25_vote_rate = float(np.mean(anomaly_top25_votes[channel_name]))
                else:
                    mean_feat = np.zeros(feature_dim, dtype=np.float32)
                    max_feat = np.zeros(feature_dim, dtype=np.float32)
                    std_feat = np.zeros(feature_dim, dtype=np.float32)
                    q75_feat = np.zeros(feature_dim, dtype=np.float32)
                    presence_ratio = 0.0
                    percentile_mean = 0.0
                    top25_vote_rate = 0.0

                feature_vector = np.concatenate(
                    [
                        mean_feat,
                        max_feat,
                        std_feat,
                        q75_feat,
                        np.asarray([presence_ratio, percentile_mean, top25_vote_rate], dtype=np.float32),
                    ],
                    axis=0,
                ).astype(np.float32, copy=False)
                rows.append(
                    {
                        "subject_id": subject_id,
                        "channel_name_norm": channel_name,
                        "label": float(labels[channel_idx]),
                        "presence_ratio": presence_ratio,
                        "percentile_mean": percentile_mean,
                        "top25_vote_rate": top25_vote_rate,
                    }
                )
                feature_rows.append(feature_vector)
                consistency_scores.append(0.6 * percentile_mean + 0.4 * top25_vote_rate)
                presence_scores.append(presence_ratio)

            patient_payloads.append(
                {
                    "subject_id": subject_id,
                    "canonical_channels": canonical_channels,
                    "labels": labels,
                    "channel_meta": channel_meta,
                    "rows": rows,
                    "feature_matrix": np.stack(feature_rows, axis=0).astype(np.float32, copy=False),
                    "consistency_scores": np.asarray(consistency_scores, dtype=np.float32),
                    "presence_scores": np.asarray(presence_scores, dtype=np.float32),
                    "n_seizures": n_seizures,
                }
            )

        return patient_payloads

    @staticmethod
    def _fit_patient_feature_scaler(patient_payloads: Sequence[Dict[str, Any]]) -> StandardScaler:
        if not patient_payloads:
            raise ValueError("No patient payloads were produced for scaler fitting.")
        x_train = np.concatenate(
            [np.asarray(payload["feature_matrix"], dtype=np.float32) for payload in patient_payloads],
            axis=0,
        )
        scaler = StandardScaler()
        scaler.fit(x_train)
        return scaler

    @staticmethod
    def _transform_patient_payloads(
        patient_payloads: Sequence[Dict[str, Any]],
        scaler: StandardScaler,
    ) -> List[Dict[str, Any]]:
        transformed = []
        for payload in patient_payloads:
            item = dict(payload)
            item["feature_matrix"] = scaler.transform(
                np.asarray(payload["feature_matrix"], dtype=np.float32)
            ).astype(np.float32, copy=False)
            transformed.append(item)
        return transformed

    @staticmethod
    def _estimate_pos_weight(patient_payloads: Sequence[Dict[str, Any]]) -> float:
        pos_count = 0.0
        neg_count = 0.0
        for payload in patient_payloads:
            labels = np.asarray(payload["labels"], dtype=np.float32)
            pos_count += float((labels == 1.0).sum())
            neg_count += float((labels == 0.0).sum())
        if pos_count <= 0.0:
            return 1.0
        return float(np.clip(neg_count / max(pos_count, 1.0), 1.0, 20.0))

    def _move_batch_to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {key: value.to(self.device) for key, value in batch.items()}

    def _compute_ranker_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        channel_mask: torch.Tensor,
        *,
        pos_weight: float,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        logits = outputs["logits"]
        scores = outputs["scores"]
        mask = channel_mask.bool()
        labels = labels.float()

        valid_logits = logits[mask]
        valid_labels = labels[mask]
        pos_weight_tensor = torch.as_tensor(float(pos_weight), device=logits.device, dtype=logits.dtype)
        bce_loss = F.binary_cross_entropy_with_logits(
            valid_logits,
            valid_labels,
            pos_weight=pos_weight_tensor,
        )

        pairwise_losses = []
        for batch_idx in range(labels.shape[0]):
            sample_mask = mask[batch_idx]
            sample_labels = labels[batch_idx][sample_mask]
            sample_logits = logits[batch_idx][sample_mask]
            pos_logits = sample_logits[sample_labels > 0.5]
            neg_logits = sample_logits[sample_labels <= 0.5]
            if pos_logits.numel() == 0 or neg_logits.numel() == 0:
                continue
            margin = float(getattr(self.args, "rank_margin", 0.25))
            diff = pos_logits.unsqueeze(1) - neg_logits.unsqueeze(0)
            pairwise_losses.append(F.softplus(margin - diff).mean())
        pairwise_loss = torch.stack(pairwise_losses).mean() if pairwise_losses else logits.sum() * 0.0

        masked_logits = logits.masked_fill(~mask, -1e9)
        log_probs = F.log_softmax(masked_logits, dim=1)
        pos_counts = (labels * mask.float()).sum(dim=1)
        listwise_mask = pos_counts > 0.0
        target = torch.zeros_like(labels)
        target[listwise_mask] = labels[listwise_mask] / pos_counts[listwise_mask].unsqueeze(1).clamp_min(1.0)
        listwise_loss = (-(target * log_probs).sum(dim=1)[listwise_mask]).mean()
        if not torch.isfinite(listwise_loss):
            listwise_loss = logits.sum() * 0.0

        valid_counts = mask.float().sum(dim=1).clamp_min(1.0)
        true_fraction = pos_counts / valid_counts
        count_fraction = outputs["predicted_count"] / valid_counts
        score_mass_fraction = scores.sum(dim=1) / valid_counts
        count_loss = F.smooth_l1_loss(count_fraction, true_fraction)
        mass_loss = F.smooth_l1_loss(score_mass_fraction, true_fraction)

        total_loss = (
            float(getattr(self.args, "rank_lambda", 1.0)) * pairwise_loss
            + float(getattr(self.args, "listwise_lambda", 1.0)) * listwise_loss
            + float(getattr(self.args, "bce_lambda", 0.25)) * bce_loss
            + float(getattr(self.args, "count_lambda", 0.20)) * count_loss
            + float(getattr(self.args, "mass_lambda", 0.05)) * mass_loss
        )
        return total_loss, {
            "loss": float(total_loss.detach().cpu()),
            "pairwise": float(pairwise_loss.detach().cpu()),
            "listwise": float(listwise_loss.detach().cpu()),
            "bce": float(bce_loss.detach().cpu()),
            "count": float(count_loss.detach().cpu()),
            "mass": float(mass_loss.detach().cpu()),
        }

    def _make_loader(
        self,
        patient_payloads: Sequence[Dict[str, Any]],
        *,
        shuffle: bool,
        seed: int,
    ) -> DataLoader:
        generator = torch.Generator()
        generator.manual_seed(int(seed))
        return DataLoader(
            PatientRankDataset(patient_payloads),
            batch_size=int(getattr(self.args, "batch_size", 4)),
            shuffle=shuffle,
            collate_fn=_collate_patient_rank_batch,
            generator=generator if shuffle else None,
        )

    def _fit_ranker(
        self,
        train_payloads: Sequence[Dict[str, Any]],
        val_payloads: Sequence[Dict[str, Any]],
        *,
        fold_seed: int,
    ) -> Tuple[PatientChannelRanker, Dict[str, object], Dict[str, float]]:
        if not train_payloads:
            raise ValueError("Training split contains no patient payloads.")

        _set_random_seed(fold_seed)
        input_dim = int(train_payloads[0]["feature_matrix"].shape[1])
        setattr(self.args, "input_dim", input_dim)
        model = self._build_model()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(getattr(self.args, "learning_rate", 5e-4)),
            weight_decay=float(getattr(self.args, "weight_decay", 1e-3)),
        )
        pos_weight = self._estimate_pos_weight(train_payloads)
        train_loader = self._make_loader(train_payloads, shuffle=True, seed=fold_seed)

        best_state = copy.deepcopy(model.state_dict())
        best_rule: Dict[str, object] = {"strategy": "predicted_count_topk", "count_scale": 0.50, "min_count": 1}
        best_summary: Dict[str, float] = {}
        best_score = -1e9
        bad_epochs = 0
        epochs = int(getattr(self.args, "epochs", 160))
        patience = int(getattr(self.args, "patience", 25))

        for epoch in range(1, epochs + 1):
            model.train()
            epoch_losses = []
            for batch in train_loader:
                batch = self._move_batch_to_device(batch)
                optimizer.zero_grad(set_to_none=True)
                outputs = model(batch["x"], batch["channel_mask"])
                loss, loss_parts = self._compute_ranker_loss(
                    outputs,
                    batch["labels"],
                    batch["channel_mask"],
                    pos_weight=pos_weight,
                )
                loss.backward()
                grad_clip = float(getattr(self.args, "grad_clip", 1.0))
                if grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                epoch_losses.append(loss_parts["loss"])

            val_outputs = self._build_patient_outputs(val_payloads, model, fold_idx=0)
            decision_rule, summary = select_best_decision_rule(val_outputs)
            val_score = (
                0.40 * float(summary["macro_auc_pr"])
                + 0.35 * float(summary["macro_topk_recall"])
                + 0.25 * float(summary["macro_f1"])
                - 0.10 * float(summary["macro_abs_count_bias_ratio"])
            )

            if val_score > best_score + 1e-5:
                best_score = val_score
                best_state = copy.deepcopy(model.state_dict())
                best_rule = dict(decision_rule)
                best_summary = dict(summary)
                bad_epochs = 0
                improved = "improved"
            else:
                bad_epochs += 1
                improved = f"no_improve={bad_epochs}/{patience}"

            if epoch == 1 or epoch % int(getattr(self.args, "log_interval", 5)) == 0 or bad_epochs == 0:
                self._log(
                    f"Epoch {epoch}/{epochs} | "
                    f"loss={float(np.mean(epoch_losses)):.4f} | "
                    f"val_f1={summary['macro_f1']:.4f} | "
                    f"val_auc_pr={summary['macro_auc_pr']:.4f} | "
                    f"val_oracle_f1={summary['macro_topk_recall']:.4f} | "
                    f"strategy={summary['strategy']} | {improved}"
                )

            if bad_epochs >= patience:
                self._log(f"Early stopping at epoch {epoch}/{epochs}.")
                break

        model.load_state_dict(best_state)
        return model, best_rule, best_summary

    def _smooth_scores_by_shaft(
        self,
        scores: np.ndarray,
        channel_meta: Sequence[Dict[str, Any]],
    ) -> np.ndarray:
        smoothed = np.asarray(scores, dtype=np.float32).copy()
        grouped_indices: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        for idx, meta in enumerate(channel_meta):
            group = str(meta.get("contact_group", ""))
            number = meta.get("contact_number")
            sort_number = int(number) if number is not None else 10**9
            grouped_indices[group].append((idx, sort_number))

        for group_items in grouped_indices.values():
            ordered = [idx for idx, _ in sorted(group_items, key=lambda item: item[1])]
            local_scores = scores[ordered]
            local_smooth = local_scores.copy()
            for local_idx, _ in enumerate(ordered):
                neighbors = []
                if local_idx > 0:
                    neighbors.append(local_scores[local_idx - 1])
                if local_idx + 1 < len(ordered):
                    neighbors.append(local_scores[local_idx + 1])
                if not neighbors:
                    continue
                neighbor_mean = float(np.mean(neighbors))
                window_start = max(0, local_idx - 1)
                window_end = min(len(ordered), local_idx + 2)
                local_window_mean = float(np.mean(local_scores[window_start:window_end]))
                local_smooth[local_idx] = (
                    0.55 * local_scores[local_idx]
                    + 0.25 * neighbor_mean
                    + 0.20 * local_window_mean
                )
            smoothed[ordered] = local_smooth
        return smoothed.astype(np.float32, copy=False)

    def _build_patient_outputs(
        self,
        patient_payloads: Sequence[Dict[str, Any]],
        model: PatientChannelRanker,
        *,
        fold_idx: int,
    ) -> List[Dict[str, Any]]:
        model.eval()
        patient_outputs: List[Dict[str, Any]] = []
        shaft_weight = float(getattr(self.args, "ranker_shaft_smooth_weight", 0.0))
        consistency_weight = float(getattr(self.args, "ranker_consistency_weight", 0.0))
        presence_weight = float(getattr(self.args, "ranker_presence_weight", 0.0))

        with torch.no_grad():
            for payload in patient_payloads:
                feature_matrix = np.asarray(payload["feature_matrix"], dtype=np.float32)
                x = torch.as_tensor(feature_matrix, dtype=torch.float32, device=self.device).unsqueeze(0)
                channel_mask = torch.ones(1, feature_matrix.shape[0], dtype=torch.bool, device=self.device)
                outputs = model(x, channel_mask)
                base_scores = outputs["scores"].squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
                predicted_count = float(outputs["predicted_count"].squeeze(0).detach().cpu())
                score_mass = float(outputs["score_mass"].squeeze(0).detach().cpu())

                consistency_scores = np.asarray(payload["consistency_scores"], dtype=np.float32)
                presence_scores = np.asarray(payload["presence_scores"], dtype=np.float32)
                shaft_scores = self._smooth_scores_by_shaft(base_scores, payload["channel_meta"])

                base_weight = max(0.05, 1.0 - shaft_weight - consistency_weight - presence_weight)
                final_scores = (
                    base_weight * base_scores
                    + shaft_weight * shaft_scores
                    + consistency_weight * consistency_scores
                    + presence_weight * presence_scores
                )
                final_scores = np.clip(final_scores, 0.0, 1.0).astype(np.float32, copy=False)

                mean_rank = np.full(len(final_scores), np.nan, dtype=np.float32)
                ordered = np.argsort(final_scores)[::-1]
                for rank_position, channel_idx in enumerate(ordered, start=1):
                    mean_rank[channel_idx] = float(rank_position)
                inverse_mean_rank = np.zeros(len(final_scores), dtype=np.float32)
                if len(final_scores) > 0:
                    inverse_mean_rank[ordered] = _rank_percentiles(final_scores)

                patient_outputs.append(
                    {
                        "subject_id": payload["subject_id"],
                        "fold_idx": int(fold_idx),
                        "canonical_channels": list(payload["canonical_channels"]),
                        "labels": np.asarray(payload["labels"], dtype=np.float32),
                        "scores": final_scores,
                        "score_mean": base_scores,
                        "score_rank_mean": shaft_scores,
                        "percentile_mean": consistency_scores,
                        "topk_vote_rate": consistency_scores,
                        "mean_seizure_rank": mean_rank,
                        "inverse_mean_seizure_rank": inverse_mean_rank,
                        "predicted_count_mean": predicted_count,
                        "score_mass_mean": score_mass,
                        "local_channel_count_mean": float(len(final_scores)),
                        "n_seizures": int(payload["n_seizures"]),
                        "run_summaries": [],
                    }
                )
        return patient_outputs

    def run(self) -> List[Dict[str, Any]]:
        all_predictions: List[Dict[str, Any]] = []
        val_ratio = float(getattr(self.args, "val_ratio", 0.2))
        base_seed = int(getattr(self.args, "random_seed", 42))
        total_folds = len(self.outer_splits)

        self._log(
            f"Starting patient-wise ranking cross-validation: {total_folds} fold(s) | "
            f"feature_family=spectral_graph_raw | "
            f"feature_scales={getattr(self.args, 'feature_scales_sec', '3,5,10')} | "
            f"epochs={getattr(self.args, 'epochs', 160)}"
        )

        for split in self.outer_splits:
            fold_idx = int(split["fold_idx"])
            fold_seed = base_seed + fold_idx

            train_subjects, val_subjects = split_train_val_subjects(
                split["train_subjects"],
                val_ratio=val_ratio,
                random_seed=base_seed,
                fold_idx=fold_idx,
            )
            test_subjects = list(split["test_subjects"])

            self._log(
                f"Fold {fold_idx}/{total_folds} - subject split ready | "
                f"train={len(train_subjects)} | val={len(val_subjects)} | test={len(test_subjects)}"
            )

            train_seizure_samples = flatten_window_samples(self.run_records, subject_ids=train_subjects)
            if not train_seizure_samples:
                raise ValueError("Training split contains no seizure samples.")
            normalizer = fit_feature_normalizer(train_seizure_samples)

            train_payloads_raw = self._build_patient_payloads(train_subjects, normalizer)
            val_payloads_raw = self._build_patient_payloads(val_subjects, normalizer)
            test_payloads_raw = self._build_patient_payloads(test_subjects, normalizer)
            scaler = self._fit_patient_feature_scaler(train_payloads_raw)
            train_payloads = self._transform_patient_payloads(train_payloads_raw, scaler)
            val_payloads = self._transform_patient_payloads(val_payloads_raw, scaler)
            test_payloads = self._transform_patient_payloads(test_payloads_raw, scaler)

            self._log(
                f"Fold {fold_idx}/{total_folds} - patient tensors ready | "
                f"train_patients={len(train_payloads)} | val_patients={len(val_payloads)} | "
                f"test_patients={len(test_payloads)} | input_dim={train_payloads[0]['feature_matrix'].shape[1]}"
            )

            model, decision_rule, summary = self._fit_ranker(
                train_payloads,
                val_payloads,
                fold_seed=fold_seed,
            )
            self._log(
                f"Fold {fold_idx}/{total_folds} - best validation summary | "
                f"strategy={summary['strategy']} | "
                f"f1={summary['macro_f1']:.4f} | "
                f"auc_pr={summary['macro_auc_pr']:.4f} | "
                f"oracle_f1={summary['macro_topk_recall']:.4f} | "
                f"bias_ratio={summary['macro_abs_count_bias_ratio']:.4f}"
            )

            test_outputs = self._build_patient_outputs(test_payloads, model, fold_idx=fold_idx)
            for patient_output in test_outputs:
                all_predictions.append(build_patient_prediction(patient_output, decision_rule))
            self._log(
                f"Fold {fold_idx}/{total_folds} complete. Generated {len(test_outputs)} test prediction record(s)."
            )

        self._log(f"Cross-validation finished. Total prediction records: {len(all_predictions)}")
        return all_predictions


__all__ = ["Exp_EZLocalization"]
