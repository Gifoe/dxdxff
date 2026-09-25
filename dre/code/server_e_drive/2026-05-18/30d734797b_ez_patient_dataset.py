from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


def _normalize_raw_waveform(
    raw_waveform: np.ndarray,
    raw_valid_samples: int,
    *,
    raw_valid_start_sample: int = 0,
    args: Any | None = None,
) -> np.ndarray:
    raw = np.asarray(raw_waveform, dtype=np.float32).copy()
    if raw.ndim != 2:
        raise ValueError("raw_waveform must have shape [channels, time].")

    valid_start = max(0, min(int(raw_valid_start_sample), int(raw.shape[-1])))
    valid_samples = max(1, min(int(raw_valid_samples), int(raw.shape[-1]) - valid_start))
    valid_end = min(int(raw.shape[-1]), valid_start + valid_samples)
    valid_segment = raw[:, valid_start:valid_end]
    raw_view = str(getattr(args, "raw_view", "baseline_normalized") if args is not None else "baseline_normalized").lower()
    if raw_view == "baseline_normalized":
        onset_sample = int(raw.shape[-1] // 2)
        baseline_end = min(valid_end, max(valid_start + 1, onset_sample))
        baseline = raw[:, valid_start:baseline_end]
        if baseline.shape[-1] < 2:
            baseline = valid_segment
    else:
        baseline = valid_segment
    channel_mean = baseline.mean(axis=-1, keepdims=True)
    channel_std = np.clip(baseline.std(axis=-1, keepdims=True), 1e-5, None)
    raw[:, valid_start:valid_end] = (valid_segment - channel_mean) / channel_std
    if valid_start > 0:
        raw[:, :valid_start] = 0.0
    if valid_end < raw.shape[-1]:
        raw[:, valid_end:] = 0.0
    return raw.astype(np.float32, copy=False)


def _as_window_tensors(sample: Dict[str, Any], feature_dim_fallback: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    window_features = np.asarray(sample.get("window_features", np.zeros((0, 0, 0))), dtype=np.float32)
    window_adjacency = np.asarray(sample.get("window_adjacency", np.zeros((0, 0, 0))), dtype=np.float32)
    window_centers = np.asarray(sample.get("window_relative_centers_sec", np.zeros((0,))), dtype=np.float32)

    if window_features.ndim == 3 and window_features.shape[0] > 0 and window_features.shape[-1] > 0:
        if window_adjacency.ndim != 3 or window_adjacency.shape[0] != window_features.shape[0]:
            num_windows, num_channels = int(window_features.shape[0]), int(window_features.shape[1])
            window_adjacency = np.zeros((num_windows, num_channels, num_channels), dtype=np.float32)
        if window_centers.shape[0] != window_features.shape[0]:
            window_centers = np.arange(window_features.shape[0], dtype=np.float32)
        return window_features, window_adjacency.astype(np.float32, copy=False), window_centers.astype(np.float32, copy=False)

    spectral = np.asarray(sample.get("spectral_features", np.zeros((0, 0))), dtype=np.float32)
    graph = np.asarray(sample.get("graph_features", np.zeros((0, 0))), dtype=np.float32)
    num_channels = int(np.asarray(sample["labels"]).shape[0])
    if spectral.ndim == 2 and graph.ndim == 2 and spectral.shape[0] == num_channels and graph.shape[0] == num_channels and (spectral.shape[1] + graph.shape[1]) > 0:
        features = np.concatenate([spectral, graph], axis=-1)[None, :, :].astype(np.float32, copy=False)
    else:
        features = np.zeros((1, num_channels, max(1, int(feature_dim_fallback))), dtype=np.float32)
    adjacency = np.zeros((features.shape[0], num_channels, num_channels), dtype=np.float32)
    centers = np.zeros((features.shape[0],), dtype=np.float32)
    return features, adjacency, centers


def _get_bool(args: Any | None, name: str, default: bool) -> bool:
    if args is None:
        return bool(default)
    value = getattr(args, name, default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _self_comparison_features(window_features: np.ndarray, window_centers: np.ndarray, args: Any | None) -> np.ndarray:
    features = np.asarray(window_features, dtype=np.float32)
    feature_view = str(getattr(args, "feature_view", "self_comparison") if args is not None else "self_comparison").lower()
    if feature_view in {"absolute", "raw", "none"}:
        return features.astype(np.float32, copy=False)
    if feature_view != "self_comparison":
        raise ValueError(f"Unsupported feature_view={feature_view!r}.")

    centers = np.asarray(window_centers, dtype=np.float32)
    pre_mask = centers < 0.0
    if centers.shape[0] != features.shape[0] or not np.any(pre_mask):
        pre_mask = np.ones((features.shape[0],), dtype=bool)

    eps = float(getattr(args, "self_compare_eps", 1e-5) if args is not None else 1e-5)
    baseline = features[pre_mask]
    pre_mean = baseline.mean(axis=0, keepdims=True)
    pre_std = np.clip(baseline.std(axis=0, keepdims=True), eps, None)
    pre_mean_t = np.broadcast_to(pre_mean, features.shape)
    delta = features - pre_mean_t
    zdelta = delta / pre_std
    ratio = np.log((np.abs(features) + eps) / (np.abs(pre_mean_t) + eps))

    parts: List[np.ndarray] = []
    if _get_bool(args, "self_compare_include_abs", True):
        parts.append(features)
    if _get_bool(args, "self_compare_include_pre_mean", False):
        parts.append(pre_mean_t.astype(np.float32, copy=False))
    if _get_bool(args, "self_compare_include_delta", True):
        parts.append(delta.astype(np.float32, copy=False))
    if _get_bool(args, "self_compare_include_zdelta", True):
        parts.append(zdelta.astype(np.float32, copy=False))
    if _get_bool(args, "self_compare_include_ratio", True):
        parts.append(ratio.astype(np.float32, copy=False))
    if _get_bool(args, "self_compare_include_channel_rank", False):
        rank = np.zeros_like(delta, dtype=np.float32)
        if delta.shape[1] > 1:
            order = np.argsort(delta, axis=1)
            rank_values = np.argsort(order, axis=1).astype(np.float32) / float(delta.shape[1] - 1)
            rank = rank_values.astype(np.float32, copy=False)
        parts.append(rank)
    if not parts:
        parts.append(features)
    return np.concatenate(parts, axis=-1).astype(np.float32, copy=False)


def _mixed_delta_adjacency(window_adjacency: np.ndarray, window_centers: np.ndarray, args: Any | None) -> np.ndarray:
    adjacency = np.asarray(window_adjacency, dtype=np.float32)
    adjacency_view = str(getattr(args, "adjacency_view", "mixed_abs_delta") if args is not None else "mixed_abs_delta").lower()
    if adjacency_view in {"absolute", "abs", "raw", "none"}:
        return adjacency.astype(np.float32, copy=False)
    if adjacency_view != "mixed_abs_delta":
        raise ValueError(f"Unsupported adjacency_view={adjacency_view!r}.")
    if adjacency.ndim != 3 or adjacency.shape[0] == 0:
        return adjacency.astype(np.float32, copy=False)

    centers = np.asarray(window_centers, dtype=np.float32)
    pre_mask = centers < 0.0
    if centers.shape[0] != adjacency.shape[0] or not np.any(pre_mask):
        pre_mask = np.ones((adjacency.shape[0],), dtype=bool)
    alpha = float(np.clip(float(getattr(args, "delta_adjacency_alpha", 0.5) if args is not None else 0.5), 0.0, 1.0))
    pre_adj = adjacency[pre_mask].mean(axis=0, keepdims=True)
    delta_abs = np.abs(adjacency - pre_adj).astype(np.float32, copy=False)
    mixed = alpha * adjacency + (1.0 - alpha) * delta_abs
    eye = np.eye(mixed.shape[-1], dtype=bool)[None, :, :]
    mixed = mixed.copy()
    mixed[eye.repeat(mixed.shape[0], axis=0)] = 0.0
    return mixed.astype(np.float32, copy=False)


def _prepare_window_tensors(
    sample: Dict[str, Any],
    *,
    feature_dim_fallback: int,
    args: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features, adjacency, centers = _as_window_tensors(sample, feature_dim_fallback=feature_dim_fallback)
    features = _self_comparison_features(features, centers, args)
    adjacency = _mixed_delta_adjacency(adjacency, centers, args)
    return features, adjacency, centers


def infer_window_feature_dim(window_samples: Iterable[Dict[str, Any]], args: Any | None = None) -> int:
    for sample in window_samples:
        features, _, _ = _prepare_window_tensors(sample, feature_dim_fallback=1, args=args)
        if features.ndim == 3 and features.shape[-1] > 0:
            return int(features.shape[-1])
    return 1


@dataclass
class WindowTensorNormalizer:
    mean: np.ndarray
    std: np.ndarray

    @property
    def feature_dim(self) -> int:
        return int(self.mean.shape[0])

    def transform(self, window_features: np.ndarray) -> np.ndarray:
        features = np.asarray(window_features, dtype=np.float32)
        if features.shape[-1] != self.feature_dim:
            raise ValueError(f"Expected feature_dim={self.feature_dim}, got {features.shape[-1]}.")
        return ((features - self.mean) / self.std).astype(np.float32, copy=False)


def fit_window_tensor_normalizer(window_samples: Iterable[Dict[str, Any]], args: Any | None = None) -> WindowTensorNormalizer:
    samples = list(window_samples)
    feature_dim = infer_window_feature_dim(samples, args=args)
    feature_sum = np.zeros((feature_dim,), dtype=np.float64)
    feature_sq_sum = np.zeros((feature_dim,), dtype=np.float64)
    count = 0
    for sample in samples:
        features, _, _ = _prepare_window_tensors(sample, feature_dim_fallback=feature_dim, args=args)
        if features.shape[-1] != feature_dim:
            continue
        flat = features.reshape(-1, feature_dim).astype(np.float64, copy=False)
        feature_sum += flat.sum(axis=0)
        feature_sq_sum += np.square(flat).sum(axis=0)
        count += int(flat.shape[0])

    if count <= 0:
        return WindowTensorNormalizer(mean=np.zeros((feature_dim,), dtype=np.float32), std=np.ones((feature_dim,), dtype=np.float32))
    mean = feature_sum / float(count)
    var = feature_sq_sum / float(count) - np.square(mean)
    std = np.sqrt(np.clip(var, 1e-8, None))
    return WindowTensorNormalizer(mean=mean.astype(np.float32), std=std.astype(np.float32))


def build_patient_examples(
    window_samples: Sequence[Dict[str, Any]],
    patient_index: Dict[str, Dict[str, Any]],
    *,
    normalizer: Optional[WindowTensorNormalizer] = None,
    subject_ids: Optional[Sequence[str]] = None,
    args: Any | None = None,
) -> List[Dict[str, Any]]:
    selected_subjects = set(subject_ids) if subject_ids is not None else None
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for sample in window_samples:
        subject_id = str(sample["subject_id"])
        if selected_subjects is not None and subject_id not in selected_subjects:
            continue
        grouped[subject_id].append(sample)

    feature_dim = int(normalizer.feature_dim) if normalizer is not None else infer_window_feature_dim(window_samples, args=args)
    positive_label = str(getattr(args, "positive_label", "nez") if args is not None else "nez").lower()
    if positive_label not in {"nez", "ez"}:
        raise ValueError(f"Unsupported positive_label={positive_label!r}; expected 'nez' or 'ez'.")
    examples: List[Dict[str, Any]] = []
    for subject_id in sorted(grouped.keys()):
        if subject_id not in patient_index:
            continue
        patient_meta = patient_index[subject_id]
        canonical_channels = list(patient_meta["canonical_channels"])
        channel_to_idx = {name: idx for idx, name in enumerate(canonical_channels)}
        labels_ez = np.asarray(patient_meta["labels"], dtype=np.float32)
        labels_nez = np.where(labels_ez >= 0.0, 1.0 - labels_ez, -1.0).astype(np.float32, copy=False)
        labels = labels_nez if positive_label == "nez" else labels_ez
        channel_mask = np.asarray(patient_meta.get("label_mask", np.ones(len(canonical_channels), dtype=bool)), dtype=bool)
        num_patient_channels = len(canonical_channels)

        seizure_features: List[np.ndarray] = []
        seizure_adjacency: List[np.ndarray] = []
        seizure_window_mask: List[np.ndarray] = []
        seizure_raw: List[np.ndarray] = []
        seizure_channel_mask: List[np.ndarray] = []
        run_ids: List[str] = []
        sample_ids: List[str] = []
        seizure_onsets: List[float] = []
        seizure_offsets: List[float] = []

        for sample in grouped[subject_id]:
            local_channels = list(sample["channel_names_norm"])
            local_to_patient = [channel_to_idx.get(channel_name) for channel_name in local_channels]
            features, adjacency, _ = _prepare_window_tensors(sample, feature_dim_fallback=feature_dim, args=args)
            if normalizer is not None:
                features = normalizer.transform(features)
            if features.shape[-1] != feature_dim:
                continue

            raw_x = _normalize_raw_waveform(
                sample["raw_waveform"],
                int(sample.get("raw_valid_samples", 1)),
                raw_valid_start_sample=int(sample.get("raw_valid_start_sample", 0)),
                args=args,
            )
            num_windows = int(features.shape[0])
            raw_samples = int(raw_x.shape[-1])
            aligned_features = np.zeros((num_windows, num_patient_channels, feature_dim), dtype=np.float32)
            aligned_adj = np.zeros((num_windows, num_patient_channels, num_patient_channels), dtype=np.float32)
            aligned_raw = np.zeros((num_patient_channels, raw_samples), dtype=np.float32)
            aligned_channel_mask = np.zeros((num_patient_channels,), dtype=bool)

            for local_idx, patient_idx in enumerate(local_to_patient):
                if patient_idx is None:
                    continue
                aligned_features[:, patient_idx, :] = features[:, local_idx, :]
                aligned_raw[patient_idx, :] = raw_x[local_idx, :]
                aligned_channel_mask[patient_idx] = True

            for src_local, src_patient in enumerate(local_to_patient):
                if src_patient is None:
                    continue
                for dst_local, dst_patient in enumerate(local_to_patient):
                    if dst_patient is None:
                        continue
                    aligned_adj[:, src_patient, dst_patient] = adjacency[:, src_local, dst_local]

            seizure_features.append(aligned_features)
            seizure_adjacency.append(aligned_adj)
            seizure_window_mask.append(np.ones((num_windows,), dtype=bool))
            seizure_raw.append(aligned_raw)
            seizure_channel_mask.append(aligned_channel_mask)
            run_ids.append(str(sample["run_id"]))
            sample_ids.append(str(sample["sample_id"]))
            seizure_onsets.append(float(sample["seizure_onset_sec"]))
            seizure_offsets.append(float(sample["seizure_offset_sec"]))

        if not seizure_features:
            continue
        examples.append(
            {
                "subject_id": subject_id,
                "canonical_channels": canonical_channels,
                "channel_meta": list(patient_meta.get("channel_meta", [])),
                "labels": labels,
                "labels_nez": labels_nez,
                "labels_ez": labels_ez,
                "label_semantics": "1=NEZ,0=EZ" if positive_label == "nez" else "1=EZ,0=NEZ",
                "channel_mask": channel_mask,
                "features": seizure_features,
                "adjacency": seizure_adjacency,
                "window_mask": seizure_window_mask,
                "raw_x": seizure_raw,
                "seizure_channel_mask": seizure_channel_mask,
                "run_ids": run_ids,
                "sample_ids": sample_ids,
                "seizure_onsets": seizure_onsets,
                "seizure_offsets": seizure_offsets,
            }
        )
    return examples


class PatientEZDataset(Dataset):
    """Patient-level EZ localization dataset.

    ``labels`` follow the configured positive class. V3-small defaults to
    NEZ-positive labels: label 1 means NEZ and label 0 means EZ. The original
    EZ-positive view is always retained as ``labels_ez``.
    """

    def __init__(self, patient_examples: Sequence[Dict[str, Any]]) -> None:
        self.patient_examples = list(patient_examples)

    def __len__(self) -> int:
        return len(self.patient_examples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.patient_examples[index]


def collate_patient_ez_batch(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not batch:
        raise ValueError("collate_patient_ez_batch received an empty batch.")

    batch_size = len(batch)
    max_seizures = max(len(item["features"]) for item in batch)
    max_windows = max(int(feature.shape[0]) for item in batch for feature in item["features"])
    max_channels = max(int(item["labels"].shape[0]) for item in batch)
    feature_dim = max(int(feature.shape[-1]) for item in batch for feature in item["features"])
    max_raw_samples = max(int(raw.shape[-1]) for item in batch for raw in item["raw_x"])

    features = torch.zeros((batch_size, max_seizures, max_windows, max_channels, feature_dim), dtype=torch.float32)
    adjacency = torch.zeros((batch_size, max_seizures, max_windows, max_channels, max_channels), dtype=torch.float32)
    raw_x = torch.zeros((batch_size, max_seizures, max_channels, max_raw_samples), dtype=torch.float32)
    labels = torch.zeros((batch_size, max_channels), dtype=torch.float32)
    labels_nez = torch.zeros((batch_size, max_channels), dtype=torch.float32)
    labels_ez = torch.zeros((batch_size, max_channels), dtype=torch.float32)
    channel_mask = torch.zeros((batch_size, max_channels), dtype=torch.bool)
    seizure_mask = torch.zeros((batch_size, max_seizures), dtype=torch.bool)
    seizure_channel_mask = torch.zeros((batch_size, max_seizures, max_channels), dtype=torch.bool)
    window_mask = torch.zeros((batch_size, max_seizures, max_windows), dtype=torch.bool)

    subject_ids: List[str] = []
    canonical_channels: List[List[str]] = []
    channel_meta: List[List[Dict[str, Any]]] = []
    run_ids: List[List[str]] = []
    sample_ids: List[List[str]] = []
    seizure_onsets: List[List[float]] = []
    seizure_offsets: List[List[float]] = []
    label_semantics: List[str] = []

    for batch_idx, item in enumerate(batch):
        num_channels = int(item["labels"].shape[0])
        labels[batch_idx, :num_channels] = torch.as_tensor(item["labels"], dtype=torch.float32)
        labels_nez[batch_idx, :num_channels] = torch.as_tensor(item["labels_nez"], dtype=torch.float32)
        labels_ez[batch_idx, :num_channels] = torch.as_tensor(item["labels_ez"], dtype=torch.float32)
        channel_mask[batch_idx, :num_channels] = torch.as_tensor(item["channel_mask"], dtype=torch.bool)
        subject_ids.append(str(item["subject_id"]))
        canonical_channels.append(list(item["canonical_channels"]))
        channel_meta.append(list(item.get("channel_meta", [])))
        run_ids.append(list(item.get("run_ids", [])))
        sample_ids.append(list(item.get("sample_ids", [])))
        seizure_onsets.append([float(value) for value in item.get("seizure_onsets", [])])
        seizure_offsets.append([float(value) for value in item.get("seizure_offsets", [])])
        label_semantics.append(str(item.get("label_semantics", "1=NEZ,0=EZ")))

        for seizure_idx, seizure_features in enumerate(item["features"]):
            seizure_features = np.asarray(seizure_features, dtype=np.float32)
            current_adj = np.asarray(item["adjacency"][seizure_idx], dtype=np.float32)
            current_raw = np.asarray(item["raw_x"][seizure_idx], dtype=np.float32)
            current_channel_mask = np.asarray(item["seizure_channel_mask"][seizure_idx], dtype=bool)
            current_window_mask = np.asarray(item["window_mask"][seizure_idx], dtype=bool)
            t = int(seizure_features.shape[0])
            c = int(seizure_features.shape[1])
            f = int(seizure_features.shape[2])
            l = int(current_raw.shape[-1])
            features[batch_idx, seizure_idx, :t, :c, :f] = torch.as_tensor(seizure_features, dtype=torch.float32)
            adjacency[batch_idx, seizure_idx, :t, :c, :c] = torch.as_tensor(current_adj, dtype=torch.float32)
            raw_x[batch_idx, seizure_idx, :c, :l] = torch.as_tensor(current_raw, dtype=torch.float32)
            seizure_channel_mask[batch_idx, seizure_idx, :c] = torch.as_tensor(current_channel_mask, dtype=torch.bool)
            window_mask[batch_idx, seizure_idx, :t] = torch.as_tensor(current_window_mask, dtype=torch.bool)
            seizure_mask[batch_idx, seizure_idx] = True

    return {
        "features": features,
        "adjacency": adjacency,
        "raw_x": raw_x,
        "labels": labels,
        "labels_nez": labels_nez,
        "labels_ez": labels_ez,
        "channel_mask": channel_mask,
        "seizure_mask": seizure_mask,
        "seizure_channel_mask": seizure_channel_mask,
        "window_mask": window_mask,
        "subject_id": subject_ids,
        "canonical_channels": canonical_channels,
        "channel_meta": channel_meta,
        "run_ids": run_ids,
        "sample_ids": sample_ids,
        "seizure_onsets": seizure_onsets,
        "seizure_offsets": seizure_offsets,
        "label_semantics": label_semantics,
    }


__all__ = [
    "PatientEZDataset",
    "WindowTensorNormalizer",
    "build_patient_examples",
    "collate_patient_ez_batch",
    "fit_window_tensor_normalizer",
    "infer_window_feature_dim",
]
