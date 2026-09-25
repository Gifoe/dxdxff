from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import os
import random
from typing import Any, Sequence

import numpy as np

from .baseline_normalization import baseline_correct_adjacency, robust_baseline_zscore
from .config import BNPDGSConfig
from .data_interface import PatientRecord, SeizureRecord
from .graph_features import compute_dynamic_graph_features
from .logging_utils import log
from .peri_onset_windows import create_peri_onset_windows
from .spectral_features import (
    compute_lowfreq_features_aligned_to_short_grid,
    compute_short_window_channel_features,
)


@dataclass
class DynamicSeizureSample:
    subject_id: str
    seizure_id: str
    channel_names: list[str]
    labels: np.ndarray
    features: np.ndarray
    adjacency: np.ndarray | None
    time_sec: np.ndarray
    baseline_mask: np.ndarray
    channel_mask: np.ndarray


@dataclass
class DynamicPatientSample:
    subject_id: str
    seizure_samples: list[DynamicSeizureSample]
    canonical_channels: list[str]
    labels: np.ndarray
    channel_meta: list[dict]


def build_dynamic_seizure_sample(
    seizure: SeizureRecord,
    args: Any | None = None,
) -> DynamicSeizureSample | None:
    cfg = BNPDGSConfig.from_args(args)
    window_set = create_peri_onset_windows(
        seizure,
        pre_sec=cfg.peri_pre_sec,
        post_sec=cfg.peri_post_sec,
        window_sec=cfg.short_window_sec,
        stride_sec=cfg.short_stride_sec,
        baseline_start_sec=cfg.baseline_start_sec,
        baseline_end_sec=cfg.baseline_end_sec,
    )
    if window_set is None:
        return None

    graph_features, adjacency_seq, _ = compute_dynamic_graph_features(
        window_set,
        graph_method=cfg.graph_method,
        topk=cfg.graph_topk,
    )

    feature_parts = []
    if cfg.use_short_features:
        short_blocks = []
        for start, end in zip(window_set.window_starts, window_set.window_ends):
            window_data = window_set.signal_segment[:, int(start) : int(end)]
            features, _ = compute_short_window_channel_features(window_data, window_set.sfreq)
            short_blocks.append(features)
        feature_parts.append(np.stack(short_blocks, axis=0).astype(np.float32, copy=False))

    if cfg.use_lowfreq_features:
        lowfreq_features, _ = compute_lowfreq_features_aligned_to_short_grid(
            window_set.signal_segment,
            window_set.sfreq,
            window_set.window_centers_sec,
            lowfreq_window_sec=cfg.low_freq_window_sec,
        )
        feature_parts.append(lowfreq_features)

    if cfg.use_graph_features:
        feature_parts.append(graph_features)

    if not feature_parts:
        raise ValueError("At least one feature family must be enabled.")
    features = np.concatenate(feature_parts, axis=-1)
    if cfg.use_baseline_normalization:
        features = robust_baseline_zscore(features, window_set.baseline_mask)
        adjacency_seq = baseline_correct_adjacency(adjacency_seq, window_set.baseline_mask, mode="robust_z")
    else:
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

    time_features = _build_time_encoding(window_set.window_centers_sec)
    time_features = np.repeat(time_features[:, None, :], repeats=features.shape[1], axis=1)
    features = np.concatenate([features, time_features], axis=-1)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

    return DynamicSeizureSample(
        subject_id=window_set.subject_id,
        seizure_id=window_set.seizure_id,
        channel_names=list(window_set.channel_names),
        labels=np.asarray(window_set.labels, dtype=np.float32),
        features=features,
        adjacency=adjacency_seq.astype(np.float32, copy=False),
        time_sec=window_set.window_centers_sec.astype(np.float32, copy=False),
        baseline_mask=window_set.baseline_mask.astype(bool),
        channel_mask=np.ones(len(window_set.channel_names), dtype=bool),
    )


def build_dynamic_patient_sample(
    patient: PatientRecord,
    args: Any | None = None,
) -> DynamicPatientSample | None:
    raw_samples = [build_dynamic_seizure_sample(seizure, args) for seizure in patient.seizures]
    raw_samples = [sample for sample in raw_samples if sample is not None]
    if not raw_samples:
        return None

    aligned_samples = [
        align_seizure_sample_to_canonical(sample, patient.canonical_channels, patient.labels) for sample in raw_samples
    ]
    return DynamicPatientSample(
        subject_id=patient.subject_id,
        seizure_samples=aligned_samples,
        canonical_channels=list(patient.canonical_channels),
        labels=np.asarray(patient.labels, dtype=np.float32),
        channel_meta=list(patient.channel_meta),
    )


def build_dynamic_patient_samples(
    patient_records: Sequence[PatientRecord],
    args: Any | None = None,
) -> list[DynamicPatientSample]:
    cfg = BNPDGSConfig.from_args(args)
    workers = _resolve_feature_workers(cfg, len(patient_records))
    samples = []
    total = len(patient_records)
    log(f"Building dynamic patient samples. patients={total}, feature_num_workers={workers}")
    if workers <= 1:
        for idx, patient in enumerate(patient_records, start=1):
            sample = build_dynamic_patient_sample(patient, cfg)
            _log_dynamic_result(idx, total, patient.subject_id, sample)
            if sample is not None:
                samples.append(sample)
    else:
        log(
            "Parallel feature extraction is starting. "
            "GPU will stay idle until this CPU preprocessing stage finishes."
        )
        payloads = [(idx, patient, cfg) for idx, patient in enumerate(patient_records, start=1)]
        completed = 0
        with ProcessPoolExecutor(max_workers=workers, initializer=_feature_worker_initializer) as executor:
            future_to_idx = {executor.submit(_build_dynamic_patient_sample_worker, payload): payload[0] for payload in payloads}
            for future in as_completed(future_to_idx):
                completed += 1
                idx, subject_id, sample, error = future.result()
                if error:
                    log(f"Dynamic sample {completed}/{total}: subject={subject_id} failed: {error}")
                    continue
                _log_dynamic_result(completed, total, subject_id, sample)
                if sample is not None:
                    samples.append(sample)
    log(f"Finished dynamic samples. kept={len(samples)}/{total}")
    samples.sort(key=lambda item: item.subject_id)
    return samples


def _resolve_feature_workers(cfg: BNPDGSConfig, total: int) -> int:
    requested = int(getattr(cfg, "feature_num_workers", 20))
    if requested <= 0 or total <= 1:
        return 1
    return max(1, min(requested, total, os.cpu_count() or 1))


def _feature_worker_initializer() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    try:
        import torch

        torch.set_num_threads(1)
        if hasattr(torch, "set_num_interop_threads"):
            try:
                torch.set_num_interop_threads(1)
            except RuntimeError:
                pass
    except Exception:
        pass


def _build_dynamic_patient_sample_worker(
    payload: tuple[int, PatientRecord, BNPDGSConfig],
) -> tuple[int, str, DynamicPatientSample | None, str | None]:
    idx, patient, cfg = payload
    try:
        sample = build_dynamic_patient_sample(patient, cfg)
        return idx, patient.subject_id, sample, None
    except Exception as exc:
        return idx, patient.subject_id, None, repr(exc)


def _log_dynamic_result(idx: int, total: int, subject_id: str, sample: DynamicPatientSample | None) -> None:
    if sample is not None:
        log(
            f"Dynamic sample {idx}/{total}: subject={subject_id}, "
            f"seizures={len(sample.seizure_samples)}, channels={len(sample.canonical_channels)}"
        )
    else:
        log(f"Dynamic sample {idx}/{total}: subject={subject_id} dropped (insufficient peri-onset data).")


def align_seizure_sample_to_canonical(
    sample: DynamicSeizureSample,
    canonical_channels: Sequence[str],
    patient_labels: np.ndarray,
) -> DynamicSeizureSample:
    channel_to_idx = {channel: idx for idx, channel in enumerate(sample.channel_names)}
    target_c = len(canonical_channels)
    t, _, f = sample.features.shape
    features = np.zeros((t, target_c, f), dtype=np.float32)
    adjacency = None if sample.adjacency is None else np.zeros((t, target_c, target_c), dtype=np.float32)
    channel_mask = np.zeros(target_c, dtype=bool)

    for target_idx, channel_name in enumerate(canonical_channels):
        source_idx = channel_to_idx.get(channel_name)
        if source_idx is None:
            continue
        features[:, target_idx, :] = sample.features[:, source_idx, :]
        channel_mask[target_idx] = True

    if sample.adjacency is not None:
        present_pairs = [(target_idx, channel_to_idx[channel]) for target_idx, channel in enumerate(canonical_channels) if channel in channel_to_idx]
        for target_i, source_i in present_pairs:
            for target_j, source_j in present_pairs:
                adjacency[:, target_i, target_j] = sample.adjacency[:, source_i, source_j]

    return DynamicSeizureSample(
        subject_id=sample.subject_id,
        seizure_id=sample.seizure_id,
        channel_names=list(canonical_channels),
        labels=np.asarray(patient_labels, dtype=np.float32),
        features=features,
        adjacency=adjacency,
        time_sec=sample.time_sec,
        baseline_mask=sample.baseline_mask,
        channel_mask=channel_mask,
    )


def collate_dynamic_patient_batch(batch: Sequence[DynamicPatientSample]) -> dict[str, Any]:
    if not batch:
        raise ValueError("collate_dynamic_patient_batch received an empty batch.")
    try:
        import torch
    except ImportError as exc:
        raise ImportError("PyTorch is required for BN-PDGS batching.") from exc

    b = len(batch)
    s_max = max(len(item.seizure_samples) for item in batch)
    c_max = max(len(item.canonical_channels) for item in batch)
    t_max = max(sample.features.shape[0] for item in batch for sample in item.seizure_samples)
    f_dim = int(batch[0].seizure_samples[0].features.shape[-1])
    use_adjacency = any(sample.adjacency is not None for item in batch for sample in item.seizure_samples)

    features = torch.zeros((b, s_max, t_max, c_max, f_dim), dtype=torch.float32)
    adjacency = torch.zeros((b, s_max, t_max, c_max, c_max), dtype=torch.float32) if use_adjacency else None
    labels = torch.zeros((b, c_max), dtype=torch.float32)
    channel_mask = torch.zeros((b, c_max), dtype=torch.bool)
    seizure_mask = torch.zeros((b, s_max), dtype=torch.bool)
    seizure_channel_mask = torch.zeros((b, s_max, c_max), dtype=torch.bool)
    baseline_mask = torch.zeros((b, s_max, t_max), dtype=torch.bool)
    sample_time_sec = torch.zeros((b, s_max, t_max), dtype=torch.float32)

    subject_ids = []
    channel_names = []
    seizure_ids = []

    for batch_idx, patient in enumerate(batch):
        c = len(patient.canonical_channels)
        labels[batch_idx, :c] = torch.as_tensor(patient.labels, dtype=torch.float32)
        channel_mask[batch_idx, :c] = True
        subject_ids.append(patient.subject_id)
        channel_names.append(list(patient.canonical_channels))
        seizure_ids.append([sample.seizure_id for sample in patient.seizure_samples])

        for seizure_idx, sample in enumerate(patient.seizure_samples):
            t = int(sample.features.shape[0])
            features[batch_idx, seizure_idx, :t, :c, :] = torch.as_tensor(sample.features, dtype=torch.float32)
            if adjacency is not None and sample.adjacency is not None:
                adjacency[batch_idx, seizure_idx, :t, :c, :c] = torch.as_tensor(sample.adjacency, dtype=torch.float32)
            seizure_mask[batch_idx, seizure_idx] = True
            seizure_channel_mask[batch_idx, seizure_idx, :c] = torch.as_tensor(sample.channel_mask, dtype=torch.bool)
            baseline_mask[batch_idx, seizure_idx, :t] = torch.as_tensor(sample.baseline_mask, dtype=torch.bool)
            sample_time_sec[batch_idx, seizure_idx, :t] = torch.as_tensor(sample.time_sec, dtype=torch.float32)

    return {
        "features": features,
        "adjacency": adjacency,
        "labels": labels,
        "channel_mask": channel_mask,
        "seizure_mask": seizure_mask,
        "seizure_channel_mask": seizure_channel_mask,
        "baseline_mask": baseline_mask,
        "time_sec": sample_time_sec[0, 0],
        "sample_time_sec": sample_time_sec,
        "subject_ids": subject_ids,
        "channel_names": channel_names,
        "seizure_ids": seizure_ids,
    }


def build_patient_splits(
    patient_records: Sequence[PatientRecord],
    strategy: str = "5fold",
    n_splits: int = 5,
    seed: int = 42,
) -> list[dict[str, Any]]:
    split_map: dict[str, set[str]] = defaultdict(set)
    for patient in patient_records:
        for seizure in patient.seizures:
            if seizure.split:
                split_map[patient.subject_id].add(str(seizure.split).lower())

    official_subjects = {subject for subject, splits in split_map.items() if splits}
    if official_subjects:
        train = []
        val = []
        test = []
        for patient in patient_records:
            splits = split_map.get(patient.subject_id, set())
            if "test" in splits:
                test.append(patient.subject_id)
            elif "val" in splits or "valid" in splits or "validation" in splits:
                val.append(patient.subject_id)
            else:
                train.append(patient.subject_id)
        return [{"fold_idx": 0, "train_subjects": train, "val_subjects": val, "test_subjects": test}]

    subjects = [patient.subject_id for patient in patient_records]
    rng = random.Random(int(seed))
    rng.shuffle(subjects)
    n = len(subjects)
    if n == 0:
        return []
    if str(strategy).lower() in {"lopo", "leave-one-patient-out"}:
        n_splits = n
    n_splits = max(1, min(int(n_splits), n))

    folds = []
    chunks = np.array_split(np.asarray(subjects, dtype=object), n_splits)
    for fold_idx in range(n_splits):
        test_subjects = [str(item) for item in chunks[fold_idx].tolist()]
        val_subjects = [str(item) for item in chunks[(fold_idx + 1) % n_splits].tolist()] if n_splits > 2 else []
        blocked = set(test_subjects) | set(val_subjects)
        train_subjects = [subject for subject in subjects if subject not in blocked]
        if not train_subjects and val_subjects:
            train_subjects, val_subjects = val_subjects, []
        folds.append(
            {
                "fold_idx": fold_idx,
                "train_subjects": train_subjects,
                "val_subjects": val_subjects,
                "test_subjects": test_subjects,
            }
        )
    return folds


def estimate_positive_class_weight(
    patient_samples: Sequence[DynamicPatientSample],
    cap: float = 20.0,
) -> float:
    pos = 0.0
    neg = 0.0
    for sample in patient_samples:
        labels = np.asarray(sample.labels, dtype=np.float32)
        pos += float((labels == 1.0).sum())
        neg += float((labels == 0.0).sum())
    if pos <= 0.0:
        return 1.0
    return float(np.clip(neg / max(pos, 1.0), 1.0, float(cap)))


def _build_time_encoding(time_sec: np.ndarray) -> np.ndarray:
    t = np.asarray(time_sec, dtype=np.float32)
    scaled = t / 10.0
    return np.stack(
        [
            t,
            scaled,
            (t < 0.0).astype(np.float32),
            (t >= 0.0).astype(np.float32),
            np.sin(np.pi * scaled),
            np.cos(np.pi * scaled),
        ],
        axis=-1,
    ).astype(np.float32, copy=False)


__all__ = [
    "DynamicPatientSample",
    "DynamicSeizureSample",
    "align_seizure_sample_to_canonical",
    "build_dynamic_patient_sample",
    "build_dynamic_patient_samples",
    "build_dynamic_seizure_sample",
    "build_patient_splits",
    "collate_dynamic_patient_batch",
    "estimate_positive_class_weight",
]
