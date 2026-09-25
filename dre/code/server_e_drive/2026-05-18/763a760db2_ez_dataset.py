from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import pickle
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from ez_features import extract_run_feature_record
from module1_file_discovery import discover_all_bids_files
from module4_time_windows import has_valid_ictal_bounds


CACHE_VERSION = 5


def _log(message: str) -> None:
    print(f"[TeChEZ-PrePostDynamic][Data] {message}", flush=True)


@dataclass
class FeatureNormalizer:
    spectral_mean: np.ndarray
    spectral_std: np.ndarray
    graph_mean: np.ndarray
    graph_std: np.ndarray

    @property
    def spectral_dim(self) -> int:
        return int(self.spectral_mean.shape[0])

    @property
    def graph_dim(self) -> int:
        return int(self.graph_mean.shape[0])

    @property
    def feature_dim(self) -> int:
        return self.spectral_dim + self.graph_dim

    def transform(self, spectral_features: np.ndarray, graph_features: np.ndarray) -> np.ndarray:
        spectral = np.asarray(spectral_features, dtype=np.float32)
        graph = np.asarray(graph_features, dtype=np.float32)
        spectral_norm = (spectral - self.spectral_mean) / self.spectral_std
        graph_norm = (graph - self.graph_mean) / self.graph_std
        combined = np.concatenate([spectral_norm, graph_norm], axis=-1)
        return combined.T.astype(np.float32, copy=False)


def _normalize_cache_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, Path)):
        path = Path(value).expanduser()
        try:
            return str(path.resolve())
        except Exception:
            return str(path)
    if isinstance(value, (list, tuple)):
        return list(value)
    return value


def _cache_signature(args: Any) -> Dict[str, Any]:
    return {
        "cache_version": CACHE_VERSION,
        "dataset_dir": _normalize_cache_value(getattr(args, "dataset_dir", ".")),
        "participants_path": _normalize_cache_value(getattr(args, "participants_path", None)),
        "subject_filter": _normalize_cache_value(getattr(args, "subject_filter", None)),
        "success_only": bool(getattr(args, "success_only", False)),
        "target_sfreq": float(getattr(args, "target_sfreq", 1000.0)),
        "prepost_context_sec": float(getattr(args, "prepost_context_sec", 30.0)),
        "raw_temporal_sfreq": float(getattr(args, "raw_temporal_sfreq", 1000.0)),
        "ez_definition": str(getattr(args, "ez_definition", "soz_or_resected")),
        "spectral_min_freq": float(getattr(args, "spectral_min_freq", 10.0)),
        "spectral_max_freq": float(getattr(args, "spectral_max_freq", 300.0)),
        "graph_edge_quantile": float(getattr(args, "graph_edge_quantile", 0.70)),
        "graph_min_edge_weight": float(getattr(args, "graph_min_edge_weight", 0.10)),
        "sliding_window_sec": float(getattr(args, "sliding_window_sec", 2.0)),
        "sliding_step_sec": float(getattr(args, "sliding_step_sec", 1.0)),
        "extract_static_features": bool(getattr(args, "extract_static_features", False)),
        "extract_window_tensors": bool(getattr(args, "extract_window_tensors", True)),
    }


def _cache_path(args: Any) -> Path:
    output_dir = Path(getattr(args, "output_dir", "outputs"))
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    signature_blob = json.dumps(_cache_signature(args), sort_keys=True, ensure_ascii=True).encode("utf-8")
    cache_key = hashlib.sha1(signature_blob).hexdigest()[:12]
    return cache_dir / f"techez_prepost_dynamic_cache_v{CACHE_VERSION}_{cache_key}.pkl"


def _feature_worker_initializer() -> None:
    torch.set_num_threads(1)
    if hasattr(torch, "set_num_interop_threads"):
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass


def _extract_run_feature_record_from_payload(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    record = extract_run_feature_record(
        payload["run_info"],
        target_sfreq=payload["target_sfreq"],
        prepost_context_sec=payload["prepost_context_sec"],
        raw_temporal_sfreq=payload["raw_temporal_sfreq"],
        ez_definition=payload["ez_definition"],
        spectral_min_freq=payload["spectral_min_freq"],
        spectral_max_freq=payload["spectral_max_freq"],
        graph_edge_quantile=payload["graph_edge_quantile"],
        graph_min_edge_weight=payload["graph_min_edge_weight"],
        sliding_window_sec=payload["sliding_window_sec"],
        sliding_step_sec=payload["sliding_step_sec"],
        extract_static_features=payload["extract_static_features"],
        extract_window_tensors=payload["extract_window_tensors"],
    )
    if record is None:
        return None
    return record.to_dict()


def _resolve_feature_num_workers(args: Any, total_runs: int) -> int:
    requested_workers = int(getattr(args, "feature_num_workers", 8))
    available_cores = os.cpu_count() or 1
    if requested_workers <= 0:
        return 1
    return max(1, min(requested_workers, available_cores, total_runs))


def _count_samples(run_records: Iterable[Dict[str, Any]]) -> int:
    return int(sum(1 for _ in run_records))


def _extract_run_records_from_cache_payload(
    cached_payload: Any,
) -> Optional[Tuple[List[Dict[str, Any]], Dict[str, Any]]]:
    if not isinstance(cached_payload, dict):
        return None
    run_records = cached_payload.get("run_records")
    patient_index = cached_payload.get("patient_index")
    if isinstance(run_records, list) and isinstance(patient_index, dict):
        return run_records, patient_index
    return None


def _load_external_cache(cache_path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    with open(cache_path, "rb") as fin:
        cached_payload = pickle.load(fin)
    result = _extract_run_records_from_cache_payload(cached_payload)
    if result is None:
        raise ValueError(f"Unsupported external cache format: {cache_path}")
    return result


def _load_cached_payload(cache_path: Path, args: Any) -> Optional[Tuple[List[Dict[str, Any]], Dict[str, Any]]]:
    with open(cache_path, "rb") as fin:
        cached_payload = pickle.load(fin)

    result = _extract_run_records_from_cache_payload(cached_payload)
    if result is None:
        return None

    if cached_payload.get("cache_version") != CACHE_VERSION:
        return None
    if cached_payload.get("signature") != _cache_signature(args):
        return None
    return result


def _channel_sort_key(channel_meta: Dict[str, Any]) -> tuple:
    number = channel_meta.get("contact_number")
    number_key = int(number) if number is not None else 10**9
    return (
        str(channel_meta.get("contact_group", "")),
        number_key,
        str(channel_meta.get("channel_name_norm", "")),
    )


def _filter_runs_to_ictal_with_bounds(runs_df):
    run_records = runs_df.to_dict(orient="records")
    keep_mask = [has_valid_ictal_bounds(record) for record in run_records]
    filtered_df = runs_df.loc[keep_mask].reset_index(drop=True)
    return filtered_df


def build_patient_index(run_records: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    grouped_runs: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for run_record in run_records:
        grouped_runs[str(run_record["subject_id"])].append(run_record)

    patient_index: Dict[str, Dict[str, Any]] = {}
    for subject_id, subject_runs in grouped_runs.items():
        channel_meta_map: Dict[str, Dict[str, Any]] = {}
        label_map: Dict[str, float] = {}

        for run in subject_runs:
            for idx, channel_name in enumerate(run["channel_names_norm"]):
                if channel_name not in channel_meta_map:
                    channel_meta_map[channel_name] = {
                        "channel_name_norm": channel_name,
                        "contact_group": run["contact_groups"][idx],
                        "contact_number": run["contact_numbers"][idx],
                    }
                label_map[channel_name] = max(float(label_map.get(channel_name, 0.0)), float(run["labels"][idx]))

        canonical_meta = sorted(channel_meta_map.values(), key=_channel_sort_key)
        canonical_channels = [item["channel_name_norm"] for item in canonical_meta]
        labels = np.asarray([label_map.get(channel_name, 0.0) for channel_name in canonical_channels], dtype=np.float32)

        patient_index[subject_id] = {
            "subject_id": subject_id,
            "canonical_channels": canonical_channels,
            "channel_meta": canonical_meta,
            "labels": labels,
            "label_mask": np.ones(len(canonical_channels), dtype=bool),
        }

    return patient_index


def build_or_load_run_records(args: Any) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    external_cache_path_raw = getattr(args, "sample_cache_path", None) or getattr(args, "window_cache_path", None)
    if external_cache_path_raw:
        external_cache_path = Path(str(external_cache_path_raw)).expanduser()
        if not external_cache_path.exists():
            raise FileNotFoundError(f"External sample cache does not exist: {external_cache_path}")
        run_records, patient_index = _load_external_cache(external_cache_path)
        _log(
            f"Loaded {len(run_records)} ictal run records and {_count_samples(run_records)} seizure samples "
            f"from external cache {external_cache_path}."
        )
        return run_records, patient_index

    cache_path = _cache_path(args)
    force_rebuild = bool(getattr(args, "force_rebuild_cache", False))
    _log(f"Preparing onset sliding-window seizure-sample cache at {cache_path}")

    if cache_path.exists() and not force_rebuild:
        cached_result = _load_cached_payload(cache_path, args)
        if cached_result is not None:
            run_records, patient_index = cached_result
            _log(
                f"Loaded {len(run_records)} ictal run records and {_count_samples(run_records)} seizure samples from cache."
            )
            return run_records, patient_index

    runs_df = discover_all_bids_files(
        getattr(args, "dataset_dir", "."),
        participants_path=getattr(args, "participants_path", None),
        subject_filter=getattr(args, "subject_filter", None),
        success_only=bool(getattr(args, "success_only", False)),
    )
    if runs_df.empty:
        raise ValueError("No BIDS runs were discovered for the pre/post onset dynamic TeChEZ pipeline.")

    total_runs_before_filter = len(runs_df)
    runs_df = _filter_runs_to_ictal_with_bounds(runs_df)
    if runs_df.empty:
        raise ValueError("No ictal runs with valid seizure onset/offset annotations were found.")

    total_runs = len(runs_df)
    feature_num_workers = _resolve_feature_num_workers(args, total_runs)
    progress_interval = max(1, total_runs // 10)
    payloads = [
        {
            "run_info": row,
            "target_sfreq": float(getattr(args, "target_sfreq", 1000.0)),
            "prepost_context_sec": float(getattr(args, "prepost_context_sec", 30.0)),
            "raw_temporal_sfreq": float(getattr(args, "raw_temporal_sfreq", 1000.0)),
            "ez_definition": str(getattr(args, "ez_definition", "soz_or_resected")),
            "spectral_min_freq": float(getattr(args, "spectral_min_freq", 10.0)),
            "spectral_max_freq": float(getattr(args, "spectral_max_freq", 300.0)),
            "graph_edge_quantile": float(getattr(args, "graph_edge_quantile", 0.70)),
            "graph_min_edge_weight": float(getattr(args, "graph_min_edge_weight", 0.10)),
            "sliding_window_sec": float(getattr(args, "sliding_window_sec", 2.0)),
            "sliding_step_sec": float(getattr(args, "sliding_step_sec", 1.0)),
            "extract_static_features": bool(getattr(args, "extract_static_features", False)),
            "extract_window_tensors": bool(getattr(args, "extract_window_tensors", True)),
        }
        for row in runs_df.to_dict(orient="records")
    ]

    ordered_run_records: List[Tuple[int, Dict[str, Any]]] = []
    _log(
        f"Discovered {total_runs_before_filter} total runs and kept {total_runs} annotated ictal runs "
        f"across {runs_df['subject_id'].nunique()} patients. Onset sliding-window feature extraction will use {feature_num_workers} worker(s)."
    )

    if feature_num_workers == 1:
        for run_idx, payload in enumerate(payloads, start=1):
            record_dict = _extract_run_feature_record_from_payload(payload)
            if record_dict is not None:
                ordered_run_records.append((run_idx, record_dict))

            if run_idx == 1 or run_idx == total_runs or run_idx % progress_interval == 0:
                _log(
                    f"Feature extraction progress: {run_idx}/{total_runs} ictal runs | "
                    f"valid seizure samples: {len(ordered_run_records)}"
                )
    else:
        with ProcessPoolExecutor(
            max_workers=feature_num_workers,
            initializer=_feature_worker_initializer,
        ) as executor:
            future_to_run_idx = {
                executor.submit(_extract_run_feature_record_from_payload, payload): run_idx
                for run_idx, payload in enumerate(payloads, start=1)
            }

            completed = 0
            for future in as_completed(future_to_run_idx):
                completed += 1
                run_idx = future_to_run_idx[future]
                record_dict = future.result()
                if record_dict is not None:
                    ordered_run_records.append((run_idx, record_dict))

                if completed == 1 or completed == total_runs or completed % progress_interval == 0:
                    _log(
                        f"Feature extraction progress: {completed}/{total_runs} ictal runs finished | "
                        f"valid seizure samples: {len(ordered_run_records)}"
                    )

    ordered_run_records.sort(key=lambda item: item[0])
    run_records = [item[1] for item in ordered_run_records]
    if not run_records:
        raise ValueError("Feature extraction finished, but no usable ictal seizure samples were produced.")

    patient_index = build_patient_index(run_records)
    cache_payload = {
        "cache_version": CACHE_VERSION,
        "signature": _cache_signature(args),
        "run_records": run_records,
        "patient_index": patient_index,
    }
    with open(cache_path, "wb") as fout:
        pickle.dump(cache_payload, fout)

    _log(
        f"Cached {len(run_records)} ictal run records, {_count_samples(run_records)} seizure samples, "
        f"and {len(patient_index)} patients to {cache_path}"
    )
    return run_records, patient_index


def flatten_window_samples(
    run_records: Iterable[Dict[str, Any]],
    *,
    subject_ids: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    selected_subjects = set(subject_ids) if subject_ids is not None else None
    samples: List[Dict[str, Any]] = []

    for run_record in run_records:
        if selected_subjects is not None and run_record["subject_id"] not in selected_subjects:
            continue

        labels = np.asarray(run_record["labels"], dtype=np.float32)
        channel_names = list(run_record["channel_names_norm"])
        sample = run_record["sample"]
        samples.append(
            {
                "subject_id": run_record["subject_id"],
                "run_id": run_record["run_id"],
                "task": run_record["task"],
                "phase_group": run_record["phase_group"],
                "sample_id": str(sample["sample_id"]),
                "analysis_phase": str(sample["analysis_phase"]),
                "start_sec": float(sample["start_sec"]),
                "end_sec": float(sample["end_sec"]),
                "seizure_onset_sec": float(sample["seizure_onset_sec"]),
                "seizure_offset_sec": float(sample["seizure_offset_sec"]),
                "ictal_duration_total_sec": float(sample["ictal_duration_total_sec"]),
                "ictal_duration_used_sec": float(sample["ictal_duration_used_sec"]),
                "feature_scales_sec": [float(value) for value in sample["feature_scales_sec"]],
                "feature_scale_used_secs": [float(value) for value in sample["feature_scale_used_secs"]],
                "raw_temporal_duration_sec": float(sample["raw_temporal_duration_sec"]),
                "raw_temporal_sfreq": float(sample["raw_temporal_sfreq"]),
                "raw_valid_samples": int(sample["raw_valid_samples"]),
                "raw_valid_start_sample": int(sample.get("raw_valid_start_sample", 0)),
                "channel_names_norm": channel_names,
                "labels": labels,
                "spectral_features": np.asarray(sample["spectral_features"], dtype=np.float32),
                "graph_features": np.asarray(sample["graph_features"], dtype=np.float32),
                "window_features": np.asarray(sample.get("window_features", np.zeros((0, 0, 0))), dtype=np.float32),
                "window_adjacency": np.asarray(sample.get("window_adjacency", np.zeros((0, 0, 0))), dtype=np.float32),
                "window_relative_centers_sec": np.asarray(
                    sample.get("window_relative_centers_sec", np.zeros((0,))),
                    dtype=np.float32,
                ),
                "raw_waveform": np.asarray(sample["raw_waveform"], dtype=np.float32),
            }
        )
    return samples


def fit_feature_normalizer(window_samples: Iterable[Dict[str, Any]]) -> FeatureNormalizer:
    spectral_sum = None
    spectral_sq_sum = None
    graph_sum = None
    graph_sq_sum = None
    spectral_count = 0
    graph_count = 0

    for sample in window_samples:
        spectral = np.asarray(sample["spectral_features"], dtype=np.float32)
        graph = np.asarray(sample["graph_features"], dtype=np.float32)

        current_spectral_sum = spectral.sum(axis=0)
        current_spectral_sq_sum = np.square(spectral).sum(axis=0)
        current_graph_sum = graph.sum(axis=0)
        current_graph_sq_sum = np.square(graph).sum(axis=0)

        spectral_sum = current_spectral_sum if spectral_sum is None else spectral_sum + current_spectral_sum
        spectral_sq_sum = current_spectral_sq_sum if spectral_sq_sum is None else spectral_sq_sum + current_spectral_sq_sum
        graph_sum = current_graph_sum if graph_sum is None else graph_sum + current_graph_sum
        graph_sq_sum = current_graph_sq_sum if graph_sq_sum is None else graph_sq_sum + current_graph_sq_sum
        spectral_count += int(spectral.shape[0])
        graph_count += int(graph.shape[0])

    if spectral_count == 0 or graph_count == 0:
        raise ValueError("Cannot fit feature normalizer because the training seizure samples are empty.")

    spectral_mean = spectral_sum / float(spectral_count)
    spectral_var = spectral_sq_sum / float(spectral_count) - np.square(spectral_mean)
    graph_mean = graph_sum / float(graph_count)
    graph_var = graph_sq_sum / float(graph_count) - np.square(graph_mean)

    return FeatureNormalizer(
        spectral_mean=np.asarray(spectral_mean, dtype=np.float32),
        spectral_std=np.sqrt(np.clip(spectral_var, 1e-8, None)).astype(np.float32),
        graph_mean=np.asarray(graph_mean, dtype=np.float32),
        graph_std=np.sqrt(np.clip(graph_var, 1e-8, None)).astype(np.float32),
    )


def estimate_positive_class_weight(window_samples: Iterable[Dict[str, Any]], cap: float = 20.0) -> float:
    pos_count = 0.0
    neg_count = 0.0
    for sample in window_samples:
        labels = np.asarray(sample["labels"], dtype=np.float32)
        pos_count += float((labels == 1.0).sum())
        neg_count += float((labels == 0.0).sum())

    if pos_count <= 0.0:
        return 1.0
    weight = neg_count / max(pos_count, 1.0)
    return float(np.clip(weight, 1.0, float(cap)))


def _normalize_raw_waveform(raw_waveform: np.ndarray, raw_valid_samples: int) -> np.ndarray:
    raw = np.asarray(raw_waveform, dtype=np.float32).copy()
    if raw.ndim != 2:
        raise ValueError("raw_waveform must have shape [channels, time].")

    valid_samples = max(1, min(int(raw_valid_samples), int(raw.shape[-1])))
    valid_segment = raw[:, :valid_samples]
    channel_mean = valid_segment.mean(axis=-1, keepdims=True)
    channel_std = valid_segment.std(axis=-1, keepdims=True)
    channel_std = np.clip(channel_std, 1e-5, None)

    raw[:, :valid_samples] = (valid_segment - channel_mean) / channel_std
    if valid_samples < raw.shape[-1]:
        raw[:, valid_samples:] = 0.0
    return raw.astype(np.float32, copy=False)


class WindowFeatureDataset(Dataset):
    def __init__(self, window_samples: Sequence[Dict[str, Any]], normalizer: FeatureNormalizer) -> None:
        self.window_samples = list(window_samples)
        self.normalizer = normalizer

    def __len__(self) -> int:
        return len(self.window_samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.window_samples[index]
        labels = np.asarray(sample["labels"], dtype=np.float32)
        feature_matrix = self.normalizer.transform(sample["spectral_features"], sample["graph_features"])
        raw_valid_samples = int(sample["raw_valid_samples"])
        raw_x = _normalize_raw_waveform(sample["raw_waveform"], raw_valid_samples)
        return {
            "x": feature_matrix,
            "raw_x": raw_x,
            "raw_valid_samples": raw_valid_samples,
            "labels": labels,
            "channel_mask": np.ones(labels.shape[0], dtype=bool),
            "subject_id": sample["subject_id"],
            "run_id": sample["run_id"],
            "sample_id": sample["sample_id"],
            "analysis_phase": sample["analysis_phase"],
            "start_sec": float(sample["start_sec"]),
            "end_sec": float(sample["end_sec"]),
            "seizure_onset_sec": float(sample["seizure_onset_sec"]),
            "seizure_offset_sec": float(sample["seizure_offset_sec"]),
            "ictal_duration_total_sec": float(sample["ictal_duration_total_sec"]),
            "ictal_duration_used_sec": float(sample["ictal_duration_used_sec"]),
            "channel_names_norm": list(sample["channel_names_norm"]),
        }


def collate_window_batch(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not batch:
        raise ValueError("collate_window_batch received an empty batch.")

    batch_size = len(batch)
    feature_dim = int(batch[0]["x"].shape[0])
    max_channels = max(int(item["labels"].shape[0]) for item in batch)
    max_raw_samples = max(int(item["raw_x"].shape[-1]) for item in batch)

    x = torch.zeros((batch_size, feature_dim, max_channels), dtype=torch.float32)
    raw_x = torch.zeros((batch_size, max_channels, max_raw_samples), dtype=torch.float32)
    raw_valid_samples = torch.zeros((batch_size,), dtype=torch.long)
    labels = torch.zeros((batch_size, max_channels), dtype=torch.float32)
    channel_mask = torch.zeros((batch_size, max_channels), dtype=torch.bool)

    subject_ids: List[str] = []
    run_ids: List[str] = []
    sample_ids: List[str] = []
    analysis_phases: List[str] = []
    start_secs: List[float] = []
    end_secs: List[float] = []
    seizure_onsets: List[float] = []
    seizure_offsets: List[float] = []
    ictal_duration_total_secs: List[float] = []
    ictal_duration_used_secs: List[float] = []
    channel_names: List[List[str]] = []

    for batch_idx, item in enumerate(batch):
        num_channels = int(item["labels"].shape[0])
        current_raw_samples = int(item["raw_x"].shape[-1])
        x[batch_idx, :, :num_channels] = torch.as_tensor(item["x"], dtype=torch.float32)
        raw_x[batch_idx, :num_channels, :current_raw_samples] = torch.as_tensor(item["raw_x"], dtype=torch.float32)
        raw_valid_samples[batch_idx] = int(item["raw_valid_samples"])
        labels[batch_idx, :num_channels] = torch.as_tensor(item["labels"], dtype=torch.float32)
        channel_mask[batch_idx, :num_channels] = torch.as_tensor(item["channel_mask"], dtype=torch.bool)

        subject_ids.append(str(item["subject_id"]))
        run_ids.append(str(item["run_id"]))
        sample_ids.append(str(item["sample_id"]))
        analysis_phases.append(str(item["analysis_phase"]))
        start_secs.append(float(item["start_sec"]))
        end_secs.append(float(item["end_sec"]))
        seizure_onsets.append(float(item["seizure_onset_sec"]))
        seizure_offsets.append(float(item["seizure_offset_sec"]))
        ictal_duration_total_secs.append(float(item["ictal_duration_total_sec"]))
        ictal_duration_used_secs.append(float(item["ictal_duration_used_sec"]))
        channel_names.append(list(item["channel_names_norm"]))

    return {
        "x": x,
        "raw_x": raw_x,
        "raw_valid_samples": raw_valid_samples,
        "labels": labels,
        "channel_mask": channel_mask,
        "subject_id": subject_ids,
        "run_id": run_ids,
        "sample_id": sample_ids,
        "analysis_phase": analysis_phases,
        "start_sec": start_secs,
        "end_sec": end_secs,
        "seizure_onset_sec": seizure_onsets,
        "seizure_offset_sec": seizure_offsets,
        "ictal_duration_total_sec": ictal_duration_total_secs,
        "ictal_duration_used_sec": ictal_duration_used_secs,
        "channel_names_norm": channel_names,
    }


__all__ = [
    "FeatureNormalizer",
    "WindowFeatureDataset",
    "build_or_load_run_records",
    "build_patient_index",
    "collate_window_batch",
    "estimate_positive_class_weight",
    "fit_feature_normalizer",
    "flatten_window_samples",
]
