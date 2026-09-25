from __future__ import annotations

import json
import pickle
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from outcome_hifos.cache_schema import CachePayload
from outcome_hifos.dataset import normalize_channel_name
from scripts.fm_baselines.preprocess import ensure_fixed_length, resample_1d_if_needed, robust_normalize_1d

from .base import FrozenFMAdapter


class FMEmbeddingError(ValueError):
    """Raised when raw windows cannot be traced and encoded safely."""


TRACE_KEY_COLUMNS = ["subject_id", "run_id", "sample_id", "window_idx", "channel_name"]
OFFICIAL_FM_REPOSITORIES = {
    "brainbert": "https://github.com/czlwang/BrainBERT",
    "cbramod": "https://github.com/wjq-learning/CBraMod",
}


def validate_official_fm_embedding_cache(path: str | Path, model_name: str) -> dict[str, Any]:
    """Fail closed unless an embedding cache records a traceable official FM load."""
    requested = Path(path)
    root = requested if requested.is_dir() else requested.parent
    audit_path = root / "audit.json"
    if not audit_path.exists():
        raise ValueError(f"Official FM embedding cache is missing {audit_path}.")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    model = str(model_name).strip().lower()
    if model not in OFFICIAL_FM_REPOSITORIES:
        raise ValueError(f"No official provenance policy is defined for FM {model!r}.")
    if str(audit.get("name", "")).lower() != model:
        raise ValueError(f"FM audit model mismatch: expected {model!r}, found {audit.get('name')!r}.")
    expected_repository = OFFICIAL_FM_REPOSITORIES[model]
    if str(audit.get("official_repository", "")).rstrip("/") != expected_repository:
        raise ValueError(f"FM official_repository must be {expected_repository!r}.")
    if model == "brainbert" and "rawbrainbert" in str(audit.get("version", "")).lower():
        raise ValueError("The internal RawBrainBERT checkpoint is not the official author BrainBERT model.")
    required = (
        "repository_commit_sha", "checkpoint_source", "checkpoint_path", "checkpoint_sha256", "checkpoint_size",
        "pretrained_modality", "expected_sampling_rate", "input_duration", "input_representation",
        "embedding_layer", "embedding_dimension", "license",
    )
    missing = [key for key in required if not str(audit.get(key, "")).strip()]
    if missing:
        raise ValueError(f"Official FM audit is missing required provenance fields: {missing}.")
    commit = str(audit["repository_commit_sha"])
    checksum = str(audit["checkpoint_sha256"])
    if len(commit) != 40 or any(char not in "0123456789abcdefABCDEF" for char in commit):
        raise ValueError("FM repository_commit_sha must be a full 40-character Git SHA.")
    if len(checksum) != 64 or any(char not in "0123456789abcdefABCDEF" for char in checksum):
        raise ValueError("FM checkpoint_sha256 must be a 64-character SHA256 digest.")
    if audit.get("missing_keys") not in ([], None) or audit.get("unexpected_keys") not in ([], None):
        raise ValueError("Official FM checkpoint load reported missing or unexpected keys.")
    if audit.get("frozen") is not True or audit.get("fine_tuned") is not False:
        raise ValueError("Official FM backbone must be frozen and not fine-tuned.")
    return audit


def _validate_chunked_trace(root: Path, manifest: pd.DataFrame, embedding_dim: int) -> dict[str, Any]:
    if manifest.duplicated(TRACE_KEY_COLUMNS).any():
        duplicate = manifest.loc[manifest.duplicated(TRACE_KEY_COLUMNS, keep=False), TRACE_KEY_COLUMNS].iloc[0].to_dict()
        raise FMEmbeddingError(f"FM manifest has duplicate trace key: {duplicate}")
    center_counts = manifest.groupby(["subject_id", "run_id", "sample_id", "window_idx"])["window_center_sec"].nunique()
    if (center_counts > 1).any():
        raise FMEmbeddingError("FM manifest has inconsistent window centers within a run/window.")
    referenced: set[tuple[str, int]] = set()
    chunk_rows_total = 0
    for chunk_name, group in manifest.groupby("chunk_file", sort=True):
        chunk_path = root / str(chunk_name)
        if not chunk_path.exists():
            raise FMEmbeddingError(f"FM chunk file does not exist: {chunk_path}")
        chunk = np.load(chunk_path, mmap_mode="r")
        if chunk.ndim != 2 or chunk.shape[1] != int(embedding_dim):
            raise FMEmbeddingError(f"FM chunk {chunk_name} has inconsistent embedding dimension {chunk.shape}.")
        chunk_rows_total += int(chunk.shape[0])
        for row_index in group["chunk_row"].astype(int):
            if row_index < 0 or row_index >= chunk.shape[0]:
                raise FMEmbeddingError(f"FM chunk row {row_index} is out of bounds for {chunk_name}.")
            key = (str(chunk_name), int(row_index))
            if key in referenced:
                raise FMEmbeddingError(f"FM chunk row is referenced more than once: {key}")
            referenced.add(key)
    if len(manifest) != chunk_rows_total or len(referenced) != len(manifest):
        raise FMEmbeddingError("FM manifest row count does not exactly match total chunk rows.")
    return {"verified": True, "manifest_row_count": len(manifest), "chunk_row_count": chunk_rows_total, "trace_key_columns": TRACE_KEY_COLUMNS}


def _preprocess(waveform: np.ndarray, sfreq: float, adapter: FrozenFMAdapter) -> np.ndarray:
    resampled, _ = resample_1d_if_needed(waveform, sfreq, adapter.spec.expected_sampling_rate)
    if adapter.spec.normalization == "robust_median_iqr":
        resampled = robust_normalize_1d(resampled)
    target_length = int(round(adapter.spec.expected_sampling_rate * adapter.spec.input_duration_sec))
    return ensure_fixed_length(resampled, target_length)


def _continuous_raw_windows(sample: dict[str, Any], adapter: FrozenFMAdapter, sfreq: float) -> np.ndarray:
    required = ("raw_waveform", "raw_valid_start_sample", "raw_valid_samples", "window_relative_centers_sec")
    missing = [key for key in required if key not in sample]
    if missing:
        raise FMEmbeddingError(
            f"Continuous raw extraction requires audited fields {required}; missing {missing}."
        )
    raw = np.asarray(sample["raw_waveform"], dtype=np.float32)
    centers = np.asarray(sample["window_relative_centers_sec"], dtype=np.float64).reshape(-1)
    if raw.ndim != 2 or centers.size == 0:
        raise FMEmbeddingError("Continuous raw waveform must be [channel,time] with non-empty window centers.")
    valid_start = int(sample["raw_valid_start_sample"])
    valid_samples = int(sample["raw_valid_samples"])
    valid_end = min(raw.shape[1], valid_start + valid_samples)
    if valid_start < 0 or valid_samples <= 0 or valid_start >= valid_end:
        raise FMEmbeddingError("Continuous raw valid sample interval is invalid.")
    source_length = max(1, int(round(float(sfreq) * adapter.spec.input_duration_sec)))
    onset_target_sample = raw.shape[1] / 2.0
    windows = np.zeros((centers.size, raw.shape[0], source_length), dtype=np.float32)
    for window_index, relative_center in enumerate(centers):
        center_sample = onset_target_sample + float(relative_center) * float(sfreq)
        requested_start = int(round(center_sample - source_length / 2.0))
        requested_end = requested_start + source_length
        copy_start = max(requested_start, valid_start, 0)
        copy_end = min(requested_end, valid_end, raw.shape[1])
        if copy_end <= copy_start:
            raise FMEmbeddingError(
                f"Window center {relative_center} sec does not overlap the audited raw valid interval."
            )
        destination_start = copy_start - requested_start
        destination_end = destination_start + (copy_end - copy_start)
        windows[window_index, :, destination_start:destination_end] = raw[:, copy_start:copy_end]
    return windows


def extract_embedding_cache(
    raw_cache: CachePayload,
    adapter: FrozenFMAdapter,
    output_dir: str | Path,
    *,
    raw_window_key: str = "raw_window_waveforms",
    batch_size: int = 64,
    raw_time_origin: str = "unknown",
) -> pd.DataFrame:
    if not adapter.loaded or adapter.model is None:
        raise RuntimeError("Frozen FM adapter must be loaded and frozen before extraction.")
    if adapter.model.training or any(parameter.requires_grad for parameter in adapter.model.parameters()):
        raise RuntimeError("Frozen FM adapter model is not in eval/frozen state.")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    chunks_dir = output / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    batch_limit = max(1, int(batch_size))
    waveforms: list[np.ndarray] = []
    pending_rows: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    chunk_index = 0
    embedding_index = 0
    embedding_dim: int | None = None
    max_buffered_waveforms = 0
    time_sources: set[str] = set()

    def flush() -> None:
        nonlocal chunk_index, embedding_index, embedding_dim, max_buffered_waveforms
        if not waveforms:
            return
        max_buffered_waveforms = max(max_buffered_waveforms, len(waveforms))
        matrix = np.stack(waveforms).astype(np.float32)
        embeddings = np.asarray(adapter.encode_batch(matrix), dtype=np.float32)
        if embeddings.ndim != 2 or embeddings.shape[0] != len(pending_rows):
            raise RuntimeError(f"Adapter returned embeddings with invalid shape {embeddings.shape} for {len(pending_rows)} rows.")
        if embedding_dim is None:
            embedding_dim = int(embeddings.shape[1])
        elif embeddings.shape[1] != embedding_dim:
            raise RuntimeError("Adapter embedding dimension changed between streaming batches.")
        chunk_name = f"chunk_{chunk_index:06d}.npy"
        np.save(chunks_dir / chunk_name, embeddings)
        for chunk_row, metadata in enumerate(pending_rows):
            rows.append(
                {
                    **metadata,
                    "embedding_index": int(embedding_index),
                    "chunk_file": f"chunks/{chunk_name}",
                    "chunk_row": int(chunk_row),
                }
            )
            embedding_index += 1
        chunk_index += 1
        waveforms.clear()
        pending_rows.clear()

    for record in raw_cache.run_records:
        sample = record.get("sample") if isinstance(record.get("sample"), dict) else {}
        sfreq = float(sample.get("raw_temporal_sfreq", sample.get("sfreq", record.get("sfreq", 0.0))))
        if sfreq <= 0:
            raise FMEmbeddingError(f"Record {record.get('run_id')} has invalid sampling frequency {sfreq}.")
        if raw_window_key in sample:
            windows = np.asarray(sample[raw_window_key], dtype=np.float32)
            raw_window_source = raw_window_key
            time_sources.add("explicit_window_tensor")
        elif "raw_waveform" in sample:
            if str(raw_time_origin) != "seizure_onset_at_raw_target_midpoint":
                raise FMEmbeddingError(
                    "Continuous raw time origin is not verified; expected 'seizure_onset_at_raw_target_midpoint' from cache-builder audit."
                )
            windows = _continuous_raw_windows(sample, adapter, sfreq)
            raw_window_source = "continuous_raw_seizure_onset_midpoint"
            time_sources.add("seizure_onset_at_raw_target_midpoint")
        else:
            raise FMEmbeddingError(
                f"Record {record.get('run_id')} is missing explicit {raw_window_key!r} and audited continuous raw fields."
            )
        if windows.ndim != 3:
            raise FMEmbeddingError(f"{raw_window_key} must be [window,channel,time], got {windows.shape}.")
        channels = record.get("channel_names_norm", sample.get("channel_names_norm", []))
        if len(channels) != windows.shape[1]:
            raise FMEmbeddingError(f"Record {record.get('run_id')} channel names do not match raw window channel axis.")
        centers = np.asarray(sample.get("window_relative_centers_sec", []), dtype=np.float32).reshape(-1)
        if centers.size != windows.shape[0]:
            raise FMEmbeddingError(f"Record {record.get('run_id')} window centers do not match raw windows.")
        for window_index in range(windows.shape[0]):
            for channel_index, channel_name in enumerate(channels):
                waveforms.append(_preprocess(windows[window_index, channel_index], sfreq, adapter))
                pending_rows.append(
                    {
                        "subject_id": str(record.get("subject_id", "")),
                        "run_id": str(record.get("run_id", "")),
                        "sample_id": str(sample.get("sample_id", record.get("run_id", ""))),
                        "window_idx": int(window_index),
                        "window_center_sec": float(centers[window_index]),
                        "channel_idx": int(channel_index),
                        "channel_name": normalize_channel_name(channel_name),
                        "raw_window_source": raw_window_source,
                    }
                )
                if len(waveforms) >= batch_limit:
                    flush()
        del windows
    flush()
    if not waveforms:
        if not rows:
            raise FMEmbeddingError("Raw cache contains no traceable channel-window waveforms.")
    assert embedding_dim is not None
    manifest = pd.DataFrame(rows)
    trace_audit = _validate_chunked_trace(output, manifest, embedding_dim)
    manifest.to_csv(output / "manifest.csv", index=False)
    with (output / "all_windows_embeddings.pkl").open("wb") as handle:
        pickle.dump(
            {
                "storage_format": "chunked_npy_v1",
                "model_name": adapter.spec.name,
                "embedding_dim": embedding_dim,
                "manifest": rows,
                "patient_index": dict(raw_cache.patient_index),
            },
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    audit = {
        **adapter.audit_info,
        "row_count": len(rows),
        "embedding_dim": embedding_dim,
        "raw_window_key": raw_window_key,
        "storage_format": "chunked_npy_v1",
        "chunk_count": chunk_index,
        "max_buffered_waveforms": max_buffered_waveforms,
        "trace_key_columns": ["subject_id", "run_id", "sample_id", "window_idx", "channel_name"],
        "pretrained_modality": adapter.audit_info.get("pretrained_modality", "EEG"),
        "input_duration": float(adapter.spec.input_duration_sec),
        "input_representation": adapter.audit_info.get("input_representation", "single-channel waveform"),
        "embedding_dimension": int(embedding_dim),
        "missing_keys": list(adapter.audit_info.get("missing_keys", [])),
        "unexpected_keys": list(adapter.audit_info.get("unexpected_keys", [])),
    }
    (output / "audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    (output / "fm_trace_integrity_audit.json").write_text(json.dumps(trace_audit, indent=2, sort_keys=True), encoding="utf-8")
    repository_root = Path(__file__).resolve().parents[2]
    builder_files = [
        repository_root / "run_baseline_three_centers" / "build_center_caches_from_patient_records.py",
        repository_root / "ez_features.py",
    ]
    builder_hash = hashlib.sha256(b"".join(path.read_bytes() for path in builder_files if path.exists())).hexdigest()
    time_origin_audit = {
        "time_origin_semantics": ",".join(sorted(time_sources)),
        "source_code_reference": "run_baseline_three_centers/build_center_caches_from_patient_records.py:_build_window_segments; ez_features.py:_build_centered_raw_waveform",
        "verified": bool(time_sources) and "unknown" not in time_sources,
        "extraction_formula": "raw_target_midpoint + window_relative_center_sec * sfreq" if "seizure_onset_at_raw_target_midpoint" in time_sources else "explicit pre-windowed tensor",
        "failure_reason": None,
        "builder_files": [str(path.relative_to(repository_root)) for path in builder_files],
        "builder_functions": ["_build_window_segments", "_build_centered_raw_waveform"],
        "source_code_hash": builder_hash,
    }
    (output / "raw_time_origin_audit.json").write_text(json.dumps(time_origin_audit, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def load_fm_embedding_as_feature_cache(path: str | Path) -> CachePayload:
    requested = Path(path)
    metadata_path = requested / "all_windows_embeddings.pkl" if requested.is_dir() else requested
    with metadata_path.open("rb") as handle:
        payload = pickle.load(handle)
    manifest = pd.DataFrame(payload.get("manifest") or [])
    patient_index = payload.get("patient_index")
    required = {"subject_id", "run_id", "sample_id", "window_idx", "window_center_sec", "channel_name", "embedding_index"}
    storage_format = str(payload.get("storage_format", "legacy_pickle"))
    embedding_dim = int(payload.get("embedding_dim", 0))
    if not required.issubset(manifest.columns) or not isinstance(patient_index, dict):
        raise FMEmbeddingError("FM embedding cache is missing embeddings, trace manifest, or patient_index.")
    if storage_format == "chunked_npy_v1":
        if embedding_dim <= 0 or not {"chunk_file", "chunk_row"}.issubset(manifest.columns):
            raise FMEmbeddingError("Chunked FM cache is missing chunk trace columns or embedding dimension.")
        _validate_chunked_trace(metadata_path.parent, manifest, embedding_dim)
        chunk_cache: dict[str, np.ndarray] = {}

        def embedding_for(row: Any) -> np.ndarray:
            name = str(row.chunk_file)
            if name not in chunk_cache:
                chunk_cache[name] = np.load(metadata_path.parent / Path(name), mmap_mode="r")
            return np.asarray(chunk_cache[name][int(row.chunk_row)], dtype=np.float32)
    else:
        embeddings = np.asarray(payload.get("embeddings"), dtype=np.float32)
        if embeddings.ndim != 2:
            raise FMEmbeddingError("Legacy FM embedding cache has no two-dimensional embeddings array.")
        embedding_dim = int(embeddings.shape[1])

        def embedding_for(row: Any) -> np.ndarray:
            return embeddings[int(row.embedding_index)]
    records = []
    for (subject_id, run_id, sample_id), group in manifest.groupby(["subject_id", "run_id", "sample_id"], sort=True):
        channels = sorted(group["channel_name"].astype(str).unique().tolist())
        windows = sorted(group["window_idx"].astype(int).unique().tolist())
        channel_lookup = {channel: index for index, channel in enumerate(channels)}
        window_lookup = {window: index for index, window in enumerate(windows)}
        tensor = np.zeros((len(windows), len(channels), embedding_dim), dtype=np.float32)
        centers = np.zeros((len(windows),), dtype=np.float32)
        for row in group.itertuples(index=False):
            tensor[window_lookup[int(row.window_idx)], channel_lookup[str(row.channel_name)]] = embedding_for(row)
            centers[window_lookup[int(row.window_idx)]] = float(row.window_center_sec)
        records.append(
            {
                "subject_id": str(subject_id),
                "run_id": str(run_id),
                "channel_names_norm": channels,
                "sample": {
                    "sample_id": str(sample_id),
                    "window_features": tensor,
                    "window_relative_centers_sec": centers,
                    "window_feature_names": [f"fm_{index}" for index in range(embedding_dim)],
                },
            }
        )
    return CachePayload(
        source_path=metadata_path.resolve(),
        payload={"run_records": records, "patient_index": patient_index, "model_name": payload.get("model_name")},
        run_records=tuple(records),
        patient_index={str(key): value for key, value in patient_index.items()},
    )


__all__ = [
    "FMEmbeddingError",
    "OFFICIAL_FM_REPOSITORIES",
    "extract_embedding_cache",
    "load_fm_embedding_as_feature_cache",
    "validate_official_fm_embedding_cache",
]
