from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .config import BNPDGSConfig
from .dynamic_dataset import DynamicPatientSample, DynamicSeizureSample, build_dynamic_patient_sample
from .logging_utils import log


FEATURE_HASH_KEYS = [
    "peri_pre_sec",
    "peri_post_sec",
    "short_window_sec",
    "short_stride_sec",
    "low_freq_window_sec",
    "low_freq_stride_sec",
    "baseline_start_sec",
    "baseline_end_sec",
    "target_sfreq",
    "bandpass_low",
    "bandpass_high",
    "graph_method",
    "graph_use_adaptive_topk",
    "graph_edge_density",
    "graph_topk_min",
    "graph_topk_max",
    "use_short_features",
    "use_lowfreq_features",
    "use_graph_features",
    "use_baseline_normalization",
    "use_self_comparison_features",
]


def feature_config_hash(cfg: BNPDGSConfig) -> str:
    payload = {key: cfg.to_dict().get(key) for key in FEATURE_HASH_KEYS}
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def extract_features_for_center(
    center_id: str,
    records: Sequence[Any],
    config: BNPDGSConfig,
    output_cache_dir: str | Path,
    force_rebuild_cache: bool = False,
) -> dict[str, Any]:
    output_cache_dir = Path(output_cache_dir)
    output_cache_dir.mkdir(parents=True, exist_ok=True)
    config_hash = feature_config_hash(config)
    manifest_path = output_cache_dir / "cache_manifest.json"
    existing = _load_manifest(manifest_path)
    if (
        not force_rebuild_cache
        and existing
        and existing.get("config_hash") == config_hash
        and all(Path(item["cache_path"]).exists() for item in existing.get("patient_files", []))
    ):
        log(f"Feature cache hit for center={center_id}: {manifest_path}")
        return existing

    patient_files = []
    skipped = []
    total_seizures = 0
    for idx, record in enumerate(records, start=1):
        if getattr(record, "center_id", center_id) != center_id:
            continue
        if len(record.canonical_channels) < 8 or float(np.asarray(record.labels).sum()) <= 0.0 or not record.seizures:
            skipped.append({"global_patient_id": record.subject_id, "reason": "audit_filter"})
            continue
        sample = build_dynamic_patient_sample(record, config)
        if sample is None or not sample.seizure_samples:
            skipped.append({"global_patient_id": record.subject_id, "reason": "no_valid_peri_onset_seizures"})
            continue
        cache_path = output_cache_dir / f"{_safe_filename(sample.subject_id)}.npz"
        save_patient_cache(sample, cache_path, config_hash)
        total_seizures += len(sample.seizure_samples)
        patient_files.append(
            {
                "global_patient_id": sample.subject_id,
                "center_id": center_id,
                "patient_id": sample.patient_id,
                "cache_path": str(cache_path),
                "n_seizures": len(sample.seizure_samples),
                "n_valid_channels": int(np.asarray(sample.seizure_samples[0].channel_mask, dtype=bool).sum()),
                "n_ez_channels": int(np.asarray(sample.labels, dtype=np.float32).sum()),
                "n_bad_channels": max(0, len(sample.canonical_channels) - int(np.asarray(sample.seizure_samples[0].channel_mask, dtype=bool).sum())),
            }
        )
        if idx == 1 or idx % 10 == 0:
            log(f"Feature cache center={center_id}: processed={idx}/{len(records)}, cached={len(patient_files)}")

    manifest = {
        "center_id": center_id,
        "n_patients_raw": len(records),
        "n_patients_cached": len(patient_files),
        "n_patients_skipped": len(skipped),
        "n_total_seizures": total_seizures,
        "config_hash": config_hash,
        "created_time": datetime.now().isoformat(timespec="seconds"),
        "patient_files": patient_files,
        "skipped": skipped,
    }
    _write_json(manifest, manifest_path)
    return manifest


def save_patient_cache(sample: DynamicPatientSample, cache_path: str | Path, config_hash: str) -> None:
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    features = np.stack([s.features for s in sample.seizure_samples], axis=0).astype(np.float32)
    adjacency = np.stack(
        [
            s.adjacency if s.adjacency is not None else np.zeros((s.features.shape[0], len(sample.canonical_channels), len(sample.canonical_channels)), dtype=np.float32)
            for s in sample.seizure_samples
        ],
        axis=0,
    ).astype(np.float32)
    valid_mask = np.stack([s.channel_mask for s in sample.seizure_samples], axis=0).astype(bool)
    time_encoding = features[..., -12:].astype(np.float32, copy=False)
    graph_features = _graph_feature_block(features, sample.feature_names or [])
    metadata = {
        "config_hash": config_hash,
        "feature_names": sample.feature_names or [],
        "center_id": sample.center_id,
        "patient_id": sample.patient_id,
        "global_patient_id": sample.subject_id,
    }
    np.savez_compressed(
        cache_path,
        global_patient_id=np.asarray(sample.subject_id),
        center_id=np.asarray(sample.center_id or ""),
        patient_id=np.asarray(sample.patient_id or ""),
        seizure_ids=np.asarray([s.seizure_id for s in sample.seizure_samples], dtype=object),
        edf_paths=np.asarray([s.edf_path or "" for s in sample.seizure_samples], dtype=object),
        onset_times=np.asarray([np.nan if s.onset_time is None else s.onset_time for s in sample.seizure_samples], dtype=np.float32),
        offset_times=np.asarray([np.nan if s.offset_time is None else s.offset_time for s in sample.seizure_samples], dtype=np.float32),
        canonical_channels=np.asarray(sample.canonical_channels, dtype=object),
        valid_channel_mask=valid_mask,
        seizure_mask=np.ones(len(sample.seizure_samples), dtype=bool),
        channel_labels=np.asarray(sample.labels, dtype=np.float32),
        X_node=features,
        A_graph=adjacency,
        graph_features=graph_features,
        time_encoding=time_encoding,
        topology_features=np.asarray(_topology_from_meta(sample.channel_meta), dtype=np.float32),
        seizure_quality_features=np.asarray([s.seizure_quality for s in sample.seizure_samples], dtype=np.float32),
        time_sec=np.stack([s.time_sec for s in sample.seizure_samples], axis=0).astype(np.float32),
        baseline_mask=np.stack([s.baseline_mask for s in sample.seizure_samples], axis=0).astype(bool),
        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
    )


def load_cached_samples(cache_sources: Sequence[str | Path]) -> list[DynamicPatientSample]:
    samples = []
    for source in cache_sources:
        manifest = _load_manifest(Path(source) / "cache_manifest.json")
        if not manifest:
            continue
        for item in manifest.get("patient_files", []):
            samples.append(load_patient_cache(item["cache_path"]))
    samples.sort(key=lambda sample: sample.subject_id)
    return samples


def load_patient_cache(cache_path: str | Path) -> DynamicPatientSample:
    cache_path = Path(cache_path)
    with np.load(cache_path, allow_pickle=True) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        features = data["X_node"].astype(np.float32)
        adjacency = data["A_graph"].astype(np.float32)
        valid_mask = data["valid_channel_mask"].astype(bool)
        labels = data["channel_labels"].astype(np.float32)
        channels = [str(item) for item in data["canonical_channels"].tolist()]
        seizure_ids = [str(item) for item in data["seizure_ids"].tolist()]
        edf_paths = [str(item) for item in data["edf_paths"].tolist()]
        onset_times = data["onset_times"].astype(np.float32)
        offset_times = data["offset_times"].astype(np.float32)
        time_sec = data["time_sec"].astype(np.float32)
        baseline_mask = data["baseline_mask"].astype(bool)
        seizure_quality = data["seizure_quality_features"].astype(np.float32)
        seizure_samples = []
        for idx, seizure_id in enumerate(seizure_ids):
            seizure_samples.append(
                DynamicSeizureSample(
                    subject_id=str(metadata["global_patient_id"]),
                    seizure_id=seizure_id,
                    center_id=str(metadata.get("center_id") or ""),
                    patient_id=str(metadata.get("patient_id") or ""),
                    channel_names=channels,
                    labels=labels,
                    features=features[idx],
                    adjacency=adjacency[idx],
                    time_sec=time_sec[idx],
                    baseline_mask=baseline_mask[idx],
                    channel_mask=valid_mask[idx],
                    seizure_quality=float(seizure_quality[idx]) if seizure_quality.size else 1.0,
                    edf_path=edf_paths[idx],
                    onset_time=None if np.isnan(onset_times[idx]) else float(onset_times[idx]),
                    offset_time=None if np.isnan(offset_times[idx]) else float(offset_times[idx]),
                )
            )
        return DynamicPatientSample(
            subject_id=str(metadata["global_patient_id"]),
            center_id=str(metadata.get("center_id") or ""),
            patient_id=str(metadata.get("patient_id") or ""),
            seizure_samples=seizure_samples,
            canonical_channels=channels,
            labels=labels,
            channel_meta=[{} for _ in channels],
            feature_names=list(metadata.get("feature_names") or []),
        )


def write_all_centers_index(cache_root: str | Path, manifests: dict[str, dict[str, Any]]) -> dict[str, Any]:
    cache_root = Path(cache_root)
    out_dir = cache_root / "all_centers"
    out_dir.mkdir(parents=True, exist_ok=True)
    index = {
        "center_manifests": {
            center_id: {
                "cache_manifest_path": str(cache_root / center_id / "cache_manifest.json"),
                "n_patients_cached": manifest.get("n_patients_cached", 0),
                "config_hash": manifest.get("config_hash"),
            }
            for center_id, manifest in manifests.items()
        },
        "patient_files": [
            item
            for manifest in manifests.values()
            for item in manifest.get("patient_files", [])
        ],
    }
    _write_json(index, out_dir / "index.json")
    _write_json(
        {
            "center_id": "all_centers",
            "config_hash": next(iter(manifests.values())).get("config_hash") if manifests else None,
            "n_patients_cached": len(index["patient_files"]),
            "index_path": str(out_dir / "index.json"),
        },
        out_dir / "cache_manifest.json",
    )
    return index


def _graph_feature_block(features: np.ndarray, feature_names: list[str]) -> np.ndarray:
    indices = [idx for idx, name in enumerate(feature_names) if any(token in name for token in ("degree_norm", "strength_norm", "clustering", "centrality", "pagerank", "kcore", "local_efficiency"))]
    if not indices:
        return np.zeros((*features.shape[:3], 0), dtype=np.float32)
    return features[..., indices].astype(np.float32, copy=False)


def _topology_from_meta(channel_meta: Sequence[dict[str, Any]]) -> np.ndarray:
    result = np.zeros((len(channel_meta), 4), dtype=np.float32)
    for idx, meta in enumerate(channel_meta):
        number = meta.get("contact_number")
        result[idx, 1] = float(number) if number is not None else float(idx + 1)
    if result.shape[0]:
        result[:, 1] /= max(float(result[:, 1].max()), 1.0)
    return result


def _load_manifest(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as fin:
        return json.load(fin)


def _write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fout:
        json.dump(payload, fout, ensure_ascii=False, indent=2)


def _safe_filename(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in name)


__all__ = [
    "extract_features_for_center",
    "feature_config_hash",
    "load_cached_samples",
    "load_patient_cache",
    "save_patient_cache",
    "write_all_centers_index",
]
