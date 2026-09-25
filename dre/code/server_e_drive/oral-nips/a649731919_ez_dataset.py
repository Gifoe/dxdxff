from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import pickle
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch

from ez_features import extract_run_feature_record
from module1_file_discovery import discover_all_bids_files


CACHE_VERSION = 3


def _log(message: str) -> None:
    print(f"[TeChEZ][Data] {message}", flush=True)


def _normalize_cache_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, Path)):
        path = Path(value).expanduser()
        try:
            return str(path.resolve())
        except Exception:
            return str(path)
    return value


def _cache_signature(args: Any) -> Dict[str, Any]:
    return {
        "cache_version": CACHE_VERSION,
        "dataset_dir": _normalize_cache_value(getattr(args, "dataset_dir", ".")),
        "participants_path": _normalize_cache_value(getattr(args, "participants_path", None)),
        "subject_filter": _normalize_cache_value(getattr(args, "subject_filter", None)),
        "success_only": bool(getattr(args, "success_only", False)),
        "feature_device": str(getattr(args, "feature_device", "cpu")),
        "target_sfreq": float(getattr(args, "target_sfreq", 512.0)),
        "win_len_sec": float(getattr(args, "win_len_sec", 15.0)),
        "step_sec": float(getattr(args, "step_sec", 5.0)),
        "ez_definition": str(getattr(args, "ez_definition", "soz_or_resected")),
    }


def _cache_path(args: Any) -> Path:
    output_dir = Path(getattr(args, "output_dir", "outputs"))
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    signature_blob = json.dumps(_cache_signature(args), sort_keys=True, ensure_ascii=True).encode("utf-8")
    cache_key = hashlib.sha1(signature_blob).hexdigest()[:12]
    return cache_dir / f"techez_cache_v{CACHE_VERSION}_{cache_key}.pkl"


def _load_cached_patient_bags(cache_path: Path, args: Any) -> Optional[List[Dict[str, Any]]]:
    with open(cache_path, "rb") as fin:
        cached_payload = pickle.load(fin)

    patient_bags = _extract_patient_bags_from_cache_payload(cached_payload)
    if patient_bags is None:
        print("Detected an unrecognized TeChEZ cache format. Rebuilding it for the current dataset.")
        return None

    if isinstance(cached_payload, list):
        print("Detected a legacy TeChEZ cache without dataset metadata. Rebuilding it for the current dataset.")
        return None

    cached_version = cached_payload.get("cache_version")
    cached_signature = cached_payload.get("signature")
    if cached_version != CACHE_VERSION or cached_signature != _cache_signature(args):
        print("Detected a stale TeChEZ cache for a different dataset/configuration. Rebuilding it.")
        return None

    return patient_bags


def _extract_patient_bags_from_cache_payload(cached_payload: Any) -> Optional[List[Dict[str, Any]]]:
    if isinstance(cached_payload, list):
        if not cached_payload:
            return []
        if isinstance(cached_payload[0], dict) and "runs" in cached_payload[0]:
            return cached_payload
        return None

    if not isinstance(cached_payload, dict):
        return None

    patient_bags = cached_payload.get("patient_bags")
    if isinstance(patient_bags, list):
        if not patient_bags:
            return []
        if isinstance(patient_bags[0], dict) and "runs" in patient_bags[0]:
            return patient_bags
    return None


def _load_external_patient_bag_cache(cache_path: Path) -> List[Dict[str, Any]]:
    with open(cache_path, "rb") as fin:
        cached_payload = pickle.load(fin)

    patient_bags = _extract_patient_bags_from_cache_payload(cached_payload)
    if patient_bags is None:
        raise ValueError(
            f"External patient-bag cache has an unsupported format: {cache_path}"
        )
    return patient_bags


def _count_runs_in_bags(patient_bags: Iterable[Dict[str, Any]]) -> int:
    return int(sum(len(bag.get("runs", [])) for bag in patient_bags))


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
        device=payload["feature_device"],
        target_sfreq=payload["target_sfreq"],
        win_len_sec=payload["win_len_sec"],
        step_sec=payload["step_sec"],
        ez_definition=payload["ez_definition"],
    )
    if record is None:
        return None
    return record.to_dict()


def _resolve_feature_num_workers(args: Any, total_runs: int) -> int:
    requested_workers = int(getattr(args, "feature_num_workers", 20))
    feature_device = str(getattr(args, "feature_device", "cpu")).strip().lower()
    available_cores = os.cpu_count() or 1
    if requested_workers <= 0:
        return 1
    if feature_device != "cpu":
        return 1
    return max(1, min(requested_workers, available_cores, total_runs))


def _channel_sort_key(channel_meta: Dict[str, Any]) -> tuple:
    number = channel_meta.get("contact_number")
    number_key = int(number) if number is not None else 10**9
    return (
        str(channel_meta.get("contact_group", "")),
        number_key,
        str(channel_meta.get("channel_name_norm", "")),
    )


def _build_patient_bag(subject_id: str, run_records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    run_records = list(run_records)
    channel_meta_map: Dict[str, Dict[str, Any]] = {}
    label_map: Dict[str, float] = {}

    for run in run_records:
        for idx, channel_name in enumerate(run["channel_names_norm"]):
            if channel_name not in channel_meta_map:
                channel_meta_map[channel_name] = {
                    "channel_name_norm": channel_name,
                    "contact_group": run["contact_groups"][idx],
                    "contact_number": run["contact_numbers"][idx],
                }
            label_map[channel_name] = max(
                float(label_map.get(channel_name, 0.0)),
                float(run["labels"][idx]),
            )

    canonical_meta = sorted(channel_meta_map.values(), key=_channel_sort_key)
    canonical_channels = [item["channel_name_norm"] for item in canonical_meta]
    canonical_index = {name: idx for idx, name in enumerate(canonical_channels)}
    patient_labels = np.asarray(
        [label_map.get(name, 0.0) for name in canonical_channels],
        dtype=np.float32,
    )

    aligned_runs: List[Dict[str, Any]] = []
    for run in run_records:
        num_windows = int(run["x_feat"].shape[0])
        patient_channels = len(canonical_channels)
        feat_dim = int(run["x_feat"].shape[-1])
        conn_dim = int(run["node_conn"].shape[-1])

        aligned_feat = np.zeros((num_windows, patient_channels, feat_dim), dtype=np.float32)
        aligned_conn = np.zeros((num_windows, patient_channels, conn_dim), dtype=np.float32)
        channel_mask = np.zeros(patient_channels, dtype=bool)
        local_to_canonical: Dict[int, int] = {}

        for local_idx, channel_name in enumerate(run["channel_names_norm"]):
            patient_idx = canonical_index[channel_name]
            local_to_canonical[local_idx] = patient_idx
            aligned_feat[:, patient_idx, :] = run["x_feat"][:, local_idx, :]
            aligned_conn[:, patient_idx, :] = run["node_conn"][:, local_idx, :]
            channel_mask[patient_idx] = True

        if run["edge_index"].size > 0:
            src = np.asarray([local_to_canonical[int(idx)] for idx in run["edge_index"][0]], dtype=np.int64)
            dst = np.asarray([local_to_canonical[int(idx)] for idx in run["edge_index"][1]], dtype=np.int64)
            edge_index = np.stack([src, dst], axis=0)
            edge_attr = np.asarray(run["edge_attr"], dtype=np.float32)
        else:
            edge_index = np.empty((2, 0), dtype=np.int64)
            edge_attr = np.zeros((num_windows, 0, 4), dtype=np.float32)

        aligned_runs.append(
            {
                "run_id": run["run_id"],
                "task": run["task"],
                "phase_group": run["phase_group"],
                "phase_ids": np.asarray(run["phase_ids"], dtype=np.int64),
                "quality_weight": float(run["quality_weight"]),
                "n_windows": int(run["n_windows"]),
                "x_feat": aligned_feat,
                "node_conn": aligned_conn,
                "edge_index": edge_index,
                "edge_attr": edge_attr,
                "channel_mask": channel_mask,
            }
        )

    return {
        "subject_id": subject_id,
        "canonical_channels": canonical_channels,
        "channel_meta": canonical_meta,
        "labels": patient_labels,
        "label_mask": np.ones(len(canonical_channels), dtype=bool),
        "runs": aligned_runs,
    }


def build_or_load_patient_bags(args: Any) -> List[Dict[str, Any]]:
    external_cache_path_raw = getattr(args, "patient_bag_cache_path", None)
    if external_cache_path_raw:
        external_cache_path = Path(str(external_cache_path_raw)).expanduser()
        _log(f"External patient-bag cache requested: {external_cache_path}")
        if not external_cache_path.exists():
            raise FileNotFoundError(
                f"External patient-bag cache does not exist: {external_cache_path}"
            )

        patient_bags = _load_external_patient_bag_cache(external_cache_path)
        _log(
            f"Loaded {len(patient_bags)} patient bags "
            f"({_count_runs_in_bags(patient_bags)} runs) from external cache."
        )
        _log("Skipping BIDS discovery and feature extraction because an external cache was provided.")
        return patient_bags

    cache_path = _cache_path(args)
    force_rebuild = bool(getattr(args, "force_rebuild_cache", False))
    _log(f"Preparing patient bags. Cache path: {cache_path}")

    if force_rebuild:
        _log("force_rebuild_cache=True, so cached patient bags will be ignored.")

    if cache_path.exists() and not force_rebuild:
        try:
            _log("Cache file found. Validating cache metadata.")
            cached_patient_bags = _load_cached_patient_bags(cache_path, args)
            if cached_patient_bags is not None:
                _log(
                    f"Loaded {len(cached_patient_bags)} patient bags "
                    f"({ _count_runs_in_bags(cached_patient_bags) } runs) from cache."
                )
                return cached_patient_bags
        except Exception as e:
            print(f"Failed to read TeChEZ cache ({e}). Rebuilding it from the dataset.")

    _log(f"Discovering BIDS runs under: {getattr(args, 'dataset_dir', '.')}")
    if getattr(args, "participants_path", None):
        _log(f"Using participants table: {getattr(args, 'participants_path', None)}")
    if getattr(args, "subject_filter", None):
        _log(f"Subject filter enabled: {getattr(args, 'subject_filter', None)}")
    _log(
        "Data options: "
        f"success_only={bool(getattr(args, 'success_only', True))} | "
        f"ez_definition={str(getattr(args, 'ez_definition', 'soz_or_resected'))} | "
        f"feature_device={getattr(args, 'feature_device', 'cpu')} | "
        f"feature_num_workers={int(getattr(args, 'feature_num_workers', 20))}"
    )
    runs_df = discover_all_bids_files(
        getattr(args, "dataset_dir", "."),
        participants_path=getattr(args, "participants_path", None),
        subject_filter=getattr(args, "subject_filter", None),
        success_only=bool(getattr(args, "success_only", True)),
    )
    if runs_df.empty:
        raise ValueError("No BIDS runs were discovered for TeChEZ.")
    _log(
        f"Discovered {len(runs_df)} runs across {runs_df['subject_id'].nunique()} subjects. "
        "Starting feature extraction."
    )

    ordered_run_records: List[tuple[int, Dict[str, Any]]] = []
    total_runs = len(runs_df)
    feature_device = str(getattr(args, "feature_device", "cpu"))
    feature_num_workers = _resolve_feature_num_workers(args, total_runs)
    progress_interval = max(1, total_runs // 10)
    payloads = [
        {
            "run_info": row,
            "feature_device": feature_device,
            "target_sfreq": float(getattr(args, "target_sfreq", 512.0)),
            "win_len_sec": float(getattr(args, "win_len_sec", 15.0)),
            "step_sec": float(getattr(args, "step_sec", 5.0)),
            "ez_definition": str(getattr(args, "ez_definition", "soz_or_resected")),
        }
        for row in runs_df.to_dict(orient="records")
    ]

    _log(
        f"Feature extraction will run on {feature_device} using "
        f"{feature_num_workers} worker(s)."
    )

    if feature_num_workers == 1:
        for run_idx, payload in enumerate(payloads, start=1):
            record_dict = _extract_run_feature_record_from_payload(payload)
            if record_dict is not None:
                ordered_run_records.append((run_idx, record_dict))

            if run_idx == 1 or run_idx == total_runs or run_idx % progress_interval == 0:
                _log(
                    f"Feature extraction progress: {run_idx}/{total_runs} runs processed | "
                    f"valid feature records: {len(ordered_run_records)}"
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

            completed_runs = 0
            for future in as_completed(future_to_run_idx):
                completed_runs += 1
                run_idx = future_to_run_idx[future]
                record_dict = future.result()
                if record_dict is not None:
                    ordered_run_records.append((run_idx, record_dict))

                if (
                    completed_runs == 1
                    or completed_runs == total_runs
                    or completed_runs % progress_interval == 0
                ):
                    _log(
                        f"Feature extraction progress: {completed_runs}/{total_runs} runs processed | "
                        f"valid feature records: {len(ordered_run_records)}"
                    )

    run_records = [record_dict for _, record_dict in sorted(ordered_run_records, key=lambda item: item[0])]
    if not run_records:
        raise ValueError("TeChEZ feature extraction produced zero valid runs.")
    _log(f"Feature extraction complete. Valid runs: {len(run_records)}.")

    grouped_runs: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for run in run_records:
        grouped_runs[run["subject_id"]].append(run)

    patient_bags = [
        _build_patient_bag(subject_id, grouped_runs[subject_id])
        for subject_id in sorted(grouped_runs.keys())
    ]
    _log(
        f"Built {len(patient_bags)} patient bags from {_count_runs_in_bags(patient_bags)} runs. "
        "Saving cache."
    )

    with open(cache_path, "wb") as fout:
        pickle.dump(
            {
                "cache_version": CACHE_VERSION,
                "signature": _cache_signature(args),
                "patient_bags": patient_bags,
            },
            fout,
        )
    _log(f"Saved patient bag cache to: {cache_path}")
    return patient_bags


__all__ = ["build_or_load_patient_bags"]
