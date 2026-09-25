from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .schemas import LedgerSchemaError, build_canonical_ledger, normalize_channel_name
from .trajectory_store import PatientTrajectoryStore, RunTrajectory
from .cache_filtering import filter_cache_records
from .utils import sha256_file


class CacheContractError(ValueError):
    pass


def load_v3_ledger(path: str | Path, *, strict: bool = True) -> tuple[pd.DataFrame, dict[str, Any]]:
    source = Path(path)
    if source.is_dir():
        files = [source / f"test_channel_predictions_neuroez_v2_fold_{fold}.csv" for fold in range(1, 6)]
        missing = [str(item) for item in files if not item.exists()]
        if missing:
            raise FileNotFoundError(f"missing frozen V3 fold files: {missing}")
        raw = pd.concat([pd.read_csv(item) for item in files], ignore_index=True)
        inputs = {str(item.resolve()): {"size": item.stat().st_size, "sha256": sha256_file(item)} for item in files}
    elif source.exists():
        raw = pd.read_csv(source)
        inputs = {str(source.resolve()): {"size": source.stat().st_size, "sha256": sha256_file(source)}}
    else:
        raise FileNotFoundError(source)
    ledger, resolved = build_canonical_ledger(raw, strict=strict)
    resolved["inputs"] = inputs
    return ledger, resolved


def _channel_names(run_record: dict[str, Any], sample: dict[str, Any], patient_meta: dict[str, Any], width: int, *, strict: bool) -> list[str] | None:
    for value in (run_record.get("canonical_channels"), sample.get("canonical_channels"), run_record.get("channel_names"), sample.get("channel_names"), patient_meta.get("canonical_channels")):
        if value is not None and len(value) == width:
            return [str(item) for item in value]
    if strict:
        raise CacheContractError(f"run {run_record.get('run_id', run_record.get('record_id', '<unknown>'))} has no real channel names")
    return None


def load_window_feature_store(path: str | Path, *, strict: bool = True, invalid_record_policy: str | None = None) -> tuple[dict[str, dict[str, list[float]]], PatientTrajectoryStore, dict[str, Any]]:
    """Read existing cache features without reading raw waveform data or labels."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    with source.open("rb") as handle:
        cache = pickle.load(handle)
    if not isinstance(cache, dict) or "run_records" not in cache or "patient_index" not in cache:
        raise CacheContractError("window cache must contain run_records and patient_index")
    records, patient_index = cache["run_records"], cache["patient_index"]
    if not isinstance(records, list) or not isinstance(patient_index, dict):
        raise CacheContractError("run_records must be a list and patient_index must be a mapping")
    policy = invalid_record_policy or ("fail" if strict else "drop")
    records, filter_audit = filter_cache_records(records, patient_index, policy=policy)
    store = PatientTrajectoryStore(feature_names=cache.get("feature_names") or cache.get("window_feature_names") or ())
    dimensions: set[int] = set()
    missing = 0
    for record in records:
        if not isinstance(record, dict):
            missing += 1
            continue
        subject = str(record.get("subject_id", ""))
        sample = record.get("sample") or {}
        values = np.asarray(sample.get("window_features"), dtype=float)
        if not subject or values.ndim != 3 or values.shape[0] == 0 or values.shape[1] == 0 or values.shape[2] == 0:
            missing += 1
            continue
        dimensions.add(int(values.shape[2]))
        names = _channel_names(record, sample, patient_index.get(subject, {}) or {}, values.shape[1], strict=True)
        if names is None:
            missing += 1
            continue
        supplied_mask = sample.get("window_mask", record.get("window_mask"))
        store.add_run(subject, RunTrajectory(str(record.get("run_id", record.get("record_id", f"run-{len(store.runs(subject))}"))), tuple(names), values, None if supplied_mask is None else np.asarray(supplied_mask, dtype=bool), np.isfinite(values)))
    if not dimensions:
        raise CacheContractError("window cache has no usable [window, channel, feature] tensors")
    if len(dimensions) != 1:
        raise CacheContractError(f"inconsistent window feature dimensions: {sorted(dimensions)}")
    static = store.static_channel_features()
    names = cache.get("feature_names") or cache.get("window_feature_names") or []
    feature_names = cache.get("feature_names") or cache.get("window_feature_names") or []
    if feature_names and len(feature_names) != next(iter(dimensions)):
        raise CacheContractError("feature_names length does not match window feature dimension")
    return static, store, {"cache_path": str(source.resolve()), "cache_sha256": sha256_file(source), "n_run_records": len(records), "invalid_run_records": missing, "filter_audit": filter_audit, "feature_dim": next(iter(dimensions)), "feature_names": list(map(str, feature_names)), "n_subjects_with_features": len(static), "trajectory_contract": "subject_to_run_to_window_to_channel", "duplicate_run_ids": store.duplicate_run_ids}


def load_hnc_oof_ledger(path: str | Path, *, semantics: str | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = pd.read_csv(path)
    required = {"subject_id", "fold_idx"}
    if not required.issubset(frame.columns):
        raise LedgerSchemaError(f"HNC OOF ledger must contain {sorted(required)}")
    score = next((name for name in ("optional_hnc_score", "hnc_oof_score", "p_hard_negative", "score") if name in frame.columns), None)
    channel = next((name for name in ("channel_name", "channel_name_norm") if name in frame.columns), None)
    if score is None or channel is None:
        raise LedgerSchemaError("HNC OOF ledger requires a channel key and OOF score")
    if semantics is None and "hnc_score_semantics" in frame:
        declared = frame["hnc_score_semantics"].dropna().astype(str).str.strip().str.lower().unique().tolist()
        if len(declared) == 1:
            semantics = declared[0]
    semantics = str(semantics or "").strip().lower()
    allowed = {"p_nez", "p_ez", "residual_nez", "residual_ez"}
    if semantics not in allowed:
        raise LedgerSchemaError(f"unknown HNC score semantics: {semantics or '<missing>'}")
    out = frame[["subject_id", "fold_idx", channel, score]].copy().rename(columns={channel: "channel_name_original", score: "optional_hnc_score"})
    out["subject_id"] = out["subject_id"].astype(str).str.strip()
    out["fold_idx"] = pd.to_numeric(out["fold_idx"], errors="raise").astype(int)
    out["channel_name_original"] = out["channel_name_original"].astype(str)
    out["channel_name_norm"] = out["channel_name_original"].map(normalize_channel_name)
    out["optional_hnc_score"] = pd.to_numeric(out["optional_hnc_score"], errors="raise")
    out["hnc_score_semantics"] = semantics
    ez_direction = semantics in {"p_ez", "residual_ez"}
    out["hnc_eject_priority"] = (-1. if ez_direction else 1.) * out["optional_hnc_score"]
    out["hnc_add_priority"] = -out["hnc_eject_priority"]
    if out.duplicated(["subject_id", "fold_idx", "channel_name_norm"], keep=False).any():
        raise LedgerSchemaError("duplicate HNC OOF subject-channel keys")
    return out, {"path": str(Path(path).resolve()), "sha256": sha256_file(path), "score_column": score, "channel_column": channel, "hnc_score_semantics": semantics, "n_rows": len(out)}
