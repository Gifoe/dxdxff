from __future__ import annotations

import pickle
from typing import Any, Sequence

import numpy as np

from extraction import InterictalBank
from ez_features import WINDOW_NODE_FEATURE_NAMES


def _align_bank_to_channels(bank: InterictalBank, channel_names: Sequence[str]) -> tuple[np.ndarray, np.ndarray] | None:
    index = {name: idx for idx, name in enumerate(bank.channel_names_norm)}
    indices = [index.get(name) for name in channel_names]
    if any(idx is None for idx in indices):
        return None
    idx_array = np.asarray(indices, dtype=int)
    return bank.features[:, idx_array, :], bank.adjacency[:, idx_array][:, :, idx_array]


def _append_physics_node_features(features: np.ndarray) -> np.ndarray:
    names = list(WINDOW_NODE_FEATURE_NAMES)
    values = np.asarray(features, dtype=np.float32)
    zeros = np.zeros(values.shape[:2], dtype=np.float32)

    def pick(name: str) -> np.ndarray:
        if name not in names:
            return zeros
        idx = names.index(name)
        return values[..., idx] if idx < values.shape[-1] else zeros

    high_gamma = pick("log_bp_high_gamma")
    total_power = pick("log_total_power")
    high_gamma_ratio = high_gamma - total_power
    phys = np.stack(
        [
            np.zeros_like(high_gamma),  # slope_abs unavailable from aggregate report; keep missing indicator.
            np.ones_like(high_gamma),   # slope_missing_indicator.
            high_gamma,
            high_gamma_ratio,
            pick("line_length_per_sec"),
            pick("rms"),
            pick("spectral_entropy"),
            pick("degree_norm"),
            pick("strength_norm"),
            pick("clustering_coeff"),
            pick("local_efficiency"),
        ],
        axis=-1,
    )
    return np.concatenate([values, phys.astype(np.float32)], axis=-1)


def _bank_stats(features_by_run: Sequence[np.ndarray], adjacency_by_run: Sequence[np.ndarray]) -> dict[str, np.ndarray]:
    run_feature_medians = np.stack([np.median(arr, axis=0) for arr in features_by_run], axis=0)
    run_feature_mads = np.stack([np.median(np.abs(arr - np.median(arr, axis=0, keepdims=True)), axis=0) for arr in features_by_run], axis=0)
    run_adj_medians = np.stack([np.median(arr, axis=0) for arr in adjacency_by_run], axis=0)
    run_adj_mads = np.stack([np.median(np.abs(arr - np.median(arr, axis=0, keepdims=True)), axis=0) for arr in adjacency_by_run], axis=0)
    stacked_features = np.concatenate(list(features_by_run), axis=0)
    return {
        "median_inter": np.median(run_feature_medians, axis=0).astype(np.float32),
        "MAD_inter": np.median(run_feature_mads, axis=0).astype(np.float32),
        "q05_inter": np.quantile(stacked_features, 0.05, axis=0).astype(np.float32),
        "q25_inter": np.quantile(stacked_features, 0.25, axis=0).astype(np.float32),
        "q75_inter": np.quantile(stacked_features, 0.75, axis=0).astype(np.float32),
        "q95_inter": np.quantile(stacked_features, 0.95, axis=0).astype(np.float32),
        "A_inter_median": np.median(run_adj_medians, axis=0).astype(np.float32),
        "A_inter_MAD": np.median(run_adj_mads, axis=0).astype(np.float32),
    }


def attach_interictal_and_make_variant(
    strict_records: Sequence[dict[str, Any]],
    banks_by_subject: dict[str, list[InterictalBank]],
    *,
    variant: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    output: list[dict[str, Any]] = []
    stats = {"records_in": len(strict_records), "records_with_interictal": 0, "records_dropped_no_interictal": 0}
    for source_record in strict_records:
        record = pickle.loads(pickle.dumps(source_record, protocol=pickle.HIGHEST_PROTOCOL))
        sample = record["sample"]
        channel_names = list(record["channel_names_norm"])
        aligned_features_by_run: list[np.ndarray] = []
        aligned_adj_by_run: list[np.ndarray] = []
        source_types: list[str] = []
        for bank in banks_by_subject.get(str(record["subject_id"]), []):
            aligned = _align_bank_to_channels(bank, channel_names)
            if aligned is None:
                continue
            features, adjacency = aligned
            if features.shape[1:] != np.asarray(sample["window_features"]).shape[1:]:
                continue
            aligned_features_by_run.append(features)
            aligned_adj_by_run.append(adjacency)
            source_types.append(bank.source_type)
        if not aligned_features_by_run:
            stats["records_dropped_no_interictal"] += 1
            continue
        stats["records_with_interictal"] += 1
        if variant in {"M2_HUP_InterPhysNode", "M3_HUP_InterPhysGraph"}:
            sample["window_features"] = _append_physics_node_features(np.asarray(sample["window_features"], dtype=np.float32))
            aligned_features_by_run = [_append_physics_node_features(arr) for arr in aligned_features_by_run]
        stats_dict = _bank_stats(aligned_features_by_run, aligned_adj_by_run)
        sample["interictal_window_features"] = np.concatenate(aligned_features_by_run, axis=0).astype(np.float32)
        sample["interictal_adjacency"] = np.concatenate(aligned_adj_by_run, axis=0).astype(np.float32)
        sample["interictal_baseline_stats"] = stats_dict
        sample["interictal_source_type"] = "+".join(sorted(set(source_types)))
        a_ictal = np.asarray(sample["window_adjacency"], dtype=np.float32)
        a_inter = stats_dict["A_inter_median"][None, :, :]
        mixed = 0.5 * a_ictal + 0.5 * np.abs(a_ictal - a_inter)
        eye = np.eye(mixed.shape[-1], dtype=bool)[None, :, :]
        mixed = mixed.copy()
        mixed[eye.repeat(mixed.shape[0], axis=0)] = 0.0
        sample["window_adjacency"] = mixed.astype(np.float32)
        record["metadata"]["adjacency_note"] = (
            "TFCCM causal adjacency skipped; using interictal-delta adjacency only."
            if variant == "M3_HUP_InterPhysGraph"
            else "Using precomputed interictal-delta adjacency."
        )
        output.append(record)
    return output, stats
