from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from outcome_hifos.cache_schema import CachePayload
from outcome_hifos.dataset import OutcomePatientExample, normalize_channel_name
from outcome_hifos.leakage_guard import assert_label_blind_tree
from task1_baselines.raw_preprocessing import RawPreprocessingConfig, extract_and_preprocess_windows


def _group(cache: CachePayload) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in cache.run_records:
        grouped[str(record.get("subject_id", ""))].append(record)
    return grouped


def _build(
    cache: CachePayload,
    references: Sequence[OutcomePatientExample],
    converter: Callable[[Mapping[str, Any]], tuple[np.ndarray, np.ndarray, dict[str, Any]]],
) -> tuple[list[OutcomePatientExample], list[dict[str, Any]]]:
    grouped = _group(cache)
    examples: list[OutcomePatientExample] = []
    audits: list[dict[str, Any]] = []
    for reference in sorted(references, key=lambda item: item.subject_id):
        records = grouped.get(reference.subject_id, [])
        if not records:
            raise ValueError(f"Token cache has no records for Task 2 patient {reference.subject_id}.")
        canonical = tuple(normalize_channel_name(value) for value in reference.canonical_channels)
        lookup = {channel: index for index, channel in enumerate(canonical)}
        runs = []
        centers_list = []
        channel_masks = []
        run_ids = []
        for record in sorted(records, key=lambda item: str(item.get("run_id", ""))):
            sample = record.get("sample") if isinstance(record.get("sample"), Mapping) else {}
            values, centers, detail = converter(record)
            local = [normalize_channel_name(value) for value in record.get("channel_names_norm", sample.get("channel_names_norm", []))]
            if values.ndim != 3 or values.shape[1] != len(local):
                raise ValueError(f"Task 2 token run {record.get('run_id')} tensor/channel mismatch.")
            aligned = np.zeros((values.shape[0], len(canonical), values.shape[2]), dtype=np.float32)
            present = np.zeros(len(canonical), dtype=bool)
            matched = 0
            for local_index, channel in enumerate(local):
                if channel in lookup:
                    aligned[:, lookup[channel]] = values[:, local_index]
                    present[lookup[channel]] = True
                    matched += 1
            if matched == 0:
                raise ValueError(f"Task 2 token run {record.get('run_id')} has no canonical channel matches.")
            runs.append(aligned)
            centers_list.append(centers)
            channel_masks.append(present)
            run_id = str(record.get("run_id", sample.get("sample_id", "")))
            run_ids.append(run_id)
            audits.append({"subject_id": reference.subject_id, "run_id": run_id, "channel_alignment_ratio": matched / max(len(local), 1), **detail})
        model_input = {"feature_runs": runs, "window_centers": centers_list, "seizure_channel_mask": channel_masks, "canonical_index": np.arange(len(canonical), dtype=np.int64)}
        assert_label_blind_tree(model_input, stage="task2_token_patient_example")
        examples.append(OutcomePatientExample(reference.subject_id, reference.center, reference.target, canonical, model_input, {"run_ids": tuple(run_ids), "token_source": "raw_or_fm"}))
    return examples, audits


def build_raw_patient_examples(cache: CachePayload, references: Sequence[OutcomePatientExample], config: RawPreprocessingConfig) -> tuple[list[OutcomePatientExample], list[dict[str, Any]]]:
    def converter(record: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        sample = record.get("sample") if isinstance(record.get("sample"), Mapping) else {}
        sfreq = float(sample.get("raw_temporal_sfreq", record.get("sfreq", 0.0)))
        values, audit = extract_and_preprocess_windows(dict(sample), original_sfreq=sfreq, config=config)
        centers = np.asarray(sample.get("window_relative_centers_sec", []), dtype=np.float32).reshape(-1)
        if len(centers) != values.shape[0]:
            raise ValueError("Raw window centers do not match extracted windows.")
        return values, centers, audit
    return _build(cache, references, converter)


def build_embedding_patient_examples(cache: CachePayload, references: Sequence[OutcomePatientExample]) -> tuple[list[OutcomePatientExample], list[dict[str, Any]]]:
    def converter(record: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        sample = record.get("sample") if isinstance(record.get("sample"), Mapping) else {}
        values = np.asarray(sample.get("window_features"), dtype=np.float32)
        centers = np.asarray(sample.get("window_relative_centers_sec", []), dtype=np.float32).reshape(-1)
        if values.ndim != 3 or len(centers) != values.shape[0]:
            raise ValueError("FM embedding tensor/center contract is invalid.")
        return values, centers, {"embedding_dim": int(values.shape[-1])}
    return _build(cache, references, converter)


__all__ = ["build_embedding_patient_examples", "build_raw_patient_examples"]
