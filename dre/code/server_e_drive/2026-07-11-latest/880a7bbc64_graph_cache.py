from __future__ import annotations

import hashlib
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .functional_graph import GraphConfig, PHASES, build_phase_graph, infer_onset_sample, is_brain_channel, normalize_channel_name, phase_bounds
from .exclusions import apply_exclusions, build_exclusion_audit, load_exclusion_manifest


def _sample(record: Mapping[str, Any]) -> Mapping[str, Any]:
    return record.get("sample") if isinstance(record.get("sample"), Mapping) else {}


def _key(record: Mapping[str, Any]) -> tuple[str, str, str]:
    sample = _sample(record)
    return str(record.get("subject_id", "")), str(record.get("run_id", "")), str(sample.get("sample_id", record.get("run_id", "")))


def _channel_metadata(record: Mapping[str, Any], channel_names: Sequence[str]) -> list[dict[str, Any]]:
    source = record.get("channel_meta", [])
    by_name = {}
    if isinstance(source, Sequence) and not isinstance(source, (str, bytes)):
        for item in source:
            if not isinstance(item, Mapping):
                continue
            name = normalize_channel_name(item.get("name", item.get("channel", item.get("channel_name", ""))))
            # status_description is intentionally never copied or read.
            by_name[name] = {key: item.get(key) for key in ("status", "bad", "is_bad") if key in item}
    return [by_name.get(normalize_channel_name(name), {}) for name in channel_names]


def build_graph_cache(
    raw_cache: Mapping[str, Any],
    feature_cache: Mapping[str, Any],
    output_dir: str | Path,
    *,
    raw_source_path: str | Path,
    config: GraphConfig | None = None,
    strict: bool = False,
    patient_filter: set[str] | None = None,
    exclusion_manifest: str | Path | None = None,
) -> dict[str, Any]:
    cfg = config or GraphConfig()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    raw_records_all = list(raw_cache.get("run_records", []))
    feature_records_all = list(feature_cache.get("run_records", []))
    exclusions = load_exclusion_manifest(exclusion_manifest)
    exclusion_audit = build_exclusion_audit(exclusions, feature_records_all, raw_records_all)
    raw_records, _ = apply_exclusions(raw_records_all, exclusions)
    feature_records, _ = apply_exclusions(feature_records_all, exclusions)
    exclusion_audit.to_csv(output / "task2_npam_exclusion_audit.csv", index=False)
    raw_counts = Counter(_key(record) for record in raw_records)
    feature_counts = Counter(_key(record) for record in feature_records)
    duplicate_raw = sorted(key for key, count in raw_counts.items() if count > 1)
    duplicate_feature = sorted(key for key, count in feature_counts.items() if count > 1)
    ambiguous = set(duplicate_raw) | set(duplicate_feature)
    if ambiguous:
        raise ValueError(f"Ambiguous duplicate patient/seizure/sample graph keys: {sorted(ambiguous)[:20]}")
    raw_lookup = {_key(record): record for record in raw_records}
    feature_lookup = {_key(record): record for record in feature_records}
    if patient_filter is not None:
        raw_lookup = {key: value for key, value in raw_lookup.items() if key[0] in patient_filter}
        feature_lookup = {key: value for key, value in feature_lookup.items() if key[0] in patient_filter}
    missing_raw = sorted(set(feature_lookup) - set(raw_lookup))
    if strict and missing_raw:
        raise ValueError(f"Feature runs missing raw match: {missing_raw[:20]}")
    source_hash = hashlib.sha256(str(Path(raw_source_path).expanduser().resolve()).encode()).hexdigest()
    graphs, patient_rows, seizure_rows, phase_rows, mapping_rows, weight_rows = [], [], [], [], [], []
    patient_summary: dict[str, Counter] = defaultdict(Counter)
    invalid_reasons = Counter()
    for key in sorted(set(feature_lookup) & set(raw_lookup)):
        feature_record, raw_record = feature_lookup[key], raw_lookup[key]
        feature_channels = [normalize_channel_name(value) for value in feature_record.get("channel_names_norm", [])]
        raw_channels = [normalize_channel_name(value) for value in raw_record.get("channel_names_norm", [])]
        if feature_channels != raw_channels:
            if strict:
                raise ValueError(f"Raw/feature channel order mismatch for {key}")
        raw_sample = _sample(raw_record)
        waveform = np.asarray(raw_sample.get("raw_waveform", []), dtype=np.float32)
        sfreq = float(raw_sample.get("raw_temporal_sfreq", raw_record.get("sfreq", 0.0)) or 0.0)
        onset, origin = infer_onset_sample(dict(raw_sample), waveform.shape[-1] if waveform.ndim == 2 else 0, sfreq)
        channel_meta = _channel_metadata(raw_record, raw_channels)
        valid_channel = np.asarray([is_brain_channel(name, meta) for name, meta in zip(raw_channels, channel_meta)], dtype=bool)
        seizure_valid = 0
        if waveform.ndim != 2 or sfreq <= 0.0 or onset is None:
            bounds = {phase: (0, 0) for phase in PHASES}
            base_reason = "missing_raw_waveform_or_sfreq" if waveform.ndim != 2 or sfreq <= 0 else origin
        else:
            bounds = phase_bounds(onset, waveform.shape[-1], sfreq)
            base_reason = ""
        for phase in PHASES:
            start, end = bounds[phase]
            if base_reason:
                graph = build_phase_graph(np.zeros((len(raw_channels), 0), dtype=np.float32), sfreq or 1.0, raw_channels, channel_valid=valid_channel, config=cfg)
                graph["invalid_reason"] = base_reason
            else:
                graph = build_phase_graph(waveform[:, start:end], sfreq, raw_channels, channel_valid=valid_channel, config=cfg)
            record = {
                "patient_key": key[0], "seizure_id": key[1], "sample_id": key[2], "phase": phase,
                "graph_source": cfg.graph_source, "sampling_frequency": sfreq, "band_low": cfg.band_low, "band_high": cfg.band_high,
                "preprocessing_hash": cfg.preprocessing_hash(), "raw_source_path_hash": source_hash,
                "raw_time_origin": origin, "phase_start_sample": int(start), "phase_end_sample": int(end),
                **graph,
            }
            graphs.append(record)
            valid = bool(graph["graph_valid"])
            seizure_valid += int(valid)
            patient_summary[key[0]]["total"] += 1
            patient_summary[key[0]]["valid"] += int(valid)
            if not valid:
                invalid_reasons[str(graph["invalid_reason"])] += 1
            phase_rows.append({"patient_key": key[0], "seizure_id": key[1], "phase": phase, "graph_valid": valid, "invalid_reason": graph["invalid_reason"], "n_nodes": graph["n_nodes"], "n_edges": graph["n_edges"], "raw_time_origin": origin})
            weight_rows.extend({"patient_key": key[0], "seizure_id": key[1], "phase": phase, "edge_weight": float(value)} for value in np.asarray(graph["edge_weight"]).reshape(-1))
        seizure_rows.append({"patient_key": key[0], "seizure_id": key[1], "n_valid_phases": seizure_valid, "graph_coverage": seizure_valid / float(len(PHASES))})
        for channel_idx, name in enumerate(raw_channels):
            mapping_rows.append({"patient_key": key[0], "seizure_id": key[1], "local_channel_index": channel_idx, "channel_name": name, "valid_brain_channel": bool(valid_channel[channel_idx]), "feature_channel_match": channel_idx < len(feature_channels) and name == feature_channels[channel_idx]})
    for patient, counts in sorted(patient_summary.items()):
        patient_rows.append({"patient_key": patient, "n_valid_phases": counts["valid"], "n_total_phases": counts["total"], "graph_coverage": counts["valid"] / max(counts["total"], 1)})
    payload = {"cache_version": "p2_q10_npam_graph_v2_three_phase", "network_phases": list(PHASES), "config": cfg.__dict__, "graphs": graphs}
    with (output / "phase_functional_graphs.pkl").open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    pd.DataFrame(patient_rows).to_csv(output / "graph_patient_coverage.csv", index=False)
    pd.DataFrame(seizure_rows).to_csv(output / "graph_seizure_coverage.csv", index=False)
    pd.DataFrame(phase_rows).to_csv(output / "graph_phase_coverage.csv", index=False)
    pd.DataFrame([{"invalid_reason": reason, "count": count} for reason, count in invalid_reasons.items()]).to_csv(output / "graph_invalid_reasons.csv", index=False)
    pd.DataFrame(mapping_rows).to_csv(output / "graph_channel_mapping_audit.csv", index=False)
    pd.DataFrame(weight_rows, columns=("patient_key", "seizure_id", "phase", "edge_weight")).to_csv(output / "graph_weight_distribution.csv", index=False)
    valid_count = sum(bool(record["graph_valid"]) for record in graphs)
    manifest = {
        "cache_version": payload["cache_version"], "graph_source": cfg.graph_source, "preprocessing_hash": cfg.preprocessing_hash(),
        "n_graphs": len(graphs), "n_valid_graphs": valid_count, "graph_phase_coverage": valid_count / max(len(graphs), 1),
        "n_patients": len(patient_summary), "n_seizures": len(seizure_rows), "n_feature_runs_missing_raw": len(missing_raw),
        "missing_raw_runs": [list(key) for key in missing_raw], "invalid_reasons": dict(invalid_reasons),
        "n_ambiguous_duplicate_keys_excluded": len(ambiguous),
        "ambiguous_duplicate_keys_excluded": [list(key) for key in sorted(ambiguous)],
        "n_raw_records_excluded_by_duplicate_key": int(sum(raw_counts[key] for key in ambiguous)),
        "n_feature_records_excluded_by_duplicate_key": int(sum(feature_counts[key] for key in ambiguous)),
        "exclusion_manifest": str(Path(exclusion_manifest).expanduser().resolve()) if exclusion_manifest else None,
        "declared_exclusion_rows": int(len(exclusions)),
        "network_phases": list(PHASES),
        "raw_preprocessing_audit": {"notch_status": "not_reapplied_cache_provenance_unknown", "reference_scheme": "cache_provenance_unknown", "bandpass_applied_here": [cfg.band_low, cfg.band_high], "unit": "scale_invariant_spearman_after_channelwise_robust_z", "nan_inf": "replaced_after_filter_input_validation"},
    }
    (output / "graph_cache_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


__all__ = ["build_graph_cache"]
