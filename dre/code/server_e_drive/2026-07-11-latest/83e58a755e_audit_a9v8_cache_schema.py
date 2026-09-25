from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.build_latent_core_targets import (  # noqa: E402
    _find_cache_feature_names,
    _get_feature_names_for_window_features,
    _get_field,
    _merge_outer_inner_record,
    _parse_physics_state_features_arg,
)


EXACT_PHYS_FEATURES = [
    "early_high_gamma_slope",
    "early_line_length_slope",
    "onset_latency_high_gamma",
    "onset_latency_line_length",
    "onset_rank_high_gamma",
    "onset_rank_line_length",
    "high_gamma_top20pct_mean",
    "line_length_top20pct_mean",
    "hfo80_150_event_rate",
    "hfo80_150_duration_fraction",
    "hfo80_150_mean_envelope_z",
    "hfo80_150_max_envelope_z",
]


def _iter_samples(obj: Any) -> Iterable[Any]:
    if isinstance(obj, list):
        yield from obj
    elif isinstance(obj, dict):
        for key in ("samples", "samples_list", "data", "records", "run_records"):
            val = obj.get(key)
            if isinstance(val, list):
                yield from val
                return
        yield obj
    else:
        yield obj


def _keys(sample: Any) -> list[str]:
    if isinstance(sample, dict):
        return sorted(str(key) for key in sample.keys())
    return sorted(k for k in dir(sample) if not k.startswith("_"))


def _get(sample: Any, key: str, default: Any = None) -> Any:
    if isinstance(sample, dict):
        return sample.get(key, default)
    return getattr(sample, key, default)


def _shape(value: Any) -> str:
    try:
        return str(tuple(np.asarray(value).shape))
    except Exception:
        return ""


def match_feature(name: str) -> tuple[str, int, bool, str]:
    norm = str(name).strip().lower()
    if norm in EXACT_PHYS_FEATURES:
        sign = -1 if "latency" in norm or "rank" in norm else 1
        return "exact_a9v3_s5", sign, True, "exact A9v3 S5 physiology feature"
    return "none", 0, False, "not in A9v3 S5 physiology feature list"


def audit_cache(
    cache_path: str | Path,
    *,
    physics_state_features: str = "",
    physics_feature_slice: str = "all",
    feature_name_source: str = "auto",
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    path = Path(cache_path)
    with open(path, "rb") as fin:
        obj = pickle.load(fin)
    raw_samples = list(_iter_samples(obj))
    samples = [_merge_outer_inner_record(sample) for sample in raw_samples]
    sample = samples[0] if samples else {}
    sample_keys = _keys(sample)
    top_level_keys = sorted(obj.keys()) if isinstance(obj, dict) else []

    feature_name_fields = [key for key in sample_keys if "feature" in key.lower() and "name" in key.lower()]
    feature_array_fields = [key for key in sample_keys if key.lower() in {"window_features", "features", "physics_state_features", "x", "channel_features"} or ("feature" in key.lower() and "name" not in key.lower())]
    subject_fields = [key for key in sample_keys if key.lower() in {"subject_id", "patient_id", "subject", "patient"}]
    channel_fields = [key for key in sample_keys if "channel" in key.lower() and "name" in key.lower()]

    cli_names = _parse_physics_state_features_arg(physics_state_features)
    feature_rows = []
    feature_name_source_used = ""
    feature_name_fallback_status = "not_attempted"
    window_feature_dim = 0
    suggestion = ""

    first_wf_sample = None
    first_wf = None
    for candidate in samples:
        wf_raw = _get_field(candidate, "window_features", None)
        if wf_raw is None:
            continue
        wf = np.asarray(wf_raw)
        if wf.ndim == 3:
            first_wf_sample = candidate
            first_wf = wf
            window_feature_dim = int(wf.shape[-1])
            break

    if first_wf_sample is not None and first_wf is not None:
        try:
            names, feature_slice, reason = _get_feature_names_for_window_features(
                first_wf_sample,
                wf=first_wf,
                cli_feature_names=cli_names,
                feature_name_source=feature_name_source,
                physics_feature_slice=physics_feature_slice,
            )
            feature_name_source_used = "cache" if reason.startswith("cache:") else "cli"
            feature_name_fallback_status = "ok"
            source_field = reason
            for idx, name in enumerate(names):
                group, sign, used, row_reason = match_feature(str(name))
                feature_rows.append(
                    {
                        "feature_index": idx,
                        "feature_name": str(name),
                        "source_field": source_field,
                        "matched_group": group,
                        "sign": sign,
                        "used": bool(used),
                        "reason": row_reason,
                    }
                )
        except ValueError as exc:
            feature_name_fallback_status = "failed_dim_mismatch" if "does not match" in str(exc) else "failed"
            suggestion = "pass --physics_feature_slice start:end" if feature_name_fallback_status == "failed_dim_mismatch" else str(exc)
    else:
        feature_name_fallback_status = "failed_missing_window_features"

    key_rows = [{"sample_index": 0, "key": key, "value_type": type(_get(sample, key)).__name__} for key in sample_keys]
    shape_rows = [{"sample_index": 0, "field": key, "shape": _shape(_get(sample, key))} for key in sample_keys]
    matched = pd.DataFrame([row for row in feature_rows if row["used"]])

    audit = {
        "cache_type": type(obj).__name__,
        "top_level_keys": top_level_keys,
        "n_samples_detected": int(len(samples)),
        "sample_type": type(sample).__name__,
        "sample_keys": sample_keys,
        "has_window_features": "window_features" in sample_keys,
        "window_features_shape_examples": [
            str(tuple(np.asarray(_get_field(candidate, "window_features")).shape))
            for candidate in samples[:5]
            if _get_field(candidate, "window_features", None) is not None
        ],
        "has_window_feature_names": "window_feature_names" in sample_keys,
        "has_feature_names": bool(feature_name_fields),
        "candidate_feature_name_fields": feature_name_fields,
        "candidate_feature_array_fields": feature_array_fields,
        "candidate_subject_fields": subject_fields,
        "candidate_channel_fields": channel_fields,
        "example_shapes": {row["field"]: row["shape"] for row in shape_rows if row["shape"]},
        "n_matched_phys_features": int(len(matched)),
        "feature_name_source_used": feature_name_source_used,
        "physics_state_features_cli_len": int(len(cli_names)),
        "window_feature_dim": int(window_feature_dim),
        "physics_feature_slice": str(physics_feature_slice),
        "feature_name_fallback_status": feature_name_fallback_status,
        "can_use_cli_feature_names": bool(window_feature_dim > 0 and len(cli_names) > 0),
        "suggestion": suggestion,
    }
    return (
        audit,
        pd.DataFrame(feature_rows, columns=["feature_index", "feature_name", "source_field", "matched_group", "sign", "used", "reason"]),
        pd.DataFrame(key_rows),
        pd.DataFrame(shape_rows),
        matched,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit A9v8 cache schema before physiology pseudo-core extraction.")
    parser.add_argument("--cache_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--physics_state_features", type=str, default="")
    parser.add_argument("--physics_feature_slice", type=str, default="all")
    parser.add_argument("--feature_name_source", type=str, default="auto", choices=["auto", "cache", "cli"])
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    audit, features, keys, shapes, matched = audit_cache(
        args.cache_path,
        physics_state_features=args.physics_state_features,
        physics_feature_slice=args.physics_feature_slice,
        feature_name_source=args.feature_name_source,
    )
    with open(output_dir / "cache_schema_audit.json", "w", encoding="utf-8") as fout:
        json.dump(audit, fout, indent=2, ensure_ascii=False, sort_keys=True)
    features.to_csv(output_dir / "cache_feature_names.csv", index=False)
    keys.to_csv(output_dir / "cache_sample_keys.csv", index=False)
    shapes.to_csv(output_dir / "cache_sample_shapes.csv", index=False)
    matched.to_csv(output_dir / "cache_matched_phys_features.csv", index=False)
    print(f"Wrote {output_dir / 'cache_schema_audit.json'}")


if __name__ == "__main__":
    main()
