"""Build A9v8 leak-free latent pseudo-core targets from A9v3 OOF teacher scores.

Physiology core score aggregates per-channel window features from the main
window cache.  Pseudo-core q uses threshold + sigmoid + mass normalisation.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import pickle
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _patient_zscore(values: pd.Series) -> pd.Series:
    arr = values.astype(float).to_numpy()
    std = float(arr.std(ddof=0))
    if std <= 1e-8:
        return pd.Series(np.zeros_like(arr), index=values.index)
    return pd.Series((arr - float(arr.mean())) / std, index=values.index)


def normalize_center(value: object) -> str:
    return str(value).strip().lower().replace(" ", "_")


def center_id_is_informative(df: pd.DataFrame) -> bool:
    if "center_id" not in df.columns:
        return False
    return int(df["center_id"].dropna().astype(str).nunique()) > 1


def observed_fold_ids(df: pd.DataFrame) -> list[int]:
    if "fold_id" not in df.columns:
        raise ValueError("No fold_id found in teacher/target dataframe.")
    folds = sorted(df["fold_id"].dropna().astype(int).unique().tolist())
    if not folds:
        raise ValueError("No fold_id found in teacher/target dataframe.")
    return folds


def _phys_fail_closed_message(reason: str) -> str:
    return (
        "No usable physiology features were extracted from cache. "
        f"Reason: {reason}. "
        "Run scripts/audit_a9v8_cache_schema.py to inspect cache schema. "
        "If you intentionally want teacher-only pseudo-core, rerun with: "
        "--phys_core_mode teacher_only --allow_teacher_only_core"
    )


def _compute_q_threshold_sigmoid(core_scores: pd.Series, mass: float, tau_q: float) -> pd.Series:
    n = len(core_scores)
    if n == 0 or mass <= 0.0:
        return pd.Series(np.zeros(n, dtype=float), index=core_scores.index)
    scores = core_scores.astype(float).to_numpy()
    k = min(max(1, int(round(mass))), n)
    threshold = float(-np.partition(-scores, k - 1)[k - 1])
    q_raw = 1.0 / (1.0 + np.exp(-(scores - threshold) / max(float(tau_q), 1e-6)))
    q_sum = max(float(q_raw.sum()), 1e-12)
    q = q_raw * (float(mass) / q_sum)
    q = np.clip(q, 0.0, 1.0)
    post_sum = float(q.sum())
    if post_sum > 0.0 and abs(post_sum - float(mass)) / max(float(mass), 1.0) > 0.3:
        q = q * (float(mass) / max(post_sum, 1e-12))
        q = np.clip(q, 0.0, 1.0)
    return pd.Series(q, index=core_scores.index)


def _entropy(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[values > 1e-12]
    if len(values) == 0:
        return 0.0
    return float(-np.sum(values * np.log(values)))


# ---------------------------------------------------------------------------
# strict feature matcher — no repr fallback, no generic gamma
# ---------------------------------------------------------------------------

_CORE_POSITIVE_PAIRS = [
    (("high_gamma", "hgamma", "gamma"), ("slope",)),
    (("high_gamma", "hgamma", "gamma"), ("top20", "top20pct", "top_pct", "peak", "max", "mean")),
    (("line_length", "linelength"), ("slope",)),
    (("line_length", "linelength"), ("top20", "top20pct", "top_pct", "peak", "max", "mean")),
    (("hfo", "ripple", "fast_ripple"), ("event_rate", "duration_fraction", "mean_envelope", "max_envelope", "envelope")),
]

_LATENCY_PATTERNS = (
    "onset_latency", "latency_high_gamma", "latency_line_length",
    "onset_rank", "rank_high_gamma", "rank_line_length",
)


def _match_feature(name: str) -> tuple[str | None, int, str]:
    """Return (group, sign, reason)."""
    low = str(name).strip().lower().replace(" ", "").replace("_", "")

    # latency / rank → -1
    for pat in _LATENCY_PATTERNS:
        pat_clean = pat.replace("_", "")
        if pat_clean in low:
            return ("latency_rank", -1, f"matched onset latency/rank pattern '{pat}'")

    # positive core pairs
    for (group_tokens, stat_tokens) in _CORE_POSITIVE_PAIRS:
        has_group = any(t.replace("_", "") in low for t in group_tokens)
        has_stat = any(t.replace("_", "") in low for t in stat_tokens)
        if has_group and has_stat:
            return ("positive_core", 1, f"matched {group_tokens[0]}+{stat_tokens[0]} core evidence")

    return (None, 0, "no matching physiology core pattern")


@dataclass(frozen=True)
class PhysCoreFeatureMatch:
    feature_index: int
    feature_name: str
    matched_group: str
    sign: int
    reason: str


_A9V8_LATENCY_PATTERNS = (
    "onset_latency", "latency_high_gamma", "latency_line_length",
    "onset_rank", "rank_high_gamma", "rank_line_length",
    "peak_time_high_gamma", "peak_time_line_length", "peaktime_highgamma", "peaktime_linelength",
)

_EXACT_S5_12_SIGNS: dict[str, tuple[str, int]] = {
    "early_high_gamma_slope": ("positive_core", 1),
    "early_line_length_slope": ("positive_core", 1),
    "onset_latency_high_gamma": ("latency_rank", -1),
    "onset_latency_line_length": ("latency_rank", -1),
    "onset_rank_high_gamma": ("latency_rank", -1),
    "onset_rank_line_length": ("latency_rank", -1),
    "high_gamma_top20pct_mean": ("positive_core", 1),
    "line_length_top20pct_mean": ("positive_core", 1),
    "hfo80_150_event_rate": ("positive_core", 1),
    "hfo80_150_duration_fraction": ("positive_core", 1),
    "hfo80_150_mean_envelope_z": ("positive_core", 1),
    "hfo80_150_max_envelope_z": ("positive_core", 1),
}

_EXPANDED_S5_16_SIGNS: dict[str, tuple[str, int]] = {
    **_EXACT_S5_12_SIGNS,
    "high_gamma_max_to_mean": ("positive_core", 1),
    "line_length_max_to_mean": ("positive_core", 1),
    "peak_time_high_gamma": ("latency_rank", -1),
    "peak_time_line_length": ("latency_rank", -1),
}


def _feature_key(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_").replace("-", "_").replace(".", "_")


def _feature_match_from_set(name: str, mapping: dict[str, tuple[str, int]]) -> tuple[str | None, int, str]:
    key = _feature_key(name)
    if key not in mapping:
        return (None, 0, "not in requested physiology core feature set")
    group, sign = mapping[key]
    return (group, sign, f"matched explicit physiology core feature set entry '{key}'")


def classify_phys_core_feature(name: str) -> tuple[str | None, int, str]:
    """Return (group, sign, reason) with latency/peak-time priority."""
    low = str(name).strip().lower().replace(" ", "").replace("_", "")

    for pat in _A9V8_LATENCY_PATTERNS:
        pat_clean = pat.replace("_", "")
        if pat_clean in low:
            return ("latency_rank", -1, f"matched onset latency/rank/peak-time pattern '{pat}'")

    for (group_tokens, stat_tokens) in _CORE_POSITIVE_PAIRS:
        has_group = any(t.replace("_", "") in low for t in group_tokens)
        has_stat = any(t.replace("_", "") in low for t in stat_tokens)
        if has_group and has_stat:
            return ("positive_core", 1, f"matched {group_tokens[0]}+{stat_tokens[0]} core evidence")

    return (None, 0, "no matching physiology core pattern")


def _match_feature(name: str) -> tuple[str | None, int, str]:
    return classify_phys_core_feature(name)


def select_phys_core_features(feature_names: Iterable[str], *, feature_set: str = "auto") -> list[PhysCoreFeatureMatch]:
    """Select physiology-core features under an explicit S5 feature-set mode."""
    mode = str(feature_set).strip().lower()
    if mode not in {"auto", "exact_s5_12", "expanded_s5_16"}:
        raise ValueError(
            f"Unsupported phys_core_feature_set={feature_set!r}; expected auto, exact_s5_12, or expanded_s5_16."
        )
    selected: list[PhysCoreFeatureMatch] = []
    for fi, fname in enumerate(feature_names):
        if mode == "exact_s5_12":
            group, sign, reason = _feature_match_from_set(str(fname), _EXACT_S5_12_SIGNS)
        elif mode == "expanded_s5_16":
            group, sign, reason = _feature_match_from_set(str(fname), _EXPANDED_S5_16_SIGNS)
        else:
            group, sign, reason = classify_phys_core_feature(str(fname))
        if group is None:
            continue
        selected.append(
            PhysCoreFeatureMatch(
                feature_index=int(fi),
                feature_name=str(fname),
                matched_group=str(group),
                sign=int(sign),
                reason=str(reason),
            )
        )
    return selected


def _iter_cache_samples(cache_obj: Any) -> Iterable[dict[str, Any]]:
    """Robust iterator over cache samples, supporting common schemas."""
    if isinstance(cache_obj, list):
        yield from cache_obj
    elif isinstance(cache_obj, dict):
        for key in ("samples", "samples_list", "data", "records", "run_records"):
            val = cache_obj.get(key)
            if isinstance(val, list):
                yield from val
                return
        # maybe the dict itself is a sample
        yield cache_obj
    else:
        yield cache_obj


def _unwrap_cache_record(record: Any) -> Any:
    if isinstance(record, dict) and "sample" in record:
        return record["sample"]
    if hasattr(record, "sample"):
        return getattr(record, "sample")
    return record


def _object_to_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    try:
        return dict(vars(value))
    except TypeError:
        return {}


def _merge_outer_inner_record(record: Any) -> Any:
    inner = _unwrap_cache_record(record)
    if inner is record:
        return inner
    outer_map = _object_to_dict(record)
    inner_map = _object_to_dict(inner)
    if outer_map or inner_map:
        merged = dict(outer_map)
        merged.update(inner_map)
        return merged
    return inner


def _get_field(sample: Any, key: str, default: Any = None) -> Any:
    if isinstance(sample, dict):
        return sample.get(key, default)
    return getattr(sample, key, default)


def _parse_physics_state_features_arg(text: str | None) -> list[str]:
    if text is None:
        return []
    return [item.strip() for item in str(text).split(",") if item.strip()]


def _norm_channel(name: str) -> str:
    return str(name).strip().lower().replace(" ", "").replace("-", "_").replace(".", "_")


def _norm_subject_id(value: object) -> str:
    return str(value).strip().lower()


def _parse_slice_spec(spec: str, total_dim: int) -> slice:
    spec = str(spec).strip().lower()
    if spec == "all":
        return slice(0, total_dim)
    parts = spec.split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid slice spec {spec!r}")
    s, e = int(parts[0]), int(parts[1])
    if s < 0 or e > total_dim or s >= e:
        raise ValueError(f"Slice {spec!r} out of range for dim {total_dim}")
    return slice(s, e)


def _find_cache_feature_names(sample: Any) -> tuple[list[str], str] | None:
    fields = (
        "window_feature_names",
        "feature_names",
        "physics_state_feature_names",
        "physics_feature_names",
    )
    for field in fields:
        names = _get_field(sample, field, None)
        if isinstance(names, (list, tuple, np.ndarray)) and len(names) > 0:
            return [str(name) for name in names], field
    for meta_field in ("metadata", "meta"):
        meta = _get_field(sample, meta_field, None)
        if isinstance(meta, dict):
            for field in fields:
                names = meta.get(field)
                if isinstance(names, (list, tuple, np.ndarray)) and len(names) > 0:
                    return [str(name) for name in names], f"{meta_field}.{field}"
    return None


def _get_feature_names_for_window_features(
    sample: Any,
    *,
    wf: np.ndarray | None,
    cli_feature_names: list[str],
    feature_name_source: str,
    physics_feature_slice: str,
) -> tuple[list[str], slice, str]:
    """Return (feature_names, feature_slice, source_reason)."""
    if wf is None:
        raise ValueError("window_features is missing.")
    if wf.ndim != 3:
        raise ValueError(f"window_features must have shape [T,C,F], got {wf.shape}.")
    total_dim = int(wf.shape[-1])
    source = str(feature_name_source).strip().lower()
    if source not in {"auto", "cache", "cli"}:
        raise ValueError(f"Unsupported feature_name_source={feature_name_source!r}.")

    cache_names = _find_cache_feature_names(sample)
    use_cache = source in {"auto", "cache"} and cache_names is not None
    if source == "cache" and cache_names is None:
        raise ValueError("cache_schema_missing_window_feature_names")
    if source == "cli":
        use_cache = False

    feature_slice = _parse_slice_spec(physics_feature_slice, total_dim)
    slice_len = int(feature_slice.stop - feature_slice.start)
    covers_all = feature_slice.start == 0 and feature_slice.stop == total_dim

    if use_cache:
        names, field = cache_names
        if len(names) != total_dim:
            raise ValueError(
                f"Cache feature names length {len(names)} does not match window_features dim {total_dim}."
            )
        return names[feature_slice], feature_slice, f"cache:{field}"

    if not cli_feature_names:
        raise ValueError("cache missing feature names and --physics_state_features is empty.")
    if covers_all and total_dim != len(cli_feature_names):
        raise ValueError(
            f"CLI physics feature names length {len(cli_feature_names)} does not match "
            f"window_features dim {total_dim}. Pass --physics_feature_slice start:end if physics "
            f"features occupy a slice."
        )
    if not covers_all and slice_len != len(cli_feature_names):
        raise ValueError(
            f"CLI physics feature names length {len(cli_feature_names)} does not match "
            f"physics_feature_slice length {slice_len}."
        )
    return list(cli_feature_names), feature_slice, "cli:physics_state_features"


# ---------------------------------------------------------------------------
# cache inspection and phys_core_score computation
# ---------------------------------------------------------------------------

def inspect_and_compute_phys_core(
    teacher: pd.DataFrame,
    cache_path: str | Path | None,
    *,
    cli_feature_names: list[str] | None = None,
    feature_name_source: str = "auto",
    physics_feature_slice: str = "all",
    phys_core_feature_set: str = "auto",
) -> tuple[pd.Series, dict[str, Any], list[dict[str, Any]]]:
    """Inspect cache, build feature audit, return phys_core_score per teacher row.

    Uses (subject_id + channel_name) keys to avoid cross-patient mixing.
    Applies feature-level patient-wise z-score *before* summing across features.
    """
    default_audit: dict[str, Any] = {
        "phys_core_score_available": False,
        "phys_core_reason": "no_cache_path",
        "matched_feature_names": [],
        "phys_core_match_key": "subject_or_patient_id + normalized_channel_name",
        "feature_name_source_used": "",
        "physics_state_features_cli_len": int(len(cli_feature_names or [])),
        "physics_feature_slice": str(physics_feature_slice),
        "phys_core_feature_set": str(phys_core_feature_set),
        "window_feature_dim": 0,
    }

    if not cache_path:
        return pd.Series(np.zeros(len(teacher)), index=teacher.index), default_audit, []

    path = Path(cache_path)
    if not path.exists():
        default_audit["phys_core_reason"] = "cache_missing"
        return pd.Series(np.zeros(len(teacher)), index=teacher.index), default_audit, []

    # load cache
    try:
        with open(path, "rb") as f:
            cache_obj = pickle.load(f)
    except Exception as exc:
        default_audit["phys_core_reason"] = f"cache_unreadable:{type(exc).__name__}"
        return pd.Series(np.zeros(len(teacher)), index=teacher.index), default_audit, []

    feature_names: list[str] = []
    feature_slice = slice(0, 0)
    source_reason = ""
    window_feature_dim = 0
    for sample in _iter_cache_samples(cache_obj):
        sample = _merge_outer_inner_record(sample)
        wf_raw = _get_field(sample, "window_features", None)
        if wf_raw is None:
            continue
        wf = np.asarray(wf_raw, dtype=np.float32)
        if wf.ndim != 3:
            continue
        window_feature_dim = int(wf.shape[-1])
        try:
            feature_names, feature_slice, source_reason = _get_feature_names_for_window_features(
                sample,
                wf=wf,
                cli_feature_names=list(cli_feature_names or []),
                feature_name_source=feature_name_source,
                physics_feature_slice=physics_feature_slice,
            )
        except ValueError as exc:
            default_audit["phys_core_reason"] = str(exc)
            default_audit["window_feature_dim"] = window_feature_dim
            raise
        if feature_names:
            break

    if not feature_names:
        default_audit["phys_core_reason"] = "cache_schema_missing_window_feature_names"
        default_audit["window_feature_dim"] = window_feature_dim
        return pd.Series(np.zeros(len(teacher)), index=teacher.index), default_audit, []
    source_used = "cache" if source_reason.startswith("cache:") else "cli"

    # build feature audit
    used_features: list[tuple[int, int, str]] = []  # (fi, sign, name)
    feaudit_rows: list[dict[str, Any]] = []
    selected_features = {
        item.feature_index: item
        for item in select_phys_core_features(feature_names, feature_set=phys_core_feature_set)
    }
    for fi, fname in enumerate(feature_names):
        selected = selected_features.get(fi)
        if selected is not None:
            group, sign, reason = selected.matched_group, selected.sign, selected.reason
            used_features.append((fi, sign, str(fname)))
        else:
            group, sign, reason = (None, 0, "not selected by physiology core feature set")
        used = selected is not None
        feaudit_rows.append({
            "feature_index": fi,
            "feature_name": str(fname),
            "matched_group": group or "none",
            "sign": sign,
            "used": used,
            "reason": reason,
            "feature_name_source": source_used,
            "feature_slice_start": int(feature_slice.start),
            "feature_slice_stop": int(feature_slice.stop),
            "phys_core_feature_set": str(phys_core_feature_set),
        })

    if not used_features:
        audit = {
            "phys_core_score_available": False,
            "phys_core_reason": "no_usable_physiology_features_in_cache",
            "matched_feature_names": [],
            "phys_core_match_key": "subject_or_patient_id + normalized_channel_name",
            "feature_name_source_used": source_used,
            "physics_state_features_cli_len": int(len(cli_feature_names or [])),
            "physics_feature_slice": str(physics_feature_slice),
            "phys_core_feature_set": str(phys_core_feature_set),
            "window_feature_dim": window_feature_dim,
        }
        return pd.Series(np.zeros(len(teacher)), index=teacher.index), audit, feaudit_rows

    # ---- collect per-(patient, channel, feature) values from cache ----
    # key: (norm_subject_id, norm_channel) → {feature_name: [values_across_seizures]}
    pc_feat: dict[tuple[str, str], dict[str, list[float]]] = {}
    valid_sample_count = 0

    for sample in _iter_cache_samples(cache_obj):
        sample = _merge_outer_inner_record(sample)
        sid = str(_get_field(sample, "subject_id", _get_field(sample, "patient_id", "")))
        ch_names = _get_field(sample, "channel_names_norm", _get_field(sample, "channel_names", []))
        wf_raw = _get_field(sample, "window_features", None)

        if wf_raw is None:
            continue
        wf = np.asarray(wf_raw, dtype=np.float32)
        if wf.ndim != 3:
            continue
        T, C, F_total = wf.shape
        if F_total != window_feature_dim:
            continue
        wf_phys = wf[:, :, feature_slice]
        _, _, F = wf_phys.shape
        if F != len(feature_names):
            raise ValueError(
                f"Selected window_features dim {F} does not match feature names length {len(feature_names)}."
            )
        valid_sample_count += 1

        sid_norm = _norm_subject_id(sid)
        ch_names_str = [str(c) for c in ch_names] if len(ch_names) == C else [f"ch{i}" for i in range(C)]
        for ci in range(C):
            pc_key = (sid_norm, _norm_channel(ch_names_str[ci]))
            feat_map = pc_feat.setdefault(pc_key, {})
            for fi, sign, fname in used_features:
                vals = wf_phys[:, ci, fi]
                finite = vals[np.isfinite(vals)]
                if len(finite) > 0:
                    feat_map.setdefault(fname, []).append(float(np.mean(finite)))

    # fail-closed: usable features found but zero valid samples
    if valid_sample_count == 0:
        raise ValueError(
            f"Found {len(used_features)} usable physiology features but zero valid cache samples "
            f"with window_features of shape [T,C,F] matching {len(feature_names)} features."
        )

    if not pc_feat:
        raise ValueError(
            f"Found {len(used_features)} usable physiology features across {valid_sample_count} samples "
            f"but zero patient-channel entries were extracted."
        )

    # ---- map teacher rows to per-feature values ----
    # per_feature_table: list of (teacher_idx, patient_id, feature_name, sign, value)
    per_feature_table: list[dict[str, Any]] = []
    matched = 0
    for ti, row in teacher.iterrows():
        # candidate subject/patient ids, deduplicated in order
        candidate_ids: list[str] = []
        if "subject_id" in row and pd.notna(row["subject_id"]):
            candidate_ids.append(str(row["subject_id"]))
        if "patient_id" in row and pd.notna(row["patient_id"]):
            candidate_ids.append(str(row["patient_id"]))
        # deduplicate while preserving order
        seen: set[str] = set()
        deduped: list[str] = []
        for cid in candidate_ids:
            normed = _norm_subject_id(cid)
            if normed not in seen:
                seen.add(normed)
                deduped.append(normed)

        channel_key = _norm_channel(str(row.get("channel_name", "")))
        feat_map: dict[str, list[float]] = {}
        for sid_norm in deduped:
            pc_key = (sid_norm, channel_key)
            if pc_key in pc_feat:
                feat_map = pc_feat[pc_key]
                break

        if feat_map:
            matched += 1
            for fi, sign, fname in used_features:
                vals = feat_map.get(fname, [])
                val = float(np.mean(vals)) if vals else 0.0
                per_feature_table.append({
                    "teacher_idx": ti,
                    "patient_id": str(row.get("patient_id", candidate_ids[0] if candidate_ids else "")),
                    "feature_name": fname,
                    "sign": sign,
                    "feature_value": val,
                })

    # fail-closed: usable features but zero teacher rows matched
    if matched == 0:
        raise ValueError(
            f"Found {len(used_features)} usable physiology features and {len(pc_feat)} patient-channel "
            f"entries in cache, but ZERO of {len(teacher)} teacher rows matched. "
            f"Check subject_id / channel_name alignment between teacher and cache."
        )

    # ---- feature-level patient-wise z-score ----
    ft_df = pd.DataFrame(per_feature_table)
    ft_df["feature_z"] = ft_df.groupby(["patient_id", "feature_name"])["feature_value"].transform(_patient_zscore)
    ft_df["feature_z"] = ft_df["feature_z"].fillna(0.0)

    # ---- weighted sum per teacher row ----
    ft_df["weighted"] = ft_df["sign"] * ft_df["feature_z"]
    phys_raw_sum = ft_df.groupby("teacher_idx")["weighted"].sum()

    # ---- patient-wise z-score of final sum ----
    phys_out = pd.Series(0.0, index=teacher.index, dtype=float)
    for ti, val in phys_raw_sum.items():
        phys_out.iloc[int(ti)] = val
    phys_z_df = teacher[["patient_id"]].copy()
    phys_z_df["_phys_raw"] = phys_out.values
    phys_z_scored = phys_z_df.groupby("patient_id")["_phys_raw"].transform(_patient_zscore).fillna(0.0)

    audit = {
        "phys_core_score_available": True,
        "phys_core_reason": (
            f"feature-level z-score of {len(used_features)} features "
            f"({sum(1 for _, s, _ in used_features if s > 0)} positive + "
            f"{sum(1 for _, s, _ in used_features if s < 0)} latency), "
            f"matched {matched}/{len(teacher)} teacher rows"
        ),
        "matched_feature_names": [fname for _, _, fname in used_features],
        "phys_core_match_key": "subject_or_patient_id + normalized_channel_name",
        "phys_core_matched_teacher_rows": matched,
        "phys_core_unmatched_teacher_rows": int(len(teacher)) - matched,
        "feature_name_source_used": source_used,
        "physics_state_features_cli_len": int(len(cli_feature_names or [])),
        "physics_feature_slice": str(physics_feature_slice),
        "phys_core_feature_set": str(phys_core_feature_set),
        "window_feature_dim": int(window_feature_dim),
    }
    return pd.Series(phys_z_scored.values, index=teacher.index, dtype=float), audit, feaudit_rows


# ---------------------------------------------------------------------------
# main target builder
# ---------------------------------------------------------------------------

def build_latent_core_targets_dataframe(
    teacher: pd.DataFrame,
    *,
    broad_centers: set[str],
    rho: float = 0.30,
    tau_q: float = 0.10,
    cache_path: str | Path | None = None,
    phys_core_mode: str = "auto_s5",
    allow_teacher_only_core: bool = False,
    require_phys_core: bool = False,
    physics_state_features: str | list[str] = "",
    feature_name_source: str = "auto",
    physics_feature_slice: str = "all",
    phys_core_feature_set: str = "auto",
) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    df = teacher.copy()
    df["center"] = df["center"].map(normalize_center)
    if "patient_id" not in df.columns:
        df["patient_id"] = df["subject_id"].astype(str)
    broad_centers = {normalize_center(center) for center in broad_centers}

    phys_core_mode = str(phys_core_mode).strip().lower()
    cli_feature_names = (
        _parse_physics_state_features_arg(physics_state_features)
        if isinstance(physics_state_features, str)
        else [str(item) for item in physics_state_features]
    )
    teacher_only_fallback_used = False
    if phys_core_mode == "teacher_only":
        if not allow_teacher_only_core:
            raise ValueError("teacher_only mode requires --allow_teacher_only_core.")
        phys = pd.Series(np.zeros(len(df), dtype=float), index=df.index)
        feaudit_rows: list[dict[str, Any]] = []
        phys_audit = {
            "phys_core_mode": phys_core_mode,
            "phys_core_score_available": False,
            "phys_core_reason": "explicit_teacher_only_mode",
            "matched_feature_names": [],
            "phys_core_matched_teacher_rows": 0,
            "phys_core_unmatched_teacher_rows": int(len(df)),
            "feature_name_source_used": "",
            "physics_state_features_cli_len": int(len(cli_feature_names)),
            "physics_feature_slice": str(physics_feature_slice),
            "phys_core_feature_set": str(phys_core_feature_set),
            "window_feature_dim": 0,
        }
        teacher_only_fallback_used = True
    elif phys_core_mode == "auto_s5":
        phys, phys_audit, feaudit_rows = inspect_and_compute_phys_core(
            df,
            cache_path,
            cli_feature_names=cli_feature_names,
            feature_name_source=feature_name_source,
            physics_feature_slice=physics_feature_slice,
            phys_core_feature_set=phys_core_feature_set,
        )
        phys_audit["phys_core_mode"] = phys_core_mode
        if not bool(phys_audit.get("phys_core_score_available", False)):
            reason = str(phys_audit.get("phys_core_reason", "unknown"))
            if require_phys_core and not allow_teacher_only_core:
                raise ValueError(_phys_fail_closed_message(reason))
            teacher_only_fallback_used = True
            phys_audit.update(
                {
                    "phys_core_score_available": False,
                    "phys_core_reason": reason,
                    "phys_core_matched_teacher_rows": 0,
                    "phys_core_unmatched_teacher_rows": int(len(df)),
                }
            )
    else:
        raise ValueError(f"Unsupported phys_core_mode={phys_core_mode!r}; expected auto_s5 or teacher_only.")
    df["phys_core_score"] = phys.values
    df["phys_core_score_available"] = bool(phys_audit.get("phys_core_score_available", False))
    df["teacher_only_fallback_used"] = bool(teacher_only_fallback_used)

    df["a9v3_oof_score_z_patient"] = df.groupby("patient_id")["a9v3_oof_score"].transform(_patient_zscore)
    df["core_score"] = df["a9v3_oof_score_z_patient"] + df["phys_core_score"]

    df["pseudo_core_q"] = 0.0
    df["core_target_type"] = "unused"

    for patient_id, group in df.groupby("patient_id", sort=False):
        idx = group.index
        center = str(group["center"].iloc[0]).lower()
        labels = group["label_ez"].astype(float)
        if center not in broad_centers:
            df.loc[idx, "pseudo_core_q"] = labels.to_numpy()
            df.loc[idx, "core_target_type"] = "strong_label"
            continue
        df.loc[idx, "core_target_type"] = "teacher_only_latent_core" if teacher_only_fallback_used else "latent_core"
        pos_idx = group.index[labels > 0.5]
        if len(pos_idx) == 0:
            continue
        k = int(round(float(labels.sum())))
        mass = max(1.0, float(rho) * float(k))
        core_scores = df.loc[pos_idx, "core_score"]
        q = _compute_q_threshold_sigmoid(core_scores, mass, tau_q)
        df.loc[pos_idx, "pseudo_core_q"] = q.to_numpy()

    audit = _build_target_audit(
        df,
        broad_centers,
        rho,
        tau_q,
        phys_audit,
        phys_core_mode=phys_core_mode,
        teacher_only_fallback_used=teacher_only_fallback_used,
    )
    return df, audit, feaudit_rows


def _build_target_audit(
    targets,
    broad_centers,
    rho,
    tau_q,
    phys_audit,
    *,
    phys_core_mode: str,
    teacher_only_fallback_used: bool,
):
    n_patients = int(targets["patient_id"].nunique())
    fold_info = {}
    for fold in observed_fold_ids(targets):
        fold = int(fold)
        train_mask = targets["fold_id"].astype(int) != fold
        contains_test = bool((targets.loc[train_mask, "fold_id"].astype(int) == fold).any())
        fold_info[f"fold{fold}"] = {
            "train_n_patients": int(targets.loc[train_mask, "patient_id"].nunique()),
            "contains_heldout_fold": contains_test,
            "heldout_fold_n_patients": int(targets.loc[~train_mask, "patient_id"].nunique()),
        }
    informative = center_id_is_informative(targets)
    return {
        "n_rows": int(len(targets)), "n_patients": n_patients,
        "fold_ids_observed": observed_fold_ids(targets),
        "fold_id_base": min(observed_fold_ids(targets)),
        "fold_train_files": fold_info,
        "broad_centers": sorted(broad_centers), "rho": float(rho), "tau_q": float(tau_q),
        "broad_center_rule": "normalized_center_string",
        "center_id_is_informative": informative,
        "center_id_warning": "" if informative else "center_id is non-informative; use normalized center string.",
        "phys_core_mode": phys_core_mode,
        "phys_core_score_available": phys_audit.get("phys_core_score_available", False),
        "phys_core_reason": phys_audit.get("phys_core_reason", ""),
        "phys_core_match_key": phys_audit.get("phys_core_match_key", "subject_or_patient_id + normalized_channel_name"),
        "matched_feature_names": phys_audit.get("matched_feature_names", []),
        "phys_core_matched_teacher_rows": phys_audit.get("phys_core_matched_teacher_rows", 0),
        "phys_core_unmatched_teacher_rows": phys_audit.get("phys_core_unmatched_teacher_rows", 0),
        "feature_name_source_used": phys_audit.get("feature_name_source_used", ""),
        "physics_state_features_cli_len": phys_audit.get("physics_state_features_cli_len", 0),
        "physics_feature_slice": phys_audit.get("physics_feature_slice", ""),
        "phys_core_feature_set": phys_audit.get("phys_core_feature_set", ""),
        "window_feature_dim": phys_audit.get("window_feature_dim", 0),
        "teacher_only_fallback_used": bool(teacher_only_fallback_used),
    }


def split_fold_train_targets(targets, n_splits=None):
    del n_splits
    result = {}
    for fold in observed_fold_ids(targets):
        ft = targets[targets["fold_id"].astype(int) != fold].copy()
        if (ft["fold_id"].astype(int) == fold).any():
            raise ValueError(f"LEAK: fold{fold} train targets contain fold {fold} test patients!")
        result[fold] = ft
    return result


def _write_summaries(targets: pd.DataFrame, output_dir: Path, *, rho: float) -> None:
    patient_rows = []
    for (pid, center, fid), group in targets.groupby(["patient_id", "center", "fold_id"], sort=False, dropna=False):
        labels = group["label_ez"].astype(float).to_numpy()
        q_vals = group["pseudo_core_q"].astype(float).to_numpy()
        ez_q = q_vals[labels > 0.5]
        n_ez = int(round(float(labels.sum())))
        mass = max(1.0, float(rho) * float(n_ez))
        patient_rows.append({
            "patient_id": str(pid), "center": str(center), "fold_id": int(fid),
            "n_channels": int(len(group)), "n_ez": n_ez,
            "sum_q": float(q_vals.sum()), "target_mass": mass,
            "effective_core_fraction": float(q_vals.sum()) / max(float(labels.sum()), 1.0),
            "q_entropy": _entropy(ez_q / max(float(ez_q.sum()), 1e-12)) if len(ez_q) > 0 else 0.0,
            "q_max": float(ez_q.max()) if len(ez_q) > 0 else 0.0,
            "q_min_positive": float(ez_q.min()) if len(ez_q) > 0 else 0.0,
            "n_core_q_gt_05": int((ez_q > 0.5).sum()),
        })
    pd.DataFrame(patient_rows).to_csv(output_dir / "latent_core_patient_summary.csv", index=False)

    center_rows = []
    for center, cdf in targets.groupby("center", sort=True, dropna=False):
        labels = cdf["label_ez"].astype(float)
        q_vals = cdf["pseudo_core_q"].astype(float)
        ez_labels = labels > 0.5
        ez_q = q_vals[ez_labels]
        cdf_pat = cdf["patient_id"]
        center_rows.append({
            "center": str(center), "n_patients": cdf_pat.nunique(),
            "mean_n_ez": float(labels.groupby(cdf_pat).sum().mean()),
            "mean_sum_q": float(q_vals.groupby(cdf_pat).sum().mean()),
            "mean_effective_core_fraction": float(
                (q_vals.groupby(cdf_pat).sum() / labels.groupby(cdf_pat).sum().clip(lower=1)).mean()
            ),
            "mean_q_entropy": float(
                ez_q.groupby(cdf_pat[ez_labels]).apply(
                    lambda x: _entropy(x.to_numpy() / max(float(x.sum()), 1e-12))).mean()
            ) if len(ez_q) > 0 else 0.0,
            "mean_q_max": float(ez_q.groupby(cdf_pat[ez_labels]).max().mean()) if len(ez_q) > 0 else 0.0,
            "mean_n_core_q_gt_05": float(
                (ez_q > 0.5).astype(int).groupby(cdf_pat[ez_labels]).sum().mean()
            ) if len(ez_q) > 0 else 0.0,
        })
    pd.DataFrame(center_rows).to_csv(output_dir / "latent_core_center_summary.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build A9v8 latent pseudo-core targets.")
    parser.add_argument("--teacher_csv", type=str, required=True)
    parser.add_argument("--cache_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--broad_centers", type=str, default="lzu,pediatric")
    parser.add_argument("--rho", type=float, default=0.30)
    parser.add_argument("--tau_q", type=float, default=0.10)
    parser.add_argument("--phys_core_mode", type=str, default="auto_s5", choices=["auto_s5", "teacher_only"])
    parser.add_argument("--allow_teacher_only_core", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--require_phys_core", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--physics_state_features", type=str, default="")
    parser.add_argument("--physics_feature_slice", type=str, default="all")
    parser.add_argument("--feature_name_source", type=str, default="auto", choices=["auto", "cache", "cli"])
    parser.add_argument("--phys_core_feature_set", type=str, default="auto", choices=["auto", "exact_s5_12", "expanded_s5_16"])
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    broad_centers = {item.strip().lower() for item in args.broad_centers.split(",") if item.strip()}

    teacher = pd.read_csv(args.teacher_csv)
    targets, audit, feaudit_rows = build_latent_core_targets_dataframe(
        teacher, broad_centers=broad_centers, rho=args.rho, tau_q=args.tau_q,
        cache_path=args.cache_path,
        phys_core_mode=args.phys_core_mode,
        allow_teacher_only_core=bool(args.allow_teacher_only_core),
        require_phys_core=bool(args.require_phys_core),
        physics_state_features=args.physics_state_features,
        feature_name_source=args.feature_name_source,
        physics_feature_slice=args.physics_feature_slice,
        phys_core_feature_set=args.phys_core_feature_set,
    )

    targets.to_csv(output_dir / "latent_core_targets_all_oof.csv", index=False)
    for fold, fold_df in split_fold_train_targets(targets).items():
        fold_df.to_csv(output_dir / f"latent_core_targets_fold{fold}_train.csv", index=False)
    _write_summaries(targets, output_dir, rho=args.rho)

    # phys_core_feature_audit.csv
    feature_audit_cols = [
        "feature_index",
        "feature_name",
        "matched_group",
        "sign",
        "used",
        "reason",
        "feature_name_source",
        "feature_slice_start",
        "feature_slice_stop",
        "phys_core_feature_set",
    ]
    pd.DataFrame(feaudit_rows, columns=feature_audit_cols).to_csv(
        output_dir / "phys_core_feature_audit.csv", index=False
    )

    with open(output_dir / "latent_core_audit.json", "w", encoding="utf-8") as fout:
        json.dump(audit, fout, indent=2, ensure_ascii=False, sort_keys=True)

    print(f"Wrote {output_dir / 'latent_core_targets_all_oof.csv'}")
    print(f"phys_core_available={audit['phys_core_score_available']}, reason={audit['phys_core_reason']}")


if __name__ == "__main__":
    main()
